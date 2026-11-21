//! Shopping Sessions: native draft pipeline.
//!
//! Architecture (following the published designs of the top public entries,
//! written from scratch here; the DuckDB pipeline in ../pipeline is the
//! correctness reference):
//!
//!   1. mmap every input file; split each file's data (after its header line)
//!      into ~4 MB chunks on line boundaries, so work is split by bytes, not
//!      files (one file can hold a quarter of the rows).
//!   2. In parallel: build the person table from the links (one thread) while
//!      every other thread parses event chunks. Each event copy is parsed and
//!      scattered into one of 256 buckets by hash(event_id).
//!   3. Per event bucket (parallel): dedup in a hash table (smallest ingest_ts
//!      wins, BEFORE validation), validate the winner, map user -> person, and
//!      scatter by hash(person). Clicks are parsed, validated and scattered by
//!      hash(person) alongside. Clicks are NOT deduplicated: copies are
//!      identical, so they cannot change a credit (same ts, same click_id) or
//!      an assist (campaigns count once).
//!   4. Per person bucket (parallel): sort events by (person, ts) and clicks by
//!      (person, ts, click_id); walk each person once, producing sessions, the
//!      credited click, assists (a sliding 7-day window of campaign counts) and
//!      the user_daily / campaign_daily aggregates in the same pass.
//!   5. Write both Parquet tables and exit without teardown.

mod csv;
mod output;
mod people;
mod ts;

use memmap2::Mmap;
use output::{CampaignRow, UserRow};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DAY: i64 = 86_400_000;
const GAP: i64 = 30 * 60 * 1000; // a gap OVER this starts a new session
const LATE: i64 = DAY; // ingest_ts - ts OVER this drops the event
const LOOKBACK: i64 = 7 * DAY; // clicks credit/assist sessions up to 7 days later
// 1024 buckets measured best (256: ~6% slower; 2048 and 4096: slower, as each
// chunk then holds too many tiny per-bucket vectors).
const BUCKET_BITS: u32 = 10;
const NB: usize = 1 << BUCKET_BITS; // buckets for dedup and for persons
// Chunk size scales with the input: ~12 chunks per thread, 4..64 MB. 4, 8 and
// 16 MB measured equal at 10M events; fixed small chunks would multiply the
// per-bucket vectors (chunks x buckets) at 60M.
const CHUNK_MIN: usize = 4 << 20;
const CHUNK_MAX: usize = 64 << 20;
const BAD_TS: i64 = i64::MIN;

const VIEW: u8 = 0;
const CART: u8 = 1;
const PURCHASE: u8 = 2;
const BAD_TYPE: u8 = 255;

use people::{hash64, PKey};

#[inline]
fn bucket_of(h: u64) -> usize {
    (h >> (64 - BUCKET_BITS)) as usize
}


/// Exact id of a person within one bucket: open addressing on (hash, PKey).
/// Comparing PKeys is an integer compare (see people::PKey).
#[inline]
fn person_id(slots: &mut [u32], names: &mut Vec<(u64, PKey)>, ph: u64, person: PKey) -> u32 {
    let mask = slots.len() - 1;
    let mut i = (ph as usize) & mask;
    loop {
        let s = slots[i];
        if s == u32::MAX {
            let id = names.len() as u32;
            slots[i] = id;
            names.push((ph, person));
            return id;
        }
        let (h, n) = names[s as usize];
        if h == ph && n.same(person) {
            return s;
        }
        i = (i + 1) & mask;
    }
}

/// One parsed copy of an event line, before dedup. 40 bytes: instead of
/// keeping the event_id and user_id slices (32 bytes), it keeps where its line
/// starts in its chunk; both are re-read from the line in the rare places they
/// are needed (comparing two copies with equal hashes, resolving a winner's user).
#[derive(Clone, Copy)]
struct EvCopy {
    h: u64, // hash64(event_id)
    ts: i64,
    ingest: i64, // i64::MAX if bad (sorts last, like NULLS LAST)
    price: i32,  // 0 unless a valid purchase
    cat: u32,    // category for purchases, u32::MAX otherwise
    off: u32,    // line start within its chunk
    ok: bool,    // this copy passes every check (validated during parse)
}

const _: () = assert!(std::mem::size_of::<EvCopy>() == 40);

/// Strip the syntax quotes of an already validated field.
#[inline]
fn unq(f: &[u8]) -> &[u8] {
    if f.first() == Some(&b'"') { &f[1..f.len() - 1] } else { f }
}

/// event_id and user_id (empty if null) of a validated event line.
#[inline]
fn line_id_user(chunk: &[u8], off: u32) -> (&[u8], &[u8]) {
    let s = &chunk[off as usize..];
    let c1 = memchr::memchr(b',', s).unwrap();
    let rest = &s[c1 + 1..];
    let c2 = memchr::memchr(b',', rest).unwrap();
    (unq(&s[..c1]), unq(&rest[..c2]))
}

/// A surviving event, ready for sessions.
#[derive(Clone, Copy)]
struct Ev {
    ph: u64,  // hash64(person)
    pid: u32, // exact person id within its bucket (set in the sessions stage)
    person: PKey,
    ts: i64,
    price: i64, // 0 unless purchase
    cat: u32,   // category index for purchases, u32::MAX otherwise
}

/// A valid click.
#[derive(Clone, Copy)]
struct ClickRaw<'a> {
    user: &'a [u8],
    id: &'a [u8],
    ts: i64,
    camp: u32,
}

/// A valid click with its person.
#[derive(Clone, Copy)]
struct Click<'a> {
    ph: u64,  // hash64(person)
    pid: u32, // exact person id within its bucket (set in the sessions stage)
    person: PKey,
    id: &'a [u8],
    ts: i64,
    camp: u32,
}

/// Per-thread scratch for the dedup stage, reused across buckets. A copy is
/// referred to as (record, index of its chunk).
struct Scratch<'r, 'a> {
    ids: Vec<&'a [u8]>,
    persons: Vec<(PKey, u64)>,
    batch: Vec<(&'r EvCopy, u32)>,
}

/// Resolve a batch of validated winners to persons (prefetched lookups) and
/// scatter them into the person buckets.
fn emit_batch<'r, 'a>(
    batch: &mut Vec<(&'r EvCopy, u32)>,
    chunks: &[&'a [u8]],
    ids: &mut Vec<&'a [u8]>,
    persons: &mut Vec<(PKey, u64)>,
    people: &people::People<'a>,
    out: &mut [Vec<Ev>],
) {
    ids.clear();
    ids.extend(batch.iter().map(|&(e, c)| line_id_user(chunks[c as usize], e.off).1));
    persons.clear();
    people.lookup_many(ids, persons);
    for (&(e, _), &(person, ph)) in batch.iter().zip(persons.iter()) {
        out[bucket_of(ph)].push(Ev { ph, pid: 0, person, ts: e.ts, price: e.price as i64, cat: e.cat });
    }
    batch.clear();
}

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false))
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

fn map_file(p: &Path) -> Option<Mmap> {
    let f = File::open(p).expect("open input");
    if f.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
        return None; // zero-byte file: no header, no data
    }
    Some(unsafe { Mmap::map(&f).expect("mmap input") })
}

fn map_all(paths: &[PathBuf]) -> Vec<Mmap> {
    paths.iter().filter_map(|p| map_file(p)).collect()
}

/// All data chunks (header skipped) of a set of mapped files.
fn data_chunks<'a>(maps: &'a [Mmap], ncpu: usize) -> Vec<&'a [u8]> {
    let total: usize = maps.iter().map(|m| m.len()).sum();
    let size = (total / (ncpu.max(1) * 12)).clamp(CHUNK_MIN, CHUNK_MAX);
    maps.iter().flat_map(|m| csv::chunks(csv::data_region(&m[..]), size)).collect()
}

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let t0 = Instant::now();
    let debug = std::env::var("DEBUG").map(|v| v == "1").unwrap_or(false);
    let step = |name: &str| {
        if debug {
            eprintln!("[{:7.3}s] {name}", t0.elapsed().as_secs_f64());
        }
    };
    let data = PathBuf::from(env("DATA_DIR", "/data"));
    let out = PathBuf::from(env("OUTPUT_DIR", "/output"));
    let ncpu: usize = env("NCPU", "8").parse().unwrap_or(8);
    rayon::ThreadPoolBuilder::new().num_threads(ncpu.max(1)).build_global().ok();

    // --- Manifest and the small lookup tables ------------------------------
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(data.join("manifest.json")).expect("manifest")).expect("manifest json");
    let win_start = ts::parse_ts(manifest["window_start"].as_str().unwrap().as_bytes()).expect("window_start");
    let win_end = ts::parse_ts(manifest["window_end"].as_str().unwrap().as_bytes()).expect("window_end");
    let click_start = win_start - LOOKBACK;

    let products_map = map_file(&data.join("products.csv"));
    let campaigns_map = map_file(&data.join("campaigns.csv"));
    let event_maps = map_all(&csv_files(&data.join("events")));
    let click_maps = map_all(&csv_files(&data.join("clicks")));
    let link_maps = map_all(&csv_files(&data.join("links")));

    // Categories: index per distinct name, plus "unknown" (a name no category
    // in the file uses) for products not in the file.
    let mut categories: Vec<&[u8]> = Vec::new();
    let mut cat_index: FxHashMap<&[u8], u32> = FxHashMap::default();
    let mut product_cat: FxHashMap<&[u8], u32> = FxHashMap::default();
    if let Some(m) = &products_map {
        csv::for_each_line(csv::data_region(&m[..]), |line| {
            if let Some([Some(p), c]) = csv::fields::<2>(line) {
                let c = c.unwrap_or(b"");
                let n = categories.len() as u32;
                let ci = *cat_index.entry(c).or_insert_with(|| {
                    categories.push(c);
                    n
                });
                product_cat.insert(p, ci);
            }
        });
    }
    let unknown_cat = categories.len() as u32;
    categories.push(b"unknown");

    let mut camp_ids: Vec<&[u8]> = Vec::new();
    let mut camp_channel: Vec<&[u8]> = Vec::new();
    let mut camp_index: FxHashMap<&[u8], u32> = FxHashMap::default();
    if let Some(m) = &campaigns_map {
        csv::for_each_line(csv::data_region(&m[..]), |line| {
            if let Some([Some(c), ch]) = csv::fields::<2>(line) {
                if !camp_index.contains_key(c) {
                    camp_index.insert(c, camp_ids.len() as u32);
                    camp_ids.push(c);
                    camp_channel.push(ch.unwrap_or(b""));
                }
            }
        });
    }
    let organic = camp_ids.len() as u32;
    camp_ids.push(b"organic");
    camp_channel.push(b"organic");
    step("lookups");

    // --- Links (one thread) || event parse + scatter by event_id -----------
    let link_regions: Vec<&[u8]> = link_maps.iter().map(|m| csv::data_region(&m[..])).collect();
    let event_chunks = data_chunks(&event_maps, ncpu);
    let click_chunks = data_chunks(&click_maps, ncpu);
    // Clicks are parsed and validated in this phase too: links run on one
    // thread and are the critical path, so the other threads would otherwise
    // go idle. Only the person lookup has to wait for the links.
    let min_len = (NB / (4 * ncpu.max(1))).max(1);
    // Phase A. Links run on one thread and were the critical path, so
    // everything that does not need people runs beside them: parsing events and
    // clicks, and choosing each event's earliest copy (dedup). Only resolving
    // users to persons waits for the links (phase B).
    #[allow(clippy::type_complexity)]
    let (people, ((ev_parts, click_raw), winners)): (
        people::People,
        ((Vec<Vec<Vec<EvCopy>>>, Vec<Vec<ClickRaw>>), Vec<Vec<(u32, u32)>>),
    ) = rayon::join(
        || {
            let p = people::build(&link_regions, ncpu);
            if debug {
                eprintln!("  links done at {:.3}s", t0.elapsed().as_secs_f64());
            }
            p
        },
        || {
            let parsed: (Vec<Vec<Vec<EvCopy>>>, Vec<Vec<ClickRaw>>) = rayon::join(
                || {
                event_chunks
                    .par_iter()
                    .map(|chunk| {
                        // ~77 bytes per event line; pre-size so buckets never grow.
                        let per = chunk.len() / 77 / NB + chunk.len() / 77 / NB / 4 + 8;
                        let mut parts: Vec<Vec<EvCopy>> = (0..NB).map(|_| Vec::with_capacity(per)).collect();
                        csv::for_each_record::<7>(chunk, |r| {
                            let [Some(id), user, ts_s, ing_s, ty, product, price_s] = *r else {
                                return;
                            };
                            // Every check except dedup is a property of the copy
                            // itself, so it is decided here; dedup still picks the
                            // earliest copy whether or not it is ok.
                            let etype = match ty {
                                Some(b"view") => VIEW,
                                Some(b"cart") => CART,
                                Some(b"purchase") => PURCHASE,
                                _ => BAD_TYPE,
                            };
                            let ts = ts_s.and_then(ts::parse_ts).unwrap_or(BAD_TS);
                            let ingest = ing_s.and_then(ts::parse_ts).unwrap_or(i64::MAX);
                            let mut ok = user.is_some()
                                && product.is_some()
                                && ts != BAD_TS
                                && ts >= win_start
                                && ts < win_end
                                && etype != BAD_TYPE
                                && ingest != i64::MAX
                                && ingest - ts <= LATE;
                            let (mut price, mut cat) = (0i32, u32::MAX);
                            if ok && etype == PURCHASE {
                                match price_s.and_then(ts::parse_price) {
                                    Some(p) => {
                                        price = p as i32;
                                        cat = *product_cat.get(product.unwrap()).unwrap_or(&unknown_cat);
                                    }
                                    None => ok = false,
                                }
                            }
                            // The line starts at the event_id field (one byte
                            // earlier if that field is quoted).
                            let mut off = id.as_ptr() as usize - chunk.as_ptr() as usize;
                            if off > 0 && chunk[off - 1] == b'"' {
                                off -= 1;
                            }
                            let h = hash64(id);
                            parts[bucket_of(h)].push(EvCopy { h, ts, ingest, price, cat, off: off as u32, ok });
                        });
                        parts
                    })
                    .collect()
                },
                || {
                    click_chunks
                        .par_iter()
                        .map(|chunk| {
                            let mut v: Vec<ClickRaw> = Vec::with_capacity(chunk.len() / 47 + 16);
                            csv::for_each_record::<4>(chunk, |r| {
                                let [Some(id), Some(user), Some(camp), Some(ts_s)] = *r else {
                                    return;
                                };
                                let Some(&camp) = camp_index.get(camp) else { return };
                                let Some(t) = ts::parse_ts(ts_s) else { return };
                                if t < click_start || t >= win_end {
                                    return;
                                }
                                v.push(ClickRaw { user, id, ts: t, camp });
                            });
                            v
                        })
                        .collect()
                },
            );
            let (ev_parts, _) = &parsed;
            // Dedup: per event bucket, open addressing on the precomputed hash
            // (low bits = slot). Keeps (chunk, index) of each event's earliest
            // copy, then keeps only winners that passed validation.
            let winners: Vec<Vec<(u32, u32)>> = (0..NB)
                .into_par_iter()
                .with_min_len(min_len)
                .map_init(
                    || (Vec::<u32>::new(), Vec::<(u32, u32)>::new()),
                    |(slots, best), b| {
                        let n: usize = ev_parts.iter().map(|p| p[b].len()).sum();
                        let cap = (n * 2).next_power_of_two().max(16);
                        let mask = cap - 1;
                        slots.clear();
                        slots.resize(cap, u32::MAX);
                        best.clear();
                        for (c, part) in ev_parts.iter().enumerate() {
                            for (k, e) in part[b].iter().enumerate() {
                                let mut i = (e.h as usize) & mask;
                                loop {
                                    let s = slots[i];
                                    if s == u32::MAX {
                                        slots[i] = best.len() as u32;
                                        best.push((c as u32, k as u32));
                                        break;
                                    }
                                    let (wc, wk) = best[s as usize];
                                    let w = &ev_parts[wc as usize][b][wk as usize];
                                    // Equal hashes are almost always the same
                                    // event_id; confirm on the actual bytes.
                                    if w.h == e.h
                                        && line_id_user(event_chunks[wc as usize], w.off).0
                                            == line_id_user(event_chunks[c], e.off).0
                                    {
                                        if e.ingest < w.ingest {
                                            best[s as usize] = (c as u32, k as u32);
                                        }
                                        break;
                                    }
                                    i = (i + 1) & mask;
                                }
                            }
                        }
                        best.iter().copied().filter(|&(c, k)| ev_parts[c as usize][b][k as usize].ok).collect()
                    },
                )
                .collect();
            if debug {
                eprintln!("  dedup select done at {:.3}s", t0.elapsed().as_secs_f64());
            }
            (parsed, winners)
        },
    );
    step(&format!("phase A: links ({} linked ids) || parse + dedup", people.len()));

    // --- Dedup + validate events || parse + validate clicks ----------------
    let (ev_by_person, click_parts): (Vec<Vec<Vec<Ev>>>, Vec<Vec<Vec<Click>>>) = rayon::join(
        || {
            // Phase B, events: resolve each winner's person and scatter by
            // person. Scratch is reused per worker thread (fresh memory costs a
            // page fault per 4 KB).
            let r = (0..NB)
                .into_par_iter()
                // At least NB/(4*ncpu) buckets per job, so per-thread scratch
                // is created ~4 times per thread, not once per bucket.
                .with_min_len(min_len)
                .map_init(
                    || Scratch { ids: Vec::new(), persons: Vec::new(), batch: Vec::new() },
                    |sc, b| {
                        // Pre-sized so the person buckets never grow and copy.
                        let w = &winners[b];
                        let per = w.len() / NB + w.len() / (NB * 4) + 8;
                        let mut out: Vec<Vec<Ev>> = (0..NB).map(|_| Vec::with_capacity(per)).collect();
                        // Resolve users 256 at a time with prefetching.
                        sc.batch.clear();
                        for &(c, k) in w {
                            sc.batch.push((&ev_parts[c as usize][b][k as usize], c));
                            if sc.batch.len() == 256 {
                                emit_batch(&mut sc.batch, &event_chunks, &mut sc.ids, &mut sc.persons, &people, &mut out);
                            }
                        }
                        emit_batch(&mut sc.batch, &event_chunks, &mut sc.ids, &mut sc.persons, &people, &mut out);
                        out
                    },
                )
                .collect();
            if debug {
                eprintln!("  event persons done at {:.3}s", t0.elapsed().as_secs_f64());
            }
            r
        },
        || {
            let r = click_raw
                .par_iter()
                .map(|raw| {
                    let per = raw.len() / NB + raw.len() / NB / 4 + 8;
                    let mut parts: Vec<Vec<Click>> = (0..NB).map(|_| Vec::with_capacity(per)).collect();
                    let mut ids: Vec<&[u8]> = Vec::with_capacity(256);
                    let mut persons = Vec::with_capacity(256);
                    for group in raw.chunks(256) {
                        ids.clear();
                        ids.extend(group.iter().map(|c| c.user));
                        persons.clear();
                        people.lookup_many(&ids, &mut persons);
                        for (c, &(person, ph)) in group.iter().zip(&persons) {
                            parts[bucket_of(ph)].push(Click { ph, pid: 0, person, id: c.id, ts: c.ts, camp: c.camp });
                        }
                    }
                    parts
                })
                .collect();
            if debug {
                eprintln!("  clicks arm done at {:.3}s", t0.elapsed().as_secs_f64());
            }
            r
        },
    );
    // Never free the parsed copies: freeing ~60k vectors cost 2.5 s on 10M
    // events (Windows heap), and the process exits without teardown anyway.
    std::mem::forget(ev_parts);
    std::mem::forget(click_raw);
    step("phase B: persons for events + clicks");

    // --- Per person bucket: sessions, credit, assists, daily aggregates -----
    // Campaign totals go into a dense [campaign][day] array, not a hash map:
    // ~19M assist increments at 10M events made the map the hot spot. Every
    // session starts inside the window, so its day is in [day0, day0 + ndays).
    let day0 = win_start.div_euclid(DAY);
    let ndays = ((win_end - 1).div_euclid(DAY) - day0 + 1) as usize;
    let ncamp = camp_ids.len(); // includes organic
    struct Acc<'a> {
        rows: Vec<UserRow<'a>>,
        camp: Vec<[i64; 4]>, // [campaign * ndays + day]: sessions, purchasing, revenue, assisted
        active_n: Vec<u32>,  // per campaign: clicks in the current 7-day window
        active_pos: Vec<u32>,
        active: Vec<u32>, // campaigns with active_n > 0
        // Buffers reused across buckets (fresh memory costs page faults).
        ev: Vec<Ev>,
        cl: Vec<Click<'a>>,
        slots: Vec<u32>,
        names: Vec<(u64, PKey)>,
    }
    let new_acc = || Acc {
        rows: Vec::new(),
        camp: vec![[0; 4]; ncamp * ndays],
        active_n: vec![0; ncamp],
        active_pos: vec![0; ncamp],
        active: Vec::new(),
        ev: Vec::new(),
        cl: Vec::new(),
        slots: Vec::new(),
        names: Vec::new(),
    };
    let accs: Vec<Acc> = (0..NB)
        .into_par_iter()
        .with_min_len(min_len) // few accumulators: each holds a 900 KB campaign array
        .fold(new_acc, |mut acc, p| {
            let mut ev = std::mem::take(&mut acc.ev);
            let mut cl = std::mem::take(&mut acc.cl);
            ev.clear();
            cl.clear();
            for x in &ev_by_person {
                ev.extend_from_slice(&x[p]);
            }
            for x in &click_parts {
                cl.extend_from_slice(&x[p]);
            }
            // Exact person ids for this bucket, so sorting compares integers.
            let cap = ((ev.len() + cl.len()) * 2).next_power_of_two().max(16);
            acc.slots.clear();
            acc.slots.resize(cap, u32::MAX);
            acc.names.clear();
            for e in ev.iter_mut() {
                e.pid = person_id(&mut acc.slots, &mut acc.names, e.ph, e.person);
            }
            for c in cl.iter_mut() {
                c.pid = person_id(&mut acc.slots, &mut acc.names, c.ph, c.person);
            }
            ev.sort_unstable_by_key(|e| (e.pid, e.ts));
            cl.sort_unstable_by(|a, b| (a.pid, a.ts).cmp(&(b.pid, b.ts)).then_with(|| a.id.cmp(b.id)));

            let mut cats: Vec<(u32, i64)> = Vec::new();
            let mut ci = 0usize; // click cursor (persons are visited in pid order)
            let mut i = 0usize;
            while i < ev.len() {
                let (pid, person) = (ev[i].pid, ev[i].person);
                let mut j = i;
                while j < ev.len() && ev[j].pid == pid {
                    j += 1;
                }
                while ci < cl.len() && cl[ci].pid < pid {
                    ci += 1;
                }
                let c0 = ci;
                while ci < cl.len() && cl[ci].pid == pid {
                    ci += 1;
                }
                let clicks = &cl[c0..ci];
                for &c in &acc.active {
                    acc.active_n[c as usize] = 0;
                }
                acc.active.clear();
                let (mut lo, mut hi) = (0usize, 0usize);

                // Running totals of the current (person, day) row.
                let (mut d_sessions, mut d_events, mut d_rev) = (0i64, 0i64, 0i64);
                cats.clear();
                let mut k = i;
                while k < j {
                    // A session: events from k while the gap stays <= GAP
                    // (equal timestamps have gap 0, so they always share it).
                    let start = ev[k].ts;
                    let mut m = k + 1;
                    while m < j && ev[m].ts - ev[m - 1].ts <= GAP {
                        m += 1;
                    }
                    let day = start.div_euclid(DAY);
                    let di = (day - day0) as usize;
                    let mut rev = 0i64;
                    let mut purchase = false;
                    for e in &ev[k..m] {
                        if e.cat != u32::MAX {
                            purchase = true;
                            rev += e.price;
                            match cats.iter_mut().find(|(c, _)| *c == e.cat) {
                                Some((_, s)) => *s += e.price,
                                None => cats.push((e.cat, e.price)),
                            }
                        }
                    }
                    // The day row is flushed after the last session of each
                    // day (below), so everything added here is this day's.
                    d_sessions += 1;
                    d_events += (m - k) as i64;
                    d_rev += rev;

                    // Assist window: clicks with start - 7d <= ts <= start.
                    while hi < clicks.len() && clicks[hi].ts <= start {
                        let c = clicks[hi].camp as usize;
                        if acc.active_n[c] == 0 {
                            acc.active_pos[c] = acc.active.len() as u32;
                            acc.active.push(c as u32);
                        }
                        acc.active_n[c] += 1;
                        hi += 1;
                    }
                    while lo < hi && clicks[lo].ts < start - LOOKBACK {
                        let c = clicks[lo].camp as usize;
                        acc.active_n[c] -= 1;
                        if acc.active_n[c] == 0 {
                            let pos = acc.active_pos[c] as usize;
                            acc.active.swap_remove(pos);
                            if pos < acc.active.len() {
                                let moved = acc.active[pos] as usize;
                                acc.active_pos[moved] = pos as u32;
                            }
                        }
                        lo += 1;
                    }
                    // Credit: latest click in the window; ties in ts -> smallest
                    // click_id, which is the first of its ts group (sorted).
                    let credited = if hi > lo {
                        let mut q = hi - 1;
                        let t = clicks[q].ts;
                        while q > lo && clicks[q - 1].ts == t {
                            q -= 1;
                        }
                        clicks[q].camp
                    } else {
                        organic
                    };
                    let a = &mut acc.camp[credited as usize * ndays + di];
                    a[0] += 1;
                    a[1] += purchase as i64;
                    a[2] += rev;
                    for &c in &acc.active {
                        acc.camp[c as usize * ndays + di][3] += 1;
                    }
                    k = m;
                    let next_day = if k < j { Some(ev[k].ts.div_euclid(DAY)) } else { None };
                    if next_day != Some(day) {
                        let top = cats
                            .iter()
                            .max_by(|a, b| a.1.cmp(&b.1).then(categories[b.0 as usize].cmp(categories[a.0 as usize])))
                            .map(|&(c, _)| categories[c as usize]);
                        acc.rows.push(UserRow {
                            user_id: person,
                            day: day as i32,
                            sessions: d_sessions,
                            events: d_events,
                            revenue_cents: d_rev,
                            top_category: top,
                        });
                        d_sessions = 0;
                        d_events = 0;
                        d_rev = 0;
                        cats.clear();
                    }
                }
                i = j;
            }
            acc.ev = ev;
            acc.cl = cl;
            acc
        })
        .collect();
    step("sessions");

    // --- Outputs -------------------------------------------------------------
    let mut camp_all: Vec<[i64; 4]> = vec![[0; 4]; ncamp * ndays];
    for acc in &accs {
        for (t, v) in camp_all.iter_mut().zip(&acc.camp) {
            for x in 0..4 {
                t[x] += v[x];
            }
        }
    }
    // One row per (campaign, day) that was credited or assisted at least once.
    let camp_rows: Vec<CampaignRow> = camp_all
        .iter()
        .enumerate()
        .filter(|(_, a)| a[0] > 0 || a[3] > 0)
        .map(|(idx, a)| {
            let c = idx / ndays;
            CampaignRow {
                campaign_id: camp_ids[c],
                channel: camp_channel[c],
                day: (day0 + (idx % ndays) as i64) as i32,
                sessions: a[0],
                purchasing_sessions: a[1],
                revenue_cents: a[2],
                assisted_sessions: a[3],
            }
        })
        .collect();
    // user_daily: one Parquet file per accumulator, written in parallel (the
    // checker reads every *.parquet in the folder; file count does not matter).
    rayon::join(
        || {
            accs.par_iter()
                .enumerate()
                .filter(|(_, a)| !a.rows.is_empty())
                .for_each(|(i, a)| output::write_user_daily(&out, i, &a.rows));
            if accs.iter().all(|a| a.rows.is_empty()) {
                output::write_user_daily(&out, 0, &[]); // an empty table still needs its schema
            }
        },
        || output::write_campaign_daily(&out, &camp_rows),
    );
    step("outputs written");
    // Files are closed. Skip freeing gigabytes of memory: exit now.
    std::process::exit(0);
}
