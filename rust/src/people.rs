//! Linked IDs -> people.
//!
//! Every valid link line joins two IDs; joining is transitive, so a person is a
//! connected component. Each person is named by the byte-wise smallest ID in it
//! (`&[u8]` ordering is byte-wise). IDs in no link are their own person and are
//! not stored: `lookup` falls back to the ID itself.
//!
//! Lookup is the hot path (one per surviving event and click), so linked IDs
//! of up to 15 bytes (every generated ID is at most 9) are packed EXACTLY into
//! a u128 (the bytes plus their length) and kept in a flat open-addressing
//! table: a probe compares integers and never touches the link files. The
//! person's name and its hash64 are stored with it, so callers get both.
//! Longer IDs use an exact byte-keyed map.

use crate::csv;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::Hasher;

/// One 64-bit hash per key: Fx over the bytes, then a full avalanche (murmur3
/// finalizer) so the top bits (buckets) and low bits (table slots) are
/// independent.
#[inline]
pub fn hash64(key: &[u8]) -> u64 {
    let mut h = FxHasher::default();
    h.write(key);
    fmix64(h.finish() ^ (key.len() as u64))
}

#[inline]
fn fmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// Exact packing of a short ID: bytes 0..len, length in the top byte. An empty
/// ID never reaches here, so a packed key is never 0 (0 marks an empty slot).
#[inline]
fn pack(id: &[u8]) -> Option<u128> {
    if id.is_empty() || id.len() > 15 {
        return None;
    }
    let mut b = [0u8; 16];
    b[..id.len()].copy_from_slice(id);
    b[15] = id.len() as u8;
    Some(u128::from_le_bytes(b))
}

#[inline(always)]
fn prefetch<T>(r: &T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(r as *const T as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

#[inline]
fn slot_hash(k: u128) -> u64 {
    fmix64((k as u64) ^ ((k >> 64) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

/// A person (or any ID) as one exact 128-bit value, so comparing two persons
/// is an integer compare that never reads the ID's bytes from the input
/// (those reads were cache misses: an unlinked person's name lives in each of
/// its event lines).
///   short (1..=15 bytes, every generated ID): the bytes, length in byte 15.
///   long  (> 15 bytes): byte 15 is 0; low 64 bits = pointer into the mapped
///         input, bits 64..120 = length. Compared by bytes (rare).
/// The pointer is valid for the whole run: the input maps are never unmapped
/// (the process exits without teardown).
#[derive(Clone, Copy)]
pub struct PKey(u128);

#[cfg(not(target_endian = "little"))]
compile_error!("PKey::bytes reads the packed bytes in little-endian order");

impl PKey {
    #[inline]
    pub fn new(id: &[u8]) -> PKey {
        match pack(id) {
            Some(k) => PKey(k),
            None => PKey((id.as_ptr() as usize as u128) | ((id.len() as u128) << 64)),
        }
    }
    #[inline]
    fn is_long(self) -> bool {
        (self.0 >> 120) == 0
    }
    /// The ID's bytes.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        if self.is_long() {
            let ptr = (self.0 as u64) as usize as *const u8;
            let len = ((self.0 >> 64) as u64 & ((1 << 56) - 1)) as usize;
            unsafe { std::slice::from_raw_parts(ptr, len) }
        } else {
            let len = (self.0 >> 120) as usize;
            let all: &[u8; 16] = unsafe { &*(self as *const PKey as *const [u8; 16]) };
            &all[..len]
        }
    }
    /// Exact equality of the IDs.
    #[inline]
    pub fn same(self, o: PKey) -> bool {
        self.0 == o.0 || (self.is_long() && o.is_long() && self.bytes() == o.bytes())
    }
}

/// The lookup table is split into shards by the top bits of the slot hash, so
/// the build can fill every shard in parallel; a lookup just picks its shard.
const SHARD_BITS: u32 = 8;
const NS: usize = 1 << SHARD_BITS;

#[inline]
fn shard_of(h: u64) -> usize {
    (h >> (64 - SHARD_BITS)) as usize
}

struct Shard {
    slots: Vec<(u128, u32)>, // packed linked ID -> rank (index into names)
    mask: usize,
}

pub struct People<'a> {
    shards: Vec<Shard>,
    long: FxHashMap<&'a [u8], u32>, // linked IDs longer than 15 bytes -> rank
    names: Vec<(PKey, u64)>,        // by rank: person name and hash64(name)
}

impl<'a> People<'a> {
    /// The person an ID belongs to, and hash64 of that person's name.
    #[inline]
    pub fn lookup(&self, id: &'a [u8]) -> (PKey, u64) {
        match pack(id) {
            Some(k) => {
                let h = slot_hash(k);
                let sh = &self.shards[shard_of(h)];
                let mut i = (h as usize) & sh.mask;
                loop {
                    let (sk, v) = sh.slots[i];
                    if sk == k {
                        return self.names[v as usize];
                    }
                    if sk == 0 {
                        break;
                    }
                    i = (i + 1) & sh.mask;
                }
            }
            None => {
                if let Some(&v) = self.long.get(id) {
                    return self.names[v as usize];
                }
            }
        }
        (PKey::new(id), hash64(id))
    }

    /// Look up many IDs at once. The table is far bigger than the CPU caches,
    /// so each lookup is a memory wait; computing a batch of slots and
    /// prefetching them all first lets those waits overlap.
    pub fn lookup_many(&self, ids: &[&'a [u8]], out: &mut Vec<(PKey, u64)>) {
        const BATCH: usize = 16;
        let mut keys = [None::<u128>; BATCH];
        let mut at = [(0usize, 0usize); BATCH]; // (shard, slot)
        for group in ids.chunks(BATCH) {
            for (g, id) in group.iter().enumerate() {
                keys[g] = pack(id);
                if let Some(k) = keys[g] {
                    let h = slot_hash(k);
                    let s = shard_of(h);
                    at[g] = (s, (h as usize) & self.shards[s].mask);
                    prefetch(&self.shards[s].slots[at[g].1]);
                }
            }
            for (g, id) in group.iter().enumerate() {
                out.push(match keys[g] {
                    Some(k) => {
                        let sh = &self.shards[at[g].0];
                        let mut i = at[g].1;
                        loop {
                            let (sk, v) = sh.slots[i];
                            if sk == k {
                                break self.names[v as usize];
                            }
                            if sk == 0 {
                                break (PKey::new(id), hash64(id));
                            }
                            i = (i + 1) & sh.mask;
                        }
                    }
                    None => self.lookup(id),
                });
            }
        }
    }

    #[cfg(test)]
    pub fn person_of(&self, id: &'a [u8]) -> Vec<u8> {
        self.lookup(id).0.bytes().to_vec()
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    // Path halving: every visited node skips to its grandparent.
    while parent[x as usize] != x {
        let gp = parent[parent[x as usize] as usize];
        parent[x as usize] = gp;
        x = gp;
    }
    x
}

/// The first 16 bytes of an ID, zero-padded, as a big-endian integer: integer
/// order equals byte order on those bytes. Ties (equal prefixes) are broken by
/// comparing the full bytes, which makes the order exactly byte-wise.
#[inline]
fn order_prefix(id: &[u8]) -> u128 {
    let mut b = [0u8; 16];
    let n = id.len().min(16);
    b[..n].copy_from_slice(&id[..n]);
    u128::from_be_bytes(b)
}

/// One end of a link line, routed to the shard that interns its ID.
#[derive(Clone, Copy)]
struct End<'a> {
    key: u128, // packed short ID (0 for a long ID)
    id: &'a [u8],
    pos: u32, // 2 * line index + side
}

/// Build the person table from the link files' data regions, in parallel:
///   1. parse link lines (split by bytes) into pairs;
///   2. intern the IDs in NS shards at once (each shard's table is small);
///   3. number the distinct IDs in byte order (parallel sort), so "byte-wise
///      smallest ID" is "smallest number";
///   4. union-find on those numbers, always keeping the smaller root: each
///      root is then its component's smallest ID, i.e. the person's name.
///      This is the only sequential part and touches only integers;
///   5. names and hashes per number, and the shard tables rewritten to map
///      packed IDs to numbers, in parallel.
/// The single-threaded version took 5 s at 60M events (3.6M linked IDs).
pub fn build<'a>(regions: &[&'a [u8]], ncpu: usize) -> People<'a> {
    // 1. Parse.
    let total: usize = regions.iter().map(|r| r.len()).sum();
    let size = (total / (ncpu.max(1) * 8)).clamp(1 << 20, 16 << 20);
    let chunks: Vec<&'a [u8]> = regions.iter().flat_map(|r| csv::chunks(r, size)).collect();
    let pairs: Vec<Vec<(&'a [u8], &'a [u8])>> = chunks
        .par_iter()
        .map(|c| {
            let mut v = Vec::with_capacity(c.len() / 20 + 8);
            // Drop malformed lines and lines where either side is empty.
            csv::for_each_record::<2>(c, |r| {
                if let [Some(a), Some(b)] = *r {
                    v.push((a, b));
                }
            });
            v
        })
        .collect();
    let mut base = Vec::with_capacity(pairs.len() + 1);
    base.push(0usize);
    for v in &pairs {
        base.push(base.last().unwrap() + v.len());
    }
    let npairs = *base.last().unwrap();

    // 2. Route each end to its shard (long IDs to the extra shard NS), then
    //    intern per shard in parallel: distinct IDs get a shard-local index.
    let routed: Vec<Vec<Vec<End<'a>>>> = pairs
        .par_iter()
        .enumerate()
        .map(|(c, v)| {
            let mut out: Vec<Vec<End>> = (0..=NS).map(|_| Vec::new()).collect();
            for (j, &(a, b)) in v.iter().enumerate() {
                for (side, id) in [(0u32, a), (1u32, b)] {
                    let pos = (2 * (base[c] + j)) as u32 + side;
                    match pack(id) {
                        Some(k) => out[shard_of(slot_hash(k))].push(End { key: k, id, pos }),
                        None => out[NS].push(End { key: 0, id, pos }),
                    }
                }
            }
            out
        })
        .collect();
    struct ShardOut<'a> {
        slots: Vec<(u128, u32)>, // packed ID -> local index (short shards)
        long: FxHashMap<&'a [u8], u32>, // id -> local index (long shard)
        uniq: Vec<&'a [u8]>,     // local index -> ID
        ends: Vec<(u32, u32)>,   // (pos, local index)
    }
    let shard_out: Vec<ShardOut<'a>> = (0..=NS)
        .into_par_iter()
        .map(|s| {
            let n: usize = routed.iter().map(|r| r[s].len()).sum();
            let mut so = ShardOut { slots: Vec::new(), long: FxHashMap::default(), uniq: Vec::new(), ends: Vec::with_capacity(n) };
            if s < NS {
                let cap = (n * 2).next_power_of_two().max(16);
                let mask = cap - 1;
                so.slots = vec![(0, 0); cap];
                for r in &routed {
                    for e in &r[s] {
                        let mut i = (slot_hash(e.key) as usize) & mask;
                        let local = loop {
                            let (k, v) = so.slots[i];
                            if k == e.key {
                                break v;
                            }
                            if k == 0 {
                                let v = so.uniq.len() as u32;
                                so.slots[i] = (e.key, v);
                                so.uniq.push(e.id);
                                break v;
                            }
                            i = (i + 1) & mask;
                        };
                        so.ends.push((e.pos, local));
                    }
                }
            } else {
                for r in &routed {
                    for e in &r[s] {
                        let next = so.uniq.len() as u32;
                        let local = *so.long.entry(e.id).or_insert(next);
                        if local == next {
                            so.uniq.push(e.id);
                        }
                        so.ends.push((e.pos, local));
                    }
                }
            }
            so
        })
        .collect();

    // 3. Number all distinct IDs in byte order.
    let mut all: Vec<(u128, &'a [u8], u32, u32)> = Vec::new(); // (prefix, id, shard, local)
    for (s, so) in shard_out.iter().enumerate() {
        for (l, id) in so.uniq.iter().enumerate() {
            all.push((order_prefix(id), id, s as u32, l as u32));
        }
    }
    all.par_sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(y.1)));
    let n = all.len();
    let mut rank_of: Vec<Vec<u32>> = shard_out.iter().map(|so| vec![0; so.uniq.len()]).collect();
    for (r, &(_, _, s, l)) in all.iter().enumerate() {
        rank_of[s as usize][l as usize] = r as u32;
    }

    // 4. Union-find on numbers; the smaller root always wins.
    let mut ends = vec![0u32; 2 * npairs];
    for (s, so) in shard_out.iter().enumerate() {
        for &(pos, l) in &so.ends {
            ends[pos as usize] = rank_of[s][l as usize];
        }
    }
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for p in 0..npairs {
        let (ra, rb) = (find(&mut parent, ends[2 * p]), find(&mut parent, ends[2 * p + 1]));
        if ra < rb {
            parent[rb as usize] = ra;
        } else if rb < ra {
            parent[ra as usize] = rb;
        }
    }
    for i in 0..n as u32 {
        find(&mut parent, i); // full compression: parent[i] is now the root
    }

    // 5. Names and hashes by number; shard tables map packed IDs to numbers.
    let root_hash: Vec<u64> = (0..n).into_par_iter().map(|i| if parent[i] as usize == i { hash64(all[i].1) } else { 0 }).collect();
    let names: Vec<(PKey, u64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let r = parent[i] as usize;
            (PKey::new(all[r].1), root_hash[r])
        })
        .collect();
    let mut shards: Vec<Shard> = Vec::with_capacity(NS);
    let mut long = FxHashMap::default();
    let mut outs: Vec<(ShardOut<'a>, Vec<u32>)> = shard_out.into_iter().zip(rank_of).collect();
    if let Some((so, ranks)) = outs.pop() {
        // The extra shard holds the long IDs.
        for (id, l) in so.long {
            long.insert(id, ranks[l as usize]);
        }
    }
    shards.extend(outs.into_par_iter().map(|(mut so, ranks)| {
        for slot in so.slots.iter_mut() {
            if slot.0 != 0 {
                slot.1 = ranks[slot.1 as usize];
            }
        }
        if so.slots.is_empty() {
            so.slots = vec![(0, 0); 1];
        }
        let mask = so.slots.len() - 1;
        Shard { slots: so.slots, mask }
    }).collect::<Vec<_>>());
    People { shards, long, names }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components() {
        let data = b"c,b\nb,a\nx,y\nU1,u1\nz,z\n,q\nbad\"line,w\n";
        let p = build(&[&data[..]], 4);
        assert_eq!(p.person_of(b"c"), b"a".to_vec());
        assert_eq!(p.person_of(b"a"), b"a".to_vec());
        assert_eq!(p.person_of(b"y"), b"x".to_vec());
        assert_eq!(p.person_of(b"u1"), b"U1".to_vec()); // upper case sorts first
        assert_eq!(p.person_of(b"z"), b"z".to_vec());
        assert_eq!(p.person_of(b"q"), b"q".to_vec()); // empty side: line dropped, q unlinked
        assert_eq!(p.person_of(b"w"), b"w".to_vec());
        assert_eq!(p.person_of(b"nobody"), b"nobody".to_vec());
        // Long IDs (over 15 bytes) take the exact byte-keyed path.
        let data2 = b"a-very-long-identifier-1,a-very-long-identifier-0\nshort,a-very-long-identifier-1\n";
        let p2 = build(&[&data2[..]], 4);
        assert_eq!(p2.person_of(b"short"), b"a-very-long-identifier-0".to_vec());
        assert_eq!(p2.lookup(b"short").1, hash64(b"a-very-long-identifier-0"));
        assert_eq!(p2.lookup(b"unlinked").1, hash64(b"unlinked"));
        // PKey: short and long IDs round-trip and compare exactly.
        let long_a = b"0123456789abcdefXYZ".to_vec();
        let long_b = b"0123456789abcdefXYZ".to_vec();
        assert!(PKey::new(&long_a).same(PKey::new(&long_b)));
        assert_eq!(PKey::new(&long_a).bytes(), &long_a[..]);
        assert!(!PKey::new(b"u1").same(PKey::new(b"U1")));
        assert!(!PKey::new(b"u1").same(PKey::new(b"u1 ")));
        assert_eq!(PKey::new(b"u1\t").bytes(), b"u1\t");
        // Batched lookups agree with single lookups (more than one batch).
        let ids: Vec<&[u8]> = (0..40).map(|i| [&b"c"[..], b"a", b"y", b"zz", b"short"][i % 5]).collect();
        let mut got = Vec::new();
        p.lookup_many(&ids, &mut got);
        for (id, g) in ids.iter().zip(&got) {
            let w = p.lookup(id);
            assert!(g.0.same(w.0) && g.1 == w.1);
        }
    }


    /// The parallel build must name every ID exactly like a plain union-find
    /// that compares bytes: random graphs with short, long, upper-case and
    /// tab-containing IDs, self links, repeated links, long chains and several
    /// link files, checked over every ID (and some unlinked ones).
    #[test]
    fn parallel_build_matches_simple_union_find() {
        use std::collections::HashMap;
        fn reference(lines: &[(Vec<u8>, Vec<u8>)]) -> HashMap<Vec<u8>, Vec<u8>> {
            let mut parent: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
            fn root(p: &mut HashMap<Vec<u8>, Vec<u8>>, x: &[u8]) -> Vec<u8> {
                let mut r = x.to_vec();
                while let Some(q) = p.get(&r) {
                    if *q == r {
                        break;
                    }
                    r = q.clone();
                }
                r
            }
            for (a, b) in lines {
                parent.entry(a.clone()).or_insert_with(|| a.clone());
                parent.entry(b.clone()).or_insert_with(|| b.clone());
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                if ra < rb {
                    parent.insert(rb, ra);
                } else if rb < ra {
                    parent.insert(ra, rb);
                }
            }
            let keys: Vec<Vec<u8>> = parent.keys().cloned().collect();
            keys.into_iter().map(|k| { let r = root(&mut parent, &k); (k, r) }).collect()
        }
        let mut x: u64 = 0xdead_beef_1234_5678;
        let mut rnd = |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        for trial in 0..60 {
            let nid = 1 + rnd(400) as usize;
            let ids: Vec<Vec<u8>> = (0..nid)
                .map(|i| match rnd(10) {
                    0 => format!("a-very-long-identifier-{i}").into_bytes(), // > 15 bytes
                    1 => format!("U{i}").into_bytes(),
                    2 => format!("u{i}\t").into_bytes(),
                    3 => format!("0123456789abcde{i}").into_bytes(), // shares a 15-byte prefix
                    _ => format!("u{i}").into_bytes(),
                })
                .collect();
            let mut lines: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for _ in 0..rnd(600) {
                let a = &ids[rnd(nid as u64) as usize];
                let b = if rnd(20) == 0 { a } else { &ids[rnd(nid as u64) as usize] };
                lines.push((a.clone(), b.clone()));
            }
            if trial % 3 == 0 {
                for w in ids.windows(2) {
                    lines.push((w[1].clone(), w[0].clone())); // one long chain
                }
            }
            // Split across several "files".
            let mut files: Vec<Vec<u8>> = vec![Vec::new(); 1 + rnd(4) as usize];
            for (k, (a, b)) in lines.iter().enumerate() {
                let f = k % files.len();
                files[f].extend_from_slice(a);
                files[f].push(b',');
                files[f].extend_from_slice(b);
                files[f].push(b'\n');
            }
            let regions: Vec<&[u8]> = files.iter().map(|f| &f[..]).collect();
            let p = build(&regions, 1 + rnd(8) as usize);
            let want = reference(&lines);
            for (id, name) in &want {
                assert_eq!(&p.person_of(id), name, "trial {trial} id {:?}", String::from_utf8_lossy(id));
                assert_eq!(p.lookup(id).1, hash64(name));
            }
            assert_eq!(p.person_of(b"never-linked"), b"never-linked".to_vec());
            assert_eq!(p.len(), want.len());
        }
    }
}
