//! A minimal Parquet writer for the two output tables (no arrow/parquet crates).
//!
//! What it writes, per the Parquet format spec:
//!   "PAR1" | row groups | FileMetaData (Thrift compact) | footer length | "PAR1"
//! Each row group has one uncompressed v1 data page per column, PLAIN encoded.
//! Strings are BYTE_ARRAY + converted type UTF8, `day` is INT32 + converted
//! type DATE (days since 1970-01-01), counts are INT64. The one nullable column
//! (top_category) carries definition levels: RLE runs of bit width 1.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

const ROW_GROUP: usize = 1 << 20;

pub struct UserRow<'a> {
    pub user_id: crate::people::PKey,
    pub day: i32,
    pub sessions: i64,
    pub events: i64,
    pub revenue_cents: i64,
    pub top_category: Option<&'a [u8]>,
}

pub struct CampaignRow<'a> {
    pub campaign_id: &'a [u8],
    pub channel: &'a [u8],
    pub day: i32,
    pub sessions: i64,
    pub purchasing_sessions: i64,
    pub revenue_cents: i64,
    pub assisted_sessions: i64,
}

// Parquet enums.
const T_INT32: i32 = 1;
const T_INT64: i32 = 2;
const T_BYTE_ARRAY: i32 = 6;
const REQUIRED: i32 = 0;
const OPTIONAL: i32 = 1;
const CONV_UTF8: i32 = 0;
const CONV_DATE: i32 = 6;
const ENC_PLAIN: i32 = 0;
const ENC_RLE: i32 = 3;

/// Thrift compact protocol, just the parts the Parquet footer needs.
struct Thrift {
    b: Vec<u8>,
    last: i16,
    stack: Vec<i16>,
}

impl Thrift {
    fn new() -> Self {
        Thrift { b: Vec::new(), last: 0, stack: Vec::new() }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.b.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.b.push(v as u8);
    }
    fn field(&mut self, id: i16, ty: u8) {
        let d = id - self.last;
        if d > 0 && d <= 15 {
            self.b.push(((d as u8) << 4) | ty);
        } else {
            self.b.push(ty);
            self.varint(((id << 1) ^ (id >> 15)) as u16 as u64);
        }
        self.last = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, 5);
        self.varint(((v << 1) ^ (v >> 31)) as u32 as u64);
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, 6);
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn bin(&mut self, id: i16, s: &[u8]) {
        self.field(id, 8);
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s);
    }
    fn list(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, 9);
        if n < 15 {
            self.b.push(((n as u8) << 4) | elem);
        } else {
            self.b.push(0xF0 | elem);
            self.varint(n as u64);
        }
    }
    fn list_i32(&mut self, v: i32) {
        self.varint(((v << 1) ^ (v >> 31)) as u32 as u64);
    }
    fn list_bin(&mut self, s: &[u8]) {
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s);
    }
    /// Begin a struct: a field (`Some(id)`) or a list element (`None`).
    fn begin(&mut self, id: Option<i16>) {
        if let Some(id) = id {
            self.field(id, 12);
        }
        self.stack.push(self.last);
        self.last = 0;
    }
    fn end(&mut self) {
        self.b.push(0);
        self.last = self.stack.pop().unwrap_or(0);
    }
}

struct ColSpec {
    name: &'static [u8],
    phys: i32,
    conv: Option<i32>,
    optional: bool,
}

struct ChunkMeta {
    offset: i64,
    size: i64,
    num_values: i64,
}

/// Encode one column's values for a slice of rows into `out` (PLAIN, plus
/// definition levels first for optional columns).
type Encoder<R> = fn(&[R], usize, &mut Vec<u8>);

fn rle_def_levels(defined: impl Iterator<Item = bool>, out: &mut Vec<u8>) {
    // RLE/bit-packed hybrid, bit width 1, RLE runs only; prefixed by its length.
    let mut body = Thrift::new(); // reuse its varint
    let mut cur: Option<bool> = None;
    let mut run = 0u64;
    let flush = |v: bool, run: u64, body: &mut Thrift| {
        body.varint(run << 1);
        body.b.push(v as u8);
    };
    for d in defined {
        match cur {
            Some(c) if c == d => run += 1,
            Some(c) => {
                flush(c, run, &mut body);
                cur = Some(d);
                run = 1;
            }
            None => {
                cur = Some(d);
                run = 1;
            }
        }
    }
    if let Some(c) = cur {
        flush(c, run, &mut body);
    }
    out.extend_from_slice(&(body.b.len() as u32).to_le_bytes());
    out.extend_from_slice(&body.b);
}

fn plain_str(s: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s);
}

fn write_table<R>(dir: &Path, part: usize, cols: &[ColSpec], rows: &[R], enc: Encoder<R>) {
    fs::create_dir_all(dir).expect("create output dir");
    let mut f = BufWriter::with_capacity(8 << 20, File::create(dir.join(format!("part-{part}.parquet"))).expect("create parquet"));
    let mut pos: i64 = 0;
    let put = |f: &mut BufWriter<File>, b: &[u8], pos: &mut i64| {
        f.write_all(b).expect("write parquet");
        *pos += b.len() as i64;
    };
    put(&mut f, b"PAR1", &mut pos);

    let mut groups: Vec<(Vec<ChunkMeta>, i64, i64)> = Vec::new(); // chunks, total bytes, rows
    let mut data = Vec::new();
    for part in rows.chunks(ROW_GROUP) {
        let mut metas = Vec::new();
        let mut total = 0i64;
        for c in 0..cols.len() {
            data.clear();
            enc(part, c, &mut data);
            let mut h = Thrift::new();
            h.i32(1, 0); // DATA_PAGE
            h.i32(2, data.len() as i32);
            h.i32(3, data.len() as i32);
            h.begin(Some(5));
            h.i32(1, part.len() as i32);
            h.i32(2, ENC_PLAIN);
            h.i32(3, ENC_RLE);
            h.i32(4, ENC_RLE);
            h.end();
            h.b.push(0); // end PageHeader
            let offset = pos;
            put(&mut f, &h.b, &mut pos);
            put(&mut f, &data, &mut pos);
            let size = pos - offset;
            total += size;
            metas.push(ChunkMeta { offset, size, num_values: part.len() as i64 });
        }
        groups.push((metas, total, part.len() as i64));
    }

    // FileMetaData
    let mut t = Thrift::new();
    t.i32(1, 1); // version
    t.list(2, 12, cols.len() + 1);
    t.begin(None); // root
    t.bin(4, b"schema");
    t.i32(5, cols.len() as i32);
    t.end();
    for c in cols {
        t.begin(None);
        t.i32(1, c.phys);
        t.i32(3, if c.optional { OPTIONAL } else { REQUIRED });
        t.bin(4, c.name);
        if let Some(conv) = c.conv {
            t.i32(6, conv);
        }
        t.end();
    }
    t.i64(3, rows.len() as i64);
    t.list(4, 12, groups.len());
    for (metas, total, nrows) in &groups {
        t.begin(None);
        t.list(1, 12, metas.len());
        for (c, m) in cols.iter().zip(metas) {
            t.begin(None); // ColumnChunk
            t.i64(2, m.offset);
            t.begin(Some(3)); // ColumnMetaData
            t.i32(1, c.phys);
            t.list(2, 5, 2);
            t.list_i32(ENC_PLAIN);
            t.list_i32(ENC_RLE);
            t.list(3, 8, 1);
            t.list_bin(c.name);
            t.i32(4, 0); // UNCOMPRESSED
            t.i64(5, m.num_values);
            t.i64(6, m.size);
            t.i64(7, m.size);
            t.i64(9, m.offset);
            t.end();
            t.end();
        }
        t.i64(2, *total);
        t.i64(3, *nrows);
        t.end();
    }
    t.b.push(0); // end FileMetaData
    put(&mut f, &t.b, &mut pos);
    put(&mut f, &(t.b.len() as u32).to_le_bytes(), &mut pos);
    put(&mut f, b"PAR1", &mut pos);
    f.flush().expect("flush parquet");
}

pub fn write_user_daily(out: &Path, part: usize, rows: &[UserRow]) {
    let cols = [
        ColSpec { name: b"user_id", phys: T_BYTE_ARRAY, conv: Some(CONV_UTF8), optional: false },
        ColSpec { name: b"day", phys: T_INT32, conv: Some(CONV_DATE), optional: false },
        ColSpec { name: b"sessions", phys: T_INT64, conv: None, optional: false },
        ColSpec { name: b"events", phys: T_INT64, conv: None, optional: false },
        ColSpec { name: b"revenue_cents", phys: T_INT64, conv: None, optional: false },
        ColSpec { name: b"top_category", phys: T_BYTE_ARRAY, conv: Some(CONV_UTF8), optional: true },
    ];
    fn enc(rows: &[UserRow], c: usize, out: &mut Vec<u8>) {
        match c {
            0 => rows.iter().for_each(|r| plain_str(r.user_id.bytes(), out)),
            1 => rows.iter().for_each(|r| out.extend_from_slice(&r.day.to_le_bytes())),
            2 => rows.iter().for_each(|r| out.extend_from_slice(&r.sessions.to_le_bytes())),
            3 => rows.iter().for_each(|r| out.extend_from_slice(&r.events.to_le_bytes())),
            4 => rows.iter().for_each(|r| out.extend_from_slice(&r.revenue_cents.to_le_bytes())),
            _ => {
                rle_def_levels(rows.iter().map(|r| r.top_category.is_some()), out);
                rows.iter().filter_map(|r| r.top_category).for_each(|s| plain_str(s, out));
            }
        }
    }
    write_table(&out.join("user_daily"), part, &cols, rows, enc);
}

pub fn write_campaign_daily(out: &Path, rows: &[CampaignRow]) {
    let cols = [
        ColSpec { name: b"campaign_id", phys: T_BYTE_ARRAY, conv: Some(CONV_UTF8), optional: false },
        ColSpec { name: b"channel", phys: T_BYTE_ARRAY, conv: Some(CONV_UTF8), optional: false },
        ColSpec { name: b"day", phys: T_INT32, conv: Some(CONV_DATE), optional: false },
        ColSpec { name: b"sessions", phys: T_INT64, conv: None, optional: false },
        ColSpec { name: b"purchasing_sessions", phys: T_INT64, conv: None, optional: false },
        ColSpec { name: b"revenue_cents", phys: T_INT64, conv: None, optional: false },
        ColSpec { name: b"assisted_sessions", phys: T_INT64, conv: None, optional: false },
    ];
    fn enc(rows: &[CampaignRow], c: usize, out: &mut Vec<u8>) {
        match c {
            0 => rows.iter().for_each(|r| plain_str(r.campaign_id, out)),
            1 => rows.iter().for_each(|r| plain_str(r.channel, out)),
            2 => rows.iter().for_each(|r| out.extend_from_slice(&r.day.to_le_bytes())),
            3 => rows.iter().for_each(|r| out.extend_from_slice(&r.sessions.to_le_bytes())),
            4 => rows.iter().for_each(|r| out.extend_from_slice(&r.purchasing_sessions.to_le_bytes())),
            5 => rows.iter().for_each(|r| out.extend_from_slice(&r.revenue_cents.to_le_bytes())),
            _ => rows.iter().for_each(|r| out.extend_from_slice(&r.assisted_sessions.to_le_bytes())),
        }
    }
    write_table(&out.join("campaign_daily"), 0, &cols, rows, enc);
}
