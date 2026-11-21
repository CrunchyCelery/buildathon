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

pub struct People<'a> {
    slots: Vec<(u128, u32)>, // packed linked ID -> index into names
    mask: usize,
    long: FxHashMap<&'a [u8], u32>, // linked IDs longer than 15 bytes
    names: Vec<(PKey, u64)>,        // person name and hash64(name)
    _ids: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> People<'a> {
    /// The person an ID belongs to, and hash64 of that person's name.
    #[inline]
    pub fn lookup(&self, id: &'a [u8]) -> (PKey, u64) {
        match pack(id) {
            Some(k) => {
                let mut i = (slot_hash(k) as usize) & self.mask;
                loop {
                    let (sk, v) = self.slots[i];
                    if sk == k {
                        return self.names[v as usize];
                    }
                    if sk == 0 {
                        break;
                    }
                    i = (i + 1) & self.mask;
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
        let mut start = [0usize; BATCH];
        for group in ids.chunks(BATCH) {
            for (g, id) in group.iter().enumerate() {
                keys[g] = pack(id);
                if let Some(k) = keys[g] {
                    start[g] = (slot_hash(k) as usize) & self.mask;
                    prefetch(&self.slots[start[g]]);
                }
            }
            for (g, id) in group.iter().enumerate() {
                out.push(match keys[g] {
                    Some(k) => {
                        let mut i = start[g];
                        loop {
                            let (sk, v) = self.slots[i];
                            if sk == k {
                                break self.names[v as usize];
                            }
                            if sk == 0 {
                                break (PKey::new(id), hash64(id));
                            }
                            i = (i + 1) & self.mask;
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

/// Interns linked IDs to dense indices. Short IDs live in the packed-u128
/// open-addressing table that also serves lookups afterwards, so the table is
/// built once (the old version interned through a byte-keyed hash map and then
/// built the lookup table in a second pass).
struct Interner<'a> {
    slots: Vec<(u128, u32)>,
    mask: usize,
    long: FxHashMap<&'a [u8], u32>,
    ids: Vec<&'a [u8]>,
    parent: Vec<u32>,
    short: usize, // IDs in `slots`
}

impl<'a> Interner<'a> {
    fn new(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(1024);
        Interner { slots: vec![(0, 0); cap], mask: cap - 1, long: FxHashMap::default(), ids: Vec::new(), parent: Vec::new(), short: 0 }
    }

    fn push(&mut self, id: &'a [u8]) -> u32 {
        let i = self.ids.len() as u32;
        self.ids.push(id);
        self.parent.push(i);
        i
    }

    fn grow(&mut self) {
        let cap = self.slots.len() * 2;
        let old = std::mem::replace(&mut self.slots, vec![(0, 0); cap]);
        self.mask = cap - 1;
        for (k, v) in old {
            if k != 0 {
                let mut j = (slot_hash(k) as usize) & self.mask;
                while self.slots[j].0 != 0 {
                    j = (j + 1) & self.mask;
                }
                self.slots[j] = (k, v);
            }
        }
    }

    #[inline]
    fn intern(&mut self, id: &'a [u8]) -> u32 {
        match pack(id) {
            Some(k) => {
                if (self.short + 1) * 2 > self.slots.len() {
                    self.grow();
                }
                let mut j = (slot_hash(k) as usize) & self.mask;
                loop {
                    let (sk, v) = self.slots[j];
                    if sk == k {
                        return v;
                    }
                    if sk == 0 {
                        let i = self.push(id);
                        self.slots[j] = (k, i);
                        self.short += 1;
                        return i;
                    }
                    j = (j + 1) & self.mask;
                }
            }
            None => {
                if let Some(&v) = self.long.get(id) {
                    return v;
                }
                let i = self.push(id);
                self.long.insert(id, i);
                i
            }
        }
    }
}

/// Build the person table from the link files' data regions.
pub fn build<'a>(regions: &[&'a [u8]]) -> People<'a> {
    // ~20 bytes per link line, two IDs per line, most IDs on several lines.
    let bytes: usize = regions.iter().map(|r| r.len()).sum();
    let mut t = Interner::new(bytes / 16);
    for region in regions {
        // Drop malformed lines and lines where either side is empty.
        csv::for_each_record::<2>(region, |r| {
            if let [Some(a), Some(b)] = *r {
                let ia = t.intern(a);
                let ib = t.intern(b);
                let (ra, rb) = (find(&mut t.parent, ia), find(&mut t.parent, ib));
                if ra != rb {
                    // Keep the root with the byte-wise smaller ID, so a root is
                    // always its component's smallest ID: that is the name.
                    if t.ids[ra as usize] <= t.ids[rb as usize] {
                        t.parent[rb as usize] = ra;
                    } else {
                        t.parent[ra as usize] = rb;
                    }
                }
            }
        });
    }
    // names[i] = (person name, its hash) for linked ID i; the table already
    // maps each packed ID to i. Each root's hash is computed once.
    let n = t.ids.len();
    let mut root_hash: Vec<Option<u64>> = vec![None; n];
    let mut names: Vec<(PKey, u64)> = Vec::with_capacity(n);
    for i in 0..n as u32 {
        let r = find(&mut t.parent, i) as usize;
        let name = t.ids[r];
        let h = *root_hash[r].get_or_insert_with(|| hash64(name));
        names.push((PKey::new(name), h));
    }
    People { slots: t.slots, mask: t.mask, long: t.long, names, _ids: std::marker::PhantomData }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components() {
        let data = b"c,b\nb,a\nx,y\nU1,u1\nz,z\n,q\nbad\"line,w\n";
        let p = build(&[&data[..]]);
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
        let p2 = build(&[&data2[..]]);
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
}
