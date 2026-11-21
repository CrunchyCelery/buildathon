//! The CSV dialect from rules.md, on raw bytes.
//!
//! - Only `\n` ends a line. `\r` is an ordinary byte of the field it is in.
//! - The first line of every file is the header; every later line is data
//!   (a later line that looks like a header is data too).
//! - A field is either bare (no `"` at all) or wrapped in quotes around the
//!   whole field (`"..."`, nothing else may contain a quote). No value ever
//!   holds a comma, so every comma separates fields.
//! - Wrong field count, a stray quote, or an empty line: malformed, dropped.
//! - An empty field (`` or `""`) is null (`None`).

use memchr::{memchr, memchr3_iter, memchr_iter};

/// The bytes after the header line (empty if the file has no data lines).
pub fn data_region(buf: &[u8]) -> &[u8] {
    match memchr(b'\n', buf) {
        Some(i) => &buf[i + 1..],
        None => &[],
    }
}

/// Split a region into chunks of roughly `target` bytes, each ending right
/// after a `\n` (or at the end of the region), so no line is cut in two.
pub fn chunks(region: &[u8], target: usize) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < region.len() {
        let want = (start + target).min(region.len());
        let end = if want >= region.len() {
            region.len()
        } else {
            match memchr(b'\n', &region[want..]) {
                Some(i) => want + i + 1,
                None => region.len(),
            }
        };
        out.push(&region[start..end]);
        start = end;
    }
    out
}

/// Call `f` for every line of a chunk (without its `\n`). A final piece with
/// no `\n` after it is a line too, unless it is empty (that is just the end of
/// the file, and an empty line would be dropped as malformed anyway).
#[inline]
pub fn for_each_line<'a>(chunk: &'a [u8], mut f: impl FnMut(&'a [u8])) {
    let mut start = 0;
    for nl in memchr_iter(b'\n', chunk) {
        f(&chunk[start..nl]);
        start = nl + 1;
    }
    if start < chunk.len() {
        f(&chunk[start..]);
    }
}

/// Split one line into exactly N fields, applying the quoting rules.
/// Returns None if the line is malformed. Each field is None when empty.
#[inline]
pub fn fields<'a, const N: usize>(line: &'a [u8]) -> Option<[Option<&'a [u8]>; N]> {
    let mut out: [Option<&'a [u8]>; N] = [None; N];
    let has_quote = memchr(b'"', line).is_some();
    let mut n = 0usize;
    let mut start = 0usize;
    let mut push = |raw: &'a [u8], n: &mut usize| -> bool {
        if *n >= N {
            return false;
        }
        let v = if has_quote && memchr(b'"', raw).is_some() {
            // Must be "..." with no quote inside.
            if raw.len() < 2 || raw[0] != b'"' || raw[raw.len() - 1] != b'"' {
                return false;
            }
            let inner = &raw[1..raw.len() - 1];
            if memchr(b'"', inner).is_some() {
                return false;
            }
            inner
        } else {
            raw
        };
        out[*n] = if v.is_empty() { None } else { Some(v) };
        *n += 1;
        true
    };
    for c in memchr_iter(b',', line) {
        if !push(&line[start..c], &mut n) {
            return None;
        }
        start = c + 1;
    }
    if !push(&line[start..], &mut n) {
        return None;
    }
    if n != N { None } else { Some(out) }
}

/// Parse every line of a chunk into exactly N fields in ONE pass: a SIMD scan
/// for `,` `"` `\n` (memchr3) visits each byte once, where `for_each_line` +
/// `fields` scan each line three times. Same rules as `fields`: a field is bare
/// (no quote) or exactly `"..."` with no quote inside; wrong field count, stray
/// quotes and empty lines are dropped. Calls `f` for each well-formed line.
#[inline]
pub fn for_each_record<'a, const N: usize>(chunk: &'a [u8], mut f: impl FnMut(&[Option<&'a [u8]>; N])) {
    let mut out: [Option<&'a [u8]>; N] = [None; N];
    let mut n = 0usize; // fields closed in this line
    let mut fstart = 0usize; // start of the current field
    let mut lstart = 0usize; // start of the current line
    let (mut q, mut q_first, mut q_last) = (0u32, 0usize, 0usize); // quotes in the current field
    let mut bad = false;
    macro_rules! close {
        ($end:expr) => {{
            let end = $end;
            if !bad {
                if n >= N {
                    bad = true;
                } else {
                    let v: Option<&'a [u8]> = if q == 0 {
                        Some(&chunk[fstart..end])
                    } else if q == 2 && q_first == fstart && q_last + 1 == end {
                        Some(&chunk[fstart + 1..end - 1])
                    } else {
                        None
                    };
                    match v {
                        Some(v) => {
                            out[n] = if v.is_empty() { None } else { Some(v) };
                            n += 1;
                        }
                        None => bad = true,
                    }
                }
            }
            #[allow(unused_assignments)]
            {
                q = 0;
            }
        }};
    }
    for pos in memchr3_iter(b',', b'"', b'\n', chunk) {
        match chunk[pos] {
            b'"' => {
                if q == 0 {
                    q_first = pos;
                }
                q_last = pos;
                q += 1;
            }
            b',' => {
                close!(pos);
                fstart = pos + 1;
            }
            _ => {
                close!(pos);
                if !bad && n == N {
                    f(&out);
                }
                n = 0;
                bad = false;
                fstart = pos + 1;
                lstart = pos + 1;
            }
        }
    }
    // A last line with no newline after it (it may end in an empty field, so
    // test where the LINE starts, not the field).
    if lstart < chunk.len() {
        close!(chunk.len());
        if !bad && n == N {
            f(&out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect() {
        assert_eq!(fields::<3>(b"x,y,z"), Some([Some(&b"x"[..]), Some(&b"y"[..]), Some(&b"z"[..])]));
        assert_eq!(fields::<3>(b"\"x\",\"y\",\"\""), Some([Some(&b"x"[..]), Some(&b"y"[..]), None]));
        assert_eq!(fields::<3>(b"x,,z"), Some([Some(&b"x"[..]), None, Some(&b"z"[..])]));
        assert_eq!(fields::<3>(b"x,y,z\r"), Some([Some(&b"x"[..]), Some(&b"y"[..]), Some(&b"z\r"[..])]));
        assert_eq!(fields::<3>(b" x,\" y\",z "), Some([Some(&b" x"[..]), Some(&b" y"[..]), Some(&b"z "[..])]));
        assert_eq!(fields::<3>(b""), None);
        assert_eq!(fields::<3>(b"x,y"), None);
        assert_eq!(fields::<3>(b"x,y,z,w"), None);
        assert_eq!(fields::<3>(b"x,y\"y,z"), None);
        assert_eq!(fields::<3>(b"\"x,y,z"), None);
        assert_eq!(fields::<3>(b"\"x\"x,y,z"), None);
        assert_eq!(fields::<3>(b"\",y,z"), None);
    }

    #[test]
    fn lines_and_chunks() {
        let region = data_region(b"h1,h2\na,b\n\nc,d\nlast");
        let mut got = Vec::new();
        for ch in chunks(region, 3) {
            for_each_line(ch, |l| got.push(l.to_vec()));
        }
        assert_eq!(got, vec![b"a,b".to_vec(), b"".to_vec(), b"c,d".to_vec(), b"last".to_vec()]);
        assert!(data_region(b"h1,h2").is_empty());
        assert!(data_region(b"h1,h2\n").is_empty());
    }

    /// The one-pass parser must agree with line-by-line `fields` on every
    /// input: random lines over the characters that matter.
    #[test]
    fn record_parser_matches_fields() {
        let alphabet: &[u8] = b"ab,\"\r \n,,\"";
        let mut x: u64 = 0x1234_5678_9abc_def1;
        for _ in 0..20_000 {
            let len = (x % 40) as usize;
            let mut text = Vec::with_capacity(len);
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                text.push(alphabet[(x % alphabet.len() as u64) as usize]);
            }
            x = x.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
            let mut want: Vec<[Option<Vec<u8>>; 3]> = Vec::new();
            for_each_line(&text, |l| {
                if let Some(r) = fields::<3>(l) {
                    want.push(r.map(|o| o.map(|v| v.to_vec())));
                }
            });
            let mut got: Vec<[Option<Vec<u8>>; 3]> = Vec::new();
            for_each_record::<3>(&text, |r| got.push(r.map(|o| o.map(|v| v.to_vec()))));
            assert_eq!(got, want, "input {:?}", String::from_utf8_lossy(&text));
        }
    }
}
