//! Timestamps and prices, exactly as rules.md defines them.

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's
/// days_from_civil). Works for any year, including before 1970.
#[inline]
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[inline]
fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[inline]
fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Two ASCII digits at s[i], s[i+1] -> value, or None.
#[inline]
fn d2(s: &[u8], i: usize) -> Option<i64> {
    let (a, b) = (s[i].wrapping_sub(b'0'), s[i + 1].wrapping_sub(b'0'));
    if a > 9 || b > 9 { None } else { Some((a * 10 + b) as i64) }
}

/// Timestamp -> epoch milliseconds (UTC), or None if it is bad.
///   F3  digits only: epoch ms, leading zeros allowed. A value too big for i64
///       cannot be in any window, so None (it would be dropped either way).
///   F1  YYYY-MM-DDTHH:MM:SSZ
///   F2  YYYY-MM-DDTHH:MM:SS+HH:MM / -HH:MM (offset subtracted to get UTC)
/// Anything else is bad: lowercase z/t, a space, fractions, no zone, +0200,
/// month 13, Feb 30, Feb 29 outside a leap year, hour 24, second 60, offset
/// hour > 23 or minute > 59.
#[inline]
pub fn parse_ts(s: &[u8]) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    if s[0].is_ascii_digit() && s.iter().all(|b| b.is_ascii_digit()) {
        let mut v: u64 = 0;
        for &b in s {
            v = v.checked_mul(10)?.checked_add((b - b'0') as u64)?;
        }
        return if v <= i64::MAX as u64 { Some(v as i64) } else { None };
    }
    let zulu = s.len() == 20 && s[19] == b'Z';
    let offset = s.len() == 25 && (s[19] == b'+' || s[19] == b'-') && s[22] == b':';
    if !(zulu || offset) {
        return None;
    }
    if s[4] != b'-' || s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':' {
        return None;
    }
    let y = d2(s, 0)? * 100 + d2(s, 2)?;
    let mo = d2(s, 5)?;
    let d = d2(s, 8)?;
    let h = d2(s, 11)?;
    let mi = d2(s, 14)?;
    let se = d2(s, 17)?;
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) || h > 23 || mi > 59 || se > 59 {
        return None;
    }
    let mut ms = days_from_civil(y, mo, d) * 86_400_000 + h * 3_600_000 + mi * 60_000 + se * 1000;
    if offset {
        let oh = d2(s, 20)?;
        let om = d2(s, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        let off = oh * 3_600_000 + om * 60_000;
        ms -= if s[19] == b'-' { -off } else { off };
    }
    Some(ms)
}

/// Purchase price: ASCII digits only, value 1..=10,000,000 ("0250" is 250).
#[inline]
pub fn parse_price(s: &[u8]) -> Option<i64> {
    if s.is_empty() || !s.iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut v: u64 = 0;
    for &b in s {
        v = v.saturating_mul(10).saturating_add((b - b'0') as u64);
        if v > 10_000_000 {
            return None;
        }
    }
    if v >= 1 { Some(v as i64) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    const F1: i64 = 1_791_071_890_000;

    #[test]
    fn timestamps() {
        let cases: &[(&str, Option<i64>)] = &[
            ("2026-10-03T23:58:10Z", Some(F1)),
            ("2026-10-03T19:58:10-04:00", Some(F1)),
            ("2026-10-04T05:28:10+05:30", Some(F1)),
            ("2026-10-03T23:58:10-00:00", Some(F1)),
            ("1791071890000", Some(F1)),
            ("0001791071890000", Some(F1)),
            ("86400000", Some(86_400_000)),
            ("99999999999999999999999", None),
            ("2028-02-29T00:00:00Z", Some(1_835_395_200_000)),
            ("2026-02-29T00:00:00Z", None),
            ("2026-02-30T00:00:00Z", None),
            ("2026-13-01T00:00:00Z", None),
            ("2026-10-03T24:00:00Z", None),
            ("2026-10-03T23:60:00Z", None),
            ("2026-10-03T23:58:60Z", None),
            ("2026-10-03T23:58:10+24:00", None),
            ("2026-10-03T23:58:10+05:60", None),
            ("2026-10-03T23:58:10+0200", None),
            ("2026-10-03T23:58:10z", None),
            ("2026-10-03t23:58:10Z", None),
            ("2026-10-03 23:58:10Z", None),
            ("2026-10-03T23:58:10", None),
            ("2026-10-03T23:58:10.000Z", None),
            ("2026-10-03T23:58:10Z\r", None),
            ("1791071890000.0", None),
            ("1.79e12", None),
            ("-5", None),
            ("١٧٩١٠٧١٨٩٠٠٠٠", None),
            ("", None),
        ];
        for (s, want) in cases {
            assert_eq!(parse_ts(s.as_bytes()), *want, "{s:?}");
        }
    }

    #[test]
    fn prices() {
        let cases: &[(&str, Option<i64>)] = &[
            ("250", Some(250)), ("0250", Some(250)), ("1", Some(1)), ("10000000", Some(10_000_000)),
            ("0", None), ("00", None), ("10000001", None), ("-5", None), ("+5", None), ("2.50", None),
            ("1000\r", None), (" 250", None), ("２５０", None), ("99999999999999999999", None), ("1e3", None),
        ];
        for (s, want) in cases {
            assert_eq!(parse_price(s.as_bytes()), *want, "{s:?}");
        }
    }
}
