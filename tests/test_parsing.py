"""
Checks the parsing layer against the tricky cases in rules.md.
Run: python tests/test_parsing.py
"""
import os
import sys
import tempfile

import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from parsing import install_macros, load_csv  # noqa: E402

F1 = 1791071890000  # 2026-10-03T23:58:10Z in epoch ms

TS_CASES = [
    # (input, expected epoch ms or None)
    ("2026-10-03T23:58:10Z", F1),
    ("2026-10-03T19:58:10-04:00", F1),          # F2, subtract the offset
    ("2026-10-04T05:28:10+05:30", F1),
    ("2026-10-03T23:58:10-00:00", F1),          # -00:00 is UTC
    ("1791071890000", F1),                      # F3
    ("0001791071890000", F1),                   # F3 leading zeros
    ("86400000", 86400000),                     # short F3: valid, just outside window
    ("99999999999999999999999", None),          # too long for 64-bit -> outside window
    ("2028-02-29T00:00:00Z", 1835395200000),    # leap day exists in 2028
    ("2026-02-29T00:00:00Z", None),             # not a leap year
    ("2026-02-30T00:00:00Z", None),
    ("2026-13-01T00:00:00Z", None),
    ("2026-10-03T24:00:00Z", None),             # hour 24
    ("2026-10-03T23:60:00Z", None),
    ("2026-10-03T23:58:60Z", None),             # second 60
    ("2026-10-03T23:58:10+24:00", None),
    ("2026-10-03T23:58:10+05:60", None),
    ("2026-10-03T23:58:10+0200", None),         # offset without colon
    ("2026-10-03T23:58:10z", None),             # lowercase z
    ("2026-10-03t23:58:10Z", None),             # lowercase t
    ("2026-10-03 23:58:10Z", None),             # space instead of T
    ("2026-10-03T23:58:10", None),              # no zone
    ("2026-10-03T23:58:10.000Z", None),         # fractional seconds
    ("2026-10-03T23:58:10Z\r", None),           # \r is part of the value
    ("1791071890000.0", None),
    ("1.79e12", None),
    ("-5", None),
    ("١٧٩١٠٧١٨٩٠٠٠٠", None),                   # non-ASCII digits
    ("", None),
]

PRICE_CASES = [
    ("250", 250), ("0250", 250), ("1", 1), ("10000000", 10000000),
    ("0", None), ("10000001", None), ("-5", None), ("+5", None), ("2.50", None),
    ("1000\r", None), (" 250", None), ("２５０", None), ("99999999999999999999", None),
]

# Raw file content -> expected rows (3 columns). The first line is the header.
CSV_CONTENT = (
    "a,b,c\n"
    "x,y,z\n"                 # plain
    '"x","y",""\n'            # quotes are syntax; "" is NULL
    "x,,z\n"                  # empty -> NULL
    "x,y,z\r\n"               # \r stays in the last field
    " x,\" y\",z \n"          # spaces are part of the value
    "\n"                      # empty line: malformed
    "x,y\n"                   # too few fields: malformed
    "x,y,z,w\n"               # too many fields: malformed
    'x,y"y,z\n'               # quote inside a field: malformed
    '"x,y,z\n'                # unbalanced quote: malformed
    '"x"x,y,z\n'              # quote not around the whole field: malformed
    '",y,z\n'                 # a lone quote: malformed
    "last,no,newline"         # final line without \n still counts
)
CSV_EXPECTED = [
    ("x", "y", "z"),
    ("x", "y", None),
    ("x", None, "z"),
    ("x", "y", "z\r"),
    (" x", " y", "z "),
    ("last", "no", "newline"),
]


def main():
    con = duckdb.connect()
    install_macros(con)
    failures = 0

    for raw, want in TS_CASES:
        got = con.execute("SELECT ts_ms(?)", [raw]).fetchone()[0]
        if got != want:
            failures += 1
            print(f"FAIL ts_ms({raw!r}) = {got}, expected {want}")

    for raw, want in PRICE_CASES:
        got = con.execute("SELECT price(?)", [raw]).fetchone()[0]
        if got != want:
            failures += 1
            print(f"FAIL price({raw!r}) = {got}, expected {want}")

    # Run the CSV cases twice: once with the \r line (exact slow path), once
    # without it (fast DuckDB reader path). Both must give the same rows.
    no_cr_content = CSV_CONTENT.replace("x,y,z\r\n", "")
    no_cr_expected = [r for r in CSV_EXPECTED if r != ("x", "y", "z\r")]
    csv_runs = [("slow path (file has \\r)", CSV_CONTENT, CSV_EXPECTED),
                ("fast path (no \\r)", no_cr_content, no_cr_expected)]
    for label, content, expected in csv_runs:
        with tempfile.TemporaryDirectory() as d:
            main_file = os.path.join(d, "part-0.csv")
            header_only = os.path.join(d, "part-1.csv")
            header_no_newline = os.path.join(d, "part-2.csv")
            for path, text in ((main_file, content), (header_only, "a,b,c\n"), (header_no_newline, "a,b,c")):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    f.write(text)
            load_csv(con, "t", [main_file, header_only, header_no_newline], ["a", "b", "c"])
            got = sorted(con.execute("SELECT * FROM t").fetchall(), key=repr)
            if got != sorted(expected, key=repr):
                failures += 1
                print(f"FAIL load_csv {label}")
                print("  got     ", got)
                print("  expected", sorted(expected, key=repr))

    total = len(TS_CASES) + len(PRICE_CASES) + len(csv_runs)
    print(f"{total - failures}/{total} parser checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
