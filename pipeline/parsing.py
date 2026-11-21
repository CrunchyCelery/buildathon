"""
Parsing layer: raw CSV lines -> DuckDB tables, following rules.md exactly.

DuckDB's CSV parsing (quotes, field splitting, error handling) is NOT used: it
repairs or skips malformed lines in its own way. Instead DuckDB's reader only
hands over whole lines (see _lines_sql), every line is checked against a strict
regex, and the fields are split and unquoted here.
"""
import glob
import mmap
import os
from concurrent.futures import ThreadPoolExecutor

# One field: either "..." wrapped in quotes around the WHOLE field, or bare.
# Neither form may contain a comma or a quote inside (no value ever does), so a
# quote anywhere else makes the line malformed. \r is an ordinary character.
FIELD = '(?:"[^",]*"|[^",]*)'


def install_macros(con):
    """SQL helper functions used by every later step."""

    # Strip the syntax quotes. An empty field, quoted or not, is NULL.
    # The line regex already guarantees a quoted field is exactly "...", so a
    # leading quote means: drop the first and last character. starts_with and
    # nullif avoid building temporary strings (x[1] makes a new 1-char string).
    con.execute("""
        CREATE OR REPLACE MACRO uq(x) AS
            CASE WHEN starts_with(x, '"') THEN nullif(substr(x, 2, length(x) - 2), '')
                 ELSE nullif(x, '') END
    """)

    # Timestamp string -> integer epoch milliseconds (UTC), or NULL if it is bad.
    #   F3: digits only (ASCII), epoch ms, leading zeros allowed. More than 18
    #       significant digits can never be inside a window, so NULL.
    #   F1: YYYY-MM-DDTHH:MM:SSZ
    #   F2: YYYY-MM-DDTHH:MM:SS+HH:MM or -HH:MM (subtract the offset to get UTC)
    # For F1/F2 the strict regex fixes the exact shape (two-digit fields, capital
    # T and Z, colon in the offset) and the time and offset ranges (hour 00-23,
    # minute/second 00-59). try_strptime then rejects impossible dates (month 13,
    # Feb 30, Feb 29 outside a leap year) and converts in one built-in call.
    # Macros are pasted in where used and cannot keep intermediate values, so
    # doing the date work in one call instead of several CASTs is what saves time.
    con.execute("""
        CREATE OR REPLACE MACRO ts_ms(s) AS
        CASE
          WHEN regexp_full_match(s, '[0-9]+') THEN
            CASE WHEN length(ltrim(s, '0')) <= 18 THEN TRY_CAST(s AS BIGINT) END
          WHEN regexp_full_match(s,
                 '[0-9]{4}-[0-9]{2}-[0-9]{2}T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])')
          THEN
            epoch_ms(try_strptime(substr(s, 1, 19), '%Y-%m-%dT%H:%M:%S'))
            - CASE WHEN length(s) = 20 THEN 0
                   ELSE (CASE WHEN substr(s, 20, 1) = '-' THEN -1 ELSE 1 END)
                        * (CAST(substr(s, 21, 2) AS BIGINT) * 3600000
                           + CAST(substr(s, 24, 2) AS BIGINT) * 60000)
              END
        END
    """)

    # Purchase price: ASCII digits only, value 1..10,000,000 ("0250" is 250). Else NULL.
    con.execute("""
        CREATE OR REPLACE MACRO price(p) AS
        CASE WHEN regexp_full_match(p, '[0-9]+') AND length(ltrim(p, '0')) <= 8
             THEN CASE WHEN CAST(p AS BIGINT) BETWEEN 1 AND 10000000 THEN CAST(p AS BIGINT) END
        END
    """)


def _sql_str(path):
    return "'" + path.replace("\\", "/").replace("'", "''") + "'"


def _has_cr(path):
    """True if the file contains a \\r byte anywhere (fast: mmap + memchr)."""
    if os.path.getsize(path) == 0:
        return False
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        return m.find(b"\r") != -1


def _lines_sql(files):
    """
    SQL producing one row per data line (header skipped), column `line`.

    Fast path: DuckDB's parallel CSV reader with one column per line (no quote
    handling, a delimiter that never occurs). It is exact for files without \\r,
    but it strips a \\r before \\n, which rules.md says is part of the value.
    So files containing any \\r take the slow, exact path: read_text + split on
    \\n only. The generator never writes \\r, so in practice that is only the
    hand-written fixtures.
    """
    # mmap.find mostly holds Python's lock, so threads help only a little
    # (0.33s -> 0.24s on 900 MB), but it is free.
    with ThreadPoolExecutor(max_workers=8) as pool:
        has_cr = list(pool.map(_has_cr, files))
    fast = [f for f, cr in zip(files, has_cr) if not cr]
    slow = [f for f, cr in zip(files, has_cr) if cr]
    parts = []
    if fast:
        file_list = "[" + ", ".join(_sql_str(f) for f in fast) + "]"
        parts.append(f"""
            SELECT line FROM read_csv({file_list}, columns = {{'line': 'VARCHAR'}},
                header = true, delim = '\\x01', quote = '', escape = '', comment = '',
                auto_detect = false, new_line = '\\n', strict_mode = false)""")
    if slow:
        file_list = "[" + ", ".join(_sql_str(f) for f in slow) + "]"
        parts.append(f"""
            SELECT unnest(string_split(content, chr(10))[2:]) AS line
            FROM read_text({file_list})""")
    return " UNION ALL ".join(parts)


def load_csv(con, table, files, columns, kind="TABLE"):
    """
    Read the given CSV files into `table`: one VARCHAR column per name in
    `columns`, quotes stripped, empty -> NULL. The header line of each file is
    skipped. Malformed lines (wrong field count, stray quotes, empty lines) are
    dropped.

    kind="VIEW" makes it lazy: nothing is stored, and the parse streams straight
    into whichever query reads it (use it for inputs that are read only once).
    """
    if not files:
        cols = ", ".join(f"NULL::VARCHAR AS {c}" for c in columns)
        con.execute(f"CREATE OR REPLACE {kind} {table} AS SELECT {cols} WHERE false")
        return
    pattern = FIELD + "(?:," + FIELD + "){" + str(len(columns) - 1) + "}"
    select = ", ".join(f"uq(f[{i + 1}]) AS {c}" for i, c in enumerate(columns))
    con.execute(f"""
        CREATE OR REPLACE {kind} {table} AS
        SELECT {select}
        FROM (
            SELECT string_split(line, ',') AS f
            FROM ({_lines_sql(files)})
            WHERE regexp_full_match(line, '{pattern}')
        )
    """)


def load_inputs(con, data_dir):
    """Create events_raw, clicks_raw, links_raw (views) and products, campaigns (tables), all VARCHAR."""
    install_macros(con)

    def csvs(sub):
        return sorted(glob.glob(os.path.join(data_dir, sub, "*.csv")))

    # Read once each, so views: parsing streams into the query that uses them.
    load_csv(con, "events_raw", csvs("events"),
             ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"], "VIEW")
    load_csv(con, "clicks_raw", csvs("clicks"), ["click_id", "user_id", "campaign_id", "ts"], "VIEW")
    load_csv(con, "links_raw", csvs("links"), ["user_id_a", "user_id_b"], "VIEW")
    load_csv(con, "products", [os.path.join(data_dir, "products.csv")], ["product_id", "category"])
    load_csv(con, "campaigns", [os.path.join(data_dir, "campaigns.csv")], ["campaign_id", "channel"])
