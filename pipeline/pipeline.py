#!/usr/bin/env python3
"""
Shopping Sessions pipeline (DuckDB).

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)

Steps, in the order rules.md gives them:
  1. Parse        raw CSV lines -> VARCHAR tables (parsing.py)
  2. Links        connected components of linked IDs -> person name
  3. Events       dedup -> validate -> drop late -> person
  4. Sessions     gap > 30 min per person -> session start and day
  5. Clicks       dedup -> validate -> person
  6. Attribution  credit each session to one click (or organic), assists
  7. Outputs      user_daily, campaign_daily
"""
import json
import os
import sys
import time

import duckdb
import numpy as np

from parsing import load_inputs

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", os.cpu_count() or 1))
MEM_GB = int(os.environ.get("MEM_GB", "8"))
DEBUG = os.environ.get("DEBUG") == "1"

DAY_MS = 86_400_000
GAP_MS = 30 * 60 * 1000          # a gap OVER this starts a new session
LATE_MS = DAY_MS                 # ingest_ts - ts OVER this is a late event
LOOKBACK_MS = 7 * DAY_MS         # clicks credit/assist sessions up to 7 days later

con = duckdb.connect()  # in-memory database
con.execute(f"SET threads = {NCPU}")
con.execute(f"SET memory_limit = '{max(1, int(MEM_GB * 0.8))}GB'")  # leave headroom for Python
con.execute(f"SET temp_directory = '{os.path.join(SCRATCH, 'duckdb_tmp')}'")
con.execute("SET preserve_insertion_order = false")

_t0 = _t = time.perf_counter()


def step(name):
    """Print how long the previous step took (DEBUG=1 only)."""
    global _t
    now = time.perf_counter()
    if DEBUG:
        print(f"[{now - _t0:7.2f}s] (+{now - _t:6.2f}s) {name}")
    _t = now


def count(table):
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


with open(os.path.join(DATA, "manifest.json")) as f:
    manifest = json.load(f)

# --- 1. Parse ------------------------------------------------------------------
# events_raw, clicks_raw, links_raw, products, campaigns: all VARCHAR, quotes
# stripped, empty -> NULL, malformed lines dropped. Installs ts_ms() and price().
load_inputs(con, DATA)
step("parse")

WINDOW_START, WINDOW_END = con.execute(
    "SELECT ts_ms(?), ts_ms(?)", [manifest["window_start"], manifest["window_end"]]
).fetchone()
CLICK_WINDOW_START = WINDOW_START - LOOKBACK_MS

# --- 2. Links -> person --------------------------------------------------------
# Every link line joins two IDs; joining is transitive, so a person is a
# connected component of the link graph. Give each distinct ID an integer index,
# find the components with a small numpy union-find (below), then name each
# component by its byte-wise smallest ID (DuckDB compares VARCHAR byte by byte,
# so min() is byte-wise).
# IDs that appear in no link are not in person_map: they are their own person.
con.execute("""
    CREATE OR REPLACE TABLE link_edges AS
    SELECT user_id_a AS a, user_id_b AS b FROM links_raw
    WHERE user_id_a IS NOT NULL AND user_id_b IS NOT NULL
""")
con.execute("""
    CREATE OR REPLACE TABLE link_ids AS
    SELECT id, (row_number() OVER () - 1)::INTEGER AS idx
    FROM (SELECT a AS id FROM link_edges UNION SELECT b FROM link_edges)
""")
n_ids = count("link_ids")
if n_ids:
    edges = con.execute("""
        SELECT ia.idx AS u, ib.idx AS v
        FROM link_edges e
        JOIN link_ids ia ON ia.id = e.a
        JOIN link_ids ib ON ib.id = e.b
    """).fetchnumpy()
    u = np.asarray(edges["u"], dtype=np.int32)
    v = np.asarray(edges["v"], dtype=np.int32)
    # Union-find in whole-array steps (no scipy: importing it alone took 0.37s).
    # labels[i] always points at an ID of the same person with index <= i.
    #   hook:     for every edge whose two ends have different roots, point the
    #             larger root at the smaller one (np.minimum.at keeps the
    #             smallest when several edges hook the same root).
    #   compress: labels = labels[labels] until every label is a root, so the
    #             next round compares roots. Chains hundreds of links long take
    #             only a few rounds because each jump halves the distance.
    # Stops when no edge joins two different roots: labels = component id.
    labels = np.arange(n_ids, dtype=np.int32)
    while True:
        lu, lv = labels[u], labels[v]
        lo, hi = np.minimum(lu, lv), np.maximum(lu, lv)
        m = lo != hi
        if not m.any():
            break
        np.minimum.at(labels, hi[m], lo[m])
        while True:
            nxt = labels[labels]
            if np.array_equal(nxt, labels):
                break
            labels = nxt
    components = {"idx": np.arange(n_ids, dtype=np.int32), "comp": labels.astype(np.int32)}  # noqa: F841
    con.execute("""
        CREATE OR REPLACE TABLE person_map AS
        WITH id_comp AS (
            SELECT l.id, c.comp FROM link_ids l JOIN components c ON c.idx = l.idx
        )
        SELECT id, min(id) OVER (PARTITION BY comp) AS person FROM id_comp
    """)
else:
    con.execute("CREATE OR REPLACE TABLE person_map (id VARCHAR, person VARCHAR)")
step(f"links: {n_ids} linked ids")

# --- 3. Events -----------------------------------------------------------------
# One query, so the 10M+ rows are only materialized once:
#   typed    convert every copy first. SQL macros are pasted in wherever they
#            are referenced, so converting BEFORE the window (which stores its
#            input) means ts_ms() runs once per row instead of once per use.
#            event_type becomes a small code (NULL = not exactly view/cart/purchase).
#            A bad value just becomes NULL here; it is rejected after dedup.
#   dedup    BEFORE validation: per event_id keep the copy with the smallest
#            ingest_ts (copies with equal ingest_ts are identical, so any one).
#            If that copy is invalid the event is gone; a later valid copy does
#            not rescue it.
#   validate the kept copy, drop late events, attach the person.
# Only purchases keep their product_id: the category is only ever used for
# top_category, which only looks at purchases (~3% of events), so the products
# join happens there instead of on every event.
# ingest_ts is always valid F1 (rules.md), so DuckDB's strptime converts it,
# about 5x faster than the fully validating ts_ms().
# Revenue only counts purchases; view/cart price is ignored whatever it holds.
con.execute(f"""
    CREATE OR REPLACE TABLE events AS
    WITH typed AS (
        SELECT event_id, user_id, product_id,
               ts_ms(ts) AS ts,
               epoch_ms(try_strptime(ingest_ts, '%Y-%m-%dT%H:%M:%SZ')) AS ingest,
               CASE event_type WHEN 'view' THEN 0 WHEN 'cart' THEN 1 WHEN 'purchase' THEN 2 END::TINYINT AS etype,
               CASE WHEN event_type = 'purchase' THEN price(price_cents) END AS price
        FROM events_raw
        WHERE event_id IS NOT NULL
    ), dedup AS (
        SELECT * FROM typed
        QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY ingest) = 1
    )
    SELECT coalesce(pm.person, t.user_id) AS person,
           t.ts,
           t.etype = 2 AS is_purchase,
           coalesce(t.price, 0) AS price,
           CASE WHEN t.etype = 2 THEN t.product_id END AS product_id
    FROM dedup t
    LEFT JOIN person_map pm ON pm.id = t.user_id
    WHERE t.user_id IS NOT NULL
      AND t.product_id IS NOT NULL
      AND t.ts BETWEEN {WINDOW_START} AND {WINDOW_END - 1}
      AND t.etype IS NOT NULL
      AND (t.etype <> 2 OR t.price IS NOT NULL)
      AND t.ingest - t.ts <= {LATE_MS}
""")
step(f"events: {count('events')} valid")

# --- 4. Sessions ---------------------------------------------------------------
# Per person, sorted by ts: a gap OVER 30 min from the previous event starts a
# new session. Equal timestamps must share a session: only one of a group of
# tied rows sees the gap to the previous distinct ts, so the running sum uses
# the default RANGE frame, which gives every tied row the same total.
con.execute(f"""
    CREATE OR REPLACE TABLE events_s AS
    WITH flagged AS (
        SELECT *,
               CASE WHEN ts - lag(ts) OVER (PARTITION BY person ORDER BY ts) <= {GAP_MS}
                    THEN 0 ELSE 1 END AS new_session
        FROM events
    )
    SELECT *, sum(new_session) OVER (PARTITION BY person ORDER BY ts) AS session_no
    FROM flagged
""")

# One row per session. The day is the UTC date of the session's FIRST event;
# every event and purchase of the session counts on that day.
con.execute("""
    CREATE OR REPLACE TABLE sessions AS
    SELECT person, session_no,
           min(ts) AS start_ts,
           (DATE '1970-01-01' + (min(ts) // 86400000)::INTEGER) AS day,
           count(*) AS n_events,
           sum(price) AS revenue,
           bool_or(is_purchase) AS has_purchase
    FROM events_s
    GROUP BY person, session_no
""")
step(f"sessions: {count('sessions')}")

# --- 5. Clicks -----------------------------------------------------------------
# Dedup on click_id (copies are identical once quotes are stripped), validate,
# attach person and channel. An inner join with campaigns drops unknown ones.
con.execute(f"""
    CREATE OR REPLACE TABLE clicks AS
    WITH dedup AS (
        -- ts_ms() inside the aggregate runs once per row (see the events note)
        SELECT click_id, any_value(user_id) AS user_id, any_value(campaign_id) AS campaign_id,
               any_value(ts_ms(ts)) AS ts
        FROM clicks_raw
        WHERE click_id IS NOT NULL
        GROUP BY click_id
    )
    SELECT t.click_id, coalesce(pm.person, t.user_id) AS person, t.campaign_id, t.ts
    FROM dedup t
    JOIN campaigns c ON c.campaign_id = t.campaign_id
    LEFT JOIN person_map pm ON pm.id = t.user_id
    WHERE t.user_id IS NOT NULL
      AND t.ts BETWEEN {CLICK_WINDOW_START} AND {WINDOW_END - 1}
""")
step(f"clicks: {count('clicks')} valid")

# --- 6. Attribution ------------------------------------------------------------
# Credit: the person's click with the latest ts <= session start, within 7 days;
# on a tie in ts, the smallest click_id. First collapse ties so each
# (person, ts) has one winner, then an ASOF join finds the latest one <= start.
con.execute(f"""
    CREATE OR REPLACE TABLE credited AS
    WITH best_at_ts AS (
        SELECT person, ts, arg_min(campaign_id, click_id) AS campaign_id
        FROM clicks
        GROUP BY person, ts
    )
    SELECT s.person, s.session_no, s.day, s.revenue, s.has_purchase,
           CASE WHEN b.ts IS NOT NULL AND s.start_ts - b.ts <= {LOOKBACK_MS}
                THEN b.campaign_id ELSE 'organic' END AS campaign_id
    FROM sessions s
    ASOF LEFT JOIN best_at_ts b
      ON b.person = s.person AND b.ts <= s.start_ts
""")
step("attribution: credit")

# Assists: every campaign with at least one of the person's clicks in
# [start - 7 days, start] assists the session, once per campaign.
# A click at time t assists every session starting in [t, t + 7 days]. Per
# (person, campaign), clicks less than 7 days apart have overlapping ranges, so
# they are merged into one interval [first click, last click + 7 days]
# (gaps-and-islands, same pattern as sessions). The merged intervals of a
# campaign never overlap, so each session falls in at most one of them: a plain
# join then counts each (session, campaign) exactly once, with no DISTINCT, and
# is aggregated straight to (campaign, day) counts. Clicks are DISTINCT on
# (person, campaign, ts) first, so ties cannot break the ROWS frame.
con.execute(f"""
    CREATE OR REPLACE TABLE assist_counts AS
    WITH c AS (SELECT DISTINCT person, campaign_id, ts FROM clicks),
    g AS (SELECT *, CASE WHEN ts - lag(ts) OVER w <= {LOOKBACK_MS} THEN 0 ELSE 1 END AS nw
          FROM c WINDOW w AS (PARTITION BY person, campaign_id ORDER BY ts)),
    grp AS (SELECT *, sum(nw) OVER (PARTITION BY person, campaign_id ORDER BY ts ROWS UNBOUNDED PRECEDING) AS gid FROM g),
    iv AS (SELECT person, campaign_id, min(ts) AS lo, max(ts) + {LOOKBACK_MS} AS hi FROM grp GROUP BY person, campaign_id, gid)
    SELECT iv.campaign_id, s.day, count(*) AS assisted_sessions
    FROM sessions s JOIN iv ON iv.person = s.person AND s.start_ts BETWEEN iv.lo AND iv.hi
    GROUP BY iv.campaign_id, s.day
""")
step("attribution: assists")

# --- 7. Outputs ----------------------------------------------------------------
os.makedirs(os.path.join(OUT, "user_daily"), exist_ok=True)
os.makedirs(os.path.join(OUT, "campaign_daily"), exist_ok=True)

# user_daily: one row per person and day with at least one session.
# top_category: highest summed purchase price that day; ties -> byte-wise
# smallest category; NULL when there were no purchases.
con.execute(f"""
    COPY (
        WITH per_day AS (
            SELECT person, day, count(*) AS sessions, sum(n_events) AS events,
                   sum(revenue) AS revenue_cents
            FROM sessions
            GROUP BY person, day
        ), cat AS (
            SELECT s.person, s.day, coalesce(p.category, 'unknown') AS category, sum(e.price) AS rev
            FROM events_s e
            JOIN sessions s ON s.person = e.person AND s.session_no = e.session_no
            LEFT JOIN products p ON p.product_id = e.product_id
            WHERE e.is_purchase
            GROUP BY ALL
        ), top AS (
            SELECT person, day, category AS top_category
            FROM cat
            QUALIFY row_number() OVER (PARTITION BY person, day ORDER BY rev DESC, category) = 1
        )
        SELECT d.person::VARCHAR AS user_id, d.day::DATE AS day, d.sessions::BIGINT AS sessions,
               d.events::BIGINT AS events, d.revenue_cents::BIGINT AS revenue_cents,
               t.top_category::VARCHAR AS top_category
        FROM per_day d
        LEFT JOIN top t ON t.person = d.person AND t.day = d.day
    ) TO '{os.path.join(OUT, "user_daily", "part-0.parquet")}' (FORMAT parquet)
""")

# campaign_daily: one row per (campaign, day) credited or assisted.
con.execute(f"""
    COPY (
        WITH cred AS (
            SELECT campaign_id, day, count(*) AS sessions,
                   count(*) FILTER (WHERE has_purchase) AS purchasing_sessions,
                   sum(revenue) AS revenue_cents
            FROM credited
            GROUP BY campaign_id, day
        ), assist AS (
            SELECT * FROM assist_counts
        ), joined AS (
            SELECT coalesce(c.campaign_id, a.campaign_id) AS campaign_id,
                   coalesce(c.day, a.day) AS day,
                   coalesce(c.sessions, 0) AS sessions,
                   coalesce(c.purchasing_sessions, 0) AS purchasing_sessions,
                   coalesce(c.revenue_cents, 0) AS revenue_cents,
                   coalesce(a.assisted_sessions, 0) AS assisted_sessions
            FROM cred c
            FULL OUTER JOIN assist a ON a.campaign_id = c.campaign_id AND a.day = c.day
        )
        SELECT j.campaign_id::VARCHAR AS campaign_id,
               CASE WHEN j.campaign_id = 'organic' THEN 'organic' ELSE cp.channel END::VARCHAR AS channel,
               j.day::DATE AS day, j.sessions::BIGINT AS sessions,
               j.purchasing_sessions::BIGINT AS purchasing_sessions,
               j.revenue_cents::BIGINT AS revenue_cents,
               j.assisted_sessions::BIGINT AS assisted_sessions
        FROM joined j
        LEFT JOIN campaigns cp ON cp.campaign_id = j.campaign_id
    ) TO '{os.path.join(OUT, "campaign_daily", "part-0.parquet")}' (FORMAT parquet)
""")
step("outputs written")

# Both Parquet files are complete and closed (COPY returns only when done).
# Exit without the normal teardown: freeing DuckDB's gigabytes of memory and the
# Python objects only costs time. Exit code 0 tells the scorer we are done.
sys.stdout.flush()
os._exit(0)
