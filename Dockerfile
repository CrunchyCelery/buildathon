# Shopping Sessions: native Rust pipeline (rust/).
# The scorers are linux/amd64 (Intel Xeon) and build this file natively, with
# network; only the run is timed, so compiling here costs nothing.
# The DuckDB pipeline (pipeline/) stays in the repo as the correctness
# reference: both produce identical output on every dataset we tested.

# --- Build stage: compile the release binary -----------------------------------
FROM rust:1.100.0-slim-bookworm AS build
WORKDIR /src
COPY rust/Cargo.toml rust/Cargo.lock ./
COPY rust/src ./src
# --locked: use exactly the dependency versions in Cargo.lock.
RUN cargo build --release --locked

# --- Run stage: just the binary on a small image --------------------------------
FROM debian:bookworm-slim

# The scorer runs the container as uid 1000; give that uid a name and a home.
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app

COPY --from=build /src/target/release/shopping_sessions /app/shopping_sessions

# At run time the root filesystem is read-only, the user is 1000:1000, there is
# no network, /data is read-only, and /output and /scratch are the only writable
# places. The binary reads DATA_DIR, OUTPUT_DIR and NCPU, writes only under
# OUTPUT_DIR, and exits 0 once both tables are written.
WORKDIR /app
CMD ["/app/shopping_sessions"]
