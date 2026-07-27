# PADOC

PADOC is a Rust tool for compressing large AI profiler traces into a compact,
queryable representation. It accepts Chrome trace JSON produced by tools such
as PyTorch Profiler and supports analysis without first rebuilding every raw
event.

The public implementation favors predictable resource use and lossless event
reconstruction. Each input file is compressed independently; PADOC does not
perform cross-rank template merging.

## Install

PADOC currently builds from source with the stable Rust toolchain:

```bash
git clone https://github.com/lvjx04/padoc-rs.git
cd padoc-rs
cargo build --release
```

The binary is written to `target/release/padoc`.

## Quick start

Compress one trace:

```bash
padoc compress trace.json --output trace.padoc
```

Inspect and verify it:

```bash
padoc inspect trace.padoc
padoc verify trace.json --artifact trace.padoc
```

Reconstruct Chrome trace JSON:

```bash
padoc decompress trace.padoc --output restored.json
```

For a directory of per-rank trace files, PADOC creates one artifact per input
and a small manifest:

```bash
padoc compress-dir ./traces --output ./artifacts --workers 4
```

`--workers` bounds file-level concurrency. It does not merge ranks or create
additional compression threads inside an artifact.

## In-situ analysis

List the available tasks:

```bash
padoc list
```

Run a task directly on an artifact:

```bash
padoc analyze trace.padoc --task operator_hotspot
padoc analyze trace.padoc --task stream_load_balance
```

The initial public task set is intentionally small. Tasks that rely on
workload-specific naming or incomplete CPU-GPU attribution remain on the
research branch until their semantics are stable.

## Supported data

PADOC currently supports Chrome trace JSON and gzip-compressed `.json.gz`
objects with a `traceEvents` array. It preserves the event fields used by AI
profilers:

- `name`, `ts`, `dur`, `cat`, `ph`, `pid`, and `tid`
- `args`, including nested JSON values
- optional `id`, `bp`, and `s`
- metadata names, process/thread coordinates, argument payloads, duplicates,
  and input order

Timestamps are normalized internally and restored during JSON reconstruction.
Event ordering may differ after decompression, but the supported event fields
are verified as a multiset.

The artifact format is versioned but is still pre-1.0. Compatibility guarantees
will begin with the first stable release.

## Design

PADOC groups events by stable signatures, stores per-instance values in typed
columns, and records flat template/instance references per stream. Large JSON
files are parsed as a stream, and artifact payloads are serialized directly
through zstd without materializing an intermediate MessagePack buffer. The
stable encoder deliberately avoids recursive call trees and cross-rank merging.

See [docs/design.md](docs/design.md) and [docs/artifact-format.md](docs/artifact-format.md)
for the current engineering contract.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Research baselines and experiment drivers are maintained separately on the
`research/baselines` branch.

## License

Apache License 2.0. See [LICENSE](LICENSE).
