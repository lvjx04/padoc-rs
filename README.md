# PADOC

PADOC is a CLI for turning large AI-profiler Chrome trace JSON files into
compact, independently usable artifacts. It reconstructs supported events
losslessly and runs a small set of analyses without first rebuilding raw JSON.

The CLI and versioned artifact behavior are PADOC's primary supported
interfaces. The Rust library is available for experimentation, but its API is
pre-1.0 and may evolve.

## Install

Build a checkout with the stable Rust toolchain:

```bash
git clone https://github.com/lvjx04/padoc-rs.git
cd padoc-rs
cargo build --release
```

Or install the current repository version directly:

```bash
cargo install --git https://github.com/lvjx04/padoc-rs
```

## Basic workflow

```bash
padoc compress trace.json --output trace.padoc
padoc inspect trace.padoc
padoc verify trace.json --artifact trace.padoc
padoc decompress trace.padoc --output restored.json
```

`inspect` prints JSON metadata, including the format version read from the
artifact header. `verify` compares all supported fields as event multisets.

Compress a directory of per-rank traces into one artifact per input plus a
manifest:

```bash
padoc compress-dir ./traces --output ./artifacts --workers 4
```

`--workers` is the maximum number of files processed concurrently. PADOC does
not merge ranks or add nested compression workers within an artifact.

## In-situ analysis

The stable task set is deliberately small:

- `operator_hotspot`: top CPU operators by total duration
- `stream_load_balance`: busy-time distribution across GPU streams

```bash
padoc list
padoc analyze trace.padoc --task operator_hotspot
```

## Supported data and artifacts

PADOC accepts Chrome trace JSON and gzip-compressed `.json.gz` inputs with a
`traceEvents` array. It preserves the supported event fields, nested JSON
arguments, optional identifiers, and metadata records. Timestamps are
normalized internally and restored during reconstruction.

Artifacts have a validated header containing their format version and codec.
The current writer emits v2; the reader accepts supported legacy versions and
rejects unknown versions, codecs, and flags.

## Verified results

These full-input results use artifact format v2:

| Dataset | Ranks | Events | Input bytes | PADOC bytes | Size/input | Compression | Wall time | Peak RSS (KiB) | Workers | Lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LeWorldModel | 2 | 3,469,389 | 927,329,976 | 38,333,493 | 4.1337% | 24.191x | 8.37s | 3,564,180 | 2 | 2/2 |
| Qwen3 | 256 | 33,813,574 | 7,422,237,193 | 310,439,485 | 4.1826% | 23.909x | 14.66s | 5,663,212 | 16 | 256/256 |
| UnifoLM world model | 4 | 80,223,071 | 24,087,743,045 | 774,296,478 | 3.2145% | 31.109x | 3m27.17s | 54,514,728 | 2 | 4/4 |
| LLaMA profiler | 1,024 | 301,288,116 | 75,106,905,369 | 2,732,778,989 | 3.6385% | 27.484x | 1m22.07s | 6,807,336 | 16 | 1,024/1,024 |

The results were produced from commit
`a439ad9e86c14c05a096c23aab893de951e9ec4f`. Compression is CPU-side and used
ordinary Chrome trace JSON inputs. Every row was verified exhaustively with
event-multiset comparison; no sampling was used.

## Resource behavior and limitations

- PADOC does not perform cross-rank merging.
- Reconstructed event order and JSON formatting may differ from the input;
  supported event content is lossless.
- One active file is retained as decoded and compressed structures, so memory
  is proportional to the largest active trace.
- `--workers` bounds concurrent files, not memory independently of file size.
- Full verification currently materializes overlapping original,
  reconstructed, and comparison representations. On two verified 4.27–4.28 GB
  deeply nested traces, verification peaked at about 56.1–56.2 GiB.

See [the design](docs/design.md) and
[artifact format](docs/artifact-format.md) for the engineering contracts.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo package --offline
```

Research baselines and experiment drivers are maintained separately on the
`research/baselines` branch.

## License

Apache License 2.0. See [LICENSE](LICENSE).
