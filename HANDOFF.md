# HANDOFF — padoc-rs initial scaffold

This document captures the state of the Rust rewrite **at the initial commit**
so anyone (you, a remote Cursor, a co-author) can pick it up without reading
the chat log.

## What this repo is

- Clean Rust rewrite of the Python `perflowai/padoc` package.
- Single Cargo crate: library + `padoc` CLI binary.
- No HTA baselines (dropped per design — paper compares against raw / gzip /
  ScalaTrace / TraceZip only).
- No Python interop — this is a hard fork.

The original Python repo is at `/Users/lvjiaxin/Work/PerFlow-AI`.  Use it
as the **reference implementation** when porting subtle behaviour.

## Quick verification

```bash
cargo test                               # 15/15 pass
cargo run --release -- list              # show all compressors / tasks
cargo run --release -- bench compress \
    --datasets ../PerFlow-AI/tests/example_trace/out-1024.json,\
../PerFlow-AI/tests/example_trace/profiler_585.json
```

## Done in this scaffold

- Cargo project, `clap` CLI bones, `tracing` logging
- Core types: `Event`, `MergeEvent`, `KernelEvent`, `MergeKernelEvent`,
  `Template`, `Node` (single enum, no class hierarchy)
- chrome-trace JSON ingestion via `simd-json`
- `CompressedTrace` serialisation as zstd-wrapped msgpack
- `TemplateCompressor` with:
  - per-stream stack-based call-tree build
  - hash-bucket O(n) sibling grouping into `SameCpu`
  - cross-instance anchor matching that recursively merges children
  - per-template name-pattern transpose / args dedup
  - SLP encoder ready (encoded but not yet written to disk — see TODO)
- `CompressorConfig` ablation switches: `enable_structural` /
  `enable_anchor_matching` / `enable_slp` / `enable_args_dedup` /
  `enable_kernel_links` / `enable_name_pattern` + 7 presets
- 5 baselines: `raw_json`, `raw_msgpack`, `gzip_json`, `gzip_msgpack`,
  `scalatrace`, `tracezip`, `padoc`
- 4 analysis tasks: `operator_hotspot`, `stream_load_balance`,
  `layer_operator_balance`, `parallel_group`
  - 3 of the 4 have a PADOC in-situ implementation
- Bench harness:
  - compression matrix (`bench compress`)
  - analysis matrix (`bench analyze`)
  - scalability sweep on synthetic traces (`bench scalability`)
  - parallel compression speedup (`bench parallel`, rayon-based)
- Synthetic trace generator with deterministic seed
- Storage breakdown + tree-shape statistics
- 15/15 unit + integration tests

## Known gaps (next steps, ordered by paper-impact)

### P0 — Compression ratio

PADOC currently sits at 25-30x on real traces, behind ScalaTrace at 30-34x.
The Python reference reaches ~50-100x.  Closing the gap:

1. **SLP on disk.** Today `Template::ts/dur/id` are `Vec<i64>`.  Switch them
   to a `SlpColumn` (or a `SlpOrRaw` enum that stays raw if SLP would be
   bigger) so arithmetic-progression timestamps land as `(start, step,
   length)` triples.  See `src/slp.rs::SlpColumn` — already implemented and
   tested; just not wired into the on-disk schema.
2. **Cross-rank template sharing.** Today every rank's `TemplateCompressor`
   builds an independent template table.  Add a global pre-pass that
   intern's templates across ranks before merging tree edges, so a
   1024-rank trace shares one template per operator instead of 1024.
3. **Args-column dedup at the column level.** `args_values` is row-major
   today (one `Vec<ArgValue>` per instance).  Switch to columnar
   (`Vec<Column>` with each column a `RawOrConst` enum) so constant /
   small-domain columns shrink to a singleton.
4. **Name-pattern dictionary.** Today every template carries its own
   `name_pattern`.  A global `Vec<String>` dict + per-template index
   shrinks repeated patterns (`layers.0.attn.qkv_proj`).

### P1 — Analysis correctness / coverage

5. Wire `lookup_gpu_event` in `compressor/call_tree.rs` so CPU launches
   actually carry the paired GPU kernel event into `KernelLaunch` nodes.
   Currently the GPU template is created and stored, but the lookup that
   pulls the `Event` payload (for kernel-launch in-situ analysis) is a
   stub.
6. Lossless decompression: `cmd_decompress` currently emits a placeholder.
   Implement `CompressedTrace::to_chrome_trace()` that walks the tree,
   reconstructs each event, and writes a chrome-trace JSON.

### P2 — Bench fidelity

7. The `bench analyze` command currently reuses an in-memory `Trace` for
   non-PADOC paths because `BaselineCompressor::decompress` is a stub for
   `raw` / `gzip`.  Make every baseline's `decompress` actually round-trip
   so the analysis-matrix numbers reflect real decompression cost.
8. Add `bench storage_breakdown` and `bench tree_stats` subcommands so the
   paper's storage / shape figures come out of one CLI.

### P3 — Cluster wiring

9. The Python `bench` harness reads a `manifest.json` describing real
   1024-rank traces.  Re-create that manifest format here and add a
   `bench compress --manifest manifest.json` path.
10. Add a `--from-dir` flag to `bench compress` so 1024 per-rank JSON files
    are loaded in parallel (rayon `from_dir_par`).

## File ownership map

| area | file(s) | when to touch |
|---|---|---|
| compressor algorithm | `src/compressor/` | P0.1, P0.4 |
| storage format | `src/event.rs`, `src/trace.rs` | P0.1, P0.3 |
| baselines | `src/baselines/` | P2.7 |
| in-situ paths | `src/analysis/` | P1.5 |
| bench CLI | `src/main.rs`, `src/bench/` | P2.8, P3.9, P3.10 |
| docs | `README.md`, `HANDOFF.md` | always update with substantive changes |

## Testing discipline

- Every algorithmic change must keep `cargo test` green.  Tests live in
  `tests/end_to_end.rs` (integration) + `src/**/tests` (units).
- `cargo build --release` is the bench compile path — most tests run on
  small synthetic traces, the realistic numbers come from
  `cargo run --release -- bench compress ...` against the example traces.
- The Python reference repo's `tests/example_trace/` has both a 1024-rank
  merged sample (`out-1024.json`, 5.3 MiB) and a single-rank profile
  (`profiler_585.json`, 70 MiB).  Use them as smoke tests for any change
  to compressor or baselines.

## Deltas from the original Python implementation

If you came from the Python repo and a behaviour surprises you, check here.

- `MergeEvent::name_nums` is `enum NameNums { Empty, Rows, Columnar }`
  rather than two parallel attributes; the columnar form is what the
  Python `compress_names()` produced.
- `Node::SameCpu::slots` is `Vec<Vec<Node>>` (per-instance trailers) — not
  flattened — so on decompress the trailers can be re-attached to the
  right instance.
- The compressed format is **always** zstd-wrapped msgpack.  The Python
  `post_zstd: bool` switch is gone; if you want raw msgpack measurement
  use the `raw_msgpack` baseline.
- HTA-shaped analysis tasks (`gpu_kernel_breakdown`,
  `comm_comp_overlap`, `temporal_breakdown`) are dropped.
- The original `bench/manifest.json` driver is not yet ported — see P3.
