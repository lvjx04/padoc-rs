# padoc — AI profiler trace compression in Rust

Clean rewrite of the original Python `perflowai/padoc` package.  Same paper,
same compression / analysis semantics, much faster pipeline and a single
crate that can be used as a CLI binary or a library.

## Quick start

```bash
# Build
cargo build --release

# List compressors and analysis tasks
./target/release/padoc list

# Compress a chrome-trace JSON file
./target/release/padoc compress trace.json -o trace.pdc

# Run an analysis task on the raw trace
./target/release/padoc analyze trace.json --task operator_hotspot

# Bench every compressor on a set of traces
./target/release/padoc bench compress --datasets a.json,b.json

# Synthetic-trace scalability sweep over GPU count
./target/release/padoc bench scalability --dimension gpus --values 1,2,4,8
```

## Architecture

```
src/
├── lib.rs                       crate-level re-exports + Error
├── main.rs                      CLI driver (clap)
├── event.rs                     Event / MergeEvent / KernelEvent / templates
├── node.rs                      compressed call-tree node enum
├── trace.rs                     Trace + CompressedTrace + chrome-trace JSON I/O
├── slp.rs                       segmented linear predictor
├── synthetic.rs                 deterministic synthetic trace generator
├── storage_breakdown.rs         per-component byte profiling
├── tree_stats.rs                tree-shape statistics
├── utils.rs                     name normalisation + logging
├── compressor/
│   ├── mod.rs
│   ├── config.rs                CompressorConfig + ablation presets
│   ├── core.rs                  TemplateCompressor driver
│   ├── call_tree.rs             per-rank tree build (CPU stack + GPU pairing)
│   └── structural.rs            SameCpu grouping + anchor matching + finalise
├── baselines/
│   ├── mod.rs                   BaselineCompressor trait + registry
│   ├── raw.rs                   raw_json / raw_msgpack
│   ├── gzip.rs                  gzip_json / gzip_msgpack
│   ├── scalatrace.rs            ScalaTrace adaptation
│   ├── tracezip.rs              TraceZip adaptation
│   └── padoc.rs                 wrapper over TemplateCompressor
├── analysis/
│   ├── mod.rs                   AnalysisTask trait + registry
│   ├── operator_hotspot.rs      [in-situ] top-N operator dur
│   ├── stream_load_balance.rs   [in-situ] per-GPU-stream busy time
│   ├── layer_operator_balance.rs[in-situ] per-layer dur
│   └── parallel_group.rs        TP/DP/PP/EP inference from NCCL ops
└── bench/
    ├── mod.rs
    ├── metrics.rs               record types
    ├── runner.rs                compression / analysis matrices
    ├── scalability.rs           synthetic sweep runner
    ├── parallel.rs              rayon-based throughput benchmark
    └── report.rs                markdown rendering
```

## Design choices

| | Python (old) | Rust (this repo) |
|---|---|---|
| node hierarchy | 5 classes | one `Node` enum |
| event hierarchy | 4 classes | `Event` + `Template` enum |
| sibling grouping | O(n²) double loop | hash-bucket O(n) |
| JSON ingest | `json` stdlib | `simd-json` |
| storage format | msgpack (no zstd) | msgpack + zstd |
| HTA baselines | yes | dropped per design |
| Python interop | n/a | none — pure Rust |

## Status (initial scaffold)

Working end-to-end on real PyTorch profiler traces (ops + kernel + nccl):

- chrome-trace JSON ingestion via `simd-json`
- TemplateCompressor with full call-tree + structural compression + anchor matching
- All 5 baselines + ablation presets round-trip
- 4 analysis tasks (3 with PADOC in-situ implementation)
- Bench harness CLI: `bench compress` / `bench analyze` / `bench scalability` / `bench parallel`
- Synthetic trace generator
- Storage breakdown + tree-shape stats
- 15/15 unit + integration tests pass

Reference numbers on `tests/example_trace/profiler_585.json` (294,918 events, 69.8 MiB):

| compressor | ratio | secs | MB/s |
|---|---:|---:|---:|
| raw_json | 1.26x | 0.45 | 156 |
| gzip_json | 21.5x | 0.75 | 93 |
| scalatrace | 34.2x | 0.28 | 252 |
| tracezip | 29.1x | 0.26 | 268 |
| padoc | 25.5x | 0.32 | 219 |

PADOC is currently behind ScalaTrace because the foundation does not yet
SLP-encode the per-template numeric arrays on disk (we rely on the trailing
zstd pass).  The remaining work to land paper-quality PADOC numbers is
captured in [HANDOFF.md](HANDOFF.md).

## License

Apache-2.0.
