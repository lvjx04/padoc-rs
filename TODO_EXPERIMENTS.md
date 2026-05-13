# Remaining experiments

What's already in `EXPERIMENTS.md` (Sections 1–7) maps onto the original paper plan as follows. The first column shows the paper section, the second whether it's done, the third the gap.

| # | Paper section | Status | Gap |
|---|---|---|---|
| T1 | Trace dataset – dense 70B 1024-rank training | ✅ | `llama_full` |
| T2 | Trace dataset – MoE 671B 1024-rank training | ❌ | **need an MoE trace** (DeepSeek-V3 / Mixtral) |
| T3 | Trace dataset – ViT | ❌ | **need a ViT trace** |
| T4 | Trace dataset – LeWorldModel | ✅ | `leworldmodel_full` |
| T5 | Extra workloads | ✅ | `qwen3_full`, `unifolm_full` |
| A1 | Analysis – operator hotspot | ✅ | done |
| A2 | Analysis – overlap (compute / comm) | ⚠️ | `stream_load_balance` covers per-stream busy time. Need a true compute–comm **overlap fraction** task that reports `overlap_us / kernel_us` per rank. |
| A3 | Analysis – per-layer operator balance | ✅ | done |
| A4 | Analysis – parallel-group / per-rank load balance | ✅ | `rank_load_balance` |
| A5 | Analysis – additional | open | candidate: stragglers / cross-rank slowest-N report; not started |
| B1 | Baselines – uncompressed JSON | ✅ | `raw_json` |
| B2 | Baselines – ScalaTrace | ✅ | cross-rank version + per-rank fallback for llama |
| B3 | Baselines – TraceZip | ✅ | cross-rank version + per-rank fallback for llama |
| B4 | Bonus: gzip(JSON) | ✅ | `gzip_json` |
| C1 | Compression ratio – every baseline × every trace | ✅ | EXPERIMENTS § 2 |
| D1 | Analysis perf – every task × every baseline × every trace | ✅ | EXPERIMENTS § 4 |
| E1 | Storage breakdown – #templates / template share / ts share / soft-link share | ⚠️ | We have **in-memory** breakdown (§ 3.1). Still need an **on-disk** byte breakdown of the PADOC zst (one number per region: templates, ts, args, name_nums, node tree, soft-link edges). Easy: extend `inspect_artifact` to optionally serialize each field separately and report sizes. |
| F1 | Analysis perf breakdown – where the time goes | ⚠️ | We have load + decompress + analyse; need an inner profile of analyse (e.g. `flamegraph` per task, or instrumented timers around column traversal vs. node-tree walk vs. hash inserts). |
| G1 | Scalability – with #GPUs | ❌ | Need PADOC compress + analyse times across `gpus = {1, 8, 64, 256, 1024}`. We have 1024 (llama) and 256 (qwen3); need synthetic 1/8/64 (or sub-sample llama). |
| G2 | Scalability – with #layers | ❌ | Need a sweep where the same workload is run at `layers = {8, 16, 32, 64, 128}`. Synthetic generator that replays the llama template tree N times. |
| G3 | Scalability – with #iterations | ❌ | Same idea, vary the number of training steps captured in the trace. |
| H1 | Parallel scalability – compress speed vs #threads | ⚠️ | Old `scalability.sh` script (now removed) had a 1/2/4/8/16/32 worker sweep. **Need to redo** under v6 typed-column code (sc1). |
| H2 | Parallel scalability – analyse speed vs #threads | ❌ | All four analyses are currently single-threaded. Decision needed: do we want to parallelise `rank_load_balance` and `operator_hotspot` over templates / ranks, or is single-thread fast enough? |
| I1 | Ablation – storage | ❌ | Toggle off, one at a time: typed numeric columns (force i64), digit packing (force `Vec<String>`), arg dictionary (force `PerInstance`), structural template tree (force per-event), zstd. Report on-disk size and in-memory size delta on `llama_full`. |
| I2 | Ablation – analyse perf | ❌ | Same toggles, report `analyze-batch` total + analyse-only seconds per task on `llama_full`. |

Legend: ✅ already in `EXPERIMENTS.md`; ⚠️ partial / needs an extra small experiment; ❌ not started.

## Suggested ordering for the next push

1. **E1 on-disk byte breakdown** — half a day; just extends `inspect_artifact` to (a) re-serialize each `CompressedTrace` field independently and (b) report the on-disk bytes per region. Plugs straight into the storage-profile figure in the paper.
2. **H1 compress-time scalability sweep** — half a day; reuse the manifest, vary `--workers` 1/2/4/8/16/32/64 with `RAYON_NUM_THREADS` matched and `taskset` pinned. Run on sc1+sc4 in parallel.
3. **A2 overlap fraction task** — one day; new `analysis::overlap_fraction` that walks the GPU template tree and reports compute kernel time, comm kernel time, and their interval-overlap per rank. Needs a `run_in_situ` and `run_raw` implementation.
4. **I1 + I2 ablation** — one day; gated by feature flags or cli overrides on the existing build. We already have the typed columns wired into `compact()`; turning each one off is `if disable_typed { return PerInstance(...) }`.
5. **G1 GPU-count scalability** — one day; sub-sample the llama profiler dir down to `{1, 8, 64, 256}` ranks (already trivially supported by passing a manifest of the chosen subset) and re-run `bench compress` + `analyze-batch`.
6. **G2 / G3 layer / iteration scalability** — needs a synthetic trace generator (we don't have one yet). Lower priority; the storage and analyse-time ablation usually carries the same paper claim.
7. **T2 / T3 missing datasets** — depends on whether we can pull a DeepSeek-V3 / Mixtral / ViT profile trace. If not, drop those rows and lean harder on llama / qwen3 / unifolm / leworldmodel.

## Things that are NOT blocking the paper draft

- F1 inner profile of analyse — only matters if reviewers ask why a particular task is slow. We can defer until we see the first review.
- H2 multi-threaded analyse — single-threaded numbers already win; if a reviewer pushes back we can add it later.
- A5 additional analysis – nice-to-have only.

## Cluster machines

- `ssh sc1` — 32 NUMA cores, 256 GiB. Used so far for small-dataset compress + analyse.
- `ssh sc4` — 64 NUMA cores, 256 GiB. Used so far for `llama_full` compress + per-rank baseline analyse.
- Both mount the same `/mnt/treasure/ljx/` so artifacts are visible from either host.
- All four PADOC artifacts live at `/mnt/treasure/ljx/artifacts_v6/{leworldmodel,qwen3,unifolm,llama}_full.padoc.zst` (rename to `/mnt/treasure/ljx/artifacts/` on the next clean run via `scripts/compress_all.sh`).
