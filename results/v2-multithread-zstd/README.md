# results/v2-multithread-zstd

**Failed optimisation attempt — kept on disk so we don't repeat it.**

## What we tried

Three changes on top of v1-baseline:

1. `compressor::core::finalize_templates` parallelised via
   `rayon::par_iter_mut().for_each` (per-template SLP / args-dedup).
2. `compressor::merge::merge_shards` Phase 2 parallelised — every
   shard's call-tree rewrite runs in parallel via `into_par_iter()`.
3. `trace.rs` — both `to_bytes` and `write_to_path` switched to a
   streaming msgpack → zstd pipeline **with `encoder.multithread(N)`
   enabled** (N = `min(available_parallelism, 16)`).

## Per-dataset numbers (commit 8b91b8a, run 2026-05-13 sc1, --workers 32)

| dataset           | parallel | merge   | serialize | total   | ratio  |
|-------------------|---------:|--------:|----------:|--------:|-------:|
| leworldmodel_full |   12.0 s |   2.7 s |     3.1 s |  17.8 s | 23.88× |
| qwen3_full        |   36.1 s |   8.6 s |    28.0 s |  72.7 s | 26.26× |
| unifolm_full      |  170.5 s |  41.4 s |    70.9 s | 282.8 s | 31.63× |
| llama_full        |   84.7 s | 147.6 s |   292.4 s | 524.7 s | 29.43× |

## Comparison vs v1-baseline

| dataset    | phase     | v1     | v2     | delta  |
|------------|-----------|-------:|-------:|-------:|
| llama_full | parallel  |   83.3 |   84.7 |  +1.4  |
| llama_full | merge     |  335.6 |  147.6 | **−188.0 ✓** |
| llama_full | serialize |   88.4 |  292.4 | **+204.0 ❌** |
| llama_full | **total** | **507.2** | **524.7** | **+17.5** |

Same shape on every dataset:

| dataset    | merge Δ | serialize Δ |
|------------|--------:|------------:|
| lewm       | +0.7    | +2.1        |
| qwen3      | −14.1   | +20.2       |
| unifolm    | −27.8   | +49.2       |
| llama      | −188.0  | +204.0      |

## Diagnosis

Parallel `merge_shards` and parallel `finalize_templates` both
delivered as expected — the merge-tree-rewrite phase becomes nearly
linear-speedup parallel work and we got 2-3× speedups proportional
to the merge bottleneck.

The serialize phase, however, regressed 3-4× across the board.
`zstd::Encoder::multithread(N)` performs poorly when the producer
(rmp_serde::write_named) writes ~MiB-scale chunks: each chunk has
to be split into per-thread jobs by zstdmt, dispatched, ack'd, and
the output reordered.  At our chunk sizes the coordination overhead
exceeds the parallel-compression gain — and on top of that
multi-threaded zstd uses smaller per-thread compression windows,
which shaves another ~0.3% off the ratio (lewm 37.48 → 37.04 MiB).

## Resolution

v3 drops `multithread()` from both `to_bytes` and `write_to_path`,
keeping single-threaded streaming msgpack→zstd.  A
`write_to_path_mt(path, level, n)` is preserved as opt-in for the
rare case (multi-GB payload AND we actually measure the threshold)
where it helps.

See `results/v3-numa-bind0/` and `results/v3-numa-interleave/` for
the post-fix numbers and the strict-NUMA vs interleave comparison.
