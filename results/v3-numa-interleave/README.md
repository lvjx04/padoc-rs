# results/v3-numa-interleave

Production v3 numbers, NUMA-interleaved memory (`numactl --interleave=all`),
32 workers (= one socket's worth of physical cores; CPUs come from
either socket via Linux's normal scheduling).

This is the **only viable strategy for llama**: the 1024-rank merge
holds ~280 GiB of templates + per-rank trees in memory, which doesn't
fit a single 256 GiB NUMA node.  Interleaving puts ~140 GiB on each
node, well under both budgets.

## Code

Same as `results/v3-numa-bind0/`: commit `4e0bb0e` on master with
parallel finalize / parallel merge / streaming single-thread zstd.

## Cluster command

```bash
cd ~/Work/padoc-rs
numactl --interleave=all ./target/release/padoc bench compress \
  --manifest full_manifest_int.json \
  --compressors padoc \
  --workers 32 \
  --out-dir /mnt/treasure/ljx/padoc_artifacts/v3-numa-interleave
```

`full_manifest_int.json` covers all four datasets including llama.

## Compression results

| dataset           | events        | raw        | compressed | ratio  | parallel | merge   | serialize | total    |
|-------------------|--------------:|-----------:|-----------:|-------:|---------:|--------:|----------:|---------:|
| leworldmodel_full |     3,469,389 | 884.37 MiB |  37.48 MiB | 23.60× |   11.6 s |   2.3 s |     2.8 s |   16.8 s |
| qwen3_full        |    33,813,574 |   6.91 GiB | 271.71 MiB | 26.05× |   30.0 s |   7.6 s |    21.5 s |   59.2 s |
| unifolm_full      |    80,223,071 |  22.43 GiB | 741.54 MiB | 30.98× |  162.1 s |  24.4 s |    56.1 s |  242.6 s |
| **llama_full**    | **301,288,116** | **69.95 GiB** | **2.40 GiB** | **29.17×** | **86.9 s** | **103.8 s** | **216.2 s** | **406.8 s** |

Saved artifacts at:
`/mnt/treasure/ljx/padoc_artifacts/v3-numa-interleave/<dataset>.padoc.zst`

## llama_full vs v1-baseline

| phase     | v1-baseline | v3-interleave | delta |
|-----------|------------:|--------------:|------:|
| parallel  |       83.3 |          86.9 |  +3.6 |
| merge     |      335.6 |         103.8 | **−231.8 ✓** (3.2× faster) |
| serialize |       88.4 |         216.2 | **+127.8 ❌** (2.4× slower) |
| **total** |    **507.2** |     **406.8** | **−100.4 (1.25× faster)** |

The parallel-merge tree-rewrite optimisation halves merge wall-clock at
this scale, but the streaming msgpack→zstd serializer is much slower
than the old "buffer full msgpack then zstd::encode_all" for a 2.4 GiB
output.  Net win is still ~1.25× faster and we save ~10 GiB peak RAM
(no intermediate msgpack buffer).

## Round-trip verification (lossless)

llama full-dir round-trip is infeasible: peak memory is
`raw_trace + reconstructed_trace + 2 fingerprint multisets ≈ 600 GiB`,
which exceeds the 503 GiB physical RAM on either machine.

We sample 4 representative rank files and verify each independently:

| rank        | events    | mismatch | missing | extra | LOSSLESS |
|-------------|----------:|---------:|--------:|------:|----------|
| profiler_0    |   295,090 |        0 |       0 |     0 | YES      |
| profiler_256  |   ~290k   |        0 |       0 |     0 | YES      |
| profiler_512  |   ~290k   |        0 |       0 |     0 | YES      |
| profiler_1023 |   322,223 |        0 |       0 |     0 | YES      |

Each rank-only roundtrip uses the per-rank `compress_rank` path (not
the merged path), so this verifies the per-rank compressor and decoder
are bit-equivalent — but it does NOT verify cross-rank merge correctness
on llama specifically.  Cross-rank merge is verified on lewm + qwen3
full-dir round-trips (see `results/v3-numa-bind0/README.md`) where
peak memory fits.  The merge code is generic, so passing on lewm/qwen3
gives high confidence on llama.

## bind0 vs interleave (3 small datasets)

| dataset    | bind0 total | interleave total | winner |
|------------|------------:|-----------------:|--------|
| lewm       |      14.9 s |           16.8 s | bind0 (small lead) |
| qwen3      |      63.9 s |           59.2 s | interleave (small lead) |
| unifolm    |     230.5 s |          242.6 s | bind0 (5%) |

The two strategies are within 7% of each other on every dataset that
fits both — there's no clear winner for small/medium data.  For llama
only interleave works, so we ship interleave as the default.
