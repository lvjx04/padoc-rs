# results/v4-bufwriter

Production v4 numbers — adds a 1 MiB `BufWriter` between rmp_serde
and the zstd `Encoder`, fixing the v3 streaming-serialize regression.

## Code

Built from commit `fb283a2` on master:

* All v3 optimisations (parallel finalize, parallel merge tree
  rewrite, single-thread streaming zstd).
* New: `to_bytes` and `write_to_path` wrap the zstd `Encoder` in a
  1 MiB `BufWriter`, coalescing rmp_serde's per-field writes (which
  are tiny: `serde_json::Value::Number` -> 5 bytes, etc.) into chunks
  large enough that zstd's per-write overhead becomes negligible.
* Validated by `examples/serialize_bench` against the v3 artifacts:
  `qwen3 17.4s -> 5.65s`, `lewm 2.42s -> 0.79s`.

## Cluster command

```bash
cd ~/Work/padoc-rs
numactl --interleave=all ./target/release/padoc bench compress \
  --manifest full_manifest_int.json \
  --compressors padoc \
  --workers 32 \
  --out-dir /mnt/treasure/ljx/padoc_artifacts/v4-bufwriter
```

## Compression results (sc1, 32 workers, --interleave=all)

| dataset           | events        | raw         | compressed | ratio  | parallel | merge   | serialize | total    |
|-------------------|--------------:|------------:|-----------:|-------:|---------:|--------:|----------:|---------:|
| leworldmodel_full |     3,469,389 |  884.37 MiB |  37.48 MiB | 23.60× |   11.1 s |   2.2 s |  **1.0 s**|   14.3 s |
| qwen3_full        |    33,813,574 |    6.91 GiB | 271.71 MiB | 26.05× |   30.7 s |   6.6 s |  **6.9 s**|   44.2 s |
| unifolm_full      |    80,223,071 |   22.43 GiB | 741.54 MiB | 30.98× |  173.2 s |  24.5 s | **20.3 s**|  217.9 s |
| **llama_full**    | **301,288,116** |**69.95 GiB**|**2.40 GiB**| **29.17×** | **87.3 s** | **110.0 s** | **65.1 s** | **262.3 s** |

Saved artifacts:
`/mnt/treasure/ljx/padoc_artifacts/v4-bufwriter/<dataset>.padoc.zst`

## Round-trip verification

| dataset            | mode             | events     | LOSSLESS |
|--------------------|------------------|-----------:|----------|
| leworldmodel_full  | full directory   |  3,469,389 | YES      |

(More verifications happen in `results/v3-numa-bind0/` since the v4
binary's compress output is bit-equivalent to v3's; the BufWriter
change only affects how bytes flow into zstd, not what bytes are
generated.)

## Evolution: v1 -> v2 -> v3 -> v4

| dataset    | v1-baseline | v2-multithread | v3-numa-interleave | v4-bufwriter | total speedup |
|------------|------------:|---------------:|-------------------:|-------------:|--------------:|
| lewm       |      13.8 s |         17.8 s |             16.8 s |   **14.3 s** |        ≈ 1.0× |
| qwen3      |      60.4 s |         72.7 s |             59.2 s |   **44.2 s** |       1.37× |
| unifolm    |     255.2 s |        282.8 s |            242.6 s |  **217.9 s** |       1.17× |
| **llama**  |   **507.2 s** |    **524.7 s** |        **406.8 s** |  **262.3 s** |  **1.94×**  |

The big llama win comes from compounding three orthogonal optimisations:

| phase     | v1 | v3 | v4 | gain | source |
|-----------|---:|---:|---:|-----:|--------|
| parallel  |  83.3 |  86.9 |  87.3 |  none | already parallel from day 1 |
| merge     | 335.6 | 103.8 | 110.0 | **3.0×** | parallel tree rewrite |
| serialize |  88.4 | 216.2 |  65.1 | **3.3×** | BufWriter(1 MiB) on zstd Encoder |

## Notes for the paper

* **Compression ratio is identical to v3** (23.60× / 26.05× / 30.98× /
  29.17×).  These are the data-side numbers; v4 only changes how we
  put the bytes onto disk.
* **Throughput at 1024 ranks: 273 MB/s** (raw input rate).  Up from
  138 MB/s in v1 — i.e. one rank-second of compress wall-clock now
  ingests two rank-seconds of profiler data.
* The serialize column is now ~25% of total wall-clock for llama;
  parallel rank compression (87 s) and merge (110 s) are the next
  optimisation targets.  Merge is dominated by sequential
  template-table dedup (Phase 1 of `merge_shards`); a tree-reduce
  variant could halve it again.
