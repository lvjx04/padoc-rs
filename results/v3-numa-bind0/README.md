# results/v3-numa-bind0

Production v3 numbers, NUMA-bound to socket 0 (`numactl --cpunodebind=0
--membind=0`), 32 workers (= 32 physical cores on socket 0).

## Code

Built from commit `4e0bb0e` on master:

* parallel `finalize_templates` (rayon, per-template SLP/args dedup)
* parallel `merge_shards` Phase 2 (rayon, per-shard call-tree rewrite)
* streaming msgpack → zstd serializer (single-threaded zstd; the v2
  multi-threaded zstd attempt regressed serialize 3-4× and was reverted —
  see `results/v2-multithread-zstd/README.md`)
* `bench compress --out-dir <DIR>` writes the artifact to disk via the
  streaming pipeline (no giant intermediate Vec<u8>).

## Cluster command

```bash
cd ~/Work/padoc-rs
numactl --cpunodebind=0 --membind=0 ./target/release/padoc bench compress \
  --manifest manifest_no_llama.json \
  --compressors padoc \
  --workers 32 \
  --out-dir /mnt/treasure/ljx/padoc_artifacts/v3-numa-bind0
```

`manifest_no_llama.json` is a 3-dataset manifest: lewm + qwen3 + unifolm.
**llama is excluded from bind0** because peak memory (~280 GiB during
merge) exceeds the per-NUMA-node 256 GiB budget; use the interleave run
(`results/v3-numa-interleave/`) for the llama_full numbers.

## Compression results

| dataset           | events       | raw        | compressed | ratio  | parallel | merge | serialize | total   |
|-------------------|-------------:|-----------:|-----------:|-------:|---------:|------:|----------:|--------:|
| leworldmodel_full |    3,469,389 | 884.37 MiB |  37.48 MiB | 23.60× |   10.5 s |  1.7 s|     2.7 s |  14.9 s |
| qwen3_full        |   33,813,574 |   6.91 GiB | 271.71 MiB | 26.05× |   34.2 s |  6.2 s|    23.5 s |  63.9 s |
| unifolm_full      |   80,223,071 |  22.43 GiB | 741.54 MiB | 30.98× |  154.0 s | 21.6 s|    54.9 s | 230.5 s |

Saved artifacts at:
`/mnt/treasure/ljx/padoc_artifacts/v3-numa-bind0/<dataset>.padoc.zst`

## Round-trip verification (lossless)

| dataset            | mode                 | events     | mismatch | missing | extra | LOSSLESS |
|--------------------|----------------------|-----------:|---------:|--------:|------:|----------|
| leworldmodel_full  | full directory       |  3,469,389 |        0 |       0 |     0 | YES      |
| qwen3_full         | full directory       | 33,813,574 |        0 |       0 |     0 | YES      |
| unifolm_full rank0 | per-rank file        | 20,015,026 |        0 |       0 |     0 | YES      |
| unifolm_full rank1 | per-rank file        | 20,107,224 |        0 |       0 |     0 | YES      |
| unifolm_full rank2 | per-rank file        | 20,107,053 |        0 |       0 |     0 | YES      |
| unifolm_full rank3 | per-rank file        | 19,993,768 |        0 |       0 |     0 | YES      |

unifolm is verified per-rank (one file at a time) instead of as a full
directory because the full-dir verify peak memory (orig + reconstructed
+ 2× fingerprint hashmap) is ~190 GiB on this dataset — fits 256 GiB
NUMA0 with little headroom.  Per-rank caps peak at ~50 GiB / rank.

For llama (verify peak ~600 GiB, doesn't fit anywhere) we sample 4 rank
files; see `results/v3-numa-interleave/README.md`.

## Phase comparison vs v1-baseline (sc1, 32 workers)

| dataset    | phase     | v1     | v3-bind0 | delta |
|------------|-----------|-------:|---------:|------:|
| lewm       | parallel  |  10.8  |   10.5   |  −0.3 |
| lewm       | merge     |   2.0  |    1.7   |  −0.3 |
| lewm       | serialize |   1.0  |    2.7   |  +1.7 |
| **lewm**   | **total** | **13.8** | **14.9** | **+1.1** |
| qwen3      | parallel  |  29.9  |   34.2   |  +4.3 |
| qwen3      | merge     |  22.7  |    6.2   | **−16.5 ✓** |
| qwen3      | serialize |   7.8  |   23.5   | **+15.7 ❌** |
| **qwen3**  | **total** | **60.4** | **63.9** | **+3.5** |
| unifolm    | parallel  | 164.3  |  154.0   |  −10.3 |
| unifolm    | merge     |  69.2  |   21.6   | **−47.6 ✓** |
| unifolm    | serialize |  21.7  |   54.9   | **+33.2 ❌** |
| **unifolm**| **total** | **255.2** | **230.5** | **−24.7 ✓** |

Net: parallel-merge optimisations land ✓; streaming msgpack→zstd
serialize is ~3× slower per-byte than the old "buffer full msgpack +
zstd::encode_all".  For the smaller datasets the merge gain is too
small to absorb the streaming serialize cost, so total is slightly
worse; for unifolm the merge gain dominates and we're 10% faster.

The serialize regression is the next optimisation target: the right
fix is probably to keep streaming for the >1 GiB output case (memory
win) and switch to the buffered path below that threshold (speed win).
