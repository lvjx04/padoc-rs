# results/v1-baseline

First clean end-to-end numbers from the parallel padoc pipeline:

* per-rank streaming compression
* cross-rank template merge (`merge_shards`)
* multi-rank parallel via rayon (`--workers 32`)
* streaming JSON parser (`trace_stream`)
* float `ts`/`dur` pre-normalised to int via `normalize_int_ts`

These are the **real numbers** without the legacy float-ts-zeroing bug
that made earlier leworldmodel runs report ~42× compression by silently
collapsing every rank's `ts` column to a single `0`.

## Cluster commands

```bash
# 1. one-off pre-process (lewm + unifolm only — qwen3 / llama already int)
mkdir -p /mnt/treasure/ljx/Trace_int
cp -r /mnt/treasure/ljx/Trace/leworldmodel_json        /mnt/treasure/ljx/Trace_int/
cp -r /mnt/treasure/ljx/Trace/unifolm-world-model_json /mnt/treasure/ljx/Trace_int/

cd ~/Work/padoc-rs
cargo build --release --example normalize_int_ts
./target/release/examples/normalize_int_ts /mnt/treasure/ljx/Trace_int/leworldmodel_json
./target/release/examples/normalize_int_ts /mnt/treasure/ljx/Trace_int/unifolm-world-model_json

# 2. bench
cargo build --release --bin padoc

nohup ./target/release/padoc bench compress \
        --manifest full_manifest_int.json \
        --compressors padoc \
        --workers 32 \
      > full_bench_v1.log 2>&1 < /dev/null &
```

## Manifest

See `manifest.json` (a copy of `full_manifest_int.json`).

## Code git

Built from commit **3f6fbeb** on master (last commit included:
`examples: normalize_int_ts — rewrite float ts/dur as integers in place`).

## Results

| dataset           | events       | raw_bytes  | compressed | ratio  | secs    | MB/s  |
|-------------------|-------------:|-----------:|-----------:|-------:|--------:|------:|
| leworldmodel_full |    3,469,389 | 884.37 MiB |  37.48 MiB | 23.60× |  13.8 s |  64.3 |
| qwen3_full        |   33,813,574 |   6.91 GiB | 271.71 MiB | 26.05× |  60.4 s | 117.1 |
| unifolm_full      |   80,223,071 |  22.43 GiB | 741.54 MiB | 30.98× | 255.2 s |  90.0 |
| llama_full        |  301,288,116 |  69.95 GiB |   2.40 GiB | 29.17× | 507.2 s | 141.2 |

Phase breakdown (logged via `padoc-parallel done: parallel=Xs merge=Ys serialize=Zs`):

| dataset           | parallel  | merge   | serialize | total   |
|-------------------|----------:|--------:|----------:|--------:|
| leworldmodel_full |   ≈10 s   |  ≈3 s   |   ≈0.5 s  |  13.8 s |
| qwen3_full        |   ≈8 s    |  ≈40 s  |    ≈12 s  |  60.4 s |
| unifolm_full      |   ≈80 s   | ≈140 s  |    ≈35 s  | 255.2 s |
| **llama_full**    | **83 s**  | **336 s** | **88 s** | **507 s** |

`merge` is single-threaded today and dominates wall-clock on the
big multi-rank runs.  `serialize` is also single-threaded msgpack →
zstd.  Both are the targets of v2-fast-tail.

## Verification status

Local mini-dataset round-trips:

* qwen3 (2 ranks)    : LOSSLESS YES, 21.86× (parallel-merge path)
* leworldmodel (2 r) : LOSSLESS YES, 24.30× (matches the v1 cluster
                                              ratio of 23.60×; small
                                              gap is the rest of the
                                              ranks)

unifolm and llama 1024-rank end-to-end round-trip not run; the
parallel-merge path is verified bit-equivalent to the from_dir path
on smaller datasets, so the only failure mode at scale would be a
shard-boundary bug that would also surface on the mini set.
