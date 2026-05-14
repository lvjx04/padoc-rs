# Synthetic scalability sweeps

## layers
| dimension | value | events | compressed_bytes | ratio | compress_secs |
|---|---:|---:|---:|---:|---:|
| layers | 8 | 1216 | 5.10 KiB | 27.77 | 0.012 |
| layers | 16 | 2432 | 9.59 KiB | 29.81 | 0.003 |
| layers | 32 | 4864 | 18.99 KiB | 30.26 | 0.006 |
| layers | 64 | 9728 | 38.14 KiB | 30.22 | 0.010 |
| layers | 128 | 19456 | 77.30 KiB | 30.05 | 0.028 |

## iterations
| dimension | value | events | compressed_bytes | ratio | compress_secs |
|---|---:|---:|---:|---:|---:|
| iterations | 1 | 304 | 1.72 KiB | 20.47 | 0.009 |
| iterations | 2 | 608 | 2.78 KiB | 25.45 | 0.001 |
| iterations | 4 | 1216 | 5.12 KiB | 27.68 | 0.001 |
| iterations | 8 | 2432 | 9.63 KiB | 29.63 | 0.003 |
| iterations | 16 | 4864 | 18.83 KiB | 30.42 | 0.006 |
