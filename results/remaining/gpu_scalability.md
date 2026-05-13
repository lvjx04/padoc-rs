
## manifest_llama_1gpus workers=32
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_1gpus | padoc | 316746 | 74.81 MiB | 2.99 MiB | 25.03 | 1.737 | 43.1 |

## manifest_llama_8gpus workers=32
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_8gpus | padoc | 2607995 | 622.99 MiB | 23.35 MiB | 26.68 | 4.665 | 133.6 |

## manifest_llama_64gpus workers=32
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_64gpus | padoc | 19544859 | 4.55 GiB | 165.23 MiB | 28.19 | 48.408 | 96.2 |

## manifest_llama_256gpus workers=32
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_256gpus | padoc | 75749224 | 17.59 GiB | 621.61 MiB | 28.97 | 115.421 | 156.0 |
