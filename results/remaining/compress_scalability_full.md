
## small workers=1
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| leworldmodel_full | padoc | 3469389 | 884.37 MiB | 37.52 MiB | 23.57 | 20.685 | 42.8 |
| qwen3_full | padoc | 33813574 | 6.91 GiB | 272.23 MiB | 26.00 | 290.937 | 24.3 |
| unifolm_full | padoc | 80223071 | 22.43 GiB | 741.08 MiB | 31.00 | 570.182 | 40.3 |

## small workers=2
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| leworldmodel_full | padoc | 3469389 | 884.37 MiB | 37.52 MiB | 23.57 | 13.687 | 64.6 |
| qwen3_full | padoc | 33813574 | 6.91 GiB | 272.23 MiB | 26.00 | 159.692 | 44.3 |
| unifolm_full | padoc | 80223071 | 22.43 GiB | 741.08 MiB | 31.00 | 321.551 | 71.4 |

## small workers=4
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| leworldmodel_full | padoc | 3469389 | 884.37 MiB | 37.52 MiB | 23.57 | 13.995 | 63.2 |
| qwen3_full | padoc | 33813574 | 6.91 GiB | 272.23 MiB | 26.00 | 88.036 | 80.4 |
| unifolm_full | padoc | 80223071 | 22.43 GiB | 741.08 MiB | 31.00 | 200.491 | 114.6 |

## small workers=8
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| leworldmodel_full | padoc | 3469389 | 884.37 MiB | 37.52 MiB | 23.57 | 13.983 | 63.2 |
| qwen3_full | padoc | 33813574 | 6.91 GiB | 272.23 MiB | 26.00 | 49.993 | 141.6 |
| unifolm_full | padoc | 80223071 | 22.43 GiB | 741.08 MiB | 31.00 | 203.894 | 112.7 |

## small workers=16
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| leworldmodel_full | padoc | 3469389 | 884.37 MiB | 37.52 MiB | 23.57 | 14.020 | 63.1 |
| qwen3_full | padoc | 33813574 | 6.91 GiB | 272.23 MiB | 26.00 | 38.413 | 184.3 |
| unifolm_full | padoc | 80223071 | 22.43 GiB | 741.08 MiB | 31.00 | 199.686 | 115.0 |

## small workers=32
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| leworldmodel_full | padoc | 3469389 | 884.37 MiB | 37.52 MiB | 23.57 | 14.161 | 62.5 |
| qwen3_full | padoc | 33813574 | 6.91 GiB | 272.23 MiB | 26.00 | 43.288 | 163.5 |
| unifolm_full | padoc | 80223071 | 22.43 GiB | 741.08 MiB | 31.00 | 206.084 | 111.5 |

## small workers=64
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| leworldmodel_full | padoc | 3469389 | 884.37 MiB | 37.52 MiB | 23.57 | 13.891 | 63.7 |
| qwen3_full | padoc | 33813574 | 6.91 GiB | 272.23 MiB | 26.00 | 60.492 | 117.0 |
| unifolm_full | padoc | 80223071 | 22.43 GiB | 741.08 MiB | 31.00 | 206.789 | 111.1 |

## llama workers=1
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_full | padoc | 301288116 | 69.95 GiB | 2.40 GiB | 29.18 | 3153.696 | 22.7 |

## llama workers=2
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_full | padoc | 301288116 | 69.95 GiB | 2.40 GiB | 29.18 | 1648.617 | 43.4 |

## llama workers=4
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_full | padoc | 301288116 | 69.95 GiB | 2.40 GiB | 29.18 | 933.970 | 76.7 |

## llama workers=8
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_full | padoc | 301288116 | 69.95 GiB | 2.40 GiB | 29.18 | 562.100 | 127.4 |

## llama workers=16
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_full | padoc | 301288116 | 69.95 GiB | 2.40 GiB | 29.18 | 397.718 | 180.1 |

## llama workers=32
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_full | padoc | 301288116 | 69.95 GiB | 2.40 GiB | 29.18 | 357.691 | 200.2 |

## llama workers=64
| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama_full | padoc | 301288116 | 69.95 GiB | 2.40 GiB | 29.18 | 452.036 | 158.5 |
