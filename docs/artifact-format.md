# Artifact format

PADOC artifacts use the `.padoc` extension.

## Header

Every artifact begins with a fixed 16-byte header:

| Offset | Size | Meaning |
|---:|---:|---|
| 0 | 8 | ASCII magic `PADOCART` |
| 8 | 2 | little-endian format version |
| 10 | 1 | payload codec (`1` = zstd) |
| 11 | 5 | reserved, currently zero |

The current format version is `2`. The reader also accepts version `1`
artifacts produced by pre-release PADOC builds when their legacy recursive
payload stays within the MessagePack decoder's safety limit.

## Payload

The remaining bytes are a zstd frame containing a named MessagePack encoding
of the compressed trace. The payload contains:

- CPU and GPU template tables;
- typed per-instance columns;
- flat template/instance references for each rank stream;
- profiler metadata payloads;
- timestamp origins used during JSON reconstruction.

Version 2 introduced bounded-depth stream references and type-tagged fallback
storage for heterogeneous JSON arguments. These changes preserve distinctions
such as `0` versus `0.0` and prevent deeply nested profiler intervals from
exceeding the MessagePack decoder's recursion limit.

Readers reject unknown versions, codecs, and non-zero reserved flags. Files are
decoded through buffered readers rather than loading and expanding the entire
artifact in a temporary byte vector.

The format is pre-1.0. A future incompatible schema change will increment the
header version.
