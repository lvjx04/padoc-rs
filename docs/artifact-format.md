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

The current format version is `1`.

## Payload

The remaining bytes are a zstd frame containing a named MessagePack encoding
of the compressed trace. The payload contains:

- CPU and GPU template tables;
- typed per-instance columns;
- per-rank call-tree roots;
- profiler metadata payloads;
- timestamp origins used during JSON reconstruction.

Readers reject unknown versions, codecs, and non-zero reserved flags. Files are
decoded through buffered readers rather than loading and expanding the entire
artifact in a temporary byte vector.

The format is pre-1.0. A future incompatible schema change will increment the
header version.
