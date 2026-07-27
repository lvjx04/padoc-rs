# Changelog

All notable user-facing changes will be documented here.

## Unreleased

### Added

- Compress, inspect, verify, decompress, and in-situ analysis CLI workflows.
- Independent directory compression with bounded file concurrency and atomic
  manifest publication.
- Streaming Chrome trace JSON and `.json.gz` input.
- Versioned artifacts with v2 writing and bounded v1 reading compatibility.
- Stable `operator_hotspot` and `stream_load_balance` in-situ tasks.

### Correctness

- Preserve supported event fields, heterogeneous nested JSON values, timestamp
  origins, and metadata coordinates, duplicates, and order.
- Verify supported events exhaustively as multisets within each stream.
- Reject unsupported artifact versions, codecs, flags, and output overwrites.
- Report the artifact format version read from the header in `padoc inspect`.

### Changed

- Define the CLI and versioned artifact behavior as the primary release
  interfaces. The experimental Rust API remains pre-1.0 and may evolve.
- Keep stable encoding flat and per-rank, without recursive call trees or
  cross-rank template merging.
