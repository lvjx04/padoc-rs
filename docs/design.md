# Design

PADOC's public implementation is built around three constraints: event-level
correctness, bounded resource use, and a small supportable interface.

## Compression pipeline

1. Parse one Chrome trace file. Files larger than 32 MiB use the streaming
   parser to avoid a full JSON value tree.
2. Normalize timestamps relative to the first event in the trace.
3. Build CPU call trees and associate GPU kernels through supported correlation
   fields when a unique match is available.
4. Group events into CPU and GPU templates. CPU and GPU signatures use separate
   namespaces.
5. Compact numeric, string, argument, phase, and name-digit columns.
6. Serialize the versioned payload through zstd.

The supported encoder policy is fixed. Research-only ablation switches are not
part of the public API.

## Directory processing

A directory is a collection of independent input traces. `compress-dir`
processes at most `--workers` files concurrently and writes one artifact for
each file. PADOC deliberately does not merge template tables across ranks.

This has three useful properties:

- peak memory is bounded by the selected number of active files;
- one failed or corrupt rank does not invalidate a global artifact;
- artifacts can be moved, analyzed, or regenerated independently.

The output manifest contains relative source and artifact names, rank ids,
event counts, and byte sizes. It does not embed machine-specific input paths.
PADOC builds a directory in a sibling staging location and publishes it only
after every artifact and the manifest succeed, so a failed input does not
leave a partial output directory.

## Correctness contract

PADOC verifies supported events as multisets within each
`(rank, pid, tid, phase)` stream. Stream ordering and the order of equal events
are not significant.

Optional numeric fields are part of a template's signature. This prevents
missing `dur` or `id` values from shifting a compact column. CPU and GPU
templates also use separate indices, so identical names and argument schemas
cannot collide across event kinds.

Chrome metadata records preserve names, original `pid`/`tid` coordinates,
argument payloads, duplicates, and input order.

## Non-goals

- Cross-rank template merging.
- Paper baseline implementations in the public crate.
- Workload-specific analysis presented as a general semantic guarantee.
- Byte-for-byte reproduction of the original JSON formatting or event order.
