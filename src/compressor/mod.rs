//! PADOC `TemplateCompressor` — template extraction + flat stream encoding.
//!
//! High-level pipeline (see `core.rs`):
//!
//! 1. **Add events to template table**: every event gets matched to (or creates)
//!    a `MergeEvent` whose signature is `(normalized_name, cat, bp, s, args_keys)`.
//! 2. **Record flat stream references** as parallel template/instance arrays.
//! 3. **Numeric finalisation**: compact `ts`, `dur`, `id`; transpose name
//!    digit fillers; dedup args.
//! 4. **Persist a bounded-depth artifact** through MessagePack and zstd.

mod config;
mod core;
mod decompress;
mod finalize;
mod flat;

pub use core::TemplateCompressor;
pub use decompress::decompress;
