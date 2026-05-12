//! PADOC `TemplateCompressor` — template extraction + structural compression.
//!
//! High-level pipeline (see `core.rs`):
//!
//! 1. **Build call tree** per stream: sort events by `(ts, -dur)` and assemble
//!    a parent-child tree using a stack (CPU events) or flat list (GPU events).
//! 2. **Add events to template table**: every event gets matched to (or creates)
//!    a `MergeEvent` whose signature is `(normalized_name, cat, bp, s, args_keys)`.
//!    Each event becomes a `Node::Cpu(template_id, instance_id)`.
//! 3. **Structural compression**: bottom-up grouping of sibling sub-trees that
//!    share a `template_index` into a `Node::SameCpu`; greedy anchor matching
//!    across instance child sequences.
//! 4. **Numeric finalisation**: SLP-encode `ts`, `dur`, `id`; transpose name
//!    digit fillers; dedup args.
//! 5. **Pair CPU launches with GPU kernels** through `correlation` ids, producing
//!    `Node::KernelLaunch` / `Node::KernelsLaunch` (the "soft-link edges").

mod call_tree;
mod config;
mod core;
mod decompress;
mod merge;
mod structural;

pub use config::{all_ablation_presets, CompressorConfig};
pub use core::TemplateCompressor;
pub use decompress::decompress;
pub use merge::{merge_shards, RankShard};
