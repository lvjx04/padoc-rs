//! `padoc` — template-based AI profiler trace compression with in-situ analysis.
//!
//! This crate is a clean Rust rewrite of the original Python implementation in
//! `perflowai/padoc`.  It keeps the same compression / analysis semantics
//! (same paper, same evaluation methodology) but with these design goals:
//!
//! * **Performance** — chrome-trace ingest via `simd-json`, columnar template
//!   storage, in-place SLP, hash-bucket dedup of similar nodes;
//! * **Simplicity** — no class hierarchies for nodes/events, just enums;
//! * **Extensibility** — every compressor/analysis task is a trait so the
//!   bench harness is uniform across baselines.
//!
//! ## Module layout
//!
//! * [`event`]        — `Event`, `MergeEvent`, `KernelEvent` and friends
//! * [`node`]         — call-tree nodes (CPU / SameCPU / KernelLaunch / GPU)
//! * [`trace`]        — `Trace`, `CompressedTrace`, JSON ingest, msgpack/zstd serialisation
//! * [`slp`]          — segmented linear predictor for ts/dur/id/name compression
//! * [`compressor`]   — the PADOC `TemplateCompressor` and `CompressorConfig`
//! * [`baselines`]    — `raw`, `gzip`, `scalatrace`, `tracezip`
//! * [`analysis`]     — `AnalysisTask` trait + core paper analyses
//! * [`bench`]        — compression matrix, analysis matrix, scalability sweeps
//! * [`synthetic`]    — parameterised synthetic trace generator
//! * [`storage_breakdown`], [`tree_stats`] — paper-side profiling helpers

pub mod analysis;
pub mod baselines;
pub mod bench;
pub mod compressor;
pub mod event;
pub mod node;
pub mod slp;
pub mod storage_breakdown;
pub mod synthetic;
pub mod trace;
pub mod trace_stream;
pub mod tree_stats;
pub mod utils;
pub mod verify;

pub use baselines::{BaselineCompressor, CompressArtifact};
pub use compressor::{CompressorConfig, TemplateCompressor};
pub use event::{Event, KernelEvent, MergeEvent, MergeKernelEvent, Phase};
pub use trace::{CompressedTrace, Trace};

/// Crate-wide error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] simd_json::Error),
    #[error("serde-json error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("msgpack encode error: {0}")]
    MsgpackEncode(#[from] rmp_serde::encode::Error),
    #[error("msgpack decode error: {0}")]
    MsgpackDecode(#[from] rmp_serde::decode::Error),
    #[error("invalid trace: {0}")]
    InvalidTrace(String),
    #[error("invalid compressed trace: {0}")]
    InvalidCompressed(String),
    #[error("verification failed: {0}")]
    Verify(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
