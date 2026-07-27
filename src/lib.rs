//! PADOC compresses AI profiler traces into a queryable, template-based
//! representation.
//!
//! This crate is a clean Rust rewrite of the original Python implementation in
//! `perflowai/padoc`.  It keeps the same compression / analysis semantics
//! (same paper, same evaluation methodology) but with these design goals:
//!
//! * **Performance** — chrome-trace ingest via `simd-json`, columnar template
//!   storage, in-place SLP, hash-bucket dedup of similar nodes;
//! * **Simplicity** — no class hierarchies for nodes/events, just enums;
//! * **Predictable resources** — each input trace is compressed independently;
//!   directory-level concurrency is bounded by the caller.
//!
//! ## Module layout
//!
//! * [`event`]        — `Event`, `MergeEvent`, `KernelEvent` and friends
//! * [`node`]         — call-tree nodes (CPU / SameCPU / KernelLaunch / GPU)
//! * [`trace`]        — `Trace`, `CompressedTrace`, JSON ingest, msgpack/zstd serialisation
//! * [`slp`]          — segmented linear predictor for ts/dur/id/name compression
//! * [`compressor`]   — template extraction and structural compression
//! * [`analysis`]     — stable in-situ analysis tasks
//! * [`synthetic`]    — deterministic traces used by examples and tests

pub mod analysis;
pub mod compressor;
pub mod event;
pub mod node;
pub mod slp;
#[doc(hidden)]
pub mod synthetic;
pub mod trace;
pub mod trace_stream;
pub mod utils;
pub mod verify;

pub use compressor::{decompress, TemplateCompressor};
pub use event::{Event, KernelEvent, MergeEvent, MergeKernelEvent, Phase};
pub use trace::{CompressedTrace, MetadataEvent, Trace};

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
