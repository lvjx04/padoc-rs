//! PADOC compresses Chrome trace JSON into a queryable, template-based
//! representation with lossless supported-event reconstruction.
//!
//! The `padoc` command-line interface and versioned artifact behavior are the
//! primary supported interfaces. This Rust library is available for
//! experimentation, but its API is pre-1.0 and may evolve between releases.
//!
//! The implementation has these design goals:
//!
//! * **Performance** — streaming chrome-trace ingest and columnar template
//!   storage with flat per-stream references;
//! * **Simplicity** — no class hierarchies for nodes/events, just enums;
//! * **Predictable resources** — each input trace is compressed independently;
//!   directory-level concurrency is bounded by the caller.
//!
//! The root re-exports the main experimental entry points. Implementation
//! modules remain visible where existing data types require them, but they are
//! not a stable API commitment before 1.0.

pub mod analysis;
pub mod compressor;
pub mod event;
pub mod node;
pub mod slp;
#[doc(hidden)]
pub mod synthetic;
pub mod trace;
mod trace_stream;
mod utils;
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
