//! Baselines: `raw`, `gzip`, `scalatrace`, `tracezip`, `padoc`.
//!
//! Every baseline implements [`BaselineCompressor`].  The bench harness
//! treats them uniformly so you can swap algorithms by name.

use crate::trace::Trace;
use crate::Result;

mod gzip;
mod padoc;
mod raw;
mod scalatrace;
mod tracezip;

pub use gzip::{GzipJsonCompressor, GzipMsgpackCompressor};
pub use padoc::PadocCompressor;
pub use raw::{RawJsonCompressor, RawMsgpackCompressor};
pub use scalatrace::ScalaTraceCompressor;
pub use tracezip::TraceZipCompressor;

/// Output of a single compression call.
#[derive(Debug)]
pub struct CompressArtifact {
    /// Compressed bytes.
    pub bytes: Vec<u8>,
    /// Wall-clock encode time.
    pub compress_secs: f64,
    /// Wall-clock decode time, if the compressor pre-decompressed for verification.
    pub decompress_secs: Option<f64>,
    /// Free-form annotations (algorithm-specific).
    pub annotations: serde_json::Map<String, serde_json::Value>,
}

impl CompressArtifact {
    pub fn new(bytes: Vec<u8>, compress_secs: f64) -> Self {
        Self {
            bytes,
            compress_secs,
            decompress_secs: None,
            annotations: serde_json::Map::new(),
        }
    }
}

/// Trait every baseline implements.  Methods are blocking and synchronous.
pub trait BaselineCompressor: Send + Sync {
    fn name(&self) -> &str;

    fn compress(&self, trace: &Trace) -> Result<CompressArtifact>;

    fn decompress(&self, bytes: &[u8]) -> Result<Trace>;

    /// Whether this compressor supports in-situ analysis for the given task
    /// (without full decompression back to `Trace`).
    fn supports_in_situ(&self, _task: &str) -> bool { false }

    /// Run an analysis task directly on the compressed bytes.
    /// Only called when `supports_in_situ(task)` returns true.
    fn run_in_situ(&self, _bytes: &[u8], _task: &str) -> Result<serde_json::Value> {
        Err(crate::Error::Other("in-situ not implemented".into()))
    }

    /// Decode the compressed bytes into an opaque in-memory payload for
    /// repeated in-situ queries. Returns None if not supported.
    fn decode_for_analysis(&self, _bytes: &[u8]) -> Result<Box<dyn std::any::Any>> {
        Err(crate::Error::Other("decode_for_analysis not implemented".into()))
    }

    /// Run an analysis task on a previously decoded payload (from `decode_for_analysis`).
    fn run_in_situ_decoded(&self, _decoded: &dyn std::any::Any, _task: &str) -> Result<serde_json::Value> {
        Err(crate::Error::Other("run_in_situ_decoded not implemented".into()))
    }
}

/// Build the canonical lookup table used by the bench CLI.
pub fn registry() -> Vec<Box<dyn BaselineCompressor>> {
    vec![
        Box::new(RawJsonCompressor::default()),
        Box::new(RawMsgpackCompressor::default()),
        Box::new(GzipJsonCompressor::default()),
        Box::new(GzipMsgpackCompressor::default()),
        Box::new(ScalaTraceCompressor::default()),
        Box::new(TraceZipCompressor::default()),
        Box::new(PadocCompressor::default()),
    ]
}
