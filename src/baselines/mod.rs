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
