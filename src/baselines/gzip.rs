//! `gzip_json` and `gzip_msgpack` — `raw` baselines piped through gzip.

use crate::baselines::raw::flatten_trace_for_baseline;
use crate::baselines::{BaselineCompressor, CompressArtifact};
use crate::trace::Trace;
use crate::Result;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;

#[derive(Default)]
pub struct GzipJsonCompressor;

impl BaselineCompressor for GzipJsonCompressor {
    fn name(&self) -> &str { "gzip_json" }
    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();
        let raw = serde_json::to_vec(&flatten_trace_for_baseline(trace))?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw)?;
        let bytes = encoder.finish()?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }
    fn decompress(&self, _bytes: &[u8]) -> Result<Trace> { Ok(Trace::empty()) }
}

#[derive(Default)]
pub struct GzipMsgpackCompressor;

impl BaselineCompressor for GzipMsgpackCompressor {
    fn name(&self) -> &str { "gzip_msgpack" }
    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();
        let raw = rmp_serde::to_vec_named(&flatten_trace_for_baseline(trace))?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw)?;
        let bytes = encoder.finish()?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }
    fn decompress(&self, _bytes: &[u8]) -> Result<Trace> { Ok(Trace::empty()) }
}
