//! Scalability sweep over GPU count, layer count and iteration count.
//!
//! Generates synthetic traces parameterised by the swept dimension and runs
//! a single compressor across them.

use serde::{Deserialize, Serialize};

use crate::baselines::{BaselineCompressor, RawJsonCompressor};
use crate::synthetic::{generate_trace, SyntheticTraceSpec};
use crate::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalabilityPoint {
    pub dimension: String,
    pub value: usize,
    pub event_count: usize,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub compress_secs: f64,
    pub ratio: f64,
}

pub fn run_scalability(
    compressor: &dyn BaselineCompressor,
    base: &SyntheticTraceSpec,
    dimension: &str,
    values: &[usize],
) -> Result<Vec<ScalabilityPoint>> {
    let mut out = Vec::new();
    for &v in values {
        let mut spec = base.clone();
        match dimension {
            "gpus" => spec.gpu_count = v,
            "layers" => spec.layer_count = v,
            "iterations" => spec.iteration_count = v,
            other => return Err(crate::Error::Other(format!("unknown sweep dimension {other}"))),
        }
        let trace = generate_trace(&spec);
        let event_count = trace.event_count();
        // Use raw JSON encoding to estimate uncompressed size (same view used by `raw_json` baseline).
        let raw_bytes = RawJsonCompressor::default().compress(&trace)?.bytes.len() as u64;
        let artifact = compressor.compress(&trace)?;
        let compressed_bytes = artifact.bytes.len() as u64;
        out.push(ScalabilityPoint {
            dimension: dimension.to_string(),
            value: v,
            event_count,
            raw_bytes,
            compressed_bytes,
            compress_secs: artifact.compress_secs,
            ratio: if compressed_bytes == 0 { 0.0 } else { raw_bytes as f64 / compressed_bytes as f64 },
        });
    }
    Ok(out)
}
