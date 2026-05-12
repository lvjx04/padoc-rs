//! Metric records emitted by the bench harness.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionRecord {
    pub compressor: String,
    pub dataset: String,
    pub event_count: usize,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub compress_secs: f64,
    pub decompress_secs: Option<f64>,
    pub ratio: f64,
    pub throughput_mb_per_sec: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisRecord {
    pub compressor: String,
    pub task: String,
    pub dataset: String,
    pub load_secs: f64,
    pub decompress_secs: f64,
    pub analysis_secs: f64,
    pub total_secs: f64,
    pub in_situ: bool,
}
