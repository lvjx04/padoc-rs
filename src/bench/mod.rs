//! Bench harness — uniform invocation of compressors and analysis tasks.
//!
//! Three entry points:
//!
//! * [`run_compression_matrix`] — every `(compressor, dataset)` cell once.
//! * [`run_analysis_matrix`]    — every `(compressor, task, dataset)` cell.
//! * [`run_scalability`]        — synthetic-trace sweep over GPUs / layers / iterations.
//! * [`run_parallel`]           — multi-thread / multi-process compression speedup.

pub mod manifest;
pub mod metrics;
pub mod parallel;
pub mod report;
pub mod runner;
pub mod scalability;

pub use manifest::{Manifest, ManifestEntry};
pub use metrics::{AnalysisRecord, CompressionRecord};
pub use parallel::{run_parallel_compression, ParallelRecord};
pub use report::{render_compression_table, render_scalability_table};
pub use runner::{
    run_analysis_matrix, run_compression_matrix, run_compression_streaming, DatasetRef,
    StreamingDataset,
};
pub use scalability::{run_scalability, ScalabilityPoint};
