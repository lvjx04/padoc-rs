//! Analysis tasks.  Every task implements [`AnalysisTask`].
//!
//! Tasks have two execution modes:
//!
//! * `run_raw(trace)` — operate on a fully-materialised `Trace`.
//! * `run_in_situ(compressed)` — optionally implemented; runs directly on a
//!   `CompressedTrace` without decompression.  The bench harness uses
//!   [`AnalysisTask::supports_in_situ`] to decide whether to skip the
//!   decode step for PADOC.
//!
//! The core paper tasks cover both compressed-template aggregation and
//! structure-aware GPU attribution:
//!
//! * `operator_hotspot`            — top-N operator/kernel by total dur.
//! * `rank_load_balance`           — per-rank GPU compute/communication balance.
//! * `layer_kernel_hotspot`        — per-layer GPU kernel hotspots via CPU->GPU links.
//! * `layer_compute_comm_overlap`  — per-layer compute/communication overlap.
//! * `layer_rank_balance`          — per-layer, per-rank GPU load balance.

use crate::trace::{CompressedTrace, Trace};
use crate::Result;
use serde_json::Value;
use std::time::Instant;

mod compute_comm_overlap;
mod kernel_class;
mod layer_gpu;
mod layer_operator_balance;
mod operator_hotspot;
mod parallel_group;
mod stream_load_balance;

pub use compute_comm_overlap::ComputeCommOverlap;
pub use layer_gpu::{LayerComputeCommOverlap, LayerKernelHotspot, LayerRankBalance};
pub use layer_operator_balance::LayerOperatorBalance;
pub use operator_hotspot::OperatorHotspot;
pub use parallel_group::ParallelGroup;
pub use stream_load_balance::StreamLoadBalance;

pub trait AnalysisTask: Send + Sync {
    fn name(&self) -> &str;
    fn run_raw(&self, trace: &Trace) -> Result<Value>;
    fn supports_in_situ(&self) -> bool {
        false
    }
    fn run_in_situ(&self, _compressed: &CompressedTrace) -> Result<Value> {
        Err(crate::Error::Other(
            "in-situ not implemented for this task".into(),
        ))
    }
}

pub(crate) fn profiling_enabled() -> bool {
    std::env::var_os("PADOC_ANALYSIS_PROFILE").is_some()
}

pub(crate) fn profiled_result(result: Value, phases: Vec<(&str, f64)>) -> Value {
    if !profiling_enabled() {
        return result;
    }
    let total_secs: f64 = phases.iter().map(|(_, secs)| *secs).sum();
    serde_json::json!({
        "result": result,
        "profile": {
            "total_profiled_secs": total_secs,
            "phases": phases.into_iter().map(|(name, secs)| {
                serde_json::json!({"name": name, "secs": secs})
            }).collect::<Vec<_>>(),
        }
    })
}

pub(crate) fn elapsed_secs(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

pub fn registry() -> Vec<Box<dyn AnalysisTask>> {
    vec![
        Box::new(OperatorHotspot::default()),
        Box::new(ParallelGroup::default()),
        Box::new(LayerKernelHotspot::default()),
        Box::new(LayerComputeCommOverlap),
        Box::new(LayerRankBalance),
    ]
}
