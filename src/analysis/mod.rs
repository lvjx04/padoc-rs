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
//! The four shipped tasks cover the access patterns the paper evaluates:
//!
//! * `operator_hotspot`        — top-N CPU operator by total dur.
//! * `stream_load_balance`     — per-GPU-stream busy time distribution.
//! * `compute_comm_overlap`    — per-rank compute/communication overlap.
//! * `layer_operator_balance`  — per-layer operator dur distribution.
//! * `parallel_group`          — TP/DP/PP/EP group inference from comm ops.

use crate::trace::{CompressedTrace, Trace};
use crate::Result;
use serde_json::Value;

mod compute_comm_overlap;
mod kernel_class;
mod layer_operator_balance;
mod operator_hotspot;
mod parallel_group;
mod stream_load_balance;

pub use compute_comm_overlap::ComputeCommOverlap;
pub use layer_operator_balance::LayerOperatorBalance;
pub use operator_hotspot::OperatorHotspot;
pub use parallel_group::ParallelGroup;
pub use stream_load_balance::StreamLoadBalance;

pub trait AnalysisTask: Send + Sync {
    fn name(&self) -> &str;
    fn run_raw(&self, trace: &Trace) -> Result<Value>;
    fn supports_in_situ(&self) -> bool { false }
    fn run_in_situ(&self, _compressed: &CompressedTrace) -> Result<Value> {
        Err(crate::Error::Other("in-situ not implemented for this task".into()))
    }
}

pub fn registry() -> Vec<Box<dyn AnalysisTask>> {
    vec![
        Box::new(OperatorHotspot::default()),
        Box::new(StreamLoadBalance::default()),
        Box::new(ComputeCommOverlap),
        Box::new(LayerOperatorBalance::default()),
        Box::new(ParallelGroup::default()),
    ]
}
