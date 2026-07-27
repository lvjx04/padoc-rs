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
//! The stable tasks cover common, unambiguous access patterns:
//!
//! * `operator_hotspot`        — top-N CPU operator by total dur.
//! * `stream_load_balance`     — per-GPU-stream busy time distribution.

use crate::trace::{CompressedTrace, Trace};
use crate::Result;
use serde_json::Value;

mod operator_hotspot;
mod stream_load_balance;

pub use operator_hotspot::OperatorHotspot;
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

pub fn registry() -> Vec<Box<dyn AnalysisTask>> {
    vec![
        Box::new(OperatorHotspot::default()),
        Box::new(StreamLoadBalance),
    ]
}
