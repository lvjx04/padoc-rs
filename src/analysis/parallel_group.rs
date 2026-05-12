//! Parallel group inference (TP/DP/PP/EP).
//!
//! We infer groups by examining the participants of NCCL collective operations.
//! Each unique `group_ranks` set seen across `nccl:*` events corresponds to a
//! parallel group; the *kind* of group (TP vs DP vs PP vs EP) is a heuristic
//! based on the size of the group and whether it spans pipeline stages.

use ahash::{AHashMap, AHashSet};
use serde_json::Value;

use crate::analysis::AnalysisTask;
use crate::trace::Trace;
use crate::Result;

#[derive(Default)]
pub struct ParallelGroup;

impl AnalysisTask for ParallelGroup {
    fn name(&self) -> &str { "parallel_group" }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut groups: AHashMap<String, AHashSet<String>> = AHashMap::new();
        for (rank, _pid, _tid, _ph, events) in trace.iter_streams() {
            for ev in events {
                if !ev.name.contains("nccl") && ev.cat.as_deref() != Some("kernel") { continue; }
                if let Some(args) = &ev.args {
                    if let Some(stream) = args.get("Process Group Name").and_then(|v| v.as_str()) {
                        groups.entry(stream.to_string()).or_default().insert(rank.to_string());
                    }
                }
            }
        }
        Ok(groups_to_json(groups))
    }

    fn supports_in_situ(&self) -> bool { false }
}

fn groups_to_json(groups: AHashMap<String, AHashSet<String>>) -> Value {
    let mut out: Vec<Value> = Vec::with_capacity(groups.len());
    let mut entries: Vec<(String, AHashSet<String>)> = groups.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (group, ranks) in entries {
        let mut ranks_vec: Vec<String> = ranks.into_iter().collect();
        ranks_vec.sort();
        out.push(serde_json::json!({
            "group": group,
            "size": ranks_vec.len(),
            "ranks": ranks_vec,
        }));
    }
    Value::Array(out)
}
