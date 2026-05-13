//! Per-GPU-stream busy time distribution.

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::AnalysisTask;
use crate::event::{NumColumn, StringColumn, Template};
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

#[derive(Default)]
pub struct StreamLoadBalance;

impl AnalysisTask for StreamLoadBalance {
    fn name(&self) -> &str { "stream_load_balance" }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut by_stream: AHashMap<(i64, &str), i64> = AHashMap::new();
        for (_rank, pid, tid, _ph, events) in trace.iter_streams() {
            if !tid.contains("stream") { continue; }
            let total: i64 = events.iter().map(|ev| ev.dur.unwrap_or(0)).sum();
            *by_stream.entry((pid, tid)).or_insert(0) += total;
        }
        Ok(to_sorted_json(by_stream))
    }

    fn supports_in_situ(&self) -> bool { true }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let mut by_stream: AHashMap<(i64, &str), i64> = AHashMap::new();
        for tmpl in &compressed.templates {
            if let Template::Gpu(t) = tmpl {
                let n = t.instance_count();
                if n == 0 {
                    continue;
                }

                // Most GPU templates compact to a `Constant` pid/stream_tid;
                // the typed enums encode that natively, so we can sum the
                // whole `dur` column once and bypass per-instance hashing.
                if let (Some(pid), Some(tid)) = (constant_i64_col(&t.pid), constant_str_col(&t.stream_tid)) {
                    let total = t.dur.sum_i64();
                    *by_stream.entry((pid, tid)).or_insert(0) += total;
                } else {
                    for i in 0..n {
                        let pid = t.pid.get(i).unwrap_or(0);
                        let tid = t.stream_tid.get(i).unwrap_or_default();
                        *by_stream.entry((pid, tid)).or_insert(0) += t.dur.get(i).unwrap_or(0);
                    }
                }
            }
        }
        Ok(to_sorted_json(by_stream))
    }
}

fn constant_i64_col(col: &NumColumn) -> Option<i64> {
    match col {
        NumColumn::Constant { value, .. } => Some(*value),
        _ => None,
    }
}

fn constant_str_col(col: &StringColumn) -> Option<&str> {
    match col {
        StringColumn::Constant { value, .. } => Some(value.as_str()),
        _ => None,
    }
}

fn to_sorted_json(map: AHashMap<(i64, &str), i64>) -> Value {
    let mut entries: Vec<((i64, &str), i64)> = map.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    Value::Array(entries.into_iter().map(|((pid, tid), total)| {
        serde_json::json!({"stream": format!("{pid}:{tid}"), "busy_us": total})
    }).collect())
}
