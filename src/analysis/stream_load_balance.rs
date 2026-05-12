//! Per-GPU-stream busy time distribution.

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::AnalysisTask;
use crate::event::Template;
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

#[derive(Default)]
pub struct StreamLoadBalance;

impl AnalysisTask for StreamLoadBalance {
    fn name(&self) -> &str { "stream_load_balance" }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut by_stream: AHashMap<String, i64> = AHashMap::new();
        for (_rank, pid, tid, _ph, events) in trace.iter_streams() {
            if !tid.contains("stream") { continue; }
            let key = format!("{}:{}", pid, tid);
            for ev in events {
                *by_stream.entry(key.clone()).or_insert(0) += ev.dur.unwrap_or(0);
            }
        }
        Ok(to_sorted_json(by_stream))
    }

    fn supports_in_situ(&self) -> bool { true }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let mut by_stream: AHashMap<String, i64> = AHashMap::new();
        for tmpl in &compressed.templates {
            if let Template::Gpu(t) = tmpl {
                for i in 0..t.instance_count() {
                    let pid = t.pid.get(i).copied().unwrap_or(0);
                    let tid = t.stream_tid.get(i).cloned().unwrap_or_default();
                    let key = format!("{}:{}", pid, tid);
                    *by_stream.entry(key).or_insert(0) += t.dur.get(i).copied().unwrap_or(0);
                }
            }
        }
        Ok(to_sorted_json(by_stream))
    }
}

fn to_sorted_json(map: AHashMap<String, i64>) -> Value {
    let mut entries: Vec<(String, i64)> = map.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    Value::Array(entries.into_iter().map(|(stream, total)| {
        serde_json::json!({"stream": stream, "busy_us": total})
    }).collect())
}
