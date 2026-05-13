//! Top-N CPU operator hotspot by total `dur`.

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::{elapsed_secs, profiled_result, AnalysisTask};
use crate::event::Template;
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

#[derive(Default)]
pub struct OperatorHotspot {
    pub top_k: usize,
}

impl AnalysisTask for OperatorHotspot {
    fn name(&self) -> &str { "operator_hotspot" }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut tally: AHashMap<String, i64> = AHashMap::new();
        for (_rank, _pid, _tid, _ph, events) in trace.iter_streams() {
            for ev in events {
                let name = crate::utils::normalize_name(&ev.name);
                *tally.entry(name).or_insert(0) += ev.dur.unwrap_or(0);
            }
        }
        Ok(top_n_to_json(tally, self.top_k.max(20)))
    }

    fn supports_in_situ(&self) -> bool { true }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let start = std::time::Instant::now();
        let mut tally: AHashMap<String, i64> = AHashMap::new();
        for tmpl in &compressed.templates {
            let name = tmpl.name_pattern().to_string();
            let total = tmpl.dur_total();
            *tally.entry(name).or_insert(0) += total;
        }
        let tally_secs = elapsed_secs(start);
        let start = std::time::Instant::now();
        let result = top_n_to_json(tally, self.top_k.max(20));
        Ok(profiled_result(result, vec![
            ("template_tally", tally_secs),
            ("sort_top_n", elapsed_secs(start)),
        ]))
    }
}

fn top_n_to_json(tally: AHashMap<String, i64>, n: usize) -> Value {
    let mut entries: Vec<(String, i64)> = tally.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.truncate(n);
    let arr: Vec<Value> = entries.into_iter().map(|(name, total)| {
        serde_json::json!({"name": name, "total_dur_us": total})
    }).collect();
    Value::Array(arr)
}

// Quiet unused import warning when only `Template` is referenced.
#[allow(dead_code)]
fn _ensure_template_used(_t: &Template) {}
