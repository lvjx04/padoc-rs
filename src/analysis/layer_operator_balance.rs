//! Per-layer operator dur distribution.  Layers are inferred from the
//! `layers.<N>.` substring of operator names (the standard PyTorch convention).

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::AnalysisTask;
use crate::trace::{CompressedTrace, Trace};
use crate::Result;
use once_cell::sync::Lazy;
use regex::Regex;

static LAYER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.layers?\.(\d+)\.|/layers?/(\d+)/").unwrap());

#[derive(Default)]
pub struct LayerOperatorBalance;

impl AnalysisTask for LayerOperatorBalance {
    fn name(&self) -> &str { "layer_operator_balance" }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut by_layer: AHashMap<i64, i64> = AHashMap::new();
        for (_rank, _pid, _tid, _ph, events) in trace.iter_streams() {
            for ev in events {
                if let Some(layer) = layer_index(&ev.name) {
                    *by_layer.entry(layer).or_insert(0) += ev.dur.unwrap_or(0);
                }
            }
        }
        Ok(to_json(by_layer))
    }

    fn supports_in_situ(&self) -> bool { true }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        // We don't have per-instance digit fillers available cheaply here yet,
        // so for in-situ we approximate using the template's first-instance
        // restored name.  This catches the common case where every event in a
        // template belongs to the same layer (true for most PyTorch traces).
        let mut by_layer: AHashMap<i64, i64> = AHashMap::new();
        for tmpl in &compressed.templates {
            // Only walk CPU templates — GPU kernel names rarely contain layers.
            if tmpl.is_gpu() { continue; }
            let name = tmpl.name_pattern();
            if let Some(layer) = layer_index(name) {
                let total: i64 = tmpl.dur().iter().sum();
                *by_layer.entry(layer).or_insert(0) += total;
            }
        }
        Ok(to_json(by_layer))
    }
}

fn layer_index(name: &str) -> Option<i64> {
    LAYER_RE.captures(name).and_then(|c| {
        c.get(1).or_else(|| c.get(2)).and_then(|m| m.as_str().parse::<i64>().ok())
    })
}

fn to_json(map: AHashMap<i64, i64>) -> Value {
    let mut entries: Vec<(i64, i64)> = map.into_iter().collect();
    entries.sort_by_key(|(k, _)| *k);
    Value::Array(entries.into_iter().map(|(layer, total)| {
        serde_json::json!({"layer": layer, "total_dur_us": total})
    }).collect())
}
