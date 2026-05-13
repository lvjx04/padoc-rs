//! Per-layer operator dur distribution — accesses the trace by **model
//! structure** (transformer layer index).  Layers are inferred from the
//! `layers.<N>.` substring of operator names (the standard PyTorch
//! convention).
//!
//! In-situ: every CPU template's `name_pattern` carries `0` placeholders
//! where digit-runs were collapsed; per-instance digits live in
//! `name_nums`.  We locate the `0` that belongs to the `layers.X.` slot,
//! then for each instance look up exactly that digit position, parse it
//! as `i64`, and credit the instance's `dur` to that layer.  This means
//! one template can fan its instances across many layers (typical: a
//! single "transformer block forward" template is used by 80+ layers in
//! a 80-block model) and we still count them per-layer.

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::{elapsed_secs, profiled_result, AnalysisTask};
use crate::slp::decode_name_nums;
use crate::trace::{CompressedTrace, Trace};
use crate::Result;
use once_cell::sync::Lazy;
use regex::Regex;

static LAYER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.layers?\.(\d+)\.|/layers?/(\d+)/").unwrap());

/// In a digit-collapsed `name_pattern`, find the index (among all `0`
/// placeholders) of the `0` that immediately follows `.layers.` or
/// `/layers/`.  Returns `None` when this template has no layer slot.
static LAYER_PATTERN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\.layers?\.0(?:\.|$)|/layers?/0(?:/|$)").unwrap());

#[derive(Default)]
pub struct LayerOperatorBalance;

impl AnalysisTask for LayerOperatorBalance {
    fn name(&self) -> &str { "layer_operator_balance" }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut by_layer: AHashMap<i64, i64> = AHashMap::new();
        for (_rank, _pid, _tid, _ph, events) in trace.iter_streams() {
            for ev in events {
                if let Some(layer) = layer_from_full_name(&ev.name) {
                    *by_layer.entry(layer).or_insert(0) += ev.dur.unwrap_or(0);
                }
            }
        }
        Ok(to_json(by_layer))
    }

    fn supports_in_situ(&self) -> bool { true }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let start = std::time::Instant::now();
        let mut by_layer: AHashMap<i64, i64> = AHashMap::new();
        for tmpl in &compressed.templates {
            // GPU kernel names rarely carry "layers.<N>." — skip to save work.
            if tmpl.is_gpu() { continue; }
            let pattern = tmpl.name_pattern();
            let Some(zero_idx) = layer_zero_index(pattern) else {
                continue;
            };
            let nums = tmpl.name_nums();
            let n = tmpl.instance_count();
            for i in 0..n {
                // `decode_name_nums` is O(K) per instance where K is the
                // number of digit-runs in the template name.  For the
                // transformer-block templates K is single digits, so this
                // stays cheap.
                let digits = decode_name_nums(nums, i);
                if let Some(d) = digits.get(zero_idx) {
                    if let Ok(layer) = d.parse::<i64>() {
                        *by_layer.entry(layer).or_insert(0) += tmpl.dur_at(i).unwrap_or(0);
                    }
                }
            }
        }
        let aggregate_secs = elapsed_secs(start);
        let start = std::time::Instant::now();
        let result = to_json(by_layer);
        Ok(profiled_result(result, vec![
            ("template_layer_aggregate", aggregate_secs),
            ("sort_json", elapsed_secs(start)),
        ]))
    }
}

fn layer_from_full_name(name: &str) -> Option<i64> {
    LAYER_RE.captures(name).and_then(|c| {
        c.get(1).or_else(|| c.get(2)).and_then(|m| m.as_str().parse::<i64>().ok())
    })
}

/// Returns the index (within the digit-fillers vector) of the `0`
/// placeholder that holds the layer number.
fn layer_zero_index(pattern: &str) -> Option<usize> {
    let m = LAYER_PATTERN_RE.find(pattern)?;
    // The `0` placeholder lives within the match; locate it.
    let zero_in_match = pattern[m.start()..m.end()].find('0')?;
    let zero_byte_pos = m.start() + zero_in_match;
    // Count how many `0` chars precede `zero_byte_pos`; that's our index
    // into the per-instance digit-fillers vector.
    Some(pattern.as_bytes()[..zero_byte_pos].iter().filter(|&&b| b == b'0').count())
}

fn to_json(map: AHashMap<i64, i64>) -> Value {
    let mut entries: Vec<(i64, i64)> = map.into_iter().collect();
    entries.sort_by_key(|(k, _)| *k);
    Value::Array(entries.into_iter().map(|(layer, total)| {
        serde_json::json!({"layer": layer, "total_dur_us": total})
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_layer_zero_in_simple_pattern() {
        // Pattern ".layers.0.attn.proj" — exactly one `0`, at index 0.
        assert_eq!(layer_zero_index("model.layers.0.attn.proj"), Some(0));
    }

    #[test]
    fn finds_layer_zero_when_other_digits_precede() {
        // Pattern ".0x0e0.layers.0." has 3 `0`s before the layer one,
        // so layer index in name_nums is 3.
        assert_eq!(layer_zero_index("addr.0x0e0.layers.0.gemm"), Some(3));
    }

    #[test]
    fn returns_none_when_no_layer_slot() {
        assert!(layer_zero_index("aten::linear").is_none());
        assert!(layer_zero_index("layers.0").is_none()); // missing trailing `.` boundary
    }

    #[test]
    fn finds_layer_in_full_name() {
        assert_eq!(layer_from_full_name("model.layers.42.attn"), Some(42));
        assert_eq!(layer_from_full_name("/transformer/layers/7/proj"), Some(7));
        assert_eq!(layer_from_full_name("not.a.layer.thing"), None);
    }
}
