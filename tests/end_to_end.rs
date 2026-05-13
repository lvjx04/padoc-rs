//! End-to-end pipeline tests: synthetic trace -> compress -> serialise -> deserialise -> analyse.

use padoc::analysis::{AnalysisTask, OperatorHotspot, StreamLoadBalance};
use padoc::baselines::{BaselineCompressor, GzipMsgpackCompressor, PadocCompressor, RawJsonCompressor, ScalaTraceCompressor, TraceZipCompressor};
use padoc::compressor::{all_ablation_presets, CompressorConfig, TemplateCompressor};
use padoc::storage_breakdown::{measure_on_disk_regions, measure_storage};
use padoc::synthetic::{generate_trace, SyntheticTraceSpec};
use padoc::trace::CompressedTrace;
use padoc::tree_stats::measure_tree_statistics;

fn small_spec() -> SyntheticTraceSpec {
    SyntheticTraceSpec {
        gpu_count: 2,
        layer_count: 3,
        iteration_count: 2,
        ops_per_layer: 4,
        op_dur_us: 50,
        seed: 42,
    }
}

#[test]
fn synthetic_trace_is_non_empty_and_deterministic() {
    let trace_a = generate_trace(&small_spec());
    let trace_b = generate_trace(&small_spec());
    assert!(trace_a.event_count() > 0);
    assert_eq!(trace_a.event_count(), trace_b.event_count());
}

#[test]
fn padoc_compress_round_trip_via_bytes() {
    let trace = generate_trace(&small_spec());
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).expect("compress");
    let bytes = compressed.to_bytes(3).expect("serialise");
    let reloaded = CompressedTrace::from_bytes(&bytes).expect("deserialise");
    assert_eq!(reloaded.templates.len(), compressed.templates.len());
    assert_eq!(reloaded.ranks.len(), compressed.ranks.len());
}

#[test]
fn every_baseline_can_compress_synthetic_trace() {
    let trace = generate_trace(&small_spec());

    let baselines: Vec<Box<dyn BaselineCompressor>> = vec![
        Box::new(RawJsonCompressor::default()),
        Box::new(GzipMsgpackCompressor::default()),
        Box::new(ScalaTraceCompressor::default()),
        Box::new(TraceZipCompressor::default()),
        Box::new(PadocCompressor::default()),
    ];

    for c in &baselines {
        let artifact = c.compress(&trace).unwrap_or_else(|e| panic!("{} failed: {}", c.name(), e));
        assert!(!artifact.bytes.is_empty(), "{} produced empty payload", c.name());
    }
}

#[test]
fn padoc_in_situ_operator_hotspot_matches_raw() {
    let trace = generate_trace(&small_spec());
    let task = OperatorHotspot { top_k: 0 };
    let raw = task.run_raw(&trace).expect("raw");

    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).expect("compress");
    let in_situ = task.run_in_situ(&compressed).expect("in-situ");

    // Both produce a top-N JSON list; they should agree on the ranking of
    // the heaviest operator (everything else can vary because the in-situ
    // path key is the digit-collapsed name).
    let raw_top = raw.as_array().unwrap().first().unwrap();
    let situ_top = in_situ.as_array().unwrap().first().unwrap();
    assert_eq!(raw_top["total_dur_us"], situ_top["total_dur_us"]);
}

#[test]
fn padoc_in_situ_stream_load_balance_runs() {
    let trace = generate_trace(&small_spec());
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).expect("compress");
    let task = StreamLoadBalance::default();
    let result = task.run_in_situ(&compressed).expect("in-situ");
    let arr = result.as_array().expect("array");
    assert!(!arr.is_empty(), "stream load balance produced no entries");
}

#[test]
fn ablation_presets_all_round_trip_through_bytes() {
    let trace = generate_trace(&small_spec());
    for (label, cfg) in all_ablation_presets() {
        let mut compressor = TemplateCompressor::with_config(cfg.clone());
        let compressed = compressor.compress(&trace).unwrap_or_else(|e| panic!("{label} compress: {e}"));
        let bytes = compressed.to_bytes(3).unwrap_or_else(|e| panic!("{label} serialise: {e}"));
        let _reload = CompressedTrace::from_bytes(&bytes).unwrap_or_else(|e| panic!("{label} deserialise: {e}"));
        let _ = cfg;
    }
}

#[test]
fn storage_breakdown_components_sum_to_total() {
    let trace = generate_trace(&small_spec());
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).unwrap();
    let breakdown = measure_storage(&compressed);
    assert!(breakdown.total_bytes > 0);
    assert!(breakdown.template_bytes > 0);
    assert_eq!(
        breakdown.total_bytes,
        breakdown.template_bytes + breakdown.structure_bytes + breakdown.metadata_bytes
    );
}

#[test]
fn on_disk_breakdown_reports_expected_regions() {
    let trace = generate_trace(&small_spec());
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).unwrap();
    let encoded = compressed.to_bytes(3).unwrap();
    let breakdown = measure_on_disk_regions(&compressed, Some(encoded.len() as u64), 3).unwrap();
    let names: Vec<&str> = breakdown.regions.iter().map(|r| r.name.as_str()).collect();
    assert!(names.contains(&"template_headers"));
    assert!(names.contains(&"ts_columns"));
    assert!(names.contains(&"dur_columns"));
    assert!(names.contains(&"rank_node_tree"));
    assert!(breakdown.regions.iter().all(|r| r.msgpack_bytes > 0));
    assert!(breakdown.regions.iter().all(|r| r.zstd_bytes > 0));
}

#[test]
fn tree_stats_have_reasonable_shape() {
    let trace = generate_trace(&small_spec());
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).unwrap();
    let stats = measure_tree_statistics(&compressed);
    assert!(stats.max_depth >= 1);
    // Mean branching can be 0 for very flat trees, but max should not be.
    assert!(stats.max_branching >= 1);
}

#[test]
fn config_default_label_is_default() {
    let cfg = CompressorConfig::default();
    assert_eq!(cfg.label, "default");
    assert!(cfg.is_default());
}
