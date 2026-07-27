use ahash::AHashMap;
use indexmap::IndexMap;
use padoc::analysis::{AnalysisTask, OperatorHotspot, StreamLoadBalance};
use padoc::event::{Event, Phase};
use padoc::synthetic::{generate_trace, SyntheticTraceSpec};
use padoc::trace::{CompressedTrace, StreamMap, Trace};
use padoc::TemplateCompressor;

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

fn round_trip(trace: &Trace) -> Trace {
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(trace).expect("compress");
    let bytes = compressed.to_bytes(3).expect("serialize");
    let reloaded = CompressedTrace::from_bytes(&bytes).expect("deserialize");
    padoc::decompress(&reloaded)
}

fn one_rank_trace(cpu_events: Vec<Event>, gpu_events: Vec<Event>) -> Trace {
    let mut streams: StreamMap = IndexMap::new();
    let mut tids = IndexMap::new();
    if !cpu_events.is_empty() {
        let mut phases = IndexMap::new();
        phases.insert(Phase::COMPLETE, cpu_events);
        tids.insert("cpu".to_string(), phases);
    }
    if !gpu_events.is_empty() {
        let mut phases = IndexMap::new();
        phases.insert(Phase::COMPLETE, gpu_events);
        tids.insert("stream 7".to_string(), phases);
    }
    streams.insert(7, tids);

    let mut trace = Trace::empty();
    trace.ranks.insert("0".into(), streams);
    trace.start_timestamp.insert("0".into(), 1_000);
    trace
}

fn event(name: &str, ts: i64, dur: Option<i64>, id: Option<i64>, tid: &str) -> Event {
    Event {
        name: name.into(),
        ts,
        dur,
        cat: Some("test".into()),
        ph: Phase::COMPLETE,
        pid: 7,
        tid: tid.into(),
        args: Some(AHashMap::from_iter([(
            "value".into(),
            serde_json::json!(ts),
        )])),
        id,
        bp: Some("e".into()),
        s: Some("g".into()),
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
fn padoc_round_trip_is_event_lossless() {
    let trace = generate_trace(&small_spec());
    let recovered = round_trip(&trace);
    let report = padoc::verify::compare_traces(&trace, &recovered);
    assert!(report.is_ok(), "{report:#?}");
}

#[test]
fn compressor_can_be_reused_for_independent_artifacts() {
    let trace = generate_trace(&small_spec());
    let mut compressor = TemplateCompressor::new();
    let first = compressor.compress(&trace).expect("first compress");
    let second = compressor.compress(&trace).expect("second compress");

    assert_eq!(
        first.to_bytes(3).expect("serialize first"),
        second.to_bytes(3).expect("serialize second")
    );
}

#[test]
fn artifact_has_a_versioned_header_and_is_deterministic() {
    let trace = generate_trace(&small_spec());

    let mut first_compressor = TemplateCompressor::new();
    let first = first_compressor.compress(&trace).expect("compress");
    let first_bytes = first.to_bytes(3).expect("serialize");

    let mut second_compressor = TemplateCompressor::new();
    let second = second_compressor.compress(&trace).expect("compress");
    let second_bytes = second.to_bytes(3).expect("serialize");

    assert_eq!(&first_bytes[..8], b"PADOCART");
    assert_eq!(u16::from_le_bytes([first_bytes[8], first_bytes[9]]), 2);
    assert_eq!(first_bytes, second_bytes);
}

#[test]
fn invalid_artifact_header_is_rejected() {
    let error = CompressedTrace::from_bytes(&[0_u8; 16]).expect_err("invalid header should fail");
    assert!(error.to_string().contains("artifact magic"));
}

#[test]
fn optional_numeric_fields_do_not_shift_between_instances() {
    let trace = one_rank_trace(
        vec![
            event("same_name", 1, Some(10), Some(100), "cpu"),
            event("same_name", 20, None, None, "cpu"),
            event("same_name", 30, Some(5), Some(300), "cpu"),
        ],
        Vec::new(),
    );
    let recovered = round_trip(&trace);
    let report = padoc::verify::compare_traces(&trace, &recovered);
    assert!(report.is_ok(), "{report:#?}");
}

#[test]
fn cpu_and_gpu_events_with_the_same_signature_do_not_collide() {
    let trace = one_rank_trace(
        vec![event("shared_name", 1, Some(10), Some(1), "cpu")],
        vec![event("shared_name", 2, Some(8), Some(2), "stream 7")],
    );
    let recovered = round_trip(&trace);
    let report = padoc::verify::compare_traces(&trace, &recovered);
    assert!(report.is_ok(), "{report:#?}");
}

#[test]
fn mixed_integer_and_float_args_round_trip_exactly() {
    let mut integer = event("kernel", 1, Some(10), None, "stream 7");
    integer.args = Some(AHashMap::from_iter([
        ("blocks per SM".into(), serde_json::json!(0)),
        ("memory bandwidth (GB/s)".into(), serde_json::json!(0)),
    ]));
    let mut float = event("kernel", 20, Some(8), None, "stream 7");
    float.args = Some(AHashMap::from_iter([
        ("blocks per SM".into(), serde_json::json!(0.0)),
        (
            "memory bandwidth (GB/s)".into(),
            serde_json::json!(14.905873071154925_f64),
        ),
    ]));
    let trace = one_rank_trace(Vec::new(), vec![integer, float]);

    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).expect("compress");
    let direct = padoc::decompress(&compressed);
    let direct_report = padoc::verify::compare_traces(&trace, &direct);
    assert!(
        direct_report.is_ok(),
        "in-memory round trip failed: {direct_report:#?}"
    );
    let bytes = compressed.to_bytes(3).expect("serialize");
    let reloaded = CompressedTrace::from_bytes(&bytes).expect("deserialize");
    let recovered = padoc::decompress(&reloaded);
    let report = padoc::verify::compare_traces(&trace, &recovered);

    assert!(report.is_ok(), "artifact round trip failed: {report:#?}");
}

#[test]
fn deeply_nested_intervals_use_a_bounded_depth_artifact() {
    let depth = 4_096_i64;
    let events = (0..depth)
        .map(|index| {
            event(
                "nested_operator",
                index,
                Some((depth - index) * 2),
                None,
                "cpu",
            )
        })
        .collect();
    let trace = one_rank_trace(events, Vec::new());

    let recovered = round_trip(&trace);
    let report = padoc::verify::compare_traces(&trace, &recovered);

    assert!(report.is_ok(), "{report:#?}");
}

#[test]
fn chrome_json_writer_restores_timestamp_origin() {
    let trace = one_rank_trace(
        vec![event("operator", 25, Some(10), None, "cpu")],
        Vec::new(),
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let output = dir.path().join("trace.json");
    trace.write_chrome_json(&output).expect("write Chrome JSON");

    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).expect("read output")).expect("parse output");
    let events = value["traceEvents"].as_array().expect("events");
    assert_eq!(events[0]["ts"], 1_025);
    assert_eq!(events[0]["ph"], "X");
}

#[test]
fn in_situ_operator_hotspot_matches_raw_total() {
    let trace = generate_trace(&small_spec());
    let task = OperatorHotspot { top_k: 0 };
    let raw = task.run_raw(&trace).expect("raw");

    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).expect("compress");
    let in_situ = task.run_in_situ(&compressed).expect("in situ");

    assert_eq!(
        raw.as_array().unwrap().first().unwrap()["total_dur_us"],
        in_situ.as_array().unwrap().first().unwrap()["total_dur_us"]
    );
}

#[test]
fn in_situ_stream_load_balance_runs() {
    let trace = generate_trace(&small_spec());
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace).expect("compress");
    let result = StreamLoadBalance.run_in_situ(&compressed).expect("in situ");
    assert!(!result.as_array().expect("array").is_empty());
}
