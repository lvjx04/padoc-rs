//! Round-trip test with the *minimal* CompressorConfig (every compression
//! technique disabled).  Used to localise whether a lossless bug lives in
//! the call-tree builder, the structural compressor, or the value coders.

use padoc::compressor::{CompressorConfig, TemplateCompressor};
use padoc::trace::Trace;
use padoc::verify::compare_traces;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: roundtrip_minimal <chrome_trace.json>");
    let cfg_label = std::env::args().nth(2).unwrap_or_else(|| "minimal".to_string());
    let cfg = match cfg_label.as_str() {
        "minimal" => CompressorConfig {
            enable_structural: false,
            enable_anchor_matching: false,
            enable_slp: false,
            enable_args_dedup: false,
            enable_kernel_links: false,
            enable_name_pattern: false,
            label: "minimal".into(),
        },
        "no_structural" => CompressorConfig {
            enable_structural: false,
            enable_anchor_matching: false,
            label: "no_structural".into(),
            ..Default::default()
        },
        "no_anchor" => CompressorConfig {
            enable_anchor_matching: false,
            label: "no_anchor".into(),
            ..Default::default()
        },
        "no_kernel_links" => CompressorConfig {
            enable_kernel_links: false,
            label: "no_kernel_links".into(),
            ..Default::default()
        },
        _ => CompressorConfig::default(),
    };

    println!("config: {}", cfg.label);
    let trace = Trace::from_file(&path)?;
    println!("loaded {} events", trace.event_count());

    let t0 = Instant::now();
    let mut comp = TemplateCompressor::with_config(cfg.clone());
    let compressed = comp.compress(&trace)?;
    println!("compress: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let recovered = padoc::compressor::decompress(&compressed);
    println!("decompress: {:.3}s -> {} events", t0.elapsed().as_secs_f64(), recovered.event_count());

    let report = compare_traces(&trace, &recovered);
    println!("orig={} recon={} matching={} mismatched={} missing_streams={} extra_streams={}",
        report.original_event_count,
        report.reconstructed_event_count,
        report.matching_events,
        report.mismatched_events,
        report.missing_streams.len(),
        report.extra_streams.len(),
    );
    if !report.stream_count_diffs.is_empty() {
        for d in &report.stream_count_diffs {
            println!("  diff: {}", d);
        }
    }
    if !report.first_mismatches.is_empty() {
        for m in report.first_mismatches.iter().take(3) {
            println!("  mis: {}", m);
        }
    }
    println!("LOSSLESS: {}", report.is_ok());
    Ok(())
}
