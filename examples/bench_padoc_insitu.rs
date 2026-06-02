//! Benchmark PADOC in-situ analysis from existing artifact files.
//!
//! Pipeline: read artifact from disk → decode CompressedTrace → run 4 tasks
//!
//! Usage:
//!   cargo run --release --example bench_padoc_insitu -- <artifact.padoc.zst> [artifact2...]

use padoc::analysis::{self, AnalysisTask};
use padoc::trace::CompressedTrace;
use std::env;
use std::time::Instant;

fn current_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let paths: Vec<String> = env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: bench_padoc_insitu <artifact.padoc.zst> [artifact2...]");
        std::process::exit(1);
    }

    let tasks: Vec<Box<dyn AnalysisTask>> = vec![
        Box::new(analysis::OperatorHotspot::default()),
        Box::new(analysis::ParallelGroup::default()),
        Box::new(analysis::GpuBubbleRate::default()),
        Box::new(analysis::LayerComputeCommOverlap),
    ];

    println!("dataset\tartifact_bytes\tload_secs\tdecode_secs\tresident_kib\ttask\tanalyze_secs\ttotal_secs");

    for path in &paths {
        let dataset = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .replace(".padoc", "");

        // Load from disk
        let load_start = Instant::now();
        let bytes = std::fs::read(path).unwrap();
        let load_secs = load_start.elapsed().as_secs_f64();
        let artifact_bytes = bytes.len() as u64;

        // Decode
        let rss_before = current_rss_kib();
        let decode_start = Instant::now();
        let compressed = CompressedTrace::from_bytes(&bytes).unwrap();
        let decode_secs = decode_start.elapsed().as_secs_f64();
        let rss_after = current_rss_kib();
        let resident_kib = rss_after.saturating_sub(rss_before);

        eprintln!("{}: {} bytes, load={:.3}s, decode={:.3}s, rss={}KiB, templates={}",
            dataset, artifact_bytes, load_secs, decode_secs, resident_kib, compressed.templates.len());

        // Run all 4 tasks
        for task in &tasks {
            let an_start = Instant::now();
            let _ = task.run_in_situ(&compressed).unwrap();
            let analyze_secs = an_start.elapsed().as_secs_f64();
            let total = load_secs + decode_secs + analyze_secs;
            println!("{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{:.6}\t{:.6}",
                dataset, artifact_bytes, load_secs, decode_secs, resident_kib,
                task.name(), analyze_secs, total);
        }
        drop(compressed);
    }
}
