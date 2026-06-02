//! Raw baseline analysis benchmark — load JSON trace → run_raw.
//!
//! Usage:
//!   cargo run --release --example bench_raw_baseline -- <trace_path_or_dir>

use padoc::analysis::{self, AnalysisTask};
use padoc::trace::Trace;
use std::env;
use std::path::Path;
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

fn load_trace(path: &Path) -> Trace {
    if path.is_dir() {
        Trace::from_dir(path).unwrap()
    } else {
        Trace::from_file(path).unwrap()
    }
}

fn main() {
    let paths: Vec<String> = env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: bench_raw_baseline <trace_path> [trace_path2...]");
        std::process::exit(1);
    }

    let tasks: Vec<Box<dyn AnalysisTask>> = vec![
        Box::new(analysis::OperatorHotspot::default()),
        Box::new(analysis::ParallelGroup::default()),
        Box::new(analysis::GpuBubbleRate::default()),
        Box::new(analysis::LayerComputeCommOverlap),
    ];

    println!("dataset\traw_bytes\tload_secs\tresident_kib\ttask\tanalyze_secs\ttotal_secs");

    for path_str in &paths {
        let path = Path::new(path_str);
        let dataset = path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let raw_bytes: u64 = if path.is_dir() {
            std::fs::read_dir(path).unwrap()
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        } else {
            std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
        };

        let rss_before = current_rss_kib();
        eprintln!("Loading: {} ({} bytes)", path_str, raw_bytes);
        let load_start = Instant::now();
        let trace = load_trace(path);
        let load_secs = load_start.elapsed().as_secs_f64();
        let rss_after = current_rss_kib();
        let resident_kib = rss_after.saturating_sub(rss_before);
        eprintln!("  {} events, load={:.3}s, rss={}KiB", trace.event_count(), load_secs, resident_kib);

        for task in &tasks {
            let an_start = Instant::now();
            let _ = task.run_raw(&trace).unwrap();
            let analyze_secs = an_start.elapsed().as_secs_f64();
            let total = load_secs + analyze_secs;
            println!("{}\t{}\t{:.6}\t{}\t{}\t{:.6}\t{:.6}",
                dataset, raw_bytes, load_secs, resident_kib,
                task.name(), analyze_secs, total);
        }
        drop(trace);
    }
}
