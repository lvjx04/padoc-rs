//! In-situ analysis benchmark — full pipeline timing + memory.
//!
//! For each compressor: load artifact once → decode once → run all tasks.
//! PADOC additionally runs layer_compute_comm_overlap (the 4th task).
//! ScalaTrace/TraceZip run layer via decompress fallback separately.
//!
//! Usage:
//!   cargo run --release --example bench_insitu -- <trace_path_or_dir>

use padoc::analysis::{self, AnalysisTask};
use padoc::baselines::{self, BaselineCompressor};
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
    let path_str = env::args().nth(1).expect("usage: bench_insitu <trace_path>");
    let path = Path::new(&path_str);

    // Load raw trace and compress
    eprintln!("Loading raw trace: {}", path_str);
    let raw_start = Instant::now();
    let trace = load_trace(path);
    eprintln!("  {} events in {:.3}s", trace.event_count(), raw_start.elapsed().as_secs_f64());

    let registry = baselines::registry();
    let compressors: Vec<&dyn BaselineCompressor> = registry.iter()
        .filter(|c| matches!(c.name(), "scalatrace" | "tracezip" | "padoc"))
        .map(|c| c.as_ref())
        .collect();

    // 3 in-situ tasks (all compressors support these)
    let insitu_tasks: Vec<Box<dyn AnalysisTask>> = vec![
        Box::new(analysis::OperatorHotspot::default()),
        Box::new(analysis::ParallelGroup::default()),
        Box::new(analysis::GpuBubbleRate::default()),
    ];
    // 4th task: only PADOC can do in-situ; others need decompress
    let layer_task = analysis::LayerComputeCommOverlap;

    // Compress and save artifacts
    struct ArtifactInfo { name: String, path: String, bytes_len: u64 }
    let mut artifacts: Vec<ArtifactInfo> = Vec::new();
    for c in &compressors {
        eprintln!("  Compressing with {}...", c.name());
        let artifact = c.compress(&trace).unwrap();
        let p = format!("/tmp/bench_insitu_{}.bin", c.name());
        std::fs::write(&p, &artifact.bytes).unwrap();
        eprintln!("    {} bytes", artifact.bytes.len());
        artifacts.push(ArtifactInfo { name: c.name().to_string(), path: p, bytes_len: artifact.bytes.len() as u64 });
    }
    drop(trace);
    eprintln!("");

    // Header
    println!("compressor\ttask\tin_situ\tartifact_bytes\tload_secs\tdecode_secs\tanalyze_secs\ttotal_secs\tresident_kib");

    // === Raw baseline: load JSON trace → run_raw directly ===
    {
        let raw_size: u64 = if path.is_dir() {
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
        let load_start = Instant::now();
        let raw_trace = load_trace(path);
        let load_secs = load_start.elapsed().as_secs_f64();
        let rss_after = current_rss_kib();
        let resident_kib = rss_after.saturating_sub(rss_before);

        for task in &insitu_tasks {
            let an_start = Instant::now();
            let _ = task.run_raw(&raw_trace).unwrap();
            let analyze_secs = an_start.elapsed().as_secs_f64();
            let total = load_secs + analyze_secs;
            println!("raw\t{}\tfalse\t{}\t{:.6}\t0.000000\t{:.6}\t{:.6}\t{}",
                task.name(), raw_size, load_secs, analyze_secs, total, resident_kib);
        }
        // layer task
        let an_start = Instant::now();
        let _ = layer_task.run_raw(&raw_trace).unwrap();
        let analyze_secs = an_start.elapsed().as_secs_f64();
        let total = load_secs + analyze_secs;
        println!("raw\t{}\tfalse\t{}\t{:.6}\t0.000000\t{:.6}\t{:.6}\t{}",
            layer_task.name(), raw_size, load_secs, analyze_secs, total, resident_kib);
        drop(raw_trace);
    }

    for (ci, c) in compressors.iter().enumerate() {
        let art = &artifacts[ci];

        // Load artifact from disk (once)
        let load_start = Instant::now();
        let bytes = std::fs::read(&art.path).unwrap();
        let load_secs = load_start.elapsed().as_secs_f64();

        // Decode (once)
        let rss_before = current_rss_kib();

        if c.name() == "padoc" {
            // PADOC: decode CompressedTrace, run all 4 tasks in-situ
            let decode_start = Instant::now();
            let compressed = padoc::trace::CompressedTrace::from_bytes(&bytes).unwrap();
            let decode_secs = decode_start.elapsed().as_secs_f64();
            let rss_after = current_rss_kib();
            let resident_kib = rss_after.saturating_sub(rss_before);

            // 3 in-situ tasks
            for task in &insitu_tasks {
                let an_start = Instant::now();
                let _ = task.run_in_situ(&compressed).unwrap();
                let analyze_secs = an_start.elapsed().as_secs_f64();
                let total = load_secs + decode_secs + analyze_secs;
                println!("{}\t{}\ttrue\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}",
                    c.name(), task.name(), art.bytes_len,
                    load_secs, decode_secs, analyze_secs, total, resident_kib);
            }
            // 4th task: layer_compute_comm_overlap
            let an_start = Instant::now();
            let _ = layer_task.run_in_situ(&compressed).unwrap();
            let analyze_secs = an_start.elapsed().as_secs_f64();
            let total = load_secs + decode_secs + analyze_secs;
            println!("{}\t{}\ttrue\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}",
                c.name(), layer_task.name(), art.bytes_len,
                load_secs, decode_secs, analyze_secs, total, resident_kib);
        } else {
            // ScalaTrace/TraceZip: run_in_situ (decode+analyze combined) for 3 tasks
            for task in &insitu_tasks {
                let rss_b = current_rss_kib();
                let insitu_start = Instant::now();
                let _ = c.run_in_situ(&bytes, task.name()).unwrap();
                let insitu_secs = insitu_start.elapsed().as_secs_f64();
                let rss_a = current_rss_kib();
                let resident_kib = rss_a.saturating_sub(rss_b);
                let total = load_secs + insitu_secs;
                println!("{}\t{}\ttrue\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}",
                    c.name(), task.name(), art.bytes_len,
                    load_secs, insitu_secs, 0.0, total, resident_kib);
            }
            // 4th task: decompress → raw (no in-situ support)
            let dec_start = Instant::now();
            let dec_trace = c.decompress(&bytes).unwrap();
            let decode_secs = dec_start.elapsed().as_secs_f64();
            let rss_a = current_rss_kib();
            let resident_kib = rss_a.saturating_sub(rss_before);
            let an_start = Instant::now();
            let _ = layer_task.run_raw(&dec_trace).unwrap();
            let analyze_secs = an_start.elapsed().as_secs_f64();
            let total = load_secs + decode_secs + analyze_secs;
            println!("{}\t{}\tfalse\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}",
                c.name(), layer_task.name(), art.bytes_len,
                load_secs, decode_secs, analyze_secs, total, resident_kib);
        }
    }
}
