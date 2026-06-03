//! Measure in-situ analysis time for all baselines on a dataset.
//! Compresses once per compressor, then runs in-situ analysis (timing only the analysis).
//!
//! Usage:
//!   cargo run --release --example bench_analysis_time -- <trace_path_or_dir>

use padoc::analysis::{self, AnalysisTask};
use padoc::baselines::{self, BaselineCompressor};
use padoc::trace::Trace;
use std::env;
use std::path::Path;
use std::time::Instant;

fn load_trace(path: &Path) -> Trace {
    if path.is_dir() { Trace::from_dir(path).unwrap() } else { Trace::from_file(path).unwrap() }
}

fn main() {
    let path_str = env::args().nth(1).expect("usage: bench_analysis_time <trace_path>");
    let path = Path::new(&path_str);
    let dataset = path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown");

    eprintln!("[{}] Loading trace...", dataset);
    let start = Instant::now();
    let trace = load_trace(path);
    eprintln!("[{}] {} events in {:.3}s", dataset, trace.event_count(), start.elapsed().as_secs_f64());

    let registry = baselines::registry();
    let compressors: Vec<&dyn BaselineCompressor> = registry.iter()
        .filter(|c| matches!(c.name(), "scalatrace" | "tracezip" | "padoc"))
        .map(|c| c.as_ref())
        .collect();

    let tasks: Vec<Box<dyn AnalysisTask>> = vec![
        Box::new(analysis::OperatorHotspot::default()),
        Box::new(analysis::ParallelGroup::default()),
        Box::new(analysis::GpuBubbleRate::default()),
        Box::new(analysis::LayerComputeCommOverlap),
    ];

    println!("dataset\tcompressor\ttask\tin_situ\tdecode_secs\tanalyze_secs\ttotal_secs");

    // Raw baseline
    {
        eprintln!("[{}] Running raw baseline...", dataset);
        for task in &tasks {
            let an_start = Instant::now();
            let _ = task.run_raw(&trace).unwrap();
            let secs = an_start.elapsed().as_secs_f64();
            println!("{}\traw\t{}\tfalse\t0.000000\t{:.6}\t{:.6}", dataset, task.name(), secs, secs);
        }
    }

    // Each compressor: compress → in-situ analyze
    for c in &compressors {
        eprintln!("[{}] Compressing with {}...", dataset, c.name());
        let artifact = c.compress(&trace).unwrap();
        eprintln!("[{}] {} artifact: {} bytes", dataset, c.name(), artifact.bytes.len());

        if c.name() == "padoc" {
            // PADOC: decode once, run all 4 tasks
            let decode_start = Instant::now();
            let compressed = padoc::trace::CompressedTrace::from_bytes(&artifact.bytes).unwrap();
            let decode_secs = decode_start.elapsed().as_secs_f64();
            eprintln!("[{}] padoc decode time: {:.3}s", dataset, decode_secs);
            for task in &tasks {
                let an_start = Instant::now();
                let _ = task.run_in_situ(&compressed).unwrap();
                let secs = an_start.elapsed().as_secs_f64();
                println!("{}\tpadoc\t{}\ttrue\t{:.6}\t{:.6}\t{:.6}", dataset, task.name(), decode_secs, secs, decode_secs + secs);
            }
        } else {
            // ScalaTrace/TraceZip: decode once, then run 3 tasks on decoded payload
            let decode_start = Instant::now();
            let decoded = c.decode_for_analysis(&artifact.bytes).unwrap();
            let decode_secs = decode_start.elapsed().as_secs_f64();
            eprintln!("[{}] {} decode time: {:.3}s", dataset, c.name(), decode_secs);

            for task in &tasks[..3] {
                let an_start = Instant::now();
                let _ = c.run_in_situ_decoded(decoded.as_ref(), task.name()).unwrap();
                let analyze_secs = an_start.elapsed().as_secs_f64();
                println!("{}\t{}\t{}\ttrue\t{:.6}\t{:.6}\t{:.6}", dataset, c.name(), task.name(), decode_secs, analyze_secs, decode_secs + analyze_secs);
            }
            drop(decoded);
            // layer: decompress + run_raw (pure analyze only)
            let dec_trace = c.decompress(&artifact.bytes).unwrap();
            let an_start = Instant::now();
            let _ = tasks[3].run_raw(&dec_trace).unwrap();
            let secs = an_start.elapsed().as_secs_f64();
            println!("{}\t{}\t{}\tfalse\t0.000000\t{:.6}\t{:.6}", dataset, c.name(), tasks[3].name(), secs, secs);
        }
    }
}
