//! Extract GPU kernel timestamps from the unifolm dataset for piecewise-linear
//! figure generation.
//!
//! Usage:
//!   cargo run --release --example extract_unifolm_ts -- <path> [--kernel <name>] [--top N] [--rank <idx>]
//!
//! If --kernel is not given, lists top kernels by instance count.
//! If --kernel is given, outputs CSV: index,ts
//!
//! Examples:
//!   # Discover top kernels from unifolm directory:
//!   cargo run --release --example extract_unifolm_ts -- /mnt/treasure/ljx/Trace_int/unifolm-world-model_json
//!
//!   # Extract ts for a specific kernel (single rank for cleaner piecewise pattern):
//!   cargo run --release --example extract_unifolm_ts -- /mnt/treasure/ljx/Trace_int/unifolm-world-model_json/global_rank0.json --kernel "ampere_fp16"
//!
//!   # Extract from whole directory:
//!   cargo run --release --example extract_unifolm_ts -- /mnt/treasure/ljx/Trace_int/unifolm-world-model_json --kernel "ampere_fp16" --top 2000

use padoc::trace::Trace;
use std::collections::HashMap;
use std::env;
use std::path::Path;

fn load_trace(path: &Path) -> Trace {
    if path.is_dir() {
        Trace::from_dir(path).unwrap()
    } else {
        Trace::from_file(path).unwrap()
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: extract_unifolm_ts <path> [--kernel <name>] [--top N] [--rank <idx>]");
        std::process::exit(1);
    }

    let path = &args[1];
    let mut kernel: Option<String> = None;
    let mut top_n: usize = 20;
    let mut rank_filter: Option<String> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--kernel" if i + 1 < args.len() => {
                kernel = Some(args[i + 1].clone());
                i += 2;
            }
            "--top" if i + 1 < args.len() => {
                top_n = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--rank" if i + 1 < args.len() => {
                rank_filter = Some(args[i + 1].clone());
                i += 2;
            }
            _ => { i += 1; }
        }
    }

    eprintln!("Loading: {}", path);
    let trace = load_trace(Path::new(path));
    eprintln!("  events: {}", trace.event_count());

    match kernel {
        None => {
            // Discovery mode: list top kernels by instance count
            let mut counts: HashMap<String, u64> = HashMap::new();
            for (rank, _pid, _tid, _ph, events) in trace.iter_streams() {
                if let Some(ref rf) = rank_filter {
                    if !rank.contains(rf.as_str()) {
                        continue;
                    }
                }
                for ev in events {
                    *counts.entry(ev.name.clone()).or_default() += 1;
                }
            }

            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));

            eprintln!("\nTop {} kernels by instance count:", top_n);
            println!("count\tname");
            for (name, count) in sorted.iter().take(top_n) {
                println!("{}\t{}", count, name);
            }
        }
        Some(kernel_name) => {
            // Extraction mode: output index,ts CSV
            let mut ts_list: Vec<i64> = Vec::new();
            for (rank, _pid, _tid, _ph, events) in trace.iter_streams() {
                if let Some(ref rf) = rank_filter {
                    if !rank.contains(rf.as_str()) {
                        continue;
                    }
                }
                for ev in events {
                    if ev.name.contains(kernel_name.as_str()) {
                        ts_list.push(ev.ts);
                    }
                }
            }
            ts_list.sort();

            let output_count = if ts_list.len() > top_n {
                // Subsample for plotting if too many instances
                top_n
            } else {
                ts_list.len()
            };

            eprintln!("{}: {} instances (outputting {})", kernel_name, ts_list.len(), output_count);

            if ts_list.len() <= top_n {
                for (i, ts) in ts_list.iter().enumerate() {
                    println!("{},{}", i, ts);
                }
            } else {
                // Subsample evenly
                let step = ts_list.len() as f64 / output_count as f64;
                for out_i in 0..output_count {
                    let idx = (out_i as f64 * step) as usize;
                    println!("{},{}", out_i, ts_list[idx]);
                }
            }
        }
    }
}
