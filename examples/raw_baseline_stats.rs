//! Compute per-region raw baseline sizes for a dataset.
//!
//! Usage:
//!   cargo run --release --example raw_baseline_stats -- <path> <dataset_name>
//!
//! `path` may be a single chrome-trace JSON file or a directory of per-rank
//! files.  Each rank is loaded and dropped individually so peak RAM stays at
//! one rank's worth (~100 MB for llama).
//!
//! Outputs a TSV with columns: dataset, region, raw_bytes

use padoc::trace::{list_trace_files, Trace};
use std::env;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: raw_baseline_stats <path> <dataset_name>");
        std::process::exit(1);
    }
    let path = Path::new(&args[1]);
    let dataset = &args[2];

    let mut total_events: u64 = 0;
    let mut total_name_bytes: u64 = 0;
    let mut total_args_bytes: u64 = 0;

    if path.is_dir() {
        let files = list_trace_files(path);
        let n = files.len();
        for (i, file) in files.iter().enumerate() {
            let trace = Trace::from_file(file)?;
            accumulate(&trace, &mut total_events, &mut total_name_bytes, &mut total_args_bytes);
            eprintln!("[{}/{}] {} events so far", i + 1, n, total_events);
        }
    } else {
        let trace = Trace::from_file(path)?;
        accumulate(&trace, &mut total_events, &mut total_name_bytes, &mut total_args_bytes);
    }

    let ts_raw = total_events * 8;
    let dur_raw = total_events * 8;
    let tree_raw = total_events * 16;
    let ids_raw = total_events * 24;

    println!("dataset\tregion\traw_bytes");
    println!("{dataset}\tts\t{ts_raw}");
    println!("{dataset}\tdur\t{dur_raw}");
    println!("{dataset}\targs\t{total_args_bytes}");
    println!("{dataset}\tnames\t{total_name_bytes}");
    println!("{dataset}\ttree_refs\t{tree_raw}");
    println!("{dataset}\tids_pids_streams\t{ids_raw}");
    println!("{dataset}\ttotal_events\t{total_events}");

    Ok(())
}

fn accumulate(
    trace: &Trace,
    total_events: &mut u64,
    total_name_bytes: &mut u64,
    total_args_bytes: &mut u64,
) {
    for (_rank, _pid, _tid, _ph, events) in trace.iter_streams() {
        for event in events {
            *total_events += 1;
            *total_name_bytes += event.name.len() as u64;
            if let Some(args) = &event.args {
                // Use msgpack encoding size for fair comparison with PADoC
                if let Ok(bytes) = rmp_serde::to_vec(args) {
                    *total_args_bytes += bytes.len() as u64;
                }
            }
        }
    }
}
