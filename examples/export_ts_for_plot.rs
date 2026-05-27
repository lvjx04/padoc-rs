//! Export per-rank ts column data for SLP visualization.
//!
//! Loads a single rank file, compresses it with PADoC, then exports ts columns
//! for selected templates. This gives clean per-rank data where ts values are
//! naturally piecewise-linear (events sorted by time within one rank).
//!
//! Usage:
//!   cargo run --release --example export_ts_for_plot -- <rank_file.json> [--top N]
//!
//! Outputs CSV: template_id, template_name, instance_count, index, ts_value, segment_id, segment_start, segment_step

use padoc::compressor::{CompressorConfig, TemplateCompressor};
use padoc::event::{NumColumn, Template};
use padoc::slp::SlpColumn;
use padoc::trace::Trace;
use std::env;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: export_ts_for_plot <rank_file.json> [--top N]");
        std::process::exit(1);
    }
    let path = Path::new(&args[1]);
    let mut top_n: usize = 5;

    let mut i = 2;
    while i < args.len() {
        if args[i] == "--top" && i + 1 < args.len() {
            top_n = args[i + 1].parse()?;
            i += 2;
        } else {
            i += 1;
        }
    }

    eprintln!("Loading trace: {}", path.display());
    let trace = Trace::from_file(path)?;
    eprintln!("  events: {}", trace.event_count());

    eprintln!("Compressing...");
    let mut compressor = TemplateCompressor::with_config(CompressorConfig::default());
    let compressed = compressor.compress(&trace)?;
    eprintln!("  templates: {}", compressed.templates.len());

    // Collect CPU templates with sufficient instances
    let mut candidates: Vec<(usize, &str, &NumColumn)> = Vec::new();
    for (idx, tmpl) in compressed.templates.iter().enumerate() {
        match tmpl {
            Template::Cpu(t) if t.ts.len() >= 50 => {
                candidates.push((idx, &t.name_pattern, &t.ts));
            }
            _ => {}
        }
    }

    // Sort by instance count descending, take top N
    candidates.sort_by(|a, b| b.2.len().cmp(&a.2.len()));
    candidates.truncate(top_n);

    // Output header
    println!("template_id,template_name,instance_count,index,ts_value,segment_id,segment_start,segment_step,segment_length");

    for (tmpl_idx, name, col) in &candidates {
        let values = extract_i64_values(col);
        let n = values.len();
        if n == 0 {
            continue;
        }
        eprintln!("  template[{}] '{}': {} instances", tmpl_idx, name, n);

        // Perform SLP encoding
        let slp = SlpColumn::encode(&values);
        eprintln!(
            "    -> {} segments (avg length: {:.1})",
            slp.segments.len(),
            n as f64 / slp.segments.len() as f64
        );

        // Output all values with segment info
        let mut global_idx: usize = 0;
        for (seg_id, seg) in slp.segments.iter().enumerate() {
            for _local_i in 0..seg.length as usize {
                // For large templates, subsample to ~3000 points
                let should_output = n <= 3000 || global_idx % (n / 3000).max(1) == 0;
                if should_output {
                    println!(
                        "{},{},{},{},{},{},{},{},{}",
                        tmpl_idx,
                        escape_csv(name),
                        n,
                        global_idx,
                        values[global_idx],
                        seg_id,
                        seg.start,
                        seg.step,
                        seg.length
                    );
                }
                global_idx += 1;
            }
        }
    }

    Ok(())
}

fn extract_i64_values(col: &NumColumn) -> Vec<i64> {
    match col {
        NumColumn::Empty => Vec::new(),
        NumColumn::Constant { len, value } => vec![*value; *len as usize],
        NumColumn::I32(v) => v.iter().map(|x| *x as i64).collect(),
        NumColumn::I64(v) => v.clone(),
    }
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
