//! Export ts column data for selected templates from a compressed artifact.
//!
//! Usage:
//!   cargo run --release --example export_ts_for_plot -- <artifact.zst> [--top N]
//!
//! Outputs CSV to stdout with columns: template_name, instance_index, ts_value, fitted_value, residual
//! Selects the top-N templates by instance count (default: 5).

use padoc::event::{NumColumn, Template};
use padoc::slp::SlpColumn;
use padoc::trace::CompressedTrace;
use std::env;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: export_ts_for_plot <artifact.zst> [--top N]");
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

    eprintln!("Loading artifact: {}", path.display());
    let ct = CompressedTrace::read_from_path(path)?;

    // Collect templates with their instance counts and ts columns
    let mut candidates: Vec<(usize, &str, &NumColumn)> = Vec::new();
    for (idx, tmpl) in ct.templates.iter().enumerate() {
        match tmpl {
            Template::Cpu(t) => {
                let count = t.ts.len();
                if count >= 100 {
                    candidates.push((idx, &t.name_pattern, &t.ts));
                }
            }
            Template::Gpu(t) => {
                let count = t.ts.len();
                if count >= 100 {
                    candidates.push((idx, &t.name_pattern, &t.ts));
                }
            }
        }
    }

    // Sort by instance count descending, take top N
    candidates.sort_by(|a, b| b.2.len().cmp(&a.2.len()));
    candidates.truncate(top_n);

    // Output header
    println!("template_id,template_name,instance_count,index,ts_value,fitted_value,residual,segment_id");

    for (tmpl_idx, name, col) in &candidates {
        let values = extract_i64_values(col);
        let n = values.len();
        if n == 0 {
            continue;
        }
        eprintln!(
            "  template[{}] '{}': {} instances",
            tmpl_idx, name, n
        );

        // Perform SLP encoding to get segments
        let slp = SlpColumn::encode(&values);

        // For each value, compute fitted and residual
        let mut global_idx: usize = 0;
        for (seg_id, seg) in slp.segments.iter().enumerate() {
            for local_i in 0..seg.length as usize {
                let fitted = seg.start.wrapping_add(seg.step.wrapping_mul(local_i as i64));
                let actual = values[global_idx];
                let residual = actual - fitted;
                // Output every value for small templates, subsample for large ones
                let should_output = n <= 2000 || global_idx % (n / 2000).max(1) == 0;
                if should_output {
                    println!(
                        "{},{},{},{},{},{},{},{}",
                        tmpl_idx,
                        escape_csv(name),
                        n,
                        global_idx,
                        actual,
                        fitted,
                        residual,
                        seg_id
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
