//! Probe name_nums content to understand size and compression potential.
//!
//! Usage:
//!   cargo run --release --example probe_name_nums -- <artifact.padoc.zst>

use padoc::event::{DigitColumn, NameNums, Template};
use padoc::trace::CompressedTrace;
use std::collections::HashSet;
use std::env;

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).expect("usage: probe_name_nums <artifact>");

    eprintln!("Loading: {}", path);
    let ct = CompressedTrace::read_from_path(&path)?;
    eprintln!("  templates: {}", ct.templates.len());

    let mut total_i32_values: u64 = 0;
    let mut total_i64_values: u64 = 0;
    let mut total_string_values: u64 = 0;
    let mut total_constant: u64 = 0;
    let mut total_columns: u64 = 0;
    let mut i32_col_count: u64 = 0;

    // Collect stats without cloning
    struct ColStats {
        name: String,
        col_idx: usize,
        len: usize,
        min_v: i32,
        max_v: i32,
        mono_inc: bool,
        mono_dec: bool,
        unique_diffs: usize,
        segments: u64,
        period: u32,
        first_values: Vec<i32>,
        first_diffs: Vec<i32>,
    }

    let mut top_cols: Vec<ColStats> = Vec::new();

    for tmpl in &ct.templates {
        let (name, nums) = match tmpl {
            Template::Cpu(t) => (&t.name_pattern, &t.name_nums),
            Template::Gpu(t) => (&t.name_pattern, &t.name_nums),
        };
        let cols = match nums {
            NameNums::Empty => continue,
            NameNums::Rows(_) => continue,
            NameNums::Columnar(c) => c,
        };
        for (ci, col) in cols.iter().enumerate() {
            total_columns += 1;
            match col {
                DigitColumn::Constant(_) => {
                    total_constant += 1;
                }
                DigitColumn::I32 { values, .. } => {
                    total_i32_values += values.len() as u64;
                    i32_col_count += 1;
                    if values.len() >= 1000 {
                        // Analyze in-place without clone
                        let n = values.len();
                        let min_v = *values.iter().min().unwrap();
                        let max_v = *values.iter().max().unwrap();
                        let mono_inc = values.windows(2).all(|w| w[1] >= w[0]);
                        let mono_dec = values.windows(2).all(|w| w[1] <= w[0]);

                        // Count unique diffs (sample first 10000)
                        let sample_n = n.min(10000);
                        let mut unique_diffs: HashSet<i32> = HashSet::new();
                        for w in values[..sample_n].windows(2) {
                            unique_diffs.insert(w[1] - w[0]);
                        }

                        // Count segments
                        let mut segments = 1u64;
                        let mut prev_diff = values[1] - values[0];
                        for w in values[1..].windows(2) {
                            let d = w[1] - w[0];
                            if d != prev_diff {
                                segments += 1;
                                prev_diff = d;
                            }
                        }

                        // Detect period (sample)
                        let mut period = 0u32;
                        if !mono_inc && !mono_dec {
                            let check_n = n.min(5000);
                            for p in 1..=(check_n / 2).min(500) {
                                let matches = values[..check_n - p].iter()
                                    .zip(values[p..check_n].iter())
                                    .filter(|(a, b)| a == b)
                                    .count();
                                if matches as f64 / (check_n - p) as f64 > 0.95 {
                                    period = p as u32;
                                    break;
                                }
                            }
                        }

                        let first_values: Vec<i32> = values[..n.min(10)].to_vec();
                        let first_diffs: Vec<i32> = values[..n.min(11)].windows(2)
                            .map(|w| w[1] - w[0]).take(10).collect();

                        top_cols.push(ColStats {
                            name: name.clone(),
                            col_idx: ci,
                            len: n,
                            min_v,
                            max_v,
                            mono_inc,
                            mono_dec,
                            unique_diffs: unique_diffs.len(),
                            segments,
                            period,
                            first_values,
                            first_diffs,
                        });
                    }
                }
                DigitColumn::I64 { values, .. } => {
                    total_i64_values += values.len() as u64;
                }
                DigitColumn::Strings(v) => {
                    total_string_values += v.len() as u64;
                }
            }
        }
    }

    println!("=== name_nums summary ===");
    println!("total columns: {}", total_columns);
    println!("  Constant: {}", total_constant);
    println!("  I32: {} columns, {} values ({:.3} GiB)",
        i32_col_count, total_i32_values,
        total_i32_values as f64 * 4.0 / 1024.0 / 1024.0 / 1024.0);
    println!("  I64: {} values ({:.3} GiB)",
        total_i64_values,
        total_i64_values as f64 * 8.0 / 1024.0 / 1024.0 / 1024.0);
    println!("  Strings: {} values", total_string_values);

    // Sort by size
    top_cols.sort_by(|a, b| b.len.cmp(&a.len));

    println!("\n=== top I32 columns (>= 1000 values) by size ===");
    for stats in top_cols.iter().take(20) {
        let avg_seg_len = stats.len as f64 / stats.segments as f64;
        let display_name = if stats.name.len() > 45 { &stats.name[..45] } else { &stats.name };
        println!("\n  [{}] col[{}]: {} values, range [{}, {}]",
            display_name, stats.col_idx, stats.len, stats.min_v, stats.max_v);
        println!("    mono_inc={} mono_dec={} period={} unique_diffs={}",
            stats.mono_inc, stats.mono_dec, stats.period, stats.unique_diffs);
        println!("    segments={} avg_seg_len={:.1}",
            stats.segments, avg_seg_len);
        println!("    first 10 values: {:?}", stats.first_values);
        println!("    first 10 diffs:  {:?}", stats.first_diffs);
    }

    // Summary: compression potential
    println!("\n=== compression potential ===");
    let mut total_slp_segments: u64 = 0;
    let mut total_analyzed_values: u64 = 0;
    let mut periodic_count = 0u32;
    let mut few_unique_count = 0u32;
    for stats in &top_cols {
        total_slp_segments += stats.segments;
        total_analyzed_values += stats.len as u64;
        if stats.period > 0 { periodic_count += 1; }
        if stats.unique_diffs <= 5 { few_unique_count += 1; }
    }
    if !top_cols.is_empty() {
        println!("  analyzed {} columns (>= 1000 values)", top_cols.len());
        println!("  total values: {} ({:.3} GiB)", total_analyzed_values,
            total_analyzed_values as f64 * 4.0 / 1024.0/1024.0/1024.0);
        println!("  total SLP segments: {} (avg seg len: {:.1})",
            total_slp_segments, total_analyzed_values as f64 / total_slp_segments as f64);
        println!("  periodic: {}/{}", periodic_count, top_cols.len());
        println!("  few unique diffs (<= 5): {}/{}", few_unique_count, top_cols.len());
    }

    Ok(())
}
