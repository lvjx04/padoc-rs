use anyhow::{Context, Result};
use padoc::event::{NumColumn, Template};
use padoc::trace::CompressedTrace;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const Q: u32 = 32;
const SCALE: i128 = 1_i128 << Q;
const HALF: i128 = SCALE / 2;
const SEGMENT_BYTES: u64 = 24;
const COLUMN_HEADER_BYTES: u64 = 64;

#[derive(Debug)]
struct Args {
    datasets: Vec<(String, PathBuf)>,
    out_dir: PathBuf,
    min_len: usize,
    max_columns: usize,
    max_values: usize,
    max_segment_len: usize,
    zstd_level: i32,
}

#[derive(Debug, Clone)]
struct ColumnCandidate {
    dataset: String,
    template_id: usize,
    template_kind: &'static str,
    column: &'static str,
    name_pattern: String,
    len: usize,
}

#[derive(Debug, Clone, Copy)]
struct ProbeOptions {
    max_values: usize,
    max_segment_len: usize,
    zstd_level: i32,
}

#[derive(Debug)]
struct SlpResult {
    mem_bytes: u64,
    segment_count: usize,
    avg_segment_len: f64,
    max_abs_residual: i64,
    raw_msgpack_bytes: u64,
    zstd_bytes: u64,
    encode_secs: f64,
    decode_ok: bool,
}

#[derive(Debug, Serialize)]
struct ColumnRow {
    dataset: String,
    template_id: usize,
    template_kind: &'static str,
    column: &'static str,
    name_pattern: String,
    original_len: usize,
    sampled_len: usize,
    truncated: bool,
    current_encoding: &'static str,
    current_mem_bytes_est: u64,
    i64_mem_bytes: u64,
    i32_applicable: bool,
    i32_mem_bytes: u64,
    constant: bool,
    constant_mem_bytes: u64,
    slp_i8_mem_bytes: u64,
    slp_i8_segments: usize,
    slp_i8_avg_segment_len: f64,
    slp_i8_max_abs_residual: i64,
    slp_i8_msgpack_bytes: u64,
    slp_i8_zstd_bytes: u64,
    slp_i8_encode_secs: f64,
    slp_i8_decode_ok: bool,
    slp_i16_mem_bytes: u64,
    slp_i16_segments: usize,
    slp_i16_avg_segment_len: f64,
    slp_i16_max_abs_residual: i64,
    slp_i16_msgpack_bytes: u64,
    slp_i16_zstd_bytes: u64,
    slp_i16_encode_secs: f64,
    slp_i16_decode_ok: bool,
    best_slp: &'static str,
    best_slp_mem_bytes: u64,
    best_slp_ratio_vs_i64: f64,
    best_slp_ratio_vs_i32: f64,
    accepted_vs_i64_gt_2: bool,
    beats_i32: bool,
    i64_msgpack_bytes: u64,
    i64_zstd_bytes: u64,
    i32_msgpack_bytes: u64,
    i32_zstd_bytes: u64,
}

#[derive(Default)]
struct Aggregate {
    columns: usize,
    values: usize,
    truncated_columns: usize,
    i64_mem: u64,
    i32_mem: u64,
    current_mem: u64,
    slp_i8_mem: u64,
    slp_i16_mem: u64,
    best_slp_mem: u64,
    i64_zstd: u64,
    i32_zstd: u64,
    slp_i8_zstd: u64,
    slp_i16_zstd: u64,
    best_slp_zstd: u64,
    accepted_columns: usize,
    beats_i32_columns: usize,
    encode_secs: f64,
}

#[derive(Serialize)]
struct EncodedSlpColumn {
    len: u32,
    q: u8,
    residual_width: u8,
    segments: Vec<EncodedSegment>,
    residuals: Residuals,
}

#[derive(Serialize)]
struct EncodedSegment {
    end: u32,
    base: i64,
    slope_q: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Residuals {
    I8(Vec<i8>),
    I16(Vec<i16>),
}

fn main() -> Result<()> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("create {}", args.out_dir.display()))?;
    let columns_path = args.out_dir.join("slp_probe_columns.tsv");
    let summary_path = args.out_dir.join("slp_probe_summary.md");
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&columns_path)?;
    let mut aggregates: BTreeMap<String, Aggregate> = BTreeMap::new();

    for (dataset, path) in &args.datasets {
        eprintln!("loading {dataset} from {}", path.display());
        let trace = CompressedTrace::read_from_path(path)?;
        let mut candidates = collect_candidates(dataset, &trace, args.min_len);
        candidates.sort_by(|a, b| b.len.cmp(&a.len));
        if args.max_columns > 0 && candidates.len() > args.max_columns {
            candidates.truncate(args.max_columns);
        }
        eprintln!("probing {dataset}: {} columns", candidates.len());

        for candidate in candidates {
            let Some(col) = get_column(&trace, candidate.template_id, candidate.column) else {
                continue;
            };
            let row = probe_column(
                &candidate,
                col,
                ProbeOptions {
                    max_values: args.max_values,
                    max_segment_len: args.max_segment_len,
                    zstd_level: args.zstd_level,
                },
            )?;
            update_aggregate(aggregates.entry(dataset.clone()).or_default(), &row);
            writer.serialize(row)?;
        }
        writer.flush()?;
    }

    write_summary(&summary_path, &args, &aggregates)?;
    eprintln!("wrote {}", columns_path.display());
    eprintln!("wrote {}", summary_path.display());
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut out = Args {
        datasets: Vec::new(),
        out_dir: PathBuf::from("/mnt/treasure/ljx/slp_probe"),
        min_len: 1024,
        max_columns: 128,
        max_values: 1_000_000,
        max_segment_len: 65_536,
        zstd_level: 3,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => {
                let value = args.next().context("--dataset requires NAME=PATH")?;
                let (name, path) = value
                    .split_once('=')
                    .context("--dataset value must be NAME=PATH")?;
                out.datasets.push((name.to_string(), PathBuf::from(path)));
            }
            "--out-dir" => {
                out.out_dir = PathBuf::from(args.next().context("--out-dir requires PATH")?)
            }
            "--min-len" => out.min_len = args.next().context("--min-len requires N")?.parse()?,
            "--max-columns" => {
                out.max_columns = args.next().context("--max-columns requires N")?.parse()?
            }
            "--max-values" => {
                out.max_values = args.next().context("--max-values requires N")?.parse()?
            }
            "--max-segment-len" => {
                out.max_segment_len = args
                    .next()
                    .context("--max-segment-len requires N")?
                    .parse()?
            }
            "--zstd-level" => {
                out.zstd_level = args.next().context("--zstd-level requires N")?.parse()?
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => anyhow::bail!("unexpected argument: {arg}"),
        }
    }

    if out.datasets.is_empty() {
        for name in [
            "leworldmodel_full",
            "qwen3_full",
            "unifolm_full",
            "llama_full",
        ] {
            let path = PathBuf::from(format!("/mnt/treasure/ljx/artifacts_v6/{name}.padoc.zst"));
            if path.exists() {
                out.datasets.push((name.to_string(), path));
            }
        }
    }

    if out.datasets.is_empty() {
        anyhow::bail!("no datasets provided and no default artifacts found");
    }
    if out.max_values == 0 {
        anyhow::bail!("--max-values must be positive");
    }
    if out.max_segment_len < 2 {
        anyhow::bail!("--max-segment-len must be at least 2");
    }
    Ok(out)
}

fn print_usage() {
    eprintln!(
        "usage: cargo run --release --example slp_probe -- \\
         [--dataset NAME=PATH ...] [--out-dir DIR] [--min-len N] \\
         [--max-columns N] [--max-values N] [--max-segment-len N] [--zstd-level N]"
    );
}

fn collect_candidates(
    dataset: &str,
    trace: &CompressedTrace,
    min_len: usize,
) -> Vec<ColumnCandidate> {
    let mut out = Vec::new();
    for (idx, template) in trace.templates.iter().enumerate() {
        match template {
            Template::Cpu(t) => {
                push_candidate(
                    &mut out,
                    dataset,
                    idx,
                    "cpu",
                    "ts",
                    &t.name_pattern,
                    t.ts.len(),
                    min_len,
                );
                push_candidate(
                    &mut out,
                    dataset,
                    idx,
                    "cpu",
                    "dur",
                    &t.name_pattern,
                    t.dur.len(),
                    min_len,
                );
            }
            Template::Gpu(t) => {
                push_candidate(
                    &mut out,
                    dataset,
                    idx,
                    "gpu",
                    "ts",
                    &t.name_pattern,
                    t.ts.len(),
                    min_len,
                );
                push_candidate(
                    &mut out,
                    dataset,
                    idx,
                    "gpu",
                    "dur",
                    &t.name_pattern,
                    t.dur.len(),
                    min_len,
                );
            }
        }
    }
    out
}

fn push_candidate(
    out: &mut Vec<ColumnCandidate>,
    dataset: &str,
    template_id: usize,
    template_kind: &'static str,
    column: &'static str,
    name_pattern: &str,
    len: usize,
    min_len: usize,
) {
    if len >= min_len {
        out.push(ColumnCandidate {
            dataset: dataset.to_string(),
            template_id,
            template_kind,
            column,
            name_pattern: name_pattern.to_string(),
            len,
        });
    }
}

fn get_column<'a>(
    trace: &'a CompressedTrace,
    template_id: usize,
    column: &str,
) -> Option<&'a NumColumn> {
    match trace.templates.get(template_id)? {
        Template::Cpu(t) => match column {
            "ts" => Some(&t.ts),
            "dur" => Some(&t.dur),
            _ => None,
        },
        Template::Gpu(t) => match column {
            "ts" => Some(&t.ts),
            "dur" => Some(&t.dur),
            _ => None,
        },
    }
}

fn probe_column(
    candidate: &ColumnCandidate,
    col: &NumColumn,
    opts: ProbeOptions,
) -> Result<ColumnRow> {
    let sampled_len = candidate.len.min(opts.max_values);
    let values: Vec<i64> = col.iter_i64().take(sampled_len).collect();
    let truncated = sampled_len < candidate.len;
    let i64_mem = 8_u64 * sampled_len as u64;
    let i32_applicable = values.iter().all(|&v| i32::try_from(v).is_ok());
    let i32_mem = if i32_applicable {
        4_u64 * sampled_len as u64
    } else {
        i64_mem
    };
    let constant = values
        .first()
        .map(|first| values.iter().all(|v| v == first))
        .unwrap_or(false);
    let constant_mem = if constant && sampled_len > 0 { 12 } else { 0 };
    let current_mem = sampled_current_mem_bytes(col, sampled_len);

    let (i64_msgpack, i64_zstd) = encoded_sizes(&values, opts.zstd_level)?;
    let (i32_msgpack, i32_zstd) = if i32_applicable {
        let v32: Vec<i32> = values.iter().map(|&v| v as i32).collect();
        encoded_sizes(&v32, opts.zstd_level)?
    } else {
        (0, 0)
    };

    let slp_i8 = encode_slp(&values, 127, 1, opts)?;
    let slp_i16 = encode_slp(&values, 32_767, 2, opts)?;
    let (best_name, best_mem) = if slp_i8.mem_bytes <= slp_i16.mem_bytes {
        ("slp_i8", slp_i8.mem_bytes)
    } else {
        ("slp_i16", slp_i16.mem_bytes)
    };
    let best_ratio_vs_i64 = ratio(i64_mem, best_mem);
    let best_ratio_vs_i32 = ratio(i32_mem, best_mem);
    let accepted = best_ratio_vs_i64 > 2.0;
    let beats_i32 = best_mem < i32_mem;

    Ok(ColumnRow {
        dataset: candidate.dataset.clone(),
        template_id: candidate.template_id,
        template_kind: candidate.template_kind,
        column: candidate.column,
        name_pattern: candidate.name_pattern.clone(),
        original_len: candidate.len,
        sampled_len,
        truncated,
        current_encoding: encoding_name(col),
        current_mem_bytes_est: current_mem,
        i64_mem_bytes: i64_mem,
        i32_applicable,
        i32_mem_bytes: i32_mem,
        constant,
        constant_mem_bytes: constant_mem,
        slp_i8_mem_bytes: slp_i8.mem_bytes,
        slp_i8_segments: slp_i8.segment_count,
        slp_i8_avg_segment_len: slp_i8.avg_segment_len,
        slp_i8_max_abs_residual: slp_i8.max_abs_residual,
        slp_i8_msgpack_bytes: slp_i8.raw_msgpack_bytes,
        slp_i8_zstd_bytes: slp_i8.zstd_bytes,
        slp_i8_encode_secs: slp_i8.encode_secs,
        slp_i8_decode_ok: slp_i8.decode_ok,
        slp_i16_mem_bytes: slp_i16.mem_bytes,
        slp_i16_segments: slp_i16.segment_count,
        slp_i16_avg_segment_len: slp_i16.avg_segment_len,
        slp_i16_max_abs_residual: slp_i16.max_abs_residual,
        slp_i16_msgpack_bytes: slp_i16.raw_msgpack_bytes,
        slp_i16_zstd_bytes: slp_i16.zstd_bytes,
        slp_i16_encode_secs: slp_i16.encode_secs,
        slp_i16_decode_ok: slp_i16.decode_ok,
        best_slp: best_name,
        best_slp_mem_bytes: best_mem,
        best_slp_ratio_vs_i64: best_ratio_vs_i64,
        best_slp_ratio_vs_i32: best_ratio_vs_i32,
        accepted_vs_i64_gt_2: accepted,
        beats_i32,
        i64_msgpack_bytes: i64_msgpack,
        i64_zstd_bytes: i64_zstd,
        i32_msgpack_bytes: i32_msgpack,
        i32_zstd_bytes: i32_zstd,
    })
}

fn encode_slp(
    values: &[i64],
    eps: i64,
    residual_width: u8,
    opts: ProbeOptions,
) -> Result<SlpResult> {
    let start_time = Instant::now();
    let mut segments = Vec::new();
    let mut residuals_i8 = if residual_width == 1 {
        Vec::with_capacity(values.len())
    } else {
        Vec::new()
    };
    let mut residuals_i16 = if residual_width == 2 {
        Vec::with_capacity(values.len())
    } else {
        Vec::new()
    };
    let mut max_abs_residual = 0_i64;
    let mut start = 0;
    while start < values.len() {
        let (end, slope_q) = longest_segment(values, start, eps, opts.max_segment_len);
        let base = values[start];
        for (offset, &value) in values[start..end].iter().enumerate() {
            let pred = predict(base, slope_q, offset);
            let residual = value
                .checked_sub(pred)
                .with_context(|| format!("residual overflow at index {}", start + offset))?;
            let abs = residual.saturating_abs();
            if abs > eps {
                anyhow::bail!(
                    "residual {} exceeds eps {} at index {}",
                    residual,
                    eps,
                    start + offset
                );
            }
            max_abs_residual = max_abs_residual.max(abs);
            match residual_width {
                1 => residuals_i8.push(i8::try_from(residual)?),
                2 => residuals_i16.push(i16::try_from(residual)?),
                _ => unreachable!(),
            }
        }
        segments.push(EncodedSegment {
            end: u32::try_from(end).context("segment end exceeds u32")?,
            base,
            slope_q,
        });
        start = end;
    }
    let encode_secs = start_time.elapsed().as_secs_f64();
    let avg_segment_len = if segments.is_empty() {
        0.0
    } else {
        values.len() as f64 / segments.len() as f64
    };
    let residual_bytes = residual_width as u64 * values.len() as u64;
    let mem_bytes = COLUMN_HEADER_BYTES + residual_bytes + SEGMENT_BYTES * segments.len() as u64;
    let encoded = EncodedSlpColumn {
        len: u32::try_from(values.len()).context("column length exceeds u32")?,
        q: Q as u8,
        residual_width,
        segments,
        residuals: if residual_width == 1 {
            Residuals::I8(residuals_i8)
        } else {
            Residuals::I16(residuals_i16)
        },
    };
    let decode_ok = verify_encoded(&encoded, values);
    let (raw_msgpack_bytes, zstd_bytes) = encoded_sizes(&encoded, opts.zstd_level)?;
    Ok(SlpResult {
        mem_bytes,
        segment_count: encoded.segments.len(),
        avg_segment_len,
        max_abs_residual,
        raw_msgpack_bytes,
        zstd_bytes,
        encode_secs,
        decode_ok,
    })
}

fn longest_segment(values: &[i64], start: usize, eps: i64, max_segment_len: usize) -> (usize, i64) {
    let n = values.len();
    let max_end = n.min(start.saturating_add(max_segment_len));
    if start + 1 >= max_end {
        return (start + 1, 0);
    }
    let base = values[start] as i128;
    let mut lo = -(1_i128 << 120);
    let mut hi = 1_i128 << 120;
    let mut best_end = start + 1;
    let mut best_lo = 0_i128;
    let mut best_hi = 0_i128;

    for (j, &value) in values.iter().enumerate().take(max_end).skip(start + 1) {
        let dt = (j - start) as i128;
        let delta = value as i128 - base;
        let pred_lo = delta - eps as i128;
        let pred_hi = delta + eps as i128;
        let x_min = pred_lo * SCALE - HALF;
        let x_max = (pred_hi + 1) * SCALE - HALF - 1;
        let slope_lo = ceil_div(x_min, dt);
        let slope_hi = floor_div(x_max, dt);
        lo = lo.max(slope_lo);
        hi = hi.min(slope_hi);
        if lo > hi {
            break;
        }
        best_end = j + 1;
        best_lo = lo;
        best_hi = hi;
    }

    let slope = best_lo + (best_hi - best_lo) / 2;
    let slope = slope.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
    (best_end, slope)
}

fn predict(base: i64, slope_q: i64, offset: usize) -> i64 {
    let x = slope_q as i128 * offset as i128;
    let rounded = floor_div(x + HALF, SCALE);
    let value = base as i128 + rounded;
    value.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

fn verify_encoded(encoded: &EncodedSlpColumn, values: &[i64]) -> bool {
    let mut pos = 0_usize;
    for segment in &encoded.segments {
        let end = segment.end as usize;
        if end > values.len() || end < pos {
            return false;
        }
        for offset in 0..(end - pos) {
            let residual = match &encoded.residuals {
                Residuals::I8(v) => *v.get(pos + offset).unwrap_or(&0) as i64,
                Residuals::I16(v) => *v.get(pos + offset).unwrap_or(&0) as i64,
            };
            let decoded = predict(segment.base, segment.slope_q, offset) + residual;
            if decoded != values[pos + offset] {
                return false;
            }
        }
        pos = end;
    }
    pos == values.len()
}

fn floor_div(a: i128, b: i128) -> i128 {
    debug_assert!(b > 0);
    let q = a / b;
    let r = a % b;
    if r < 0 {
        q - 1
    } else {
        q
    }
}

fn ceil_div(a: i128, b: i128) -> i128 {
    -floor_div(-a, b)
}

fn encoded_sizes<T: Serialize>(value: &T, zstd_level: i32) -> Result<(u64, u64)> {
    let raw = rmp_serde::to_vec_named(value)?;
    let zstd = zstd::bulk::compress(&raw, zstd_level)?;
    Ok((raw.len() as u64, zstd.len() as u64))
}

fn encoding_name(col: &NumColumn) -> &'static str {
    match col {
        NumColumn::Empty => "empty",
        NumColumn::Constant { .. } => "constant",
        NumColumn::I32(_) => "i32",
        NumColumn::I64(_) => "i64",
    }
}

fn sampled_current_mem_bytes(col: &NumColumn, sampled_len: usize) -> u64 {
    match col {
        NumColumn::Empty => 0,
        NumColumn::Constant { .. } => 12,
        NumColumn::I32(_) => 4_u64 * sampled_len as u64,
        NumColumn::I64(_) => 8_u64 * sampled_len as u64,
    }
}

fn ratio(numer: u64, denom: u64) -> f64 {
    if denom == 0 {
        0.0
    } else {
        numer as f64 / denom as f64
    }
}

fn update_aggregate(agg: &mut Aggregate, row: &ColumnRow) {
    agg.columns += 1;
    agg.values += row.sampled_len;
    agg.truncated_columns += usize::from(row.truncated);
    agg.i64_mem += row.i64_mem_bytes;
    agg.i32_mem += row.i32_mem_bytes;
    agg.current_mem += row.current_mem_bytes_est;
    agg.slp_i8_mem += row.slp_i8_mem_bytes;
    agg.slp_i16_mem += row.slp_i16_mem_bytes;
    agg.best_slp_mem += row.best_slp_mem_bytes;
    agg.i64_zstd += row.i64_zstd_bytes;
    agg.i32_zstd += row.i32_zstd_bytes;
    agg.slp_i8_zstd += row.slp_i8_zstd_bytes;
    agg.slp_i16_zstd += row.slp_i16_zstd_bytes;
    agg.best_slp_zstd += if row.best_slp == "slp_i8" {
        row.slp_i8_zstd_bytes
    } else {
        row.slp_i16_zstd_bytes
    };
    agg.accepted_columns += usize::from(row.accepted_vs_i64_gt_2);
    agg.beats_i32_columns += usize::from(row.beats_i32);
    agg.encode_secs += row.slp_i8_encode_secs + row.slp_i16_encode_secs;
}

fn write_summary(
    path: &PathBuf,
    args: &Args,
    aggregates: &BTreeMap<String, Aggregate>,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# SLP Timestamp/Duration Probe")?;
    writeln!(file)?;
    writeln!(
        file,
        "This is an offline probe. It does not change the PADOC artifact format."
    )?;
    writeln!(file)?;
    writeln!(file, "## Settings")?;
    writeln!(file)?;
    writeln!(file, "| Setting | Value |")?;
    writeln!(file, "|---|---:|")?;
    writeln!(file, "| min column length | {} |", args.min_len)?;
    writeln!(file, "| max columns per dataset | {} |", args.max_columns)?;
    writeln!(
        file,
        "| max sampled values per column | {} |",
        args.max_values
    )?;
    writeln!(file, "| max segment length | {} |", args.max_segment_len)?;
    writeln!(file, "| fixed-point Q | {} |", Q)?;
    writeln!(file, "| zstd level | {} |", args.zstd_level)?;
    writeln!(file)?;
    writeln!(file, "## Aggregate Results")?;
    writeln!(file)?;
    writeln!(
        file,
        "| Dataset | Columns | Values | Truncated cols | Best SLP mem / i64 mem | Best SLP vs i64 | Best SLP vs i32 | Accepted cols | Beat i32 cols | Best SLP zstd / i64 zstd | Encode secs |"
    )?;
    writeln!(
        file,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )?;
    for (dataset, agg) in aggregates {
        writeln!(
            file,
            "| `{}` | {} | {} | {} | {} / {} | {:.2}x | {:.2}x | {} | {} | {} / {} | {:.3} |",
            dataset,
            agg.columns,
            agg.values,
            agg.truncated_columns,
            agg.best_slp_mem,
            agg.i64_mem,
            ratio(agg.i64_mem, agg.best_slp_mem),
            ratio(agg.i32_mem, agg.best_slp_mem),
            agg.accepted_columns,
            agg.beats_i32_columns,
            agg.best_slp_zstd,
            agg.i64_zstd,
            agg.encode_secs
        )?;
    }
    writeln!(file)?;
    writeln!(
        file,
        "Acceptance criterion for the prototype is `best_slp_ratio_vs_i64 > 2.0`."
    )?;
    writeln!(file, "A separate `beats_i32` flag records whether the SLP result is also smaller than the current i32-style memory baseline.")?;
    Ok(())
}
