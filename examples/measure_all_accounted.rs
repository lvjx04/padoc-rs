//! Measure accounted for all compressors on a single dataset.
//! Compresses sequentially to avoid OOM: compress one → save → drop → next.
//!
//! Usage:
//!   cargo run --release --example measure_all_accounted -- <trace_path_or_dir>

use padoc::baselines::{self, BaselineCompressor};
use padoc::trace::{CompressedTrace, Trace};
use std::env;
use std::mem::size_of;
use std::path::Path;
use std::time::Instant;

fn load_trace(path: &Path) -> Trace {
    if path.is_dir() { Trace::from_dir(path).unwrap() } else { Trace::from_file(path).unwrap() }
}

fn main() {
    let path_str = env::args().nth(1).expect("usage: measure_all_accounted <trace_path>");
    let path = Path::new(&path_str);
    let dataset = path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown");

    println!("dataset\tcompressor\tartifact_bytes\taccounted_bytes\taccounted_gib");

    // ScalaTrace
    {
        eprintln!("[{}] Loading trace for scalatrace...", dataset);
        let trace = load_trace(path);
        let registry = baselines::registry();
        let c = registry.iter().find(|c| c.name() == "scalatrace").unwrap();
        eprintln!("[{}] Compressing with scalatrace...", dataset);
        let artifact = c.compress(&trace).unwrap();
        let artifact_bytes = artifact.bytes.len() as u64;
        drop(trace);
        eprintln!("[{}] Measuring scalatrace accounted...", dataset);
        let accounted = scalatrace_accounted(&artifact.bytes);
        let gib = accounted as f64 / 1024.0 / 1024.0 / 1024.0;
        println!("{}\tscalatrace\t{}\t{}\t{:.3}", dataset, artifact_bytes, accounted, gib);
        eprintln!("[{}] scalatrace: artifact={} accounted={:.3} GiB", dataset, artifact_bytes, gib);
    }

    // TraceZip
    {
        eprintln!("[{}] Loading trace for tracezip...", dataset);
        let trace = load_trace(path);
        let registry = baselines::registry();
        let c = registry.iter().find(|c| c.name() == "tracezip").unwrap();
        eprintln!("[{}] Compressing with tracezip...", dataset);
        let artifact = c.compress(&trace).unwrap();
        let artifact_bytes = artifact.bytes.len() as u64;
        drop(trace);
        eprintln!("[{}] Measuring tracezip accounted...", dataset);
        let accounted = tracezip_accounted(&artifact.bytes);
        let gib = accounted as f64 / 1024.0 / 1024.0 / 1024.0;
        println!("{}\ttracezip\t{}\t{}\t{:.3}", dataset, artifact_bytes, accounted, gib);
        eprintln!("[{}] tracezip: artifact={} accounted={:.3} GiB", dataset, artifact_bytes, gib);
    }

    // PADOC (from existing artifact if available, otherwise compress)
    {
        eprintln!("[{}] Loading trace for padoc...", dataset);
        let trace = load_trace(path);
        let registry = baselines::registry();
        let c = registry.iter().find(|c| c.name() == "padoc").unwrap();
        eprintln!("[{}] Compressing with padoc...", dataset);
        let artifact = c.compress(&trace).unwrap();
        let artifact_bytes = artifact.bytes.len() as u64;
        drop(trace);
        eprintln!("[{}] Measuring padoc accounted...", dataset);
        let ct = CompressedTrace::from_bytes(&artifact.bytes).unwrap();
        let accounted = padoc_accounted(&ct);
        let gib = accounted as f64 / 1024.0 / 1024.0 / 1024.0;
        println!("{}\tpadoc\t{}\t{}\t{:.3}", dataset, artifact_bytes, accounted, gib);
        eprintln!("[{}] padoc: artifact={} accounted={:.3} GiB", dataset, artifact_bytes, gib);
    }
}

fn padoc_accounted(ct: &CompressedTrace) -> u64 {
    use padoc::event::{ArgColumn, DigitColumn, NameNums, NumColumn, StringColumn, Template};

    let mut total: u64 = 0;
    for tmpl in &ct.templates {
        match tmpl {
            Template::Cpu(t) => {
                total += num_bytes(&t.ts) as u64;
                total += num_bytes(&t.dur) as u64;
                total += num_bytes(&t.id) as u64;
                total += args_total(&t.args_columns) as u64;
                total += name_nums_total(&t.name_nums) as u64;
            }
            Template::Gpu(t) => {
                total += num_bytes(&t.ts) as u64;
                total += num_bytes(&t.dur) as u64;
                total += num_bytes(&t.pid) as u64;
                total += str_col_bytes(&t.stream_tid) as u64;
                total += args_total(&t.args_columns) as u64;
                total += name_nums_total(&t.name_nums) as u64;
            }
        }
    }
    // arena
    if let Some(arenas) = ct.arenas.as_ref() {
        for processes in arenas.values() {
            for threads in processes.values() {
                for phases in threads.values() {
                    for (arena, _) in phases.values() {
                        total += arena.heap_bytes() as u64;
                    }
                }
            }
        }
    }
    total
}

fn num_bytes(col: &padoc::event::NumColumn) -> usize {
    match col {
        padoc::event::NumColumn::Empty => 0,
        padoc::event::NumColumn::Constant { .. } => 12,
        padoc::event::NumColumn::I32(v) => v.capacity() * 4,
        padoc::event::NumColumn::I64(v) => v.capacity() * 8,
        padoc::event::NumColumn::Slp(slp) => slp.heap_bytes(),
    }
}

fn str_col_bytes(col: &padoc::event::StringColumn) -> usize {
    match col {
        padoc::event::StringColumn::Empty => 0,
        padoc::event::StringColumn::Constant { value, .. } => size_of::<String>() + value.capacity(),
        padoc::event::StringColumn::PerInstance(v) => {
            v.capacity() * size_of::<String>() + v.iter().map(|s| s.capacity()).sum::<usize>()
        }
    }
}

fn args_total(cols: &[padoc::event::ArgColumn]) -> usize {
    cols.iter().map(|col| match col {
        padoc::event::ArgColumn::Constant(v) => v.to_string().len(),
        padoc::event::ArgColumn::I32(v) => v.capacity() * 4,
        padoc::event::ArgColumn::I64(v) => v.capacity() * 8,
        padoc::event::ArgColumn::F64(v) => v.capacity() * 8,
        padoc::event::ArgColumn::Bool(v) => v.capacity(),
        padoc::event::ArgColumn::Str(v) => v.capacity() * size_of::<String>() + v.iter().map(|s| s.capacity()).sum::<usize>(),
        padoc::event::ArgColumn::StrDict { dict, ids } => ids.capacity() * 4 + dict.capacity() * size_of::<String>() + dict.iter().map(|s| s.capacity()).sum::<usize>(),
        padoc::event::ArgColumn::PerInstance(v) => v.capacity() * size_of::<serde_json::Value>() + v.iter().map(|x| x.to_string().len()).sum::<usize>(),
        padoc::event::ArgColumn::SlpI32(slp) => slp.heap_bytes(),
    }).sum()
}

fn name_nums_total(nums: &padoc::event::NameNums) -> usize {
    match nums {
        padoc::event::NameNums::Empty => 0,
        padoc::event::NameNums::Rows(rows) => {
            rows.capacity() * size_of::<Vec<String>>()
                + rows.iter().map(|r| r.capacity() * size_of::<String>() + r.iter().map(|s| s.capacity()).sum::<usize>()).sum::<usize>()
        }
        padoc::event::NameNums::Columnar(cols) => {
            cols.capacity() * size_of::<padoc::event::DigitColumn>()
                + cols.iter().map(|c| match c {
                    padoc::event::DigitColumn::Constant(v) => v.capacity(),
                    padoc::event::DigitColumn::I32 { values, .. } => values.capacity() * 4,
                    padoc::event::DigitColumn::I64 { values, .. } => values.capacity() * 8,
                    padoc::event::DigitColumn::Strings(v) => v.capacity() * size_of::<String>() + v.iter().map(|s| s.capacity()).sum::<usize>(),
                }).sum::<usize>()
        }
    }
}

// --- ScalaTrace/TraceZip payload structs (for deserialization) ---

fn scalatrace_accounted(bytes: &[u8]) -> u64 {
    let raw = zstd::stream::decode_all(bytes).unwrap();
    let payload: ScalaTracePayload = rmp_serde::from_slice(&raw).unwrap();
    let mut total: u64 = 0;
    for s in &payload.dict { total += s.capacity() as u64 + size_of::<String>() as u64; }
    total += payload.dict.capacity() as u64 * size_of::<String>() as u64;
    for t in &payload.types { total += t.arg_key_ids.capacity() as u64 * 4; }
    total += payload.types.capacity() as u64 * 24;
    for r in &payload.rsds { total += r.pattern.capacity() as u64 * 4; }
    total += payload.rsds.capacity() as u64 * 32;
    for s in &payload.streams {
        total += s.rsd_ids.capacity() as u64 * 4;
        total += s.payload_name_ids.capacity() as u64 * 4;
        total += s.ts.capacity() as u64 * 8;
        total += s.dur_present.capacity() as u64;
        total += s.dur.capacity() as u64 * 8;
        total += s.id_present.capacity() as u64;
        total += s.ids.capacity() as u64 * 8;
        total += s.bp_dict_id_plus1.capacity() as u64 * 4;
        total += s.s_dict_id_plus1.capacity() as u64 * 4;
        total += s.args.capacity() as u64 * size_of::<Vec<serde_json::Value>>() as u64;
        for row in &s.args { total += row.capacity() as u64 * size_of::<serde_json::Value>() as u64; }
    }
    total += payload.streams.capacity() as u64 * 200; // approx StreamPayload size
    total
}

fn tracezip_accounted(bytes: &[u8]) -> u64 {
    let raw = zstd::stream::decode_all(bytes).unwrap();
    let payload: TraceZipPayload = rmp_serde::from_slice(&raw).unwrap();
    let mut total: u64 = 0;
    for s in &payload.dict_strings { total += s.capacity() as u64 + size_of::<String>() as u64; }
    total += payload.dict_strings.capacity() as u64 * size_of::<String>() as u64;
    for s in &payload.schemas { total += s.arg_key_ids.capacity() as u64 * 4; }
    total += payload.schemas.capacity() as u64 * 32;
    total += payload.streams.capacity() as u64 * 32;
    for b in &payload.global_buckets {
        total += b.stream_ids.capacity() as u64 * 4;
        total += b.ts_offsets.capacity() as u64 * 8;
        total += b.dur_present.capacity() as u64;
        total += b.dur.capacity() as u64 * 8;
        total += b.id_present.capacity() as u64;
        total += b.ids.capacity() as u64 * 8;
        total += b.cat_dict_id_plus1.capacity() as u64 * 4;
        total += b.bp_dict_id_plus1.capacity() as u64 * 4;
        total += b.s_dict_id_plus1.capacity() as u64 * 4;
        total += b.arg_present.capacity() as u64 * size_of::<Vec<bool>>() as u64;
        for row in &b.arg_present { total += row.capacity() as u64; }
        total += b.arg_values.capacity() as u64 * size_of::<Vec<serde_json::Value>>() as u64;
        for row in &b.arg_values { total += row.capacity() as u64 * size_of::<serde_json::Value>() as u64; }
    }
    total += payload.global_buckets.capacity() as u64 * 200;
    total
}

#[derive(serde::Deserialize)] struct ScalaTracePayload { dict: Vec<String>, types: Vec<StType>, rsds: Vec<StRsd>, streams: Vec<StStream> }
#[derive(serde::Deserialize)] struct StType { #[allow(dead_code)] name_id: u32, #[allow(dead_code)] cat_id_plus1: u32, arg_key_ids: Vec<u32> }
#[derive(serde::Deserialize)] struct StRsd { pattern: Vec<u32>, #[allow(dead_code)] repeats: u32 }
#[derive(serde::Deserialize)] struct StStream { #[allow(dead_code)] rank_id: u32, #[allow(dead_code)] pid: i64, #[allow(dead_code)] tid_id: u32, #[allow(dead_code)] ph: u8, rsd_ids: Vec<u32>, payload_name_ids: Vec<u32>, ts: Vec<i64>, dur_present: Vec<bool>, dur: Vec<i64>, id_present: Vec<bool>, ids: Vec<i64>, bp_dict_id_plus1: Vec<u32>, s_dict_id_plus1: Vec<u32>, args: Vec<Vec<serde_json::Value>> }
#[derive(serde::Deserialize)] struct TraceZipPayload { dict_strings: Vec<String>, schemas: Vec<TzSchema>, streams: Vec<TzStream>, global_buckets: Vec<TzBucket> }
#[derive(serde::Deserialize)] struct TzSchema { #[allow(dead_code)] name_dict_id: u32, arg_key_ids: Vec<u32> }
#[derive(serde::Deserialize)] struct TzStream { #[allow(dead_code)] rank_dict_id: u32, #[allow(dead_code)] pid: i64, #[allow(dead_code)] tid_dict_id: u32, #[allow(dead_code)] ph: u8, #[allow(dead_code)] time_base: i64 }
#[derive(serde::Deserialize)] struct TzBucket { #[allow(dead_code)] schema_id: u32, stream_ids: Vec<u32>, ts_offsets: Vec<i64>, dur_present: Vec<bool>, dur: Vec<i64>, id_present: Vec<bool>, ids: Vec<i64>, cat_dict_id_plus1: Vec<u32>, bp_dict_id_plus1: Vec<u32>, s_dict_id_plus1: Vec<u32>, arg_present: Vec<Vec<bool>>, arg_values: Vec<Vec<serde_json::Value>> }
