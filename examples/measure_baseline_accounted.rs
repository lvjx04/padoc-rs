//! Measure accounted resident memory of ScalaTrace/TraceZip payloads
//! from existing compressed artifact files.
//!
//! Usage:
//!   cargo run --release --example measure_baseline_accounted -- <scalatrace.bin> <tracezip.bin>

use std::env;
use std::mem::size_of;
use std::path::Path;
use std::time::Instant;

/// Calculate accounted bytes for a ScalaTrace payload by deserializing
/// and measuring each field's heap allocation.
fn scalatrace_accounted(bytes: &[u8]) -> u64 {
    let raw = zstd::stream::decode_all(bytes).unwrap();
    let payload: ScalaTracePayload = rmp_serde::from_slice(&raw).unwrap();

    let mut total: u64 = 0;

    // dict strings
    for s in &payload.dict {
        total += s.capacity() as u64 + size_of::<String>() as u64;
    }
    total += payload.dict.capacity() as u64 * size_of::<String>() as u64;

    // types
    total += payload.types.capacity() as u64 * size_of::<ScalaTraceType>() as u64;
    for t in &payload.types {
        total += t.arg_key_ids.capacity() as u64 * 4;
    }

    // rsds
    total += payload.rsds.capacity() as u64 * size_of::<ScalaTraceRsd>() as u64;
    for r in &payload.rsds {
        total += r.pattern.capacity() as u64 * 4;
    }

    // streams (the bulk)
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
        // args: Vec<Vec<serde_json::Value>>
        total += s.args.capacity() as u64 * size_of::<Vec<serde_json::Value>>() as u64;
        for row in &s.args {
            total += row.capacity() as u64 * size_of::<serde_json::Value>() as u64;
        }
    }
    total += payload.streams.capacity() as u64 * size_of::<ScalaTraceStream>() as u64;

    total
}

/// Calculate accounted bytes for a TraceZip payload.
fn tracezip_accounted(bytes: &[u8]) -> u64 {
    let raw = zstd::stream::decode_all(bytes).unwrap();
    let payload: TraceZipPayload = rmp_serde::from_slice(&raw).unwrap();

    let mut total: u64 = 0;

    // dict strings
    for s in &payload.dict_strings {
        total += s.capacity() as u64 + size_of::<String>() as u64;
    }
    total += payload.dict_strings.capacity() as u64 * size_of::<String>() as u64;

    // schemas
    total += payload.schemas.capacity() as u64 * size_of::<TraceZipSchema>() as u64;
    for s in &payload.schemas {
        total += s.arg_key_ids.capacity() as u64 * 4;
    }

    // streams meta
    total += payload.streams.capacity() as u64 * size_of::<TraceZipStreamMeta>() as u64;

    // global buckets (the bulk)
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
        // arg_present: Vec<Vec<bool>>
        total += b.arg_present.capacity() as u64 * size_of::<Vec<bool>>() as u64;
        for row in &b.arg_present {
            total += row.capacity() as u64;
        }
        // arg_values: Vec<Vec<serde_json::Value>>
        total += b.arg_values.capacity() as u64 * size_of::<Vec<serde_json::Value>>() as u64;
        for row in &b.arg_values {
            total += row.capacity() as u64 * size_of::<serde_json::Value>() as u64;
        }
    }
    total += payload.global_buckets.capacity() as u64 * size_of::<TraceZipBucket>() as u64;

    total
}

// Mirror the serde structs from the baselines (just for size calculation)
#[derive(serde::Deserialize)]
struct ScalaTracePayload {
    dict: Vec<String>,
    types: Vec<ScalaTraceType>,
    rsds: Vec<ScalaTraceRsd>,
    streams: Vec<ScalaTraceStream>,
}

#[derive(serde::Deserialize)]
struct ScalaTraceType {
    #[allow(dead_code)]
    name_id: u32,
    #[allow(dead_code)]
    cat_id_plus1: u32,
    arg_key_ids: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct ScalaTraceRsd {
    pattern: Vec<u32>,
    #[allow(dead_code)]
    repeats: u32,
}

#[derive(serde::Deserialize)]
struct ScalaTraceStream {
    #[allow(dead_code)]
    rank_id: u32,
    #[allow(dead_code)]
    pid: i64,
    #[allow(dead_code)]
    tid_id: u32,
    #[allow(dead_code)]
    ph: u8,
    rsd_ids: Vec<u32>,
    payload_name_ids: Vec<u32>,
    ts: Vec<i64>,
    dur_present: Vec<bool>,
    dur: Vec<i64>,
    id_present: Vec<bool>,
    ids: Vec<i64>,
    bp_dict_id_plus1: Vec<u32>,
    s_dict_id_plus1: Vec<u32>,
    args: Vec<Vec<serde_json::Value>>,
}

#[derive(serde::Deserialize)]
struct TraceZipPayload {
    dict_strings: Vec<String>,
    schemas: Vec<TraceZipSchema>,
    streams: Vec<TraceZipStreamMeta>,
    global_buckets: Vec<TraceZipBucket>,
}

#[derive(serde::Deserialize)]
struct TraceZipSchema {
    #[allow(dead_code)]
    name_dict_id: u32,
    arg_key_ids: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct TraceZipStreamMeta {
    #[allow(dead_code)]
    rank_dict_id: u32,
    #[allow(dead_code)]
    pid: i64,
    #[allow(dead_code)]
    tid_dict_id: u32,
    #[allow(dead_code)]
    ph: u8,
    #[allow(dead_code)]
    time_base: i64,
}

#[derive(serde::Deserialize)]
struct TraceZipBucket {
    #[allow(dead_code)]
    schema_id: u32,
    stream_ids: Vec<u32>,
    ts_offsets: Vec<i64>,
    dur_present: Vec<bool>,
    dur: Vec<i64>,
    id_present: Vec<bool>,
    ids: Vec<i64>,
    cat_dict_id_plus1: Vec<u32>,
    bp_dict_id_plus1: Vec<u32>,
    s_dict_id_plus1: Vec<u32>,
    arg_present: Vec<Vec<bool>>,
    arg_values: Vec<Vec<serde_json::Value>>,
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: measure_baseline_accounted <artifact_file> [artifact_file2...]");
        eprintln!("  File name must contain 'scalatrace' or 'tracezip' to select the decoder.");
        eprintln!("  Example: measure_baseline_accounted /tmp/bench_insitu_scalatrace.bin /tmp/bench_insitu_tracezip.bin");
        std::process::exit(1);
    }

    println!("file\tcompressor\tartifact_bytes\taccounted_bytes\taccounted_gib");

    for path_str in &args {
        let path = Path::new(path_str);
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

        let compressor_name = if filename.contains("scalatrace") {
            "scalatrace"
        } else if filename.contains("tracezip") {
            "tracezip"
        } else {
            eprintln!("  SKIP {}: cannot determine compressor from filename", path_str);
            continue;
        };

        eprintln!("Loading artifact: {} ({})", path_str, compressor_name);
        let start = Instant::now();
        let bytes = std::fs::read(path).unwrap();
        let artifact_bytes = bytes.len() as u64;
        eprintln!("  {} bytes read in {:.3}s", artifact_bytes, start.elapsed().as_secs_f64());

        let accounted = match compressor_name {
            "scalatrace" => scalatrace_accounted(&bytes),
            "tracezip" => tracezip_accounted(&bytes),
            _ => 0,
        };

        let gib = accounted as f64 / 1024.0 / 1024.0 / 1024.0;
        println!("{}\t{}\t{}\t{}\t{:.3}", filename, compressor_name, artifact_bytes, accounted, gib);
        eprintln!("  accounted={} bytes ({:.3} GiB)", accounted, gib);
    }
}
