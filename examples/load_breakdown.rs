//! Measure how long the two implicit decompression steps take when loading a
//! .pdc file: (1) zstd decode of the whole blob, (2) msgpack decode of the
//! resulting bytes.

use padoc::compressor::TemplateCompressor;
use padoc::trace::{CompressedTrace, Trace};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let trace_path = std::env::args().nth(1).expect("usage: load_breakdown <chrome_trace.json>");
    println!("trace: {}", trace_path);
    let t0 = Instant::now();
    let trace = Trace::from_file(&trace_path)?;
    println!("  load chrome-trace JSON   : {:>8.3} ms", t0.elapsed().as_secs_f64() * 1e3);

    let t0 = Instant::now();
    let mut comp = TemplateCompressor::new();
    let compressed = comp.compress(&trace)?;
    let bytes = compressed.to_bytes(3)?;
    println!("  compress + encode .pdc   : {:>8.3} ms  ({} bytes)", t0.elapsed().as_secs_f64() * 1e3, bytes.len());

    // The actual question: how much of `from_bytes` is zstd vs msgpack?
    let t0 = Instant::now();
    let raw = zstd::stream::decode_all(&bytes[..])?;
    let zstd_secs = t0.elapsed().as_secs_f64();
    println!("  step ① zstd::decode_all  : {:>8.3} ms  ({} bytes -> {} bytes)", zstd_secs * 1e3, bytes.len(), raw.len());

    let t0 = Instant::now();
    let _decoded: CompressedTrace = rmp_serde::from_slice(&raw)?;
    let mp_secs = t0.elapsed().as_secs_f64();
    println!("  step ② rmp_serde::from_  : {:>8.3} ms", mp_secs * 1e3);

    println!("  total load (① + ②)      : {:>8.3} ms", (zstd_secs + mp_secs) * 1e3);
    Ok(())
}
