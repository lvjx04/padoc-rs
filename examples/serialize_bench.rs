//! Compare serialization strategies on an existing CompressedTrace
//! artifact.  Used to pick a winner between
//!
//!   buffered  : rmp_serde::to_vec_named + zstd::encode_all
//!   streamed  : rmp_serde::write_named -> zstd::Encoder (current)
//!   buf+stream: rmp_serde::write_named -> BufWriter -> zstd::Encoder
//!
//! Run as:
//!
//!   cargo run --release --example serialize_bench -- <ARTIFACT.zst>
//!
//! Prints elapsed times + final compressed size for each strategy.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use padoc::trace::CompressedTrace;

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: serialize_bench <ARTIFACT.zst>"))?
        .into();

    eprintln!("loading {}", path.display());
    let load_start = Instant::now();
    let compressed = CompressedTrace::read_from_path(&path)?;
    eprintln!(
        "loaded in {:.3} s ({} templates, {} ranks)",
        load_start.elapsed().as_secs_f64(),
        compressed.templates.len(),
        compressed.ranks.len(),
    );

    let level = 3;

    // (1) buffered: msgpack -> Vec, then zstd::encode_all
    {
        let t0 = Instant::now();
        let payload = rmp_serde::to_vec_named(&compressed)?;
        let t_msgpack = t0.elapsed().as_secs_f64();
        let payload_len = payload.len();
        let t1 = Instant::now();
        let comp = zstd::stream::encode_all(&payload[..], level)?;
        let t_zstd = t1.elapsed().as_secs_f64();
        eprintln!(
            "[buffered]   msgpack={:.3}s ({:>6.2} GiB)  zstd={:.3}s ({:>6.2} MiB)  total={:.3}s",
            t_msgpack,
            payload_len as f64 / 1024.0 / 1024.0 / 1024.0,
            t_zstd,
            comp.len() as f64 / 1024.0 / 1024.0,
            t_msgpack + t_zstd,
        );
        // drop payload to free memory before next test
        drop(payload);
        drop(comp);
    }

    // (2) streamed (current): rmp_serde::write_named -> zstd::Encoder<Vec<u8>>
    {
        let t0 = Instant::now();
        let out: Vec<u8> = Vec::with_capacity(8 * 1024 * 1024);
        let mut encoder = zstd::stream::Encoder::new(out, level)?;
        rmp_serde::encode::write_named(&mut encoder, &compressed)?;
        let comp = encoder.finish()?;
        let t = t0.elapsed().as_secs_f64();
        eprintln!(
            "[streamed]   total={:.3}s  ({:>6.2} MiB)",
            t,
            comp.len() as f64 / 1024.0 / 1024.0
        );
        drop(comp);
    }

    // (3) buf+stream: BufWriter wrap over zstd::Encoder (1 MiB internal buf)
    {
        let t0 = Instant::now();
        let out: Vec<u8> = Vec::with_capacity(8 * 1024 * 1024);
        let encoder = zstd::stream::Encoder::new(out, level)?;
        let mut buf_enc = std::io::BufWriter::with_capacity(1 << 20, encoder);
        rmp_serde::encode::write_named(&mut buf_enc, &compressed)?;
        buf_enc.flush()?;
        let encoder = buf_enc
            .into_inner()
            .map_err(|e| anyhow!("buf flush: {e}"))?;
        let comp = encoder.finish()?;
        let t = t0.elapsed().as_secs_f64();
        eprintln!(
            "[bufstream]  total={:.3}s  ({:>6.2} MiB)",
            t,
            comp.len() as f64 / 1024.0 / 1024.0
        );
        drop(comp);
    }

    Ok(())
}
