//! `gzip_json` and `gzip_msgpack` — baselines that gzip a faithful
//! per-event representation of the trace.  Both are *lossless* with
//! respect to round-tripping through `Event`.
//!
//! `gzip_json` serializes the same flattened structure that
//! `serialise_trace` emits (one events array per rank), then gzips it.
//! Decompression reads the JSON back, parses each event row, and
//! re-builds the [`Trace`] tree.  Lossless wrt all of `Event`'s
//! fields including `cat`, `bp`, `s`, optional `dur`/`id`, and `args`.
//!
//! `gzip_msgpack` does the same with msgpack-named encoding instead
//! of JSON.

use crate::baselines::raw::flatten_trace_for_baseline;
use crate::baselines::{BaselineCompressor, CompressArtifact};
use crate::event::{Event, Phase};
use crate::trace::Trace;
use crate::Result;
use ahash::AHashMap;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde_json::Value;
use std::io::{Read, Write};

#[derive(Default)]
pub struct GzipJsonCompressor;

impl BaselineCompressor for GzipJsonCompressor {
    fn name(&self) -> &str { "gzip_json" }
    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();
        let raw = serde_json::to_vec(&flatten_trace_for_baseline(trace))?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw)?;
        let bytes = encoder.finish()?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }
    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let mut dec = GzDecoder::new(bytes);
        let mut raw = Vec::new();
        dec.read_to_end(&mut raw)?;
        let value: Value = serde_json::from_slice(&raw)?;
        rebuild_trace_from_flat(&value)
    }
}

#[derive(Default)]
pub struct GzipMsgpackCompressor;

impl BaselineCompressor for GzipMsgpackCompressor {
    fn name(&self) -> &str { "gzip_msgpack" }
    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();
        let raw = rmp_serde::to_vec_named(&flatten_trace_for_baseline(trace))?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw)?;
        let bytes = encoder.finish()?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }
    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let mut dec = GzDecoder::new(bytes);
        let mut raw = Vec::new();
        dec.read_to_end(&mut raw)?;
        let value: Value = rmp_serde::from_slice(&raw)?;
        rebuild_trace_from_flat(&value)
    }
}

/// Inverse of `serialise_trace`: take the `{rank: [events]}` JSON tree
/// and rebuild a [`Trace`] by re-routing each event into its
/// (rank, pid, tid, ph) cell.
pub(crate) fn rebuild_trace_from_flat(value: &Value) -> Result<Trace> {
    let mut trace = Trace::empty();
    let obj = value.as_object().ok_or_else(|| crate::Error::Other(
        "gzip baseline: expected top-level object".to_string()
    ))?;
    for (rank, events_val) in obj {
        let events_arr = events_val.as_array().ok_or_else(|| crate::Error::Other(
            format!("gzip baseline: rank {} did not contain an events array", rank)
        ))?;
        for ev_val in events_arr {
            let row = ev_val.as_object().ok_or_else(|| crate::Error::Other(
                "gzip baseline: event was not an object".to_string()
            ))?;
            let event = row_to_event(row);
            let pid = row.get("pid").and_then(|v| v.as_i64()).unwrap_or(0);
            let tid = row.get("tid").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let ph = row.get("ph").and_then(|v| v.as_str())
                .and_then(|s| s.chars().next())
                .map(|c| Phase(c as u8))
                .unwrap_or(Phase(b'X'));
            trace.ranks
                .entry(rank.clone()).or_default()
                .entry(pid).or_default()
                .entry(tid).or_default()
                .entry(ph).or_default()
                .push(event);
        }
    }
    Ok(trace)
}

fn row_to_event(row: &serde_json::Map<String, Value>) -> Event {
    let name = row.get("name").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let ts = row.get("ts").and_then(|v| v.as_i64()).unwrap_or(0);
    let dur = row.get("dur").and_then(|v| v.as_i64());
    let cat = row.get("cat").and_then(|v| v.as_str()).map(|s| s.to_string());
    let ph = row.get("ph").and_then(|v| v.as_str())
        .and_then(|s| s.chars().next())
        .map(|c| Phase(c as u8))
        .unwrap_or(Phase(b'X'));
    let pid = row.get("pid").and_then(|v| v.as_i64()).unwrap_or(0);
    let tid = row.get("tid").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let id = row.get("id").and_then(|v| v.as_i64());
    let bp = row.get("bp").and_then(|v| v.as_str()).map(|s| s.to_string());
    let s = row.get("s").and_then(|v| v.as_str()).map(|s| s.to_string());
    let args = row.get("args").and_then(|v| v.as_object()).map(|o| {
        let mut map: AHashMap<String, Value> = AHashMap::new();
        for (k, v) in o {
            map.insert(k.clone(), v.clone());
        }
        map
    });
    Event {
        name, ts, dur, cat, ph, pid, tid,
        args,
        id, bp, s,
    }
}
