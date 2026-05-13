//! `raw_json` and `raw_msgpack` — no compression at all, for baseline ratios.

use crate::baselines::{BaselineCompressor, CompressArtifact};
use crate::trace::Trace;
use crate::Result;

#[derive(Default)]
pub struct RawJsonCompressor;

impl BaselineCompressor for RawJsonCompressor {
    fn name(&self) -> &str { "raw_json" }
    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();
        let bytes = serde_json::to_vec(&serialise_trace(trace))?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }
    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        crate::baselines::gzip::rebuild_trace_from_flat(&value)
    }
}

#[derive(Default)]
pub struct RawMsgpackCompressor;

impl BaselineCompressor for RawMsgpackCompressor {
    fn name(&self) -> &str { "raw_msgpack" }
    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();
        let bytes = rmp_serde::to_vec_named(&serialise_trace(trace))?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }
    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let value: serde_json::Value = rmp_serde::from_slice(bytes)?;
        crate::baselines::gzip::rebuild_trace_from_flat(&value)
    }
}

/// Flat JSON view of a Trace — every event in one list per rank.  Used by raw / gzip baselines.
pub(crate) fn serialise_trace(trace: &Trace) -> serde_json::Value {
    let mut ranks_obj = serde_json::Map::new();
    for (rank, streams) in &trace.ranks {
        let mut events_arr = Vec::new();
        for (pid, threads) in streams {
            for (tid, phases) in threads {
                for (ph, events) in phases {
                    for ev in events {
                        let mut row = serde_json::Map::new();
                        row.insert("name".into(), serde_json::Value::String(ev.name.clone()));
                        row.insert("ph".into(), serde_json::Value::String((ev.ph.0 as char).to_string()));
                        row.insert("ts".into(), serde_json::json!(ev.ts));
                        if let Some(d) = ev.dur {
                            row.insert("dur".into(), serde_json::json!(d));
                        }
                        row.insert("pid".into(), serde_json::json!(pid));
                        row.insert("tid".into(), serde_json::Value::String(tid.clone()));
                        if let Some(c) = &ev.cat { row.insert("cat".into(), serde_json::Value::String(c.clone())); }
                        if let Some(i) = ev.id { row.insert("id".into(), serde_json::json!(i)); }
                        if let Some(b) = &ev.bp { row.insert("bp".into(), serde_json::Value::String(b.clone())); }
                        if let Some(s) = &ev.s { row.insert("s".into(), serde_json::Value::String(s.clone())); }
                        if let Some(a) = &ev.args {
                            let mut args = serde_json::Map::new();
                            for (k, v) in a { args.insert(k.clone(), v.clone()); }
                            row.insert("args".into(), serde_json::Value::Object(args));
                        }
                        let _ = ph; // already encoded into row.
                        events_arr.push(serde_json::Value::Object(row));
                    }
                }
            }
        }
        ranks_obj.insert(rank.clone(), serde_json::Value::Array(events_arr));
    }
    serde_json::Value::Object(ranks_obj)
}

pub(crate) fn flatten_trace_for_baseline(trace: &Trace) -> serde_json::Value {
    serialise_trace(trace)
}
