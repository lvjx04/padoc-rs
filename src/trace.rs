//! `Trace` (raw chrome-trace) and `CompressedTrace` (PADOC output).
//!
//! Public surface:
//!
//! * [`Trace::from_file`], [`Trace::from_dir`] — load chrome-trace JSON
//!   (single file or per-rank directory).  Uses `simd-json` for speed.
//! * [`Trace::write_chrome_json`] — round-trip back to chrome-trace JSON
//!   (used by analysis tasks that need raw events for comparison).
//! * [`CompressedTrace::write_to_path`] / [`CompressedTrace::read_from_path`]
//!   — zstd-wrapped msgpack persistence.

use crate::event::{ArgValue, Event, Phase, Template};
use crate::node::Node;
use crate::Result;
use ahash::{AHashMap, AHashSet};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One rank's events grouped by `(pid, tid, ph)`.
pub type StreamMap = IndexMap<i64, IndexMap<String, IndexMap<Phase, Vec<Event>>>>;

/// Top-level container — one entry per rank.
#[derive(Debug, Default)]
pub struct Trace {
    pub ranks: BTreeMap<String, StreamMap>,
    pub metadata: AHashMap<String, AHashMap<String, serde_json::Value>>,
    pub start_timestamp: AHashMap<String, i64>,
}

impl Trace {
    pub fn empty() -> Self {
        Trace::default()
    }

    pub fn rank_ids(&self) -> Vec<String> {
        self.ranks.keys().cloned().collect()
    }

    pub fn iter_streams(&self) -> impl Iterator<Item = (&str, i64, &str, Phase, &[Event])> {
        self.ranks.iter().flat_map(|(rank, processes)| {
            processes.iter().flat_map(move |(pid, threads)| {
                threads.iter().flat_map(move |(tid, phases)| {
                    phases
                        .iter()
                        .map(move |(ph, events)| (rank.as_str(), *pid, tid.as_str(), *ph, events.as_slice()))
                })
            })
        })
    }

    /// Total event count.  O(streams) (events are cheap to count).
    pub fn event_count(&self) -> usize {
        self.iter_streams().map(|(_, _, _, _, events)| events.len()).sum()
    }

    /// Load a single chrome-trace JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        // simd-json caps inputs at ~4 GiB. Fall back to serde_json for very
        // large traces (the unifolm rank dumps are 6 GiB each).
        let trace = if bytes.len() > 3 * 1024 * 1024 * 1024 {
            parse_chrome_trace_bytes_serde(&bytes, path)?
        } else {
            parse_chrome_trace_bytes(bytes, path)?
        };
        Ok(trace)
    }

    /// Load every `*.json` (and `*.json.gz`) in a directory, treating each
    /// file as a separate rank.  Sequential by default; switch to parallel
    /// at the bench-harness level if needed.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut combined = Trace::empty();
        let entries = list_trace_files(path);
        for entry in entries {
            let single = Self::from_file(&entry)?;
            combined.merge(single);
        }
        Ok(combined)
    }

    fn merge(&mut self, other: Trace) {
        for (rank, streams) in other.ranks {
            self.ranks.entry(rank).or_default().extend(streams);
        }
        for (rank, meta) in other.metadata {
            self.metadata.entry(rank).or_default().extend(meta);
        }
        for (rank, ts) in other.start_timestamp {
            self.start_timestamp.insert(rank, ts);
        }
    }
}

/// Return every chrome-trace file under `dir`, sorted.  Used both by
/// `Trace::from_dir` (which loads them all into a single in-memory trace)
/// and by the streaming bench runner (which loads one file at a time so
/// 1024-rank datasets don't exhaust RAM).
pub fn list_trace_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".json") || n.ends_with(".json.gz"))
                .unwrap_or(false)
        {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Parse a single chrome-trace JSON payload.  Implementation is
/// `simd-json`-based for big files.
fn parse_chrome_trace_bytes(mut bytes: Vec<u8>, source_path: &Path) -> Result<Trace> {
    use simd_json::OwnedValue as Value;
    use simd_json::prelude::*;

    let root: Value = simd_json::to_owned_value(&mut bytes)?;
    let root_obj = match root {
        Value::Object(obj) => obj,
        _ => return Err(crate::Error::InvalidTrace("expected JSON object".into())),
    };

    // distributedInfo.rank
    let rank = root_obj
        .get("distributedInfo")
        .and_then(|v| match v {
            Value::Object(o) => o.get("rank"),
            _ => None,
        })
        .and_then(|v| v.as_i64())
        .map(|n| n.to_string())
        .unwrap_or_else(|| {
            source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_owned)
                .unwrap_or_else(|| "0".to_string())
        });

    let trace_events = match root_obj.get("traceEvents") {
        Some(Value::Array(arr)) => arr,
        _ => return Err(crate::Error::InvalidTrace("missing traceEvents array".into())),
    };

    let mut streams: StreamMap = IndexMap::new();
    let mut metadata: AHashMap<String, serde_json::Value> = AHashMap::new();

    // Two-pass: first pass collects every event; we need to know the rank's
    // minimum ts so we can normalise (matches PerFlow-AI Python behaviour
    // where each rank is shifted to its own time origin).
    let mut staging: Vec<StagingEvent> = Vec::with_capacity(trace_events.len());

    for raw in trace_events {
        let obj = match raw {
            Value::Object(o) => o,
            _ => continue,
        };

        let ph = obj.get("ph").and_then(|v| v.as_str()).map(|s| s.as_bytes()[0]).unwrap_or(b'X');
        let phase = Phase(ph);

        let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or_default().to_string();

        let pid: i64 = obj.get("pid").and_then(|v| v.as_i64()).unwrap_or(0);
        let raw_tid: String = match obj.get("tid").cloned().unwrap_or(Value::Static(simd_json::StaticNode::Null)) {
            Value::String(s) => s.into(),
            Value::Static(simd_json::StaticNode::I64(n)) => n.to_string(),
            Value::Static(simd_json::StaticNode::U64(n)) => n.to_string(),
            _ => "0".to_string(),
        };

        if phase == Phase::METADATA {
            let value = simd_to_serde(obj.get("args").cloned().unwrap_or(Value::Static(simd_json::StaticNode::Null)));
            metadata.insert(name, value);
            continue;
        }

        // Normalise tid for HIP/ROCm and PyTorch GPU traces:
        //   * if `args.stream` is present, it is a GPU stream id   -> tid := "stream <id>"
        //   * else if cat == "gpu_user_annotation",                  -> tid := "stream <tid>"
        //   * else leave as-is.
        let stream_in_args = obj.get("args").and_then(|args| match args {
            Value::Object(o) => o.get("stream").and_then(|v| v.as_i64().map(|n| n.to_string()).or_else(|| v.as_str().map(str::to_owned))),
            _ => None,
        });
        let cat = obj.get("cat").and_then(|v| v.as_str()).map(str::to_owned);

        let tid = if let Some(stream) = stream_in_args {
            format!("stream {}", stream)
        } else if cat.as_deref() == Some("gpu_user_annotation") {
            format!("stream {}", raw_tid)
        } else {
            raw_tid
        };

        let event = build_event(obj, name, pid, tid.clone(), phase);
        staging.push(StagingEvent { event, pid, tid, phase });
    }

    // Per-rank ts origin: subtract the minimum ts so the column is small.
    let start_ts = staging.iter().map(|s| s.event.ts).min().unwrap_or(0);
    for s in staging.iter_mut() {
        s.event.ts -= start_ts;
    }
    for s in staging {
        streams
            .entry(s.pid)
            .or_default()
            .entry(s.tid)
            .or_default()
            .entry(s.phase)
            .or_default()
            .push(s.event);
    }

    let mut trace = Trace::empty();
    trace.ranks.insert(rank.clone(), streams);
    trace.start_timestamp.insert(rank.clone(), start_ts);
    let mut rank_meta: AHashMap<String, serde_json::Value> = AHashMap::new();
    rank_meta.extend(metadata);
    trace.metadata.insert(rank, rank_meta);
    Ok(trace)
}

struct StagingEvent {
    event: Event,
    pid: i64,
    tid: String,
    phase: Phase,
}

fn build_event(
    obj: &simd_json::owned::Object,
    name: String,
    pid: i64,
    tid: String,
    phase: Phase,
) -> Event {
    use simd_json::OwnedValue as Value;
    use simd_json::prelude::*;
    let ts = obj.get("ts").and_then(|v| v.as_i64()).unwrap_or(0);
    let dur = obj.get("dur").and_then(|v| v.as_i64());
    let cat = obj.get("cat").and_then(|v| v.as_str()).map(str::to_owned);
    let id = obj.get("id").and_then(|v| v.as_i64());
    let bp = obj.get("bp").and_then(|v| v.as_str()).map(str::to_owned);
    let s = obj.get("s").and_then(|v| v.as_str()).map(str::to_owned);

    let args = obj.get("args").cloned().and_then(|v| match v {
        Value::Object(o) => {
            let unboxed = *o;
            let mut map = ahash::AHashMap::with_capacity(unboxed.len());
            for (k, v) in unboxed {
                map.insert(k, simd_to_serde(v));
            }
            Some(map)
        }
        _ => None,
    });

    Event {
        name,
        ts,
        dur,
        cat,
        ph: phase,
        pid,
        tid,
        args,
        id,
        bp,
        s,
    }
}

/// `serde_json` based parser used for files that exceed `simd-json`'s 4 GiB
/// cap.  Slower (no SIMD) but no size limit.  Mirrors the simd-json path.
fn parse_chrome_trace_bytes_serde(bytes: &[u8], source_path: &Path) -> Result<Trace> {
    use serde_json::Value;

    let root: Value = serde_json::from_slice(bytes)?;
    let root_obj = root
        .as_object()
        .ok_or_else(|| crate::Error::InvalidTrace("expected JSON object".into()))?;

    let rank = root_obj
        .get("distributedInfo")
        .and_then(|v| v.as_object())
        .and_then(|o| o.get("rank"))
        .and_then(|v| v.as_i64())
        .map(|n| n.to_string())
        .unwrap_or_else(|| {
            source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_owned)
                .unwrap_or_else(|| "0".to_string())
        });

    let trace_events = root_obj
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .ok_or_else(|| crate::Error::InvalidTrace("missing traceEvents array".into()))?;

    let mut streams: StreamMap = IndexMap::new();
    let mut metadata: AHashMap<String, serde_json::Value> = AHashMap::new();
    let mut staging: Vec<StagingEvent> = Vec::with_capacity(trace_events.len());

    for raw in trace_events {
        let obj = match raw.as_object() {
            Some(o) => o,
            None => continue,
        };

        let ph = obj.get("ph").and_then(|v| v.as_str()).map(|s| s.as_bytes()[0]).unwrap_or(b'X');
        let phase = Phase(ph);
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let pid: i64 = obj.get("pid").and_then(|v| v.as_i64()).unwrap_or(0);
        let raw_tid: String = match obj.get("tid") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => "0".to_string(),
        };

        if phase == Phase::METADATA {
            let value = obj.get("args").cloned().unwrap_or(Value::Null);
            metadata.insert(name, value);
            continue;
        }

        let stream_in_args = obj.get("args").and_then(|args| args.as_object()).and_then(|a| {
            a.get("stream").and_then(|v| {
                v.as_i64()
                    .map(|n| n.to_string())
                    .or_else(|| v.as_str().map(str::to_owned))
            })
        });
        let cat = obj.get("cat").and_then(|v| v.as_str()).map(str::to_owned);

        let tid = if let Some(stream) = stream_in_args {
            format!("stream {}", stream)
        } else if cat.as_deref() == Some("gpu_user_annotation") {
            format!("stream {}", raw_tid)
        } else {
            raw_tid
        };

        let event = build_event_serde(obj, name, pid, tid.clone(), phase);
        staging.push(StagingEvent { event, pid, tid, phase });
    }

    let start_ts = staging.iter().map(|s| s.event.ts).min().unwrap_or(0);
    for s in staging.iter_mut() {
        s.event.ts -= start_ts;
    }
    for s in staging {
        streams
            .entry(s.pid)
            .or_default()
            .entry(s.tid)
            .or_default()
            .entry(s.phase)
            .or_default()
            .push(s.event);
    }

    let mut trace = Trace::empty();
    trace.ranks.insert(rank.clone(), streams);
    trace.start_timestamp.insert(rank.clone(), start_ts);
    let mut rank_meta: AHashMap<String, serde_json::Value> = AHashMap::new();
    rank_meta.extend(metadata);
    trace.metadata.insert(rank, rank_meta);
    Ok(trace)
}

fn build_event_serde(
    obj: &serde_json::Map<String, serde_json::Value>,
    name: String,
    pid: i64,
    tid: String,
    phase: Phase,
) -> Event {
    use serde_json::Value;
    let ts = obj.get("ts").and_then(|v| v.as_i64()).unwrap_or(0);
    let dur = obj.get("dur").and_then(|v| v.as_i64());
    let cat = obj.get("cat").and_then(|v| v.as_str()).map(str::to_owned);
    let id = obj.get("id").and_then(|v| v.as_i64());
    let bp = obj.get("bp").and_then(|v| v.as_str()).map(str::to_owned);
    let s = obj.get("s").and_then(|v| v.as_str()).map(str::to_owned);

    let args = obj.get("args").and_then(|v| match v {
        Value::Object(m) => {
            let mut map = ahash::AHashMap::with_capacity(m.len());
            for (k, v) in m {
                map.insert(k.clone(), v.clone());
            }
            Some(map)
        }
        _ => None,
    });

    Event { name, ts, dur, cat, ph: phase, pid, tid, args, id, bp, s }
}

fn simd_to_serde(v: simd_json::OwnedValue) -> serde_json::Value {
    use simd_json::OwnedValue as V;
    match v {
        V::Static(s) => match s {
            simd_json::StaticNode::Null => serde_json::Value::Null,
            simd_json::StaticNode::Bool(b) => serde_json::Value::Bool(b),
            simd_json::StaticNode::I64(n) => serde_json::Value::Number(n.into()),
            simd_json::StaticNode::U64(n) => serde_json::Value::Number(n.into()),
            simd_json::StaticNode::F64(n) => serde_json::Number::from_f64(n)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
        },
        V::String(s) => serde_json::Value::String(s.into()),
        V::Array(arr) => serde_json::Value::Array(arr.into_iter().map(simd_to_serde).collect()),
        V::Object(obj) => {
            let unboxed = *obj;
            let mut m = serde_json::Map::with_capacity(unboxed.len());
            for (k, v) in unboxed {
                m.insert(k, simd_to_serde(v));
            }
            serde_json::Value::Object(m)
        }
    }
}

// ---------------------------------------------------------------------------
// CompressedTrace
// ---------------------------------------------------------------------------

/// Output of `TemplateCompressor`.  Self-contained: can be serialised to
/// disk via [`CompressedTrace::write_to_path`] and reloaded for in-situ
/// analysis or full decompression.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompressedTrace {
    pub templates: Vec<Template>,
    /// `rank -> pid -> tid -> ph -> root_node`
    pub ranks: BTreeMap<String, BTreeMap<i64, BTreeMap<String, BTreeMap<u8, Node>>>>,
    pub metadata: AHashMap<String, AHashMap<String, serde_json::Value>>,
    pub start_timestamp: AHashMap<String, i64>,
}

impl CompressedTrace {
    /// Persist to disk: msgpack-encoded then zstd-compressed.
    pub fn write_to_path(&self, path: impl AsRef<Path>, zstd_level: i32) -> Result<()> {
        let mut buf = Vec::new();
        rmp_serde::encode::write_named(&mut buf, self)?;
        let compressed = zstd::stream::encode_all(&buf[..], zstd_level)?;
        std::fs::write(path, compressed)?;
        Ok(())
    }

    /// Read back what `write_to_path` produced.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Encode to a self-contained byte blob (zstd-wrapped msgpack).
    pub fn to_bytes(&self, zstd_level: i32) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        rmp_serde::encode::write_named(&mut buf, self)?;
        let compressed = zstd::stream::encode_all(&buf[..], zstd_level)?;
        Ok(compressed)
    }

    /// Decode the byte blob produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let raw = zstd::stream::decode_all(bytes)?;
        let trace: CompressedTrace = rmp_serde::from_slice(&raw)?;
        Ok(trace)
    }
}

// Used by metadata, args, etc. — guarantees deterministic key ordering for ranks.
#[allow(dead_code)]
fn ordered_keys(m: &AHashMap<String, AHashMap<String, ArgValue>>) -> Vec<String> {
    let mut keys: AHashSet<&str> = AHashSet::new();
    for (_, sub) in m {
        for k in sub.keys() {
            keys.insert(k);
        }
    }
    let mut out: Vec<String> = keys.into_iter().map(str::to_owned).collect();
    out.sort();
    out
}
