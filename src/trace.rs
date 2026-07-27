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

use crate::event::{Event, Phase, Template};
use crate::node::Node;
use crate::Result;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One rank's events grouped by `(pid, tid, ph)`.
pub type StreamMap = IndexMap<i64, IndexMap<String, IndexMap<Phase, Vec<Event>>>>;

/// Chrome trace metadata record.
///
/// Metadata remains outside the event streams, but retains its original
/// process/thread coordinates and input order for faithful reconstruction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetadataEvent {
    pub name: String,
    pub pid: i64,
    pub tid: String,
    pub args: Option<serde_json::Value>,
}

/// Top-level container — one entry per rank.
#[derive(Debug, Default)]
pub struct Trace {
    pub ranks: BTreeMap<String, StreamMap>,
    pub metadata: BTreeMap<String, Vec<MetadataEvent>>,
    pub start_timestamp: BTreeMap<String, i64>,
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
                    phases.iter().map(move |(ph, events)| {
                        (rank.as_str(), *pid, tid.as_str(), *ph, events.as_slice())
                    })
                })
            })
        })
    }

    /// Total event count.  O(streams) (events are cheap to count).
    pub fn event_count(&self) -> usize {
        self.iter_streams()
            .map(|(_, _, _, _, events)| events.len())
            .sum()
    }

    /// Load a single chrome-trace JSON file.
    ///
    /// Two ingest paths:
    ///
    /// * **Streaming** (default) — `serde_json::Deserializer::from_reader` driven
    ///   by a hand-rolled `Visitor` that pulls one event at a time off the
    ///   `traceEvents` array.  Peak memory tracks the **decoded** trace size
    ///   (~0.5–1× the JSON file), not the JSON-tree expansion (~5–10×).
    ///   Used for everything ≥ a small file threshold so 1024-rank +
    ///   parallel-worker setups don't blow up RAM.
    /// * **simd-json** — fast SIMD parser, but builds a full owned tree
    ///   first.  Reserved for tiny files (<32 MiB) where the absolute
    ///   parsing speed wins and the tree is too small to matter.
    ///
    /// Both paths produce identical [`Trace`] structures.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if is_gzip_path(path) {
            return crate::trace_stream::parse_chrome_trace_gzip(path);
        }
        let metadata = std::fs::metadata(path)?;
        let size = metadata.len();
        // Files this small load + parse faster via simd-json's full-tree
        // path; memory is a non-issue at this size.
        const SIMD_FAST_PATH_LIMIT: u64 = 32 * 1024 * 1024;
        if size <= SIMD_FAST_PATH_LIMIT {
            let bytes = std::fs::read(path)?;
            return parse_chrome_trace_bytes(bytes, path);
        }
        crate::trace_stream::parse_chrome_trace_stream(path)
    }

    /// Force the legacy full-tree parser.  Useful for benchmarking and as a
    /// fallback when the streaming path hits an unexpected JSON shape.
    pub fn from_file_full_tree(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = read_trace_bytes(path)?;
        if bytes.len() > 3 * 1024 * 1024 * 1024 {
            parse_chrome_trace_bytes_serde(&bytes, path)
        } else {
            parse_chrome_trace_bytes(bytes, path)
        }
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

    /// Write a single-rank trace as Chrome trace JSON.
    ///
    /// PADOC stores timestamps relative to each rank's first event. This
    /// method restores the original timestamp origin while streaming events
    /// to disk, so the complete JSON document is never built in memory.
    pub fn write_chrome_json(&self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;

        if self.ranks.len() != 1 {
            return Err(crate::Error::InvalidTrace(format!(
                "Chrome JSON output requires exactly one rank, found {}",
                self.ranks.len()
            )));
        }

        let (rank, _) = self
            .ranks
            .first_key_value()
            .ok_or_else(|| crate::Error::InvalidTrace("trace has no ranks".into()))?;
        let start_ts = self.start_timestamp.get(rank).copied().unwrap_or(0);
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

        writer.write_all(b"{\"distributedInfo\":{\"rank\":")?;
        if let Ok(numeric_rank) = rank.parse::<i64>() {
            serde_json::to_writer(&mut writer, &numeric_rank)?;
        } else {
            serde_json::to_writer(&mut writer, rank)?;
        }
        writer.write_all(b"},\"traceEvents\":[")?;

        let mut first = true;
        if let Some(metadata) = self.metadata.get(rank) {
            for metadata_event in metadata {
                write_json_separator(&mut writer, &mut first)?;
                let mut event = serde_json::Map::new();
                event.insert(
                    "name".into(),
                    serde_json::Value::String(metadata_event.name.clone()),
                );
                event.insert("ph".into(), serde_json::Value::String("M".into()));
                event.insert(
                    "pid".into(),
                    serde_json::Value::Number(metadata_event.pid.into()),
                );
                event.insert(
                    "tid".into(),
                    serde_json::Value::String(metadata_event.tid.clone()),
                );
                if let Some(args) = &metadata_event.args {
                    event.insert("args".into(), args.clone());
                }
                serde_json::to_writer(&mut writer, &event)?;
            }
        }

        for (_, _, _, _, events) in self.iter_streams() {
            for event in events {
                write_json_separator(&mut writer, &mut first)?;
                let value = chrome_event_value(event, start_ts)?;
                serde_json::to_writer(&mut writer, &value)?;
            }
        }

        writer.write_all(b"]}\n")?;
        writer.flush()?;
        Ok(())
    }
}

fn is_gzip_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".json.gz"))
}

fn trace_source_stem(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let stem = name
        .strip_suffix(".json.gz")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(name);
    Some(stem.to_owned())
}

fn read_trace_bytes(path: &Path) -> Result<Vec<u8>> {
    if !is_gzip_path(path) {
        return Ok(std::fs::read(path)?);
    }

    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn write_json_separator(writer: &mut impl std::io::Write, first: &mut bool) -> Result<()> {
    if *first {
        *first = false;
    } else {
        writer.write_all(b",")?;
    }
    Ok(())
}

fn chrome_event_value(event: &Event, start_ts: i64) -> Result<serde_json::Value> {
    let absolute_ts = event.ts.checked_add(start_ts).ok_or_else(|| {
        crate::Error::InvalidTrace(format!(
            "timestamp overflow while restoring {} + {}",
            event.ts, start_ts
        ))
    })?;

    let mut object = serde_json::Map::new();
    object.insert("name".into(), serde_json::Value::String(event.name.clone()));
    object.insert("ts".into(), serde_json::Value::Number(absolute_ts.into()));
    if let Some(dur) = event.dur {
        object.insert("dur".into(), serde_json::Value::Number(dur.into()));
    }
    if let Some(cat) = &event.cat {
        object.insert("cat".into(), serde_json::Value::String(cat.clone()));
    }
    object.insert(
        "ph".into(),
        serde_json::Value::String(event.ph.as_char().to_string()),
    );
    object.insert("pid".into(), serde_json::Value::Number(event.pid.into()));
    object.insert("tid".into(), serde_json::Value::String(event.tid.clone()));
    if let Some(args) = &event.args {
        object.insert("args".into(), serde_json::to_value(args)?);
    }
    if let Some(id) = event.id {
        object.insert("id".into(), serde_json::Value::Number(id.into()));
    }
    if let Some(bp) = &event.bp {
        object.insert("bp".into(), serde_json::Value::String(bp.clone()));
    }
    if let Some(scope) = &event.s {
        object.insert("s".into(), serde_json::Value::String(scope.clone()));
    }
    Ok(serde_json::Value::Object(object))
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
    use simd_json::prelude::*;
    use simd_json::OwnedValue as Value;

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
        .and_then(|v| {
            v.as_i64()
                .map(|rank| rank.to_string())
                .or_else(|| v.as_str().map(str::to_owned))
        })
        .unwrap_or_else(|| trace_source_stem(source_path).unwrap_or_else(|| "0".to_string()));

    let trace_events = match root_obj.get("traceEvents") {
        Some(Value::Array(arr)) => arr,
        _ => {
            return Err(crate::Error::InvalidTrace(
                "missing traceEvents array".into(),
            ))
        }
    };

    let mut streams: StreamMap = IndexMap::new();
    let mut metadata: Vec<MetadataEvent> = Vec::new();

    // Two-pass: first pass collects every event; we need to know the rank's
    // minimum ts so we can normalise (matches PerFlow-AI Python behaviour
    // where each rank is shifted to its own time origin).
    let mut staging: Vec<StagingEvent> = Vec::with_capacity(trace_events.len());

    for raw in trace_events {
        let obj = match raw {
            Value::Object(o) => o,
            _ => continue,
        };

        let ph = obj
            .get("ph")
            .and_then(|v| v.as_str())
            .and_then(|s| s.as_bytes().first().copied())
            .unwrap_or(b'X');
        let phase = Phase(ph);

        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        // pid/tid can be process-label strings (e.g. "GPU 0") in some
        // chrome-trace dialects; truncate floats and tolerate strings.
        let pid: i64 = obj
            .get("pid")
            .and_then(|v| {
                v.as_i64()
                    .or_else(|| v.as_f64().map(|f| f as i64))
                    .or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok()))
            })
            .unwrap_or(0);
        let raw_tid: String = match obj
            .get("tid")
            .cloned()
            .unwrap_or(Value::Static(simd_json::StaticNode::Null))
        {
            Value::String(s) => s,
            Value::Static(simd_json::StaticNode::I64(n)) => n.to_string(),
            Value::Static(simd_json::StaticNode::U64(n)) => n.to_string(),
            _ => "0".to_string(),
        };

        if phase == Phase::METADATA {
            let args = obj.get("args").cloned().map(simd_to_serde);
            metadata.push(MetadataEvent {
                name,
                pid,
                tid: raw_tid,
                args,
            });
            continue;
        }

        // Normalise tid for HIP/ROCm and PyTorch GPU traces:
        //   * if `args.stream` is present, it is a GPU stream id   -> tid := "stream <id>"
        //   * else if cat == "gpu_user_annotation",                  -> tid := "stream <tid>"
        //   * else leave as-is.
        let stream_in_args = obj.get("args").and_then(|args| match args {
            Value::Object(o) => o.get("stream").and_then(|v| {
                v.as_i64()
                    .map(|n| n.to_string())
                    .or_else(|| v.as_str().map(str::to_owned))
            }),
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
        staging.push(StagingEvent {
            event,
            pid,
            tid,
            phase,
        });
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
    trace.metadata.insert(rank, metadata);
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
    use simd_json::prelude::*;
    use simd_json::OwnedValue as Value;
    // simd-json's `.as_i64()` returns None for f64 numbers; chrome-traces
    // emitted by Kineto+ROCm write `ts`/`dur` as floats.  Falling through
    // to `unwrap_or(0)` would silently zero out every such event's
    // timestamp.  Truncate floats explicitly to match the streaming
    // parser's behaviour and the legacy serde_json path.
    let ts = obj
        .get("ts")
        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))
        .unwrap_or(0);
    let dur = obj
        .get("dur")
        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)));
    let cat = obj.get("cat").and_then(|v| v.as_str()).map(str::to_owned);
    let id = obj
        .get("id")
        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)));
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
        .and_then(|v| {
            v.as_i64()
                .map(|rank| rank.to_string())
                .or_else(|| v.as_str().map(str::to_owned))
        })
        .unwrap_or_else(|| trace_source_stem(source_path).unwrap_or_else(|| "0".to_string()));

    let trace_events = root_obj
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .ok_or_else(|| crate::Error::InvalidTrace("missing traceEvents array".into()))?;

    let mut streams: StreamMap = IndexMap::new();
    let mut metadata: Vec<MetadataEvent> = Vec::new();
    let mut staging: Vec<StagingEvent> = Vec::with_capacity(trace_events.len());

    for raw in trace_events {
        let obj = match raw.as_object() {
            Some(o) => o,
            None => continue,
        };

        let ph = obj
            .get("ph")
            .and_then(|v| v.as_str())
            .and_then(|s| s.as_bytes().first().copied())
            .unwrap_or(b'X');
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
            metadata.push(MetadataEvent {
                name,
                pid,
                tid: raw_tid,
                args: obj.get("args").cloned(),
            });
            continue;
        }

        let stream_in_args = obj
            .get("args")
            .and_then(|args| args.as_object())
            .and_then(|a| {
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
        staging.push(StagingEvent {
            event,
            pid,
            tid,
            phase,
        });
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
    trace.metadata.insert(rank, metadata);
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
        V::String(s) => serde_json::Value::String(s),
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
pub type CompressedRankMap = BTreeMap<String, BTreeMap<i64, BTreeMap<String, BTreeMap<u8, Node>>>>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompressedTrace {
    pub templates: Vec<Template>,
    /// `rank -> pid -> tid -> ph -> root_node`
    pub ranks: CompressedRankMap,
    pub metadata: BTreeMap<String, Vec<MetadataEvent>>,
    pub start_timestamp: BTreeMap<String, i64>,
}

/// Current on-disk PADOC artifact format.
pub const ARTIFACT_FORMAT_VERSION: u16 = 2;
const MIN_SUPPORTED_ARTIFACT_FORMAT_VERSION: u16 = 1;
const ARTIFACT_MAGIC: &[u8; 8] = b"PADOCART";
const ARTIFACT_CODEC_ZSTD: u8 = 1;
const ARTIFACT_HEADER_LEN: usize = 16;

impl CompressedTrace {
    /// Persist a versioned PADOC artifact without overwriting an existing file.
    ///
    /// Streaming pipeline: msgpack chunks flow directly into a zstd
    /// encoder wrapped around a buffered file writer, so neither the raw
    /// msgpack output (~10 GiB on a 1024-rank profiler trace) nor the
    /// compressed blob (~2.4 GiB) is ever fully buffered in memory.
    ///
    pub fn write_to_path(&self, path: impl AsRef<Path>, zstd_level: i32) -> Result<()> {
        use std::io::Write;

        let path = path.as_ref();
        if path.exists() {
            return Err(crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("refusing to overwrite {}", path.display()),
            )));
        }

        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let temp = tempfile::NamedTempFile::new_in(parent)?;
        let mut writer = std::io::BufWriter::with_capacity(8 * 1024 * 1024, temp);
        write_artifact_header(&mut writer)?;
        let encoder = zstd::stream::Encoder::new(writer, zstd_level)?;
        let mut buffered_encoder = std::io::BufWriter::with_capacity(1 << 20, encoder);
        rmp_serde::encode::write_named(&mut buffered_encoder, self)?;
        buffered_encoder.flush()?;
        let encoder = buffered_encoder
            .into_inner()
            .map_err(|e| crate::Error::Other(format!("flush artifact encoder: {}", e.error())))?;
        let mut writer = encoder.finish()?;
        writer.flush()?;
        let temp = writer
            .into_inner()
            .map_err(|e| crate::Error::Other(format!("flush artifact file: {}", e.error())))?;
        temp.as_file().sync_all()?;
        temp.persist_noclobber(path)
            .map_err(|error| crate::Error::Io(error.error))?;
        Ok(())
    }

    /// Read a PADOC artifact with bounded I/O buffering.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_from_path_with_format_version(path).map(|(trace, _)| trace)
    }

    /// Read a PADOC artifact and return the validated on-disk format version.
    ///
    /// This is useful for metadata tools that must report the artifact header
    /// rather than the version emitted by the current build.
    pub fn read_from_path_with_format_version(path: impl AsRef<Path>) -> Result<(Self, u16)> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::with_capacity(8 * 1024 * 1024, file);
        let version = read_artifact_header(&mut reader)?;
        let decoder = zstd::stream::read::Decoder::new(reader)?;
        Ok((rmp_serde::from_read(decoder)?, version))
    }

    /// Encode to a self-contained, versioned artifact byte blob.
    ///
    /// Streams msgpack output straight into a single-threaded zstd encoder
    /// — the full uncompressed msgpack payload is never materialised, only
    /// the final compressed buffer.
    ///
    /// The payload encoder is intentionally single-threaded. Directory-level
    /// parallelism processes independent trace files instead of adding a
    /// second, nested source of concurrency here.
    pub fn to_bytes(&self, zstd_level: i32) -> Result<Vec<u8>> {
        let mut out: Vec<u8> = Vec::with_capacity(8 * 1024 * 1024);
        write_artifact_header(&mut out)?;
        let encoder = zstd::stream::Encoder::new(out, zstd_level)?;
        let mut buf_enc = std::io::BufWriter::with_capacity(1 << 20, encoder);
        rmp_serde::encode::write_named(&mut buf_enc, self)?;
        use std::io::Write;
        buf_enc.flush()?;
        let encoder = buf_enc
            .into_inner()
            .map_err(|e| crate::Error::Other(format!("flush BufWriter: {}", e.error())))?;
        Ok(encoder.finish()?)
    }

    /// Decode the byte blob produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        read_artifact_header(&mut cursor)?;
        let decoder = zstd::stream::read::Decoder::new(cursor)?;
        Ok(rmp_serde::from_read(decoder)?)
    }
}

fn write_artifact_header(writer: &mut impl std::io::Write) -> Result<()> {
    let mut header = [0_u8; ARTIFACT_HEADER_LEN];
    header[..8].copy_from_slice(ARTIFACT_MAGIC);
    header[8..10].copy_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
    header[10] = ARTIFACT_CODEC_ZSTD;
    writer.write_all(&header)?;
    Ok(())
}

fn read_artifact_header(reader: &mut impl std::io::Read) -> Result<u16> {
    let mut header = [0_u8; ARTIFACT_HEADER_LEN];
    reader.read_exact(&mut header)?;
    if &header[..8] != ARTIFACT_MAGIC {
        return Err(crate::Error::InvalidCompressed(
            "invalid PADOC artifact magic".into(),
        ));
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    if !(MIN_SUPPORTED_ARTIFACT_FORMAT_VERSION..=ARTIFACT_FORMAT_VERSION).contains(&version) {
        return Err(crate::Error::InvalidCompressed(format!(
            "unsupported artifact format version {version}; supported versions are \
             {MIN_SUPPORTED_ARTIFACT_FORMAT_VERSION}..={ARTIFACT_FORMAT_VERSION}"
        )));
    }
    if header[10] != ARTIFACT_CODEC_ZSTD {
        return Err(crate::Error::InvalidCompressed(format!(
            "unsupported artifact codec {}",
            header[10]
        )));
    }
    if header[11..].iter().any(|byte| *byte != 0) {
        return Err(crate::Error::InvalidCompressed(
            "artifact header uses unsupported flags".into(),
        ));
    }
    Ok(version)
}
