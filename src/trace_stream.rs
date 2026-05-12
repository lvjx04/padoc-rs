//! Streaming chrome-trace parser.
//!
//! The default parsers in `trace.rs` build a full `simd_json::OwnedValue`
//! (or `serde_json::Value`) tree before walking it.  For a 5.7 GiB
//! `unifolm` rank that intermediate tree balloons to 30-60 GiB resident,
//! which dominates compression peak memory and forces tiny worker
//! counts.
//!
//! This module instead drives `serde_json::Deserializer` with a hand-
//! written `Visitor` chain that pulls one event at a time off the
//! `traceEvents` array, immediately decodes it into our `Event` struct,
//! and inserts it into the per-rank `StreamMap`.  Peak memory therefore
//! tracks the **final, decoded** trace size — typically 0.5-1× the JSON
//! file size on disk — instead of the raw JSON-tree expansion.
//!
//! The parser preserves all semantics of the existing simd-json path:
//!
//! * `distributedInfo.rank` becomes the trace's rank id (falling back to
//!   the file stem).
//! * Phase-`M` (metadata) events are split out into `Trace::metadata`.
//! * `args.stream` and `cat == "gpu_user_annotation"` rewrite the `tid`
//!   into `"stream <id>"` so HIP/ROCm GPU events end up on the right
//!   per-stream lane.
//! * `tid` is accepted as either an integer or a string.
//! * Per-rank timestamps are normalised relative to the rank's smallest
//!   `ts` (matches the legacy Python pipeline so analysis is invariant).

use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ahash::AHashMap;
use indexmap::IndexMap;
use serde::de::{self, DeserializeSeed, Deserializer as _, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;

use crate::event::{Event, Phase};
use crate::trace::{StreamMap, Trace};
use crate::Result;

/// 8 MiB BufReader window — large enough that serde_json's lookahead never
/// thrashes between buffer refills on a typical trace.
const BUFFER_SIZE: usize = 8 * 1024 * 1024;

pub fn parse_chrome_trace_stream(path: &Path) -> Result<Trace> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(BUFFER_SIZE, file);
    let mut de = serde_json::Deserializer::from_reader(reader);

    let mut state = ParserState {
        rank: None,
        streams: IndexMap::new(),
        metadata: AHashMap::new(),
        min_ts: i64::MAX,
        source_stem: path.file_stem().and_then(|s| s.to_str()).map(str::to_owned),
    };

    de.deserialize_map(TopLevelVisitor { state: &mut state })
        .map_err(|e| crate::Error::Other(format!("streaming chrome-trace parse failed: {e}")))?;

    // Per-rank ts origin: subtract the smallest ts so the column is small.
    let start_ts = if state.min_ts == i64::MAX { 0 } else { state.min_ts };
    if start_ts != 0 {
        for threads in state.streams.values_mut() {
            for phases in threads.values_mut() {
                for events in phases.values_mut() {
                    for ev in events.iter_mut() {
                        ev.ts -= start_ts;
                    }
                }
            }
        }
    }

    let rank = state
        .rank
        .or_else(|| state.source_stem.clone())
        .unwrap_or_else(|| "0".to_string());

    let mut trace = Trace::empty();
    trace.ranks.insert(rank.clone(), state.streams);
    trace.start_timestamp.insert(rank.clone(), start_ts);
    trace.metadata.insert(rank, state.metadata);
    Ok(trace)
}

struct ParserState {
    rank: Option<String>,
    streams: StreamMap,
    metadata: AHashMap<String, serde_json::Value>,
    min_ts: i64,
    source_stem: Option<String>,
}

impl ParserState {
    fn absorb(&mut self, raw: RawEvent) {
        let phase_byte = raw
            .ph
            .as_ref()
            .and_then(|s| s.as_bytes().first().copied())
            .unwrap_or(b'X');
        let phase = Phase(phase_byte);
        let name = raw.name.unwrap_or_default();

        if phase == Phase::METADATA {
            let val = raw
                .args
                .map(serde_json::Value::Object)
                .unwrap_or(serde_json::Value::Null);
            self.metadata.insert(name, val);
            return;
        }

        let pid = raw.pid.unwrap_or(0);
        let raw_tid = raw.tid.unwrap_or_else(|| "0".to_string());
        let cat = raw.cat;

        let stream_id = raw.args.as_ref().and_then(|m| {
            m.get("stream").and_then(|v| {
                v.as_i64()
                    .map(|n| n.to_string())
                    .or_else(|| v.as_str().map(str::to_owned))
            })
        });

        let tid = if let Some(stream) = stream_id {
            format!("stream {}", stream)
        } else if cat.as_deref() == Some("gpu_user_annotation") {
            format!("stream {}", raw_tid)
        } else {
            raw_tid
        };

        let ts = raw.ts.unwrap_or(0);
        if ts < self.min_ts {
            self.min_ts = ts;
        }

        let args = raw.args.map(|m| {
            let mut out = AHashMap::with_capacity(m.len());
            for (k, v) in m {
                out.insert(k, v);
            }
            out
        });

        let event = Event {
            name,
            ts,
            dur: raw.dur,
            cat,
            ph: phase,
            pid,
            tid: tid.clone(),
            args,
            id: raw.id,
            bp: raw.bp,
            s: raw.s,
        };

        self.streams
            .entry(pid)
            .or_default()
            .entry(tid)
            .or_default()
            .entry(phase)
            .or_default()
            .push(event);
    }
}

// ---------------------------------------------------------------------------
// Top-level visitor: handles `traceEvents`, `distributedInfo`, ignores
// everything else (displayTimeUnit, otherData, ...).
// ---------------------------------------------------------------------------

struct TopLevelVisitor<'a> {
    state: &'a mut ParserState,
}

impl<'de> Visitor<'de> for TopLevelVisitor<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "chrome-trace JSON object")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> std::result::Result<Self::Value, M::Error> {
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "traceEvents" => {
                    map.next_value_seed(TraceEventsSeed { state: self.state })?;
                }
                "distributedInfo" => {
                    let v: serde_json::Value = map.next_value()?;
                    if let Some(rank) = v.get("rank").and_then(|r| r.as_i64()) {
                        self.state.rank = Some(rank.to_string());
                    }
                }
                _ => {
                    let _: de::IgnoredAny = map.next_value()?;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// `traceEvents` — array of events streamed one-by-one.
// ---------------------------------------------------------------------------

struct TraceEventsSeed<'a> {
    state: &'a mut ParserState,
}

impl<'de> DeserializeSeed<'de> for TraceEventsSeed<'_> {
    type Value = ();

    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> std::result::Result<(), D::Error> {
        deserializer.deserialize_seq(TraceEventsSeqVisitor { state: self.state })
    }
}

struct TraceEventsSeqVisitor<'a> {
    state: &'a mut ParserState,
}

impl<'de> Visitor<'de> for TraceEventsSeqVisitor<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "traceEvents array")
    }

    fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> std::result::Result<Self::Value, S::Error> {
        while let Some(raw) = seq.next_element::<RawEvent>()? {
            self.state.absorb(raw);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RawEvent — chrome-trace event row, decoded straight from the stream.
// `tid` accepts either string or integer (chrome-trace lets either).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawEvent {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    ph: Option<String>,
    /// Chrome traces emitted by some profilers (notably Kineto + ROCm) write
    /// `ts` and `dur` as floating-point microseconds, even though the spec
    /// says integer.  Round to i64 here to match the legacy `.as_i64()`
    /// path which silently truncated.
    #[serde(default, deserialize_with = "deserialize_lossy_i64")]
    ts: Option<i64>,
    #[serde(default, deserialize_with = "deserialize_lossy_i64")]
    dur: Option<i64>,
    #[serde(default)]
    cat: Option<String>,
    #[serde(default, deserialize_with = "deserialize_lossy_i64")]
    pid: Option<i64>,
    #[serde(default, deserialize_with = "deserialize_tid")]
    tid: Option<String>,
    #[serde(default)]
    args: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, deserialize_with = "deserialize_lossy_i64")]
    id: Option<i64>,
    #[serde(default)]
    bp: Option<String>,
    #[serde(default)]
    s: Option<String>,
}

fn deserialize_lossy_i64<'de, D: de::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<i64>, D::Error> {
    struct LossyI64Visitor;
    impl<'de> Visitor<'de> for LossyI64Visitor {
        type Value = Option<i64>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "integer (possibly written as float, string label, null, or absent)")
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v))
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v as i64))
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
            // Round-half-toward-zero (same as `.as_i64()` truncation).
            Ok(Some(v as i64))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
            // chrome-trace lets pid/tid be process-label strings ("Spans",
            // "GPU 0", ...).  The legacy `.as_i64()` path returned 0 for
            // non-numeric strings; keep that behaviour.
            Ok(Some(v.parse::<i64>().unwrap_or(0)))
        }
        fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
            self.visit_str(&v)
        }
        fn visit_bool<E: de::Error>(self, _v: bool) -> std::result::Result<Self::Value, E> {
            Ok(Some(0))
        }
        fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D2: de::Deserializer<'de>>(
            self,
            d: D2,
        ) -> std::result::Result<Self::Value, D2::Error> {
            d.deserialize_any(self)
        }
    }
    d.deserialize_any(LossyI64Visitor)
}

fn deserialize_tid<'de, D: de::Deserializer<'de>>(d: D) -> std::result::Result<Option<String>, D::Error> {
    struct TidVisitor;
    impl<'de> Visitor<'de> for TidVisitor {
        type Value = Option<String>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "tid (string or integer or null)")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_owned()))
        }
        fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
            Ok(Some(v))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D2: de::Deserializer<'de>>(
            self,
            d: D2,
        ) -> std::result::Result<Self::Value, D2::Error> {
            d.deserialize_any(self)
        }
    }
    d.deserialize_any(TidVisitor)
}
