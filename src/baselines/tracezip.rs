//! TraceZip adaptation for AI traces (ICSE'25 / arXiv:2502.06318).
//!
//! TraceZip targets distributed tracing spans.  Adapted here:
//!
//! * Each event = one span.  `Event.name` is the SRT (Span Retrieval Tree) key.
//! * Universal columns (one per event in the span): `cat`, `bp`, `s`, `args.*`
//!   keys whose distinct value count is small.  Each universal value is
//!   replaced by its dictionary id.
//! * Local columns (per stream): `ts`, `dur`, `id`, plus high-cardinality
//!   `args.*` values that aren't worth dictionary-encoding.
//! * Per-stream `time_base = min(ts)` is subtracted from `ts` so the
//!   numeric range is small.
//!
//! This is the "PADOC without structural / SLP" baseline — its compression
//! ratio should beat raw/gzip but lose to PADOC because it doesn't exploit
//! call-tree structure.

use crate::baselines::{BaselineCompressor, CompressArtifact};
use crate::trace::Trace;
use crate::Result;
use ahash::AHashMap;
use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct TraceZipCompressor;

#[derive(Serialize, Deserialize)]
struct TraceZipPayload {
    /// Global string dictionary; `dict_strings[i]` is the i-th distinct string.
    dict_strings: Vec<String>,
    /// Span Retrieval Tree per stream.
    streams: Vec<TraceZipStream>,
}

#[derive(Serialize, Deserialize)]
struct TraceZipStream {
    rank: String,
    pid: i64,
    tid: String,
    ph: u8,
    time_base: i64,
    /// SRT bucket per distinct event name.  Each bucket carries every event
    /// instance that shares that name.
    srt: Vec<SrtNode>,
}

#[derive(Serialize, Deserialize)]
struct SrtNode {
    name_dict_id: u32,
    cat_dict_id: Option<u32>,
    ts_offsets: Vec<i64>,
    dur: Vec<i64>,
    ids: Vec<i64>,
    /// Universal arg keys (dict-encoded) and their per-instance dict ids.
    universal_arg_keys: Vec<u32>,
    universal_args: Vec<Vec<u32>>,
    /// Local (high-cardinality) args dumped as raw JSON.
    local_arg_keys: Vec<u32>,
    local_args: Vec<Vec<serde_json::Value>>,
}

impl BaselineCompressor for TraceZipCompressor {
    fn name(&self) -> &str { "tracezip" }

    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();

        let mut dict: StringDict = StringDict::default();
        let mut streams: Vec<TraceZipStream> = Vec::new();

        for (rank, processes) in &trace.ranks {
            for (pid, threads) in processes {
                for (tid, phases) in threads {
                    for (ph, events) in phases {
                        if events.is_empty() { continue; }
                        let mut srt_buckets: AHashMap<String, Vec<&crate::event::Event>> = AHashMap::new();
                        let mut time_base = i64::MAX;
                        for ev in events {
                            time_base = time_base.min(ev.ts);
                            srt_buckets.entry(ev.name.clone()).or_default().push(ev);
                        }
                        let mut srt: Vec<SrtNode> = Vec::with_capacity(srt_buckets.len());
                        for (name, bucket) in srt_buckets {
                            let name_id = dict.intern(&name);
                            let cat_id = bucket[0].cat.as_ref().map(|c| dict.intern(c));
                            let mut ts_offsets = Vec::with_capacity(bucket.len());
                            let mut dur = Vec::with_capacity(bucket.len());
                            let mut ids = Vec::with_capacity(bucket.len());

                            // Discover universal vs local keys: count distinct values per arg key.
                            let mut value_distinct: AHashMap<String, ahash::AHashSet<String>> = AHashMap::new();
                            for ev in &bucket {
                                if let Some(args) = &ev.args {
                                    for (k, v) in args {
                                        value_distinct.entry(k.clone()).or_default().insert(v.to_string());
                                    }
                                }
                            }
                            let mut universal_keys: Vec<String> = Vec::new();
                            let mut local_keys: Vec<String> = Vec::new();
                            for (k, vs) in &value_distinct {
                                let dict_threshold = (bucket.len() / 4).max(8);
                                if vs.len() <= dict_threshold {
                                    universal_keys.push(k.clone());
                                } else {
                                    local_keys.push(k.clone());
                                }
                            }
                            universal_keys.sort();
                            local_keys.sort();

                            let universal_key_ids: Vec<u32> = universal_keys.iter().map(|k| dict.intern(k)).collect();
                            let local_key_ids: Vec<u32> = local_keys.iter().map(|k| dict.intern(k)).collect();

                            let mut universal_args: Vec<Vec<u32>> = Vec::with_capacity(bucket.len());
                            let mut local_args: Vec<Vec<serde_json::Value>> = Vec::with_capacity(bucket.len());

                            for ev in &bucket {
                                ts_offsets.push(ev.ts - time_base);
                                dur.push(ev.dur.unwrap_or(0));
                                ids.push(ev.id.unwrap_or(0));
                                let mut u_row = Vec::with_capacity(universal_keys.len());
                                let mut l_row = Vec::with_capacity(local_keys.len());
                                for k in &universal_keys {
                                    let v = ev.args.as_ref().and_then(|a| a.get(k)).cloned().unwrap_or(serde_json::Value::Null);
                                    u_row.push(dict.intern(&v.to_string()));
                                }
                                for k in &local_keys {
                                    let v = ev.args.as_ref().and_then(|a| a.get(k)).cloned().unwrap_or(serde_json::Value::Null);
                                    l_row.push(v);
                                }
                                universal_args.push(u_row);
                                local_args.push(l_row);
                            }

                            srt.push(SrtNode {
                                name_dict_id: name_id,
                                cat_dict_id: cat_id,
                                ts_offsets,
                                dur,
                                ids,
                                universal_arg_keys: universal_key_ids,
                                universal_args,
                                local_arg_keys: local_key_ids,
                                local_args,
                            });
                        }

                        streams.push(TraceZipStream {
                            rank: rank.clone(),
                            pid: *pid,
                            tid: tid.clone(),
                            ph: ph.0,
                            time_base: if time_base == i64::MAX { 0 } else { time_base },
                            srt,
                        });
                    }
                }
            }
        }

        let payload = TraceZipPayload { dict_strings: dict.into_strings(), streams };
        let mut buf = Vec::new();
        rmp_serde::encode::write_named(&mut buf, &payload)?;
        let bytes = zstd::stream::encode_all(&buf[..], 3)?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }

    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let raw = zstd::stream::decode_all(bytes)?;
        let payload: TraceZipPayload = rmp_serde::from_slice(&raw)?;
        let dict = payload.dict_strings;
        let mut trace = Trace::empty();
        for stream in payload.streams {
            let mut events: Vec<crate::event::Event> = Vec::new();
            for node in &stream.srt {
                let name = dict.get(node.name_dict_id as usize).cloned().unwrap_or_default();
                let cat = node.cat_dict_id.and_then(|id| dict.get(id as usize)).cloned();
                let universal_keys: Vec<String> = node.universal_arg_keys.iter()
                    .filter_map(|i| dict.get(*i as usize).cloned()).collect();
                let local_keys: Vec<String> = node.local_arg_keys.iter()
                    .filter_map(|i| dict.get(*i as usize).cloned()).collect();
                for i in 0..node.ts_offsets.len() {
                    let mut args = ahash::AHashMap::new();
                    if let (Some(u_row), false) = (node.universal_args.get(i), universal_keys.is_empty()) {
                        for (k, v_id) in universal_keys.iter().zip(u_row.iter()) {
                            let raw = dict.get(*v_id as usize).cloned().unwrap_or_default();
                            let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap_or(serde_json::Value::String(raw));
                            args.insert(k.clone(), parsed);
                        }
                    }
                    if let Some(l_row) = node.local_args.get(i) {
                        for (k, v) in local_keys.iter().zip(l_row.iter()) {
                            args.insert(k.clone(), v.clone());
                        }
                    }
                    events.push(crate::event::Event {
                        name: name.clone(),
                        ts: stream.time_base + node.ts_offsets[i],
                        dur: Some(node.dur[i]),
                        cat: cat.clone(),
                        ph: crate::event::Phase(stream.ph),
                        pid: stream.pid,
                        tid: stream.tid.clone(),
                        args: if args.is_empty() { None } else { Some(args) },
                        id: Some(node.ids[i]),
                        bp: None,
                        s: None,
                    });
                }
            }
            trace.ranks
                .entry(stream.rank.clone()).or_default()
                .entry(stream.pid).or_default()
                .entry(stream.tid.clone()).or_default()
                .entry(crate::event::Phase(stream.ph)).or_default()
                .extend(events);
        }
        Ok(trace)
    }
}

#[derive(Default)]
struct StringDict {
    index: AHashMap<String, u32>,
    items: Vec<String>,
}

impl StringDict {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.index.get(s) {
            return id;
        }
        let id = self.items.len() as u32;
        self.items.push(s.to_string());
        self.index.insert(s.to_string(), id);
        id
    }
    fn into_strings(self) -> Vec<String> { self.items }
}
