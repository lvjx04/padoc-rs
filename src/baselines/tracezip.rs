//! TraceZip adaptation for AI traces (ICSE'25 / arXiv:2502.06318).
//!
//! TraceZip targets distributed tracing spans.  Adapted here:
//!
//! * Each event = one span.  `Event.name` is the SRT (Span Retrieval Tree) key.
//! * Per-stream events are bucketed by name into SRT nodes.  Each node
//!   stores parallel column arrays so inter-event redundancy across the
//!   bucket is compressed by zstd at the end.
//! * Per-stream `time_base = min(ts)` is subtracted from `ts` so the
//!   numeric range is small.
//! * Universal columns (one per event in the bucket): arg keys whose
//!   distinct value count is small are dictionary-encoded; high-cardinality
//!   keys are stored verbatim.
//!
//! Lossless: every `Event` field (`cat`, `bp`, `s`, optional `dur`/`id`,
//! and `args` including absent vs. present-with-null distinction) is
//! preserved across compress/decompress.

use crate::baselines::{BaselineCompressor, CompressArtifact};
use crate::event::{Event, Phase};
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
    /// One Span Retrieval Tree per stream.
    streams: Vec<TraceZipStream>,
}

#[derive(Serialize, Deserialize)]
struct TraceZipStream {
    rank: String,
    pid: i64,
    tid: String,
    ph: u8,
    time_base: i64,
    /// SRT bucket per distinct event name.
    srt: Vec<SrtNode>,
}

#[derive(Serialize, Deserialize)]
struct SrtNode {
    name_dict_id: u32,
    /// Per-event scalar columns.
    ts_offsets: Vec<i64>,
    /// `dur_present[i]` = true means `dur[i]` is meaningful; false = original was None.
    dur_present: Vec<bool>,
    dur: Vec<i64>,
    id_present: Vec<bool>,
    ids: Vec<i64>,
    /// Per-event optional cat / bp / s as dict-id, 0 = absent (offset by +1).
    cat_dict_id_plus1: Vec<u32>,
    bp_dict_id_plus1: Vec<u32>,
    s_dict_id_plus1: Vec<u32>,
    /// args.* keys dict-encoded.  `arg_keys` is the union across this bucket.
    arg_keys: Vec<u32>,
    /// Per-event arg-presence bitmap and arg values (parallel to `arg_keys`).
    /// `arg_present[i]` is a Vec<bool> of length arg_keys.len();
    /// `arg_values[i]` only stores values for keys where `arg_present[i][k]`.
    arg_present: Vec<Vec<bool>>,
    arg_values: Vec<Vec<serde_json::Value>>,
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
                        let mut srt_buckets: AHashMap<String, Vec<&Event>> = AHashMap::new();
                        let mut time_base = i64::MAX;
                        for ev in events {
                            time_base = time_base.min(ev.ts);
                            srt_buckets.entry(ev.name.clone()).or_default().push(ev);
                        }
                        let mut srt: Vec<SrtNode> = Vec::with_capacity(srt_buckets.len());
                        for (name, bucket) in srt_buckets {
                            let name_id = dict.intern(&name);

                            // Discover arg keys union.
                            let mut keys_set: ahash::AHashSet<String> = ahash::AHashSet::new();
                            for ev in &bucket {
                                if let Some(args) = &ev.args {
                                    for k in args.keys() { keys_set.insert(k.clone()); }
                                }
                            }
                            let mut arg_key_strs: Vec<String> = keys_set.into_iter().collect();
                            arg_key_strs.sort();
                            let arg_keys: Vec<u32> = arg_key_strs.iter().map(|k| dict.intern(k)).collect();

                            let mut ts_offsets = Vec::with_capacity(bucket.len());
                            let mut dur_present = Vec::with_capacity(bucket.len());
                            let mut dur = Vec::with_capacity(bucket.len());
                            let mut id_present = Vec::with_capacity(bucket.len());
                            let mut ids = Vec::with_capacity(bucket.len());
                            let mut cat_dict_id_plus1 = Vec::with_capacity(bucket.len());
                            let mut bp_dict_id_plus1 = Vec::with_capacity(bucket.len());
                            let mut s_dict_id_plus1 = Vec::with_capacity(bucket.len());
                            let mut arg_present = Vec::with_capacity(bucket.len());
                            let mut arg_values = Vec::with_capacity(bucket.len());

                            for ev in &bucket {
                                ts_offsets.push(ev.ts - time_base);
                                match ev.dur {
                                    Some(d) => { dur_present.push(true); dur.push(d); },
                                    None    => { dur_present.push(false); dur.push(0); },
                                }
                                match ev.id {
                                    Some(i) => { id_present.push(true); ids.push(i); },
                                    None    => { id_present.push(false); ids.push(0); },
                                }
                                cat_dict_id_plus1.push(ev.cat.as_ref().map(|c| dict.intern(c) + 1).unwrap_or(0));
                                bp_dict_id_plus1.push(ev.bp.as_ref().map(|b| dict.intern(b) + 1).unwrap_or(0));
                                s_dict_id_plus1.push(ev.s.as_ref().map(|s| dict.intern(s) + 1).unwrap_or(0));

                                let mut presence = Vec::with_capacity(arg_key_strs.len());
                                let mut values = Vec::new();
                                for k in &arg_key_strs {
                                    match ev.args.as_ref().and_then(|a| a.get(k)) {
                                        Some(v) => { presence.push(true); values.push(v.clone()); },
                                        None    => { presence.push(false); },
                                    }
                                }
                                arg_present.push(presence);
                                arg_values.push(values);
                            }

                            srt.push(SrtNode {
                                name_dict_id: name_id,
                                ts_offsets,
                                dur_present, dur,
                                id_present, ids,
                                cat_dict_id_plus1,
                                bp_dict_id_plus1,
                                s_dict_id_plus1,
                                arg_keys,
                                arg_present,
                                arg_values,
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
        let lookup = |id_plus1: u32| -> Option<String> {
            if id_plus1 == 0 { None } else { dict.get((id_plus1 - 1) as usize).cloned() }
        };
        let mut trace = Trace::empty();
        for stream in payload.streams {
            let mut events: Vec<Event> = Vec::new();
            for node in &stream.srt {
                let name = dict.get(node.name_dict_id as usize).cloned().unwrap_or_default();
                let arg_keys: Vec<String> = node.arg_keys.iter()
                    .filter_map(|i| dict.get(*i as usize).cloned()).collect();
                let n = node.ts_offsets.len();
                for i in 0..n {
                    let mut args = AHashMap::new();
                    if let Some(presence) = node.arg_present.get(i) {
                        let mut vi = 0usize;
                        for (k, &p) in arg_keys.iter().zip(presence.iter()) {
                            if p {
                                let v = node.arg_values.get(i)
                                    .and_then(|row| row.get(vi))
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                args.insert(k.clone(), v);
                                vi += 1;
                            }
                        }
                    }
                    let dur = if *node.dur_present.get(i).unwrap_or(&false) {
                        Some(*node.dur.get(i).unwrap_or(&0))
                    } else { None };
                    let id = if *node.id_present.get(i).unwrap_or(&false) {
                        Some(*node.ids.get(i).unwrap_or(&0))
                    } else { None };
                    let cat = lookup(*node.cat_dict_id_plus1.get(i).unwrap_or(&0));
                    let bp  = lookup(*node.bp_dict_id_plus1 .get(i).unwrap_or(&0));
                    let s   = lookup(*node.s_dict_id_plus1  .get(i).unwrap_or(&0));
                    events.push(Event {
                        name: name.clone(),
                        ts: stream.time_base + node.ts_offsets[i],
                        dur, cat,
                        ph: Phase(stream.ph),
                        pid: stream.pid,
                        tid: stream.tid.clone(),
                        args: if args.is_empty() { None } else { Some(args) },
                        id, bp, s,
                    });
                }
            }
            trace.ranks
                .entry(stream.rank.clone()).or_default()
                .entry(stream.pid).or_default()
                .entry(stream.tid.clone()).or_default()
                .entry(Phase(stream.ph)).or_default()
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
