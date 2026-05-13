//! TraceZip adaptation for AI traces (ICSE'25 / arXiv:2502.06318) —
//! **cross-rank** version.
//!
//! TraceZip targets distributed tracing spans.  Adapted here:
//!
//! * Each event = one span.  `Event.name` is the SRT (Span Retrieval Tree) key.
//! * Cross-rank merging:
//!     - **Global string dict** — every distinct string (event name,
//!       cat, bp, s, arg-key) is interned ONCE across the whole trace.
//!     - **Global SRT schema** — `(name_id, sorted_arg_key_ids)` is
//!       interned globally so all 1024 ranks share one schema entry
//!       per `(name, arg-keys)` combination.  A 1024-rank training
//!       trace with 200 distinct event names yields 200 schemas, not
//!       1024 × 200.
//!     - **Per-stream data buckets** — a stream's events that match a
//!       given schema are stored in a `BucketData` carrying parallel
//!       column arrays (ts_offsets, dur, args).  Schemas are shared,
//!       data isn't (events differ per rank).
//! * Per-stream `time_base = min(ts)` is subtracted from `ts` so the
//!   numeric range is small (good for zstd).
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
    /// Global string dictionary: event names, cats, bp/s strings,
    /// arg-key names, and rank/tid strings all share this dict.
    dict_strings: Vec<String>,
    /// Global SRT schema pool.  Indexed by `schema_id`.  Each schema is
    /// a `(name_id, arg_key_ids)` tuple — defines what columns a bucket
    /// will carry.
    schemas: Vec<SrtSchema>,
    /// One stream per `(rank, pid, tid, ph)`.
    streams: Vec<TraceZipStream>,
}

#[derive(Serialize, Deserialize)]
struct SrtSchema {
    name_dict_id: u32,
    /// Sorted dict ids of args.* keys ever seen for this name across the
    /// whole trace.  Per-event presence is stored as a bitmap.
    arg_key_ids: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct TraceZipStream {
    rank_dict_id: u32,
    pid: i64,
    tid_dict_id: u32,
    ph: u8,
    time_base: i64,
    /// Per-stream bucket data: one entry per global schema this
    /// stream's events touched.
    buckets: Vec<BucketData>,
}

#[derive(Serialize, Deserialize)]
struct BucketData {
    /// Index into the GLOBAL `schemas` table.
    schema_id: u32,
    ts_offsets: Vec<i64>,
    dur_present: Vec<bool>,
    dur: Vec<i64>,
    id_present: Vec<bool>,
    ids: Vec<i64>,
    /// 0 = absent, otherwise dict_id+1 (refers to GLOBAL `dict_strings`).
    cat_dict_id_plus1: Vec<u32>,
    bp_dict_id_plus1: Vec<u32>,
    s_dict_id_plus1: Vec<u32>,
    /// Per-event arg-presence bitmap.  Length matches
    /// `schemas[schema_id].arg_key_ids.len()`.  `arg_values[i]` holds
    /// values only for keys where `arg_present[i][k]` is true.
    arg_present: Vec<Vec<bool>>,
    arg_values: Vec<Vec<serde_json::Value>>,
}

impl BaselineCompressor for TraceZipCompressor {
    fn name(&self) -> &str { "tracezip" }

    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();

        let mut dict = StringDict::default();
        // Global SRT schema interning: (name_id, sorted arg_key_ids).
        let mut schema_index: AHashMap<(u32, Vec<u32>), u32> = AHashMap::new();
        let mut schemas: Vec<SrtSchema> = Vec::new();

        // Pass 1: build per-stream name buckets and the GLOBAL arg-key
        // union per name.  We need a global arg-key union (not just
        // per-stream) so that schemas are stable across ranks — so a
        // first pass scans every stream's events to populate
        // `name_to_arg_keys`, then a second pass dict-encodes events
        // against the resulting global schemas.
        let mut name_to_arg_keys: AHashMap<String, ahash::AHashSet<String>> = AHashMap::new();
        for (_rank, processes) in &trace.ranks {
            for (_pid, threads) in processes {
                for (_tid, phases) in threads {
                    for (_ph, events) in phases {
                        for ev in events {
                            let entry = name_to_arg_keys.entry(ev.name.clone()).or_default();
                            if let Some(args) = &ev.args {
                                for k in args.keys() {
                                    if !entry.contains(k.as_str()) {
                                        entry.insert(k.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pre-intern the global schemas in one go.  After this loop,
        // `schema_index[(name_id, arg_key_ids)] -> schema_id` lets us
        // route each event to the right BucketData in O(1).
        let mut name_to_schema_id: AHashMap<String, u32> = AHashMap::new();
        for (name, key_set) in &name_to_arg_keys {
            let name_id = dict.intern(name);
            let mut arg_key_strs: Vec<String> = key_set.iter().cloned().collect();
            arg_key_strs.sort();
            let arg_key_ids: Vec<u32> = arg_key_strs.iter().map(|k| dict.intern(k)).collect();
            let key = (name_id, arg_key_ids.clone());
            let schema_id = if let Some(&id) = schema_index.get(&key) {
                id
            } else {
                let id = schemas.len() as u32;
                schemas.push(SrtSchema { name_dict_id: name_id, arg_key_ids });
                schema_index.insert(key, id);
                id
            };
            name_to_schema_id.insert(name.clone(), schema_id);
        }

        // Pass 2: per-stream encode using the global schemas.
        let mut streams: Vec<TraceZipStream> = Vec::new();
        for (rank, processes) in &trace.ranks {
            let rank_dict_id = dict.intern(rank);
            for (pid, threads) in processes {
                for (tid, phases) in threads {
                    let tid_dict_id = dict.intern(tid);
                    for (ph, events) in phases {
                        if events.is_empty() { continue; }
                        let mut time_base = i64::MAX;
                        for ev in events { time_base = time_base.min(ev.ts); }

                        // Bucket events by schema_id (which is uniquely
                        // determined by event name).
                        let mut buckets: AHashMap<u32, BucketData> = AHashMap::new();
                        for ev in events {
                            let schema_id = match name_to_schema_id.get(&ev.name) {
                                Some(&id) => id,
                                None => continue, // shouldn't happen given pass 1
                            };
                            // Clone the arg-key strings out of the dict
                            // so we can call `dict.intern(cat)` etc.
                            // below without overlapping borrows.
                            let arg_keys: Vec<String> = schemas[schema_id as usize].arg_key_ids
                                .iter().map(|id| dict.items[*id as usize].clone()).collect();
                            let bucket = buckets.entry(schema_id).or_insert_with(|| BucketData {
                                schema_id,
                                ts_offsets: Vec::new(),
                                dur_present: Vec::new(), dur: Vec::new(),
                                id_present: Vec::new(),  ids: Vec::new(),
                                cat_dict_id_plus1: Vec::new(),
                                bp_dict_id_plus1: Vec::new(),
                                s_dict_id_plus1: Vec::new(),
                                arg_present: Vec::new(),
                                arg_values: Vec::new(),
                            });
                            bucket.ts_offsets.push(ev.ts - time_base);
                            match ev.dur {
                                Some(d) => { bucket.dur_present.push(true); bucket.dur.push(d); }
                                None    => { bucket.dur_present.push(false); bucket.dur.push(0); }
                            }
                            match ev.id {
                                Some(i) => { bucket.id_present.push(true); bucket.ids.push(i); }
                                None    => { bucket.id_present.push(false); bucket.ids.push(0); }
                            }
                            bucket.cat_dict_id_plus1.push(ev.cat.as_ref().map(|c| dict.intern(c) + 1).unwrap_or(0));
                            bucket.bp_dict_id_plus1.push(ev.bp.as_ref().map(|b| dict.intern(b) + 1).unwrap_or(0));
                            bucket.s_dict_id_plus1.push(ev.s.as_ref().map(|s| dict.intern(s) + 1).unwrap_or(0));
                            let mut presence = Vec::with_capacity(arg_keys.len());
                            let mut values   = Vec::new();
                            for k in &arg_keys {
                                match ev.args.as_ref().and_then(|a| a.get(k)) {
                                    Some(v) => { presence.push(true); values.push(v.clone()); }
                                    None    => { presence.push(false); }
                                }
                            }
                            bucket.arg_present.push(presence);
                            bucket.arg_values.push(values);
                        }
                        // Stable order: sort by schema_id so decompress
                        // order is deterministic.
                        let mut bucket_vec: Vec<BucketData> = buckets.into_values().collect();
                        bucket_vec.sort_by_key(|b| b.schema_id);
                        streams.push(TraceZipStream {
                            rank_dict_id,
                            pid: *pid,
                            tid_dict_id,
                            ph: ph.0,
                            time_base: if time_base == i64::MAX { 0 } else { time_base },
                            buckets: bucket_vec,
                        });
                    }
                }
            }
        }

        let payload = TraceZipPayload {
            dict_strings: dict.into_strings(),
            schemas,
            streams,
        };
        let mut buf = Vec::new();
        rmp_serde::encode::write_named(&mut buf, &payload)?;
        let bytes = zstd::stream::encode_all(&buf[..], 3)?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }

    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let raw = zstd::stream::decode_all(bytes)?;
        let payload: TraceZipPayload = rmp_serde::from_slice(&raw)?;
        let dict = payload.dict_strings;
        let lookup = |id: u32| -> &str { dict.get(id as usize).map(|s| s.as_str()).unwrap_or("") };
        let lookup_opt = |id_plus1: u32| -> Option<String> {
            if id_plus1 == 0 { None } else { dict.get((id_plus1 - 1) as usize).cloned() }
        };
        let mut trace = Trace::empty();
        for stream in payload.streams {
            let mut events: Vec<Event> = Vec::new();
            for bucket in &stream.buckets {
                let schema = match payload.schemas.get(bucket.schema_id as usize) {
                    Some(s) => s,
                    None => continue,
                };
                let name = lookup(schema.name_dict_id).to_string();
                let arg_keys: Vec<String> = schema.arg_key_ids.iter()
                    .map(|id| lookup(*id).to_string()).collect();
                let n = bucket.ts_offsets.len();
                for i in 0..n {
                    let mut args = AHashMap::new();
                    if let Some(presence) = bucket.arg_present.get(i) {
                        let mut vi = 0usize;
                        for (k, &p) in arg_keys.iter().zip(presence.iter()) {
                            if p {
                                let v = bucket.arg_values.get(i)
                                    .and_then(|row| row.get(vi))
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                args.insert(k.clone(), v);
                                vi += 1;
                            }
                        }
                    }
                    let dur = if *bucket.dur_present.get(i).unwrap_or(&false) {
                        Some(*bucket.dur.get(i).unwrap_or(&0))
                    } else { None };
                    let id = if *bucket.id_present.get(i).unwrap_or(&false) {
                        Some(*bucket.ids.get(i).unwrap_or(&0))
                    } else { None };
                    let cat = lookup_opt(*bucket.cat_dict_id_plus1.get(i).unwrap_or(&0));
                    let bp  = lookup_opt(*bucket.bp_dict_id_plus1 .get(i).unwrap_or(&0));
                    let s   = lookup_opt(*bucket.s_dict_id_plus1  .get(i).unwrap_or(&0));
                    events.push(Event {
                        name: name.clone(),
                        ts: stream.time_base + bucket.ts_offsets[i],
                        dur, cat,
                        ph: Phase(stream.ph),
                        pid: stream.pid,
                        tid: lookup(stream.tid_dict_id).to_string(),
                        args: if args.is_empty() { None } else { Some(args) },
                        id, bp, s,
                    });
                }
            }
            trace.ranks
                .entry(lookup(stream.rank_dict_id).to_string()).or_default()
                .entry(stream.pid).or_default()
                .entry(lookup(stream.tid_dict_id).to_string()).or_default()
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
