//! TraceZip adaptation for AI traces (ICSE'25 / arXiv:2502.06318) —
//! **global bucket** version.
//!
//! TraceZip targets distributed tracing spans.  Adapted here:
//!
//! * Each event = one span.  `Event.name` is the SRT (Span Retrieval Tree) key.
//! * Global merging:
//!     - **Global string dict** — every distinct string (event name,
//!       cat, bp, s, arg-key) is interned ONCE across the whole trace.
//!     - **Global SRT schema** — `(name_id, sorted_arg_key_ids)` is
//!       interned globally so all ranks share one schema entry per
//!       `(name, arg-keys)` combination.
//!     - **Global buckets** — all events matching a schema are stored in
//!       ONE global bucket with a `stream_ids` column indicating which
//!       stream each event belongs to.  This enables per-schema
//!       aggregate queries (e.g. operator hotspot) without decompression.
//! * Per-stream metadata (rank, pid, tid, ph, time_base) stored separately.
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
use serde_json::Value;

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
    /// Stream metadata: one entry per `(rank, pid, tid, ph)`.
    streams: Vec<StreamMeta>,
    /// Global buckets: one per schema, containing ALL events across all
    /// streams that match this schema.
    global_buckets: Vec<GlobalBucket>,
}

#[derive(Serialize, Deserialize)]
struct SrtSchema {
    name_dict_id: u32,
    /// Sorted dict ids of args.* keys ever seen for this name across the
    /// whole trace.  Per-event presence is stored as a bitmap.
    arg_key_ids: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct StreamMeta {
    rank_dict_id: u32,
    pid: i64,
    tid_dict_id: u32,
    ph: u8,
    time_base: i64,
}

#[derive(Serialize, Deserialize)]
struct GlobalBucket {
    /// Index into the GLOBAL `schemas` table.
    schema_id: u32,
    /// Which stream each event belongs to.
    stream_ids: Vec<u32>,
    /// ts offset relative to the owning stream's time_base.
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
    /// `schemas[schema_id].arg_key_ids.len()`.
    arg_present: Vec<Vec<bool>>,
    arg_values: Vec<Vec<serde_json::Value>>,
}

impl BaselineCompressor for TraceZipCompressor {
    fn name(&self) -> &str { "tracezip" }

    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();

        let mut dict = StringDict::default();
        let mut schema_index: AHashMap<(u32, Vec<u32>), u32> = AHashMap::new();
        let mut schemas: Vec<SrtSchema> = Vec::new();

        // Pass 1: build global arg-key union per name.
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

        // Pre-intern global schemas.
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

        // Pass 2: build streams and global buckets.
        let mut streams: Vec<StreamMeta> = Vec::new();
        let mut stream_index: AHashMap<(u32, i64, u32, u8), u32> = AHashMap::new();
        let mut buckets: AHashMap<u32, GlobalBucket> = AHashMap::new();

        for (rank, processes) in &trace.ranks {
            let rank_dict_id = dict.intern(rank);
            for (pid, threads) in processes {
                for (tid, phases) in threads {
                    let tid_dict_id = dict.intern(tid);
                    for (ph, events) in phases {
                        if events.is_empty() { continue; }

                        // Get or create stream.
                        let stream_key = (rank_dict_id, *pid, tid_dict_id, ph.0);
                        let stream_id = if let Some(&id) = stream_index.get(&stream_key) {
                            id
                        } else {
                            let mut time_base = i64::MAX;
                            for ev in events { time_base = time_base.min(ev.ts); }
                            let id = streams.len() as u32;
                            streams.push(StreamMeta {
                                rank_dict_id,
                                pid: *pid,
                                tid_dict_id,
                                ph: ph.0,
                                time_base: if time_base == i64::MAX { 0 } else { time_base },
                            });
                            stream_index.insert(stream_key, id);
                            id
                        };
                        let time_base = streams[stream_id as usize].time_base;

                        for ev in events {
                            let schema_id = match name_to_schema_id.get(&ev.name) {
                                Some(&id) => id,
                                None => continue,
                            };

                            let arg_keys: Vec<String> = schemas[schema_id as usize].arg_key_ids
                                .iter().map(|id| dict.items[*id as usize].clone()).collect();

                            let bucket = buckets.entry(schema_id).or_insert_with(|| GlobalBucket {
                                schema_id,
                                stream_ids: Vec::new(),
                                ts_offsets: Vec::new(),
                                dur_present: Vec::new(), dur: Vec::new(),
                                id_present: Vec::new(), ids: Vec::new(),
                                cat_dict_id_plus1: Vec::new(),
                                bp_dict_id_plus1: Vec::new(),
                                s_dict_id_plus1: Vec::new(),
                                arg_present: Vec::new(),
                                arg_values: Vec::new(),
                            });

                            bucket.stream_ids.push(stream_id);
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
                            let mut values = Vec::new();
                            for k in &arg_keys {
                                match ev.args.as_ref().and_then(|a| a.get(k)) {
                                    Some(v) => { presence.push(true); values.push(v.clone()); }
                                    None    => { presence.push(false); }
                                }
                            }
                            bucket.arg_present.push(presence);
                            bucket.arg_values.push(values);
                        }
                    }
                }
            }
        }

        // Sort buckets by schema_id for deterministic output.
        let mut global_buckets: Vec<GlobalBucket> = buckets.into_values().collect();
        global_buckets.sort_by_key(|b| b.schema_id);

        let payload = TraceZipPayload {
            dict_strings: dict.into_strings(),
            schemas,
            streams,
            global_buckets,
        };
        let mut buf = Vec::new();
        rmp_serde::encode::write_named(&mut buf, &payload)?;
        let bytes = zstd::stream::encode_all(&buf[..], 3)?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }

    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let raw = zstd::stream::decode_all(bytes)?;
        let payload: TraceZipPayload = rmp_serde::from_slice(&raw)?;
        let dict = &payload.dict_strings;
        let lookup = |id: u32| -> &str { dict.get(id as usize).map(|s| s.as_str()).unwrap_or("") };
        let lookup_opt = |id_plus1: u32| -> Option<String> {
            if id_plus1 == 0 { None } else { dict.get((id_plus1 - 1) as usize).cloned() }
        };

        let mut trace = Trace::empty();
        for bucket in &payload.global_buckets {
            let schema = match payload.schemas.get(bucket.schema_id as usize) {
                Some(s) => s,
                None => continue,
            };
            let name = lookup(schema.name_dict_id).to_string();
            let arg_keys: Vec<String> = schema.arg_key_ids.iter()
                .map(|id| lookup(*id).to_string()).collect();

            let n = bucket.stream_ids.len();
            for i in 0..n {
                let stream_id = bucket.stream_ids[i] as usize;
                let stream = match payload.streams.get(stream_id) {
                    Some(s) => s,
                    None => continue,
                };

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
                let bp  = lookup_opt(*bucket.bp_dict_id_plus1.get(i).unwrap_or(&0));
                let s   = lookup_opt(*bucket.s_dict_id_plus1.get(i).unwrap_or(&0));

                let ev = Event {
                    name: name.clone(),
                    ts: stream.time_base + bucket.ts_offsets[i],
                    dur, cat,
                    ph: Phase(stream.ph),
                    pid: stream.pid,
                    tid: lookup(stream.tid_dict_id).to_string(),
                    args: if args.is_empty() { None } else { Some(args) },
                    id, bp, s,
                };

                trace.ranks
                    .entry(lookup(stream.rank_dict_id).to_string()).or_default()
                    .entry(stream.pid).or_default()
                    .entry(lookup(stream.tid_dict_id).to_string()).or_default()
                    .entry(Phase(stream.ph)).or_default()
                    .push(ev);
            }
        }
        Ok(trace)
    }

    fn supports_in_situ(&self, task: &str) -> bool {
        matches!(task, "operator_hotspot" | "rank_load_balance" | "gpu_bubble_rate")
    }

    fn run_in_situ(&self, bytes: &[u8], task: &str) -> Result<Value> {
        let raw = zstd::stream::decode_all(bytes)?;
        let payload: TraceZipPayload = rmp_serde::from_slice(&raw)?;
        match task {
            "operator_hotspot" => in_situ_operator_hotspot(&payload),
            "rank_load_balance" => in_situ_rank_load_balance(&payload),
            "gpu_bubble_rate" => in_situ_gpu_bubble_rate(&payload),
            _ => Err(crate::Error::Other(format!("unsupported in-situ task: {task}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// In-situ analysis implementations
// ---------------------------------------------------------------------------

fn in_situ_operator_hotspot(payload: &TraceZipPayload) -> Result<Value> {
    let dict = &payload.dict_strings;
    let mut tally: AHashMap<&str, i64> = AHashMap::new();
    for bucket in &payload.global_buckets {
        let schema = &payload.schemas[bucket.schema_id as usize];
        let name = dict[schema.name_dict_id as usize].as_str();
        let total: i64 = bucket.dur.iter()
            .zip(bucket.dur_present.iter())
            .filter(|(_, &p)| p)
            .map(|(&d, _)| d)
            .sum();
        *tally.entry(name).or_insert(0) += total;
    }
    let mut entries: Vec<(&str, i64)> = tally.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.truncate(20);
    let arr: Vec<Value> = entries.into_iter().map(|(name, total)| {
        serde_json::json!({"name": name, "total_dur_us": total})
    }).collect();
    Ok(Value::Array(arr))
}

fn in_situ_rank_load_balance(payload: &TraceZipPayload) -> Result<Value> {
    let dict = &payload.dict_strings;
    let mut compute: AHashMap<u32, i64> = AHashMap::new(); // rank_dict_id -> dur
    let mut comm: AHashMap<u32, i64> = AHashMap::new();

    for bucket in &payload.global_buckets {
        let schema = &payload.schemas[bucket.schema_id as usize];
        let name = dict[schema.name_dict_id as usize].as_str();
        // Check if this bucket's events are GPU kernels (cat == "kernel")
        // We check per-event since cat can vary within a bucket.
        let is_nccl = crate::analysis::kernel_class::is_nccl_kernel(name);
        let n = bucket.stream_ids.len();
        for i in 0..n {
            let cat_id = bucket.cat_dict_id_plus1[i];
            if cat_id == 0 { continue; }
            let cat = dict[(cat_id - 1) as usize].as_str();
            if cat != "kernel" { continue; }
            if !bucket.dur_present[i] { continue; }
            let dur = bucket.dur[i];
            let stream_id = bucket.stream_ids[i] as usize;
            let rank_dict_id = payload.streams[stream_id].rank_dict_id;
            if is_nccl {
                *comm.entry(rank_dict_id).or_insert(0) += dur;
            } else {
                *compute.entry(rank_dict_id).or_insert(0) += dur;
            }
        }
    }

    // Convert to named map
    let mut compute_named: AHashMap<String, i64> = AHashMap::new();
    let mut comm_named: AHashMap<String, i64> = AHashMap::new();
    for (rank_id, dur) in compute {
        compute_named.insert(dict[rank_id as usize].clone(), dur);
    }
    for (rank_id, dur) in comm {
        comm_named.insert(dict[rank_id as usize].clone(), dur);
    }
    Ok(load_balance_json(&compute_named, &comm_named))
}

fn in_situ_gpu_bubble_rate(payload: &TraceZipPayload) -> Result<Value> {
    let dict = &payload.dict_strings;
    // Collect per-rank GPU kernel (ts, dur) events.
    let mut per_rank: AHashMap<u32, Vec<(i64, i64)>> = AHashMap::new();

    for bucket in &payload.global_buckets {
        let n = bucket.stream_ids.len();
        for i in 0..n {
            let cat_id = bucket.cat_dict_id_plus1[i];
            if cat_id == 0 { continue; }
            let cat = dict[(cat_id - 1) as usize].as_str();
            if cat != "kernel" { continue; }
            if !bucket.dur_present[i] { continue; }
            let stream_id = bucket.stream_ids[i] as usize;
            let stream = &payload.streams[stream_id];
            let ts = stream.time_base + bucket.ts_offsets[i];
            let dur = bucket.dur[i];
            per_rank.entry(stream.rank_dict_id).or_default().push((ts, dur));
        }
    }

    // Compute bubble rate per rank.
    let mut results: Vec<Value> = Vec::new();
    let mut ranks: Vec<u32> = per_rank.keys().copied().collect();
    ranks.sort();
    for rank_id in ranks {
        let events = per_rank.get_mut(&rank_id).unwrap();
        events.sort_unstable();
        let rank_name = dict[rank_id as usize].as_str();
        if events.is_empty() {
            results.push(serde_json::json!({
                "rank": rank_name, "bubble_rate": 0.0,
                "busy_us": 0, "total_span_us": 0, "kernel_count": 0,
            }));
            continue;
        }
        let first_ts = events[0].0;
        let mut last_end: i64 = first_ts;
        let mut busy_window_end = first_ts;
        let mut busy_total: i64 = 0;
        for &(ts, dur) in events.iter() {
            let end = ts + dur;
            if end > last_end { last_end = end; }
            if ts > busy_window_end {
                busy_total += dur;
                busy_window_end = end;
            } else if end > busy_window_end {
                busy_total += end - busy_window_end;
                busy_window_end = end;
            }
        }
        let total_span = last_end - first_ts;
        let bubble_rate = if total_span > 0 {
            1.0 - busy_total as f64 / total_span as f64
        } else { 0.0 };
        results.push(serde_json::json!({
            "rank": rank_name,
            "bubble_rate": bubble_rate,
            "busy_us": busy_total,
            "total_span_us": total_span,
            "kernel_count": events.len(),
        }));
    }
    Ok(Value::Array(results))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_balance_json(compute: &AHashMap<String, i64>, comm: &AHashMap<String, i64>) -> Value {
    let mut ranks: Vec<&String> = compute.keys().chain(comm.keys()).collect();
    ranks.sort();
    ranks.dedup();
    let compute_vals: Vec<i64> = ranks.iter().map(|r| *compute.get(*r).unwrap_or(&0)).collect();
    let comm_vals: Vec<i64> = ranks.iter().map(|r| *comm.get(*r).unwrap_or(&0)).collect();
    let rank_names: Vec<&str> = ranks.iter().map(|r| r.as_str()).collect();
    serde_json::json!({
        "ranks": rank_names,
        "compute_busy_us": compute_vals,
        "comm_busy_us": comm_vals,
        "compute": metric_summary(&compute_vals),
        "comm": metric_summary(&comm_vals),
    })
}

fn metric_summary(values: &[i64]) -> Value {
    if values.is_empty() {
        return serde_json::json!({"max_us":0,"min_us":0,"mean_us":0.0,"stddev_us":0.0,"cv":0.0,"imbalance_max_min_over_mean":0.0});
    }
    let max_v = *values.iter().max().unwrap();
    let min_v = *values.iter().min().unwrap();
    let n = values.len() as f64;
    let mean = values.iter().sum::<i64>() as f64 / n;
    let var = values.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    serde_json::json!({
        "max_us": max_v, "min_us": min_v, "mean_us": mean,
        "stddev_us": stddev,
        "cv": if mean > 0.0 { stddev / mean } else { 0.0 },
        "imbalance_max_min_over_mean": if mean > 0.0 { (max_v - min_v) as f64 / mean } else { 0.0 },
    })
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
