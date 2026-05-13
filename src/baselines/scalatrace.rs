//! ScalaTrace adaptation for AI traces.
//!
//! The original ScalaTrace targets MPI traces using Regular Section
//! Descriptors (RSD/PRSD).  Adapted here:
//!
//! * Per-rank, per-stream event sequence is the input.
//! * Each event's "type" is `(normalized_name, cat, sorted_arg_keys)`.
//! * RSD = `(pattern, repeats)` describing a repeating sub-sequence.
//!   The greedy encoder runs once per stream over the type-id sequence.
//! * Per-event scalar payloads (`name`, `ts`, `dur`, `id`, `bp`, `s`,
//!   `args`) are stored as parallel arrays so the encoder is bit-exact
//!   even when the values vary across events sharing a structural
//!   pattern.
//!
//! Lossless: every `Event` field round-trips exactly.

use crate::baselines::{BaselineCompressor, CompressArtifact};
use crate::event::{Event, Phase};
use crate::trace::Trace;
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct ScalaTraceCompressor;

#[derive(Serialize, Deserialize)]
struct ScalaTracePayload {
    streams: Vec<StreamPayload>,
}

#[derive(Serialize, Deserialize)]
struct StreamPayload {
    rank: String,
    pid: i64,
    tid: String,
    ph: u8,
    /// Type table: type_id -> (normalized_name, cat?, arg_keys[]).
    types: Vec<TypeEntry>,
    /// RSDs encoding the type-id sequence (structural compression).
    /// The flattened sequence is reconstructible from `rsds`.
    rsds: Vec<Rsd>,
    /// Per-event scalars in original sequence order; len == flattened RSD length.
    payload_names: Vec<String>,
    ts: Vec<i64>,
    dur_present: Vec<bool>,
    dur: Vec<i64>,
    id_present: Vec<bool>,
    ids: Vec<i64>,
    /// Optional bp / s as dict-id+1 (0 = absent).  We share the type
    /// table's name dict for these too, but we use a per-stream
    /// auxiliary dict for simplicity.
    aux_strings: Vec<String>,
    bp_dict_id_plus1: Vec<u32>,
    s_dict_id_plus1: Vec<u32>,
    /// Per-event args matching the *event's type's* `arg_keys` order.
    /// The actual JSON values are stored verbatim per event since they
    /// vary; the keys come from `types[type_id].arg_keys`.
    args: Vec<Vec<serde_json::Value>>,
}

#[derive(Serialize, Deserialize)]
struct TypeEntry {
    name: String,
    cat: Option<String>,
    arg_keys: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Rsd {
    /// Repeating sub-pattern (type ids).
    pattern: Vec<u32>,
    /// Number of times the pattern repeats consecutively.
    repeats: u32,
}

impl BaselineCompressor for ScalaTraceCompressor {
    fn name(&self) -> &str { "scalatrace" }

    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();
        let mut streams: Vec<StreamPayload> = Vec::new();
        for (rank, processes) in &trace.ranks {
            for (pid, threads) in processes {
                for (tid, phases) in threads {
                    for (ph, events) in phases {
                        if events.is_empty() { continue; }
                        let mut types: Vec<TypeEntry> = Vec::new();
                        let mut type_index: ahash::AHashMap<(String, Option<String>, Vec<String>), u32> = ahash::AHashMap::new();
                        let mut sequence: Vec<u32> = Vec::with_capacity(events.len());
                        let mut payload_names: Vec<String> = Vec::with_capacity(events.len());
                        let mut ts: Vec<i64> = Vec::with_capacity(events.len());
                        let mut dur_present: Vec<bool> = Vec::with_capacity(events.len());
                        let mut dur: Vec<i64> = Vec::with_capacity(events.len());
                        let mut id_present: Vec<bool> = Vec::with_capacity(events.len());
                        let mut ids: Vec<i64> = Vec::with_capacity(events.len());

                        let mut aux_dict = StringDict::default();
                        let mut bp_dict_id_plus1 = Vec::with_capacity(events.len());
                        let mut s_dict_id_plus1  = Vec::with_capacity(events.len());

                        let mut args_payload: Vec<Vec<serde_json::Value>> = Vec::with_capacity(events.len());
                        for ev in events {
                            let normalized = crate::utils::normalize_name(&ev.name);
                            let mut arg_keys: Vec<String> = ev.args.as_ref()
                                .map(|a| a.keys().cloned().collect())
                                .unwrap_or_default();
                            arg_keys.sort();
                            let key = (normalized.clone(), ev.cat.clone(), arg_keys.clone());
                            let tid_id = if let Some(&id) = type_index.get(&key) {
                                id
                            } else {
                                let id = types.len() as u32;
                                types.push(TypeEntry {
                                    name: normalized.clone(),
                                    cat: ev.cat.clone(),
                                    arg_keys: arg_keys.clone(),
                                });
                                type_index.insert(key, id);
                                id
                            };
                            sequence.push(tid_id);
                            payload_names.push(ev.name.clone());
                            ts.push(ev.ts);
                            match ev.dur {
                                Some(d) => { dur_present.push(true); dur.push(d); },
                                None    => { dur_present.push(false); dur.push(0); },
                            }
                            match ev.id {
                                Some(i) => { id_present.push(true); ids.push(i); },
                                None    => { id_present.push(false); ids.push(0); },
                            }
                            bp_dict_id_plus1.push(ev.bp.as_ref().map(|b| aux_dict.intern(b) + 1).unwrap_or(0));
                            s_dict_id_plus1 .push(ev.s .as_ref().map(|s| aux_dict.intern(s) + 1).unwrap_or(0));
                            let row: Vec<serde_json::Value> = arg_keys.iter()
                                .map(|k| ev.args.as_ref().and_then(|a| a.get(k.as_str())).cloned().unwrap_or(serde_json::Value::Null))
                                .collect();
                            args_payload.push(row);
                        }

                        let rsds = encode_rsd(&sequence);
                        streams.push(StreamPayload {
                            rank: rank.clone(),
                            pid: *pid,
                            tid: tid.clone(),
                            ph: ph.0,
                            types,
                            rsds,
                            payload_names,
                            ts,
                            dur_present, dur,
                            id_present, ids,
                            aux_strings: aux_dict.into_strings(),
                            bp_dict_id_plus1,
                            s_dict_id_plus1,
                            args: args_payload,
                        });
                    }
                }
            }
        }
        let mut buf = Vec::new();
        rmp_serde::encode::write_named(&mut buf, &ScalaTracePayload { streams })?;
        let bytes = zstd::stream::encode_all(&buf[..], 3)?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }

    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let raw = zstd::stream::decode_all(bytes)?;
        let payload: ScalaTracePayload = rmp_serde::from_slice(&raw)?;
        let mut trace = Trace::empty();
        for stream in payload.streams {
            // Re-expand the type-id sequence from the RSDs.
            let sequence: Vec<u32> = expand_rsd(&stream.rsds);
            let lookup = |id_plus1: u32| -> Option<String> {
                if id_plus1 == 0 { None } else { stream.aux_strings.get((id_plus1 - 1) as usize).cloned() }
            };
            let mut events = Vec::with_capacity(stream.payload_names.len());
            for (i, name) in stream.payload_names.iter().enumerate() {
                let type_id = sequence.get(i).copied().unwrap_or(0) as usize;
                let ty = stream.types.get(type_id);
                let cat = ty.and_then(|t| t.cat.clone());
                let arg_keys: &[String] = ty.map(|t| t.arg_keys.as_slice()).unwrap_or_default();

                let mut args = ahash::AHashMap::new();
                if let Some(row) = stream.args.get(i) {
                    for (k, v) in arg_keys.iter().zip(row.iter()) {
                        args.insert(k.clone(), v.clone());
                    }
                }
                let dur = if *stream.dur_present.get(i).unwrap_or(&false) {
                    Some(*stream.dur.get(i).unwrap_or(&0))
                } else { None };
                let id = if *stream.id_present.get(i).unwrap_or(&false) {
                    Some(*stream.ids.get(i).unwrap_or(&0))
                } else { None };
                let bp = lookup(*stream.bp_dict_id_plus1.get(i).unwrap_or(&0));
                let s  = lookup(*stream.s_dict_id_plus1 .get(i).unwrap_or(&0));
                events.push(Event {
                    name: name.clone(),
                    ts: stream.ts.get(i).copied().unwrap_or(0),
                    dur,
                    cat,
                    ph: Phase(stream.ph),
                    pid: stream.pid,
                    tid: stream.tid.clone(),
                    args: if args.is_empty() { None } else { Some(args) },
                    id, bp, s,
                });
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

/// Greedy RSD encoder: scan once, detect `(period, repeats)` runs.
fn encode_rsd(sequence: &[u32]) -> Vec<Rsd> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < sequence.len() {
        let mut best: Option<(usize, u32)> = None;
        for period in 1..=((sequence.len() - i) / 2).min(64) {
            let pattern = &sequence[i..i + period];
            let mut repeats: u32 = 1;
            let mut j = i + period;
            while j + period <= sequence.len() && &sequence[j..j + period] == pattern {
                repeats += 1;
                j += period;
            }
            if repeats > 1 {
                if best.map(|(p, r)| (period * (repeats as usize)).cmp(&(p * (r as usize)))).unwrap_or(std::cmp::Ordering::Greater)
                    == std::cmp::Ordering::Greater
                {
                    best = Some((period, repeats));
                }
            }
        }
        let (period, repeats) = best.unwrap_or((1, 1));
        out.push(Rsd { pattern: sequence[i..i + period].to_vec(), repeats });
        i += period * (repeats as usize);
    }
    out
}

fn expand_rsd(rsds: &[Rsd]) -> Vec<u32> {
    let mut out = Vec::new();
    for r in rsds {
        for _ in 0..r.repeats {
            out.extend_from_slice(&r.pattern);
        }
    }
    out
}

#[derive(Default)]
struct StringDict {
    index: ahash::AHashMap<String, u32>,
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
