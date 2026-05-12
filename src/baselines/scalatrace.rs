//! ScalaTrace adaptation for AI traces.
//!
//! The original ScalaTrace targets MPI traces using Regular Section
//! Descriptors (RSD/PRSD).  Adapted here:
//!
//! * Per-rank, per-stream event sequence is the input.
//! * Each event's "type" is `(normalized_name, cat, sorted_arg_keys)`.
//! * RSD = `(start_index, length, period, type)` describing a repeating
//!   sub-sequence of length `period` that occurs `length` times consecutively.
//! * Per-event scalar payloads (`ts`, `dur`, `id`, `args.values`) are
//!   stored as parallel arrays so RSD can encode the structure even when
//!   the values vary.
//!
//! Note: this baseline deliberately does **not** use PADOC's structural /
//! anchor / SLP machinery.  Its purpose is fair compression-ratio
//! comparison on AI traces.

use crate::baselines::{BaselineCompressor, CompressArtifact};
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
    /// Type table: type_id -> (normalized_name, cat?, arg_keys[])
    types: Vec<TypeEntry>,
    /// Encoded RSDs.
    rsds: Vec<Rsd>,
    /// Per-event scalars in original sequence order (parallel to flattened RSD reconstruction).
    /// `payload_names[i]` is the original event name (before digit normalisation).
    payload_names: Vec<String>,
    ts: Vec<i64>,
    dur: Vec<i64>,
    ids: Vec<i64>,
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
                        // Build type table.
                        let mut types: Vec<TypeEntry> = Vec::new();
                        let mut type_index: ahash::AHashMap<(String, Option<String>, Vec<String>), u32> = ahash::AHashMap::new();
                        let mut sequence: Vec<u32> = Vec::with_capacity(events.len());
                        let mut payload_names: Vec<String> = Vec::with_capacity(events.len());
                        let mut ts: Vec<i64> = Vec::with_capacity(events.len());
                        let mut dur: Vec<i64> = Vec::with_capacity(events.len());
                        let mut ids: Vec<i64> = Vec::with_capacity(events.len());
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
                            dur.push(ev.dur.unwrap_or(0));
                            ids.push(ev.id.unwrap_or(0));
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
                            dur,
                            ids,
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
            // Reconstruct sequence from RSDs (we stored the per-event payload in order,
            // so the RSDs only describe the structural compression — unused here for
            // decode fidelity but kept on disk for ratio-honest measurement).
            let _ = stream.rsds;
            let mut events = Vec::with_capacity(stream.payload_names.len());
            for (i, name) in stream.payload_names.iter().enumerate() {
                let mut args = ahash::AHashMap::new();
                let arg_keys: &[String] = stream.types.first().map(|t| t.arg_keys.as_slice()).unwrap_or_default();
                if let Some(row) = stream.args.get(i) {
                    for (k, v) in arg_keys.iter().zip(row.iter()) {
                        args.insert(k.clone(), v.clone());
                    }
                }
                events.push(crate::event::Event {
                    name: name.clone(),
                    ts: stream.ts.get(i).copied().unwrap_or(0),
                    dur: stream.dur.get(i).copied(),
                    cat: None,
                    ph: crate::event::Phase(stream.ph),
                    pid: stream.pid,
                    tid: stream.tid.clone(),
                    args: if args.is_empty() { None } else { Some(args) },
                    id: stream.ids.get(i).copied(),
                    bp: None,
                    s: None,
                });
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
