//! ScalaTrace adaptation for AI traces — **cross-rank** version.
//!
//! The original ScalaTrace targets MPI traces using Regular Section
//! Descriptors (RSD/PRSD).  Adapted here:
//!
//! * Each `(rank, pid, tid, ph)` stream contributes its own type-id
//!   sequence, encoded as a list of RSDs.
//! * Cross-rank merging:
//!     - **Global type table** — every distinct `(normalized_name,
//!       cat, sorted_arg_keys)` triple is interned ONCE across the
//!       whole trace.  In practice that means a 1024-rank training
//!       trace shares one entry per kernel name, not 1024 copies.
//!     - **Global RSD pool** — every distinct `(pattern, repeats)` is
//!       interned once.  Same outer-loop-step pattern in different
//!       ranks now points to the same RSD id.  Streams store a
//!       `Vec<u32>` of RSD ids.
//!     - **Global string dict** — name strings, `bp`, `s`, and the
//!       arg-key vectors are dict-encoded against a shared dict.
//! * Per-event scalar payloads (`name`, `ts`, `dur`, `id`, `bp`, `s`,
//!   `args` values) stay per-stream because they vary across events
//!   that share a structural pattern.  zstd then squashes the
//!   resulting columns.
//!
//! Lossless: every `Event` field round-trips exactly.

use crate::baselines::{BaselineCompressor, CompressArtifact};
use crate::event::{Event, Phase};
use crate::trace::Trace;
use crate::Result;
use ahash::AHashMap;
use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct ScalaTraceCompressor;

#[derive(Serialize, Deserialize)]
struct ScalaTracePayload {
    /// Global string dict: name strings (event names + cats + bp/s
    /// + arg keys) are all interned here so cross-rank repetition
    /// becomes a single u32.
    dict: Vec<String>,
    /// Global type table.  Indexed by `type_id`.
    types: Vec<TypeEntry>,
    /// Global RSD pool.  Streams reference RSDs by index.
    rsds: Vec<Rsd>,
    streams: Vec<StreamPayload>,
}

#[derive(Serialize, Deserialize)]
struct TypeEntry {
    /// Dict id of the normalized event name.
    name_id: u32,
    /// Dict id+1 of the cat (0 = absent).
    cat_id_plus1: u32,
    /// Dict ids of arg keys, sorted, deduped.
    arg_key_ids: Vec<u32>,
}

#[derive(Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
struct Rsd {
    /// Repeating sub-pattern (type ids).
    pattern: Vec<u32>,
    /// Number of times the pattern repeats consecutively.
    repeats: u32,
}

#[derive(Serialize, Deserialize)]
struct StreamPayload {
    /// Dict id of `rank`.  rank strings are interned the same way as
    /// op names — multiple streams within one rank share an id.
    rank_id: u32,
    pid: i64,
    /// Dict id of `tid`.
    tid_id: u32,
    ph: u8,
    /// Indices into the global `rsds` pool, in stream order.  Expanded
    /// they yield this stream's full type-id sequence (parallel to
    /// `payload_*` arrays below).
    rsd_ids: Vec<u32>,
    /// Per-event scalars in original sequence order; len == flattened
    /// RSD length.
    /// Names are stored verbatim because they keep digit fillers
    /// (e.g. `layers.5.attn`) — but each name's distinct strings are
    /// interned through `dict`, so cross-rank duplicates collapse.
    payload_name_ids: Vec<u32>,
    ts: Vec<i64>,
    dur_present: Vec<bool>,
    dur: Vec<i64>,
    id_present: Vec<bool>,
    ids: Vec<i64>,
    /// 0 = absent, otherwise dict_id+1 (refers to the GLOBAL `dict`).
    bp_dict_id_plus1: Vec<u32>,
    s_dict_id_plus1: Vec<u32>,
    /// Per-event args matching the *event's type's* `arg_key_ids`
    /// order (looked up via `types[rsd-expanded type_id]`).  Values
    /// vary across events so they stay verbatim.
    args: Vec<Vec<serde_json::Value>>,
}

impl BaselineCompressor for ScalaTraceCompressor {
    fn name(&self) -> &str { "scalatrace" }

    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();

        let mut dict = StringDict::default();
        // Per-stream type-id sequences first; we then RSD-encode them
        // and intern the resulting RSDs into a global pool.
        struct StreamPre {
            rank_id: u32,
            pid: i64,
            tid_id: u32,
            ph: u8,
            sequence: Vec<u32>,
            payload_name_ids: Vec<u32>,
            ts: Vec<i64>,
            dur_present: Vec<bool>,
            dur: Vec<i64>,
            id_present: Vec<bool>,
            ids: Vec<i64>,
            bp_dict_id_plus1: Vec<u32>,
            s_dict_id_plus1: Vec<u32>,
            args: Vec<Vec<serde_json::Value>>,
        }
        let mut prelim: Vec<StreamPre> = Vec::new();

        // Global type interning: `(name_id, cat_id_plus1, arg_key_ids)` tuple.
        // `arg_key_ids` is a Vec<u32>, so we hash by string form for stability.
        let mut type_index: AHashMap<(u32, u32, Vec<u32>), u32> = AHashMap::new();
        let mut types: Vec<TypeEntry> = Vec::new();

        for (rank, processes) in &trace.ranks {
            let rank_id = dict.intern(rank);
            for (pid, threads) in processes {
                for (tid, phases) in threads {
                    let tid_id = dict.intern(tid);
                    for (ph, events) in phases {
                        if events.is_empty() { continue; }
                        let mut sequence: Vec<u32> = Vec::with_capacity(events.len());
                        let mut payload_name_ids: Vec<u32> = Vec::with_capacity(events.len());
                        let mut ts: Vec<i64> = Vec::with_capacity(events.len());
                        let mut dur_present: Vec<bool> = Vec::with_capacity(events.len());
                        let mut dur: Vec<i64> = Vec::with_capacity(events.len());
                        let mut id_present: Vec<bool> = Vec::with_capacity(events.len());
                        let mut ids: Vec<i64> = Vec::with_capacity(events.len());
                        let mut bp_dict_id_plus1: Vec<u32> = Vec::with_capacity(events.len());
                        let mut s_dict_id_plus1: Vec<u32> = Vec::with_capacity(events.len());
                        let mut args_payload: Vec<Vec<serde_json::Value>> = Vec::with_capacity(events.len());

                        for ev in events {
                            let normalized = crate::utils::normalize_name(&ev.name);
                            let name_id = dict.intern(&normalized);
                            let cat_id_plus1 = ev.cat.as_ref().map(|c| dict.intern(c) + 1).unwrap_or(0);
                            let mut arg_key_ids: Vec<u32> = ev.args.as_ref()
                                .map(|a| a.keys().map(|k| dict.intern(k)).collect())
                                .unwrap_or_default();
                            arg_key_ids.sort();
                            // Re-derive arg_keys (string form) in stable order — needed to pull
                            // the row's args in the same order as arg_key_ids.
                            let arg_keys: Vec<String> = arg_key_ids.iter()
                                .map(|id| dict.items[*id as usize].clone())
                                .collect();

                            let key = (name_id, cat_id_plus1, arg_key_ids.clone());
                            let type_id = if let Some(&id) = type_index.get(&key) {
                                id
                            } else {
                                let id = types.len() as u32;
                                types.push(TypeEntry {
                                    name_id,
                                    cat_id_plus1,
                                    arg_key_ids,
                                });
                                type_index.insert(key, id);
                                id
                            };
                            sequence.push(type_id);
                            payload_name_ids.push(dict.intern(&ev.name));
                            ts.push(ev.ts);
                            match ev.dur {
                                Some(d) => { dur_present.push(true); dur.push(d); }
                                None    => { dur_present.push(false); dur.push(0); }
                            }
                            match ev.id {
                                Some(i) => { id_present.push(true); ids.push(i); }
                                None    => { id_present.push(false); ids.push(0); }
                            }
                            bp_dict_id_plus1.push(ev.bp.as_ref().map(|b| dict.intern(b) + 1).unwrap_or(0));
                            s_dict_id_plus1.push(ev.s.as_ref().map(|s| dict.intern(s) + 1).unwrap_or(0));
                            let row: Vec<serde_json::Value> = arg_keys.iter()
                                .map(|k| ev.args.as_ref().and_then(|a| a.get(k.as_str())).cloned().unwrap_or(serde_json::Value::Null))
                                .collect();
                            args_payload.push(row);
                        }

                        prelim.push(StreamPre {
                            rank_id,
                            pid: *pid,
                            tid_id,
                            ph: ph.0,
                            sequence,
                            payload_name_ids,
                            ts,
                            dur_present, dur,
                            id_present, ids,
                            bp_dict_id_plus1,
                            s_dict_id_plus1,
                            args: args_payload,
                        });
                    }
                }
            }
        }

        // Cross-rank RSD pool.  Each stream's sequence is RSD-encoded
        // independently; the resulting RSDs are interned globally so
        // the same `(pattern, repeats)` shared by N streams costs one
        // entry plus N small refs.
        let mut rsd_index: AHashMap<Rsd, u32> = AHashMap::new();
        let mut rsds_global: Vec<Rsd> = Vec::new();
        let mut streams: Vec<StreamPayload> = Vec::with_capacity(prelim.len());
        for s in prelim {
            let local_rsds = encode_rsd(&s.sequence);
            let mut rsd_ids: Vec<u32> = Vec::with_capacity(local_rsds.len());
            for r in local_rsds {
                let id = if let Some(&id) = rsd_index.get(&r) {
                    id
                } else {
                    let id = rsds_global.len() as u32;
                    rsds_global.push(r.clone());
                    rsd_index.insert(r, id);
                    id
                };
                rsd_ids.push(id);
            }
            streams.push(StreamPayload {
                rank_id: s.rank_id,
                pid: s.pid,
                tid_id: s.tid_id,
                ph: s.ph,
                rsd_ids,
                payload_name_ids: s.payload_name_ids,
                ts: s.ts,
                dur_present: s.dur_present,
                dur: s.dur,
                id_present: s.id_present,
                ids: s.ids,
                bp_dict_id_plus1: s.bp_dict_id_plus1,
                s_dict_id_plus1: s.s_dict_id_plus1,
                args: s.args,
            });
        }

        let payload = ScalaTracePayload {
            dict: dict.into_strings(),
            types,
            rsds: rsds_global,
            streams,
        };
        let mut buf = Vec::new();
        rmp_serde::encode::write_named(&mut buf, &payload)?;
        let bytes = zstd::stream::encode_all(&buf[..], 3)?;
        Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
    }

    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let raw = zstd::stream::decode_all(bytes)?;
        let payload: ScalaTracePayload = rmp_serde::from_slice(&raw)?;
        let dict = payload.dict;
        let lookup = |id: u32| -> &str { dict.get(id as usize).map(|s| s.as_str()).unwrap_or("") };
        let lookup_opt = |id_plus1: u32| -> Option<String> {
            if id_plus1 == 0 { None } else { dict.get((id_plus1 - 1) as usize).cloned() }
        };
        let mut trace = Trace::empty();
        for stream in payload.streams {
            // Re-expand the type-id sequence: concatenate every referenced RSD.
            let mut sequence: Vec<u32> = Vec::new();
            for rid in &stream.rsd_ids {
                let r = payload.rsds.get(*rid as usize)
                    .ok_or_else(|| crate::Error::Other(format!("scalatrace decode: rsd id {rid} out of bounds")))?;
                for _ in 0..r.repeats {
                    sequence.extend_from_slice(&r.pattern);
                }
            }
            let mut events = Vec::with_capacity(stream.payload_name_ids.len());
            for (i, name_id) in stream.payload_name_ids.iter().enumerate() {
                let type_id = sequence.get(i).copied().unwrap_or(0) as usize;
                let ty = payload.types.get(type_id);
                let cat = ty.and_then(|t| {
                    if t.cat_id_plus1 == 0 { None }
                    else { Some(lookup(t.cat_id_plus1 - 1).to_string()) }
                });
                let arg_keys: Vec<String> = ty
                    .map(|t| t.arg_key_ids.iter().map(|id| lookup(*id).to_string()).collect())
                    .unwrap_or_default();
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
                let bp = lookup_opt(*stream.bp_dict_id_plus1.get(i).unwrap_or(&0));
                let s  = lookup_opt(*stream.s_dict_id_plus1 .get(i).unwrap_or(&0));
                events.push(Event {
                    name: lookup(*name_id).to_string(),
                    ts: stream.ts.get(i).copied().unwrap_or(0),
                    dur,
                    cat,
                    ph: Phase(stream.ph),
                    pid: stream.pid,
                    tid: lookup(stream.tid_id).to_string(),
                    args: if args.is_empty() { None } else { Some(args) },
                    id, bp, s,
                });
            }
            trace.ranks
                .entry(lookup(stream.rank_id).to_string()).or_default()
                .entry(stream.pid).or_default()
                .entry(lookup(stream.tid_id).to_string()).or_default()
                .entry(Phase(stream.ph)).or_default()
                .extend(events);
        }
        Ok(trace)
    }
}

/// Greedy RSD encoder: scan once, detect `(period, repeats)` runs.
/// Period bounded to 64 to keep the inner loop O(n·p) tractable on
/// sequences with millions of events.
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
            if repeats > 1
                && best.map(|(p, r)| (period * (repeats as usize)).cmp(&(p * (r as usize))))
                    .unwrap_or(std::cmp::Ordering::Greater) == std::cmp::Ordering::Greater
            {
                best = Some((period, repeats));
            }
        }
        let (period, repeats) = best.unwrap_or((1, 1));
        out.push(Rsd { pattern: sequence[i..i + period].to_vec(), repeats });
        i += period * (repeats as usize);
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
