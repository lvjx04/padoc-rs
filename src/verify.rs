//! Lossless verification — compares two `Trace`s field-by-field.
//!
//! The compressor pipeline is supposed to be lossless.  Bench harnesses and
//! tests use [`compare_traces`] to confirm round-trip equivalence.
//!
//! The comparison ignores:
//!
//! * stream insertion order — the original chrome-trace JSON has no fixed
//!   per-stream order;
//! * within-stream ordering — events with equal `ts` may swap;
//! * "empty" `args` vs missing `args` — chrome-trace allows either.
//!
//! Everything else (event count, name, cat, ts, dur, ph, pid, tid, args,
//! id, bp, s) must match exactly.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::event::Event;
use crate::trace::Trace;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct VerifyReport {
    pub original_event_count: usize,
    pub reconstructed_event_count: usize,
    pub matching_events: usize,
    pub mismatched_events: usize,
    pub missing_streams: Vec<String>,
    pub extra_streams: Vec<String>,
    pub first_mismatches: Vec<String>,
    /// Streams whose event counts don't match — useful to spot where events
    /// went missing during round-trip.
    pub stream_count_diffs: Vec<String>,
}

impl VerifyReport {
    pub fn is_ok(&self) -> bool {
        self.mismatched_events == 0
            && self.missing_streams.is_empty()
            && self.extra_streams.is_empty()
            && self.original_event_count == self.reconstructed_event_count
    }
}

/// Compare two traces.  `original` is the raw chrome-trace; `reconstructed`
/// is what comes back out of `padoc::decompress(...)`.
pub fn compare_traces(original: &Trace, reconstructed: &Trace) -> VerifyReport {
    let mut report = VerifyReport::default();
    report.original_event_count = original.event_count();
    report.reconstructed_event_count = reconstructed.event_count();

    let orig_streams = collect_streams(original);
    let recon_streams = collect_streams(reconstructed);

    for key in orig_streams.keys() {
        if !recon_streams.contains_key(key) {
            report.missing_streams.push(key.clone());
        }
    }
    for key in recon_streams.keys() {
        if !orig_streams.contains_key(key) {
            report.extra_streams.push(key.clone());
        }
    }

    for (key, orig_events) in &orig_streams {
        let recon_events = match recon_streams.get(key) {
            Some(v) => v,
            None => continue,
        };
        if orig_events.len() != recon_events.len() && report.stream_count_diffs.len() < 10 {
            report
                .stream_count_diffs
                .push(format!("{} : orig={} recon={}", key, orig_events.len(), recon_events.len()));
        }
        compare_stream(key, orig_events, recon_events, &mut report);
    }
    report
}

fn collect_streams(trace: &Trace) -> AHashMap<String, Vec<Event>> {
    let mut out: AHashMap<String, Vec<Event>> = AHashMap::new();
    for (rank, pid_map) in &trace.ranks {
        for (pid, tid_map) in pid_map {
            for (tid, ph_map) in tid_map {
                for (ph, events) in ph_map {
                    let key = format!("{}|{}|{}|{}", rank, pid, tid, ph.0 as char);
                    out.entry(key).or_default().extend(events.iter().cloned());
                }
            }
        }
    }
    out
}

/// Multiset-based stream comparison: chrome-trace makes no guarantees about
/// the order of events that share `(ts, name, dur, ph)`, so any order-sensitive
/// comparison would flag spurious mismatches between two genuinely equivalent
/// streams.  We compute a stable per-event fingerprint (every observable field,
/// args sorted by key for stability) and compare the resulting `fingerprint ->
/// count` maps.
fn compare_stream(key: &str, orig: &[Event], recon: &[Event], report: &mut VerifyReport) {
    let mut orig_counts: AHashMap<String, i64> = AHashMap::new();
    for e in orig {
        *orig_counts.entry(event_fingerprint(e)).or_insert(0) += 1;
    }
    let mut recon_counts: AHashMap<String, i64> = AHashMap::new();
    for e in recon {
        *recon_counts.entry(event_fingerprint(e)).or_insert(0) += 1;
    }

    // Walk the union of fingerprints; anything where orig and recon counts
    // disagree is reported as a mismatch.  `matching_events` is the size of
    // the multiset intersection.
    let mut keys: AHashMap<&String, ()> = AHashMap::new();
    for k in orig_counts.keys() { keys.insert(k, ()); }
    for k in recon_counts.keys() { keys.insert(k, ()); }

    for fp in keys.keys() {
        let oc = orig_counts.get(*fp).copied().unwrap_or(0);
        let rc = recon_counts.get(*fp).copied().unwrap_or(0);
        let matched = oc.min(rc);
        report.matching_events += matched as usize;
        let diff = (oc - rc).abs();
        if diff > 0 {
            report.mismatched_events += diff as usize;
            if report.first_mismatches.len() < 5 {
                report.first_mismatches.push(format!(
                    "stream={} fingerprint={:?} orig_count={} recon_count={}",
                    key, fp, oc, rc
                ));
            }
        }
    }
}

/// Build a stable, content-only fingerprint for one `Event`.
/// Args (which use a hash map) are serialised with keys sorted to keep the
/// fingerprint independent of insertion order.
fn event_fingerprint(e: &Event) -> String {
    let mut s = String::with_capacity(256);
    s.push_str(&e.name);
    s.push('\x01');
    s.push_str(&e.ts.to_string());
    s.push('\x01');
    s.push_str(&e.dur.map(|x| x.to_string()).unwrap_or_default());
    s.push('\x01');
    s.push_str(&e.ph.0.to_string());
    s.push('\x01');
    s.push_str(&e.pid.to_string());
    s.push('\x01');
    s.push_str(&e.tid);
    s.push('\x01');
    s.push_str(e.cat.as_deref().unwrap_or(""));
    s.push('\x01');
    s.push_str(&e.id.map(|x| x.to_string()).unwrap_or_default());
    s.push('\x01');
    s.push_str(e.bp.as_deref().unwrap_or(""));
    s.push('\x01');
    s.push_str(e.s.as_deref().unwrap_or(""));
    s.push('\x01');
    if let Some(args) = e.args.as_ref() {
        let mut keys: Vec<&String> = args.keys().collect();
        keys.sort();
        for k in keys {
            s.push_str(k);
            s.push('=');
            // serde_json::Value's `Display` keeps numbers / arrays / objects
            // canonical for our uses; objects from chrome-trace use serde's
            // default key order which is stable per construction here.
            s.push_str(&args.get(k).map(|v| v.to_string()).unwrap_or_default());
            s.push(';');
        }
    }
    s
}

