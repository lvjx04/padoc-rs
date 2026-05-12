//! Core event types.
//!
//! `Event` mirrors a chrome-trace `traceEvent` row. `MergeEvent` is the
//! template-side aggregate for a group of events that share the same
//! `(normalized_name, cat, bp, s, args_keys)` signature. `KernelEvent` and
//! `MergeKernelEvent` are the GPU-side counterparts (they additionally
//! carry per-instance `pid/tid/ph` because GPU events are pulled from the
//! CPU stream via correlation IDs).
//!
//! ## Why no inheritance
//!
//! The Python implementation used class polymorphism (CPUNode / SameCPUNode /
//! KernelLaunchNode / GPUNode and Event / MergeEvent / KernelEvent /
//! MergeKernelEvent).  In Rust we collapse the closely-related shapes into
//! enums so each variant is one fixed memory layout, `match` is exhaustive,
//! and `dyn` dispatch is gone.

use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// chrome-trace event "phase" — `X` (complete), `M` (metadata), `i` (instant), etc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Phase(pub u8);

impl Phase {
    pub const COMPLETE: Phase = Phase(b'X');
    pub const METADATA: Phase = Phase(b'M');
    pub const INSTANT: Phase = Phase(b'i');

    pub fn as_char(self) -> char {
        self.0 as char
    }
}

impl Default for Phase {
    fn default() -> Self { Phase::COMPLETE }
}

/// Free-form arg payload.  We use `serde_json::Value` so chrome-trace's
/// arbitrary nested args round-trip losslessly without a custom enum.
pub type ArgValue = serde_json::Value;
pub type Args = AHashMap<String, ArgValue>;

/// One raw chrome-trace event before compression.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Event {
    pub name: String,
    pub ts: i64,
    /// Duration in microseconds.  Some events (instants, metadata) have no dur.
    pub dur: Option<i64>,
    /// `cat` field if present.
    pub cat: Option<String>,
    /// Phase character, default `X`.
    pub ph: Phase,
    pub pid: i64,
    pub tid: String,
    pub args: Option<Args>,
    /// Optional `id` field for async events.
    pub id: Option<i64>,
    /// `bp` (binding point) — rarely used but kept for fidelity.
    pub bp: Option<String>,
    /// `s` (scope) — rarely used.
    pub s: Option<String>,
}

impl Event {
    /// Stable signature used to bucket events into the same template.
    ///
    /// Matches the Python `is_same_event(template, event)` semantics:
    /// `(normalized_name, cat, bp, s, sorted_args_keys)`.
    pub fn template_signature(&self) -> EventSignature {
        let normalized = crate::utils::normalize_name(&self.name);
        let mut arg_keys: SmallVec<[String; 8]> = self
            .args
            .as_ref()
            .map(|a| a.keys().cloned().collect())
            .unwrap_or_default();
        arg_keys.sort();
        EventSignature {
            normalized_name: normalized,
            cat: self.cat.clone(),
            bp: self.bp.clone(),
            s: self.s.clone(),
            arg_keys,
        }
    }
}

/// Hashable identity of a template — every event with the same signature
/// gets folded into the same `MergeEvent`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EventSignature {
    pub normalized_name: String,
    pub cat: Option<String>,
    pub bp: Option<String>,
    pub s: Option<String>,
    pub arg_keys: SmallVec<[String; 8]>,
}

/// Aggregated CPU template.  Every column is parallel: index `i` describes
/// the `i`-th physical event that was folded into this template.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MergeEvent {
    /// Digit-collapsed name pattern (single string, shared across instances).
    pub name_pattern: String,
    /// Per-instance digit fillers; each row reconstructs one instance's name.
    /// Stored columnar after `compress_names()` (= `Vec<Vec<i64>>` transposed).
    pub name_nums: NameNums,

    pub cat: Option<String>,
    pub bp: Option<String>,
    pub s: Option<String>,

    /// Sorted args key set — every instance's args values are stored aligned to this.
    pub arg_keys: Vec<String>,
    /// One row per instance; row[i] is parallel to `arg_keys`.
    pub args_values: Vec<Vec<ArgValue>>,

    pub ts: Vec<i64>,
    pub dur: Vec<i64>,
    pub id: Vec<i64>,
}

impl MergeEvent {
    pub fn instance_count(&self) -> usize {
        self.ts.len()
    }
}

/// Columnar representation of per-instance digit fillers.  Empty when the
/// raw name had no digits at all (in which case `name_pattern == name`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum NameNums {
    #[default]
    Empty,
    /// Per-instance digit lists, **as appended** (row-major).
    /// Cleared once `compress_names()` has folded into `Columnar` form.
    Rows(Vec<Vec<i64>>),
    /// Transposed form: outer Vec is one entry per `0` in `name_pattern`,
    /// inner Vec has one entry per instance.  Singletons collapse to a 1-elem column.
    Columnar(Vec<Vec<i64>>),
}

/// GPU-side raw event (pulled from a `correlation` arg).  Mirrors the Python
/// `KernelEvent`.
#[derive(Clone, Debug)]
pub struct KernelEvent {
    pub event: Event,
    /// The pid the originating cpu launch belonged to.
    pub pid: i64,
    /// Stream id, typically the suffix after `"stream "`.
    pub stream_tid: String,
    pub ph: Phase,
}

/// Aggregated GPU template.  Like `MergeEvent` but additionally carries
/// per-instance `pid/stream_tid/ph` because GPU events are reattached to
/// CPU-side launches via correlation IDs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MergeKernelEvent {
    pub name_pattern: String,
    pub name_nums: NameNums,
    pub cat: Option<String>,
    pub arg_keys: Vec<String>,
    pub args_values: Vec<Vec<ArgValue>>,
    pub ts: Vec<i64>,
    pub dur: Vec<i64>,
    pub pid: Vec<i64>,
    pub stream_tid: Vec<String>,
    pub ph: Vec<Phase>,
}

impl MergeKernelEvent {
    pub fn instance_count(&self) -> usize {
        self.ts.len()
    }
}

/// Discriminated template — a flat Vec of these is the templates table of a
/// compressed trace.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Template {
    Cpu(MergeEvent),
    Gpu(MergeKernelEvent),
}

impl Template {
    pub fn instance_count(&self) -> usize {
        match self {
            Template::Cpu(t) => t.instance_count(),
            Template::Gpu(t) => t.instance_count(),
        }
    }

    pub fn dur(&self) -> &[i64] {
        match self {
            Template::Cpu(t) => &t.dur,
            Template::Gpu(t) => &t.dur,
        }
    }

    pub fn ts(&self) -> &[i64] {
        match self {
            Template::Cpu(t) => &t.ts,
            Template::Gpu(t) => &t.ts,
        }
    }

    pub fn name_pattern(&self) -> &str {
        match self {
            Template::Cpu(t) => &t.name_pattern,
            Template::Gpu(t) => &t.name_pattern,
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Template::Gpu(_))
    }
}

/// Helper: discover every distinct arg key across a slice of events.  Used
/// when initialising a fresh `MergeEvent` for the very first instance.
pub fn collect_arg_keys(events: &[&Event]) -> Vec<String> {
    let mut seen: AHashSet<&str> = AHashSet::new();
    let mut keys: Vec<String> = Vec::new();
    for ev in events {
        if let Some(args) = ev.args.as_ref() {
            for k in args.keys() {
                if seen.insert(k.as_str()) {
                    keys.push(k.clone());
                }
            }
        }
    }
    keys.sort();
    keys
}
