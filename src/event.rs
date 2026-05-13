//! Core event types.
//!
//! `Event` mirrors a chrome-trace `traceEvent` row.  `MergeEvent` is the
//! template-side aggregate for a group of events that share the same
//! `(normalized_name, cat, bp, s, args_keys)` signature.  `KernelEvent` and
//! `MergeKernelEvent` are the GPU-side counterparts (they additionally
//! carry per-instance `pid/stream_tid/ph` because GPU events are pulled
//! from the CPU stream via correlation IDs).
//!
//! ## Storage layout
//!
//! Per-instance columns are typed **enums**, not raw `Vec<i64>` / `Vec<String>`.
//! Each column auto-collapses to a constant when every instance shares the
//! same value, and integer columns downcast to `i32` when the observed range
//! fits.  This keeps the in-memory and on-disk sizes close to what the
//! original Python (`numpy` int8/int16/int32) implementation achieved.
//!
//! Build path: append one event at a time (the `Empty` / `I64` / `Strings`
//! variants are mutable).  Finalize path: `to_compact()` shrinks every column
//! into the most compact variant.

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
    fn default() -> Self {
        Phase::COMPLETE
    }
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

// ---------------------------------------------------------------------------
// Compact per-instance columns
// ---------------------------------------------------------------------------

/// Compact integer column.  Detects constants and downcasts to `i32` when
/// every observed value fits in `i32::MIN..=i32::MAX`.  `Empty` represents an
/// optional column that received no values (e.g. CPU `id` for events without
/// an async id).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum NumColumn {
    #[default]
    Empty,
    Constant {
        len: u32,
        value: i64,
    },
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl NumColumn {
    pub fn len(&self) -> usize {
        match self {
            NumColumn::Empty => 0,
            NumColumn::Constant { len, .. } => *len as usize,
            NumColumn::I32(v) => v.len(),
            NumColumn::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<i64> {
        match self {
            NumColumn::Empty => None,
            NumColumn::Constant { len, value } => {
                if i < *len as usize {
                    Some(*value)
                } else {
                    None
                }
            }
            NumColumn::I32(v) => v.get(i).map(|x| *x as i64),
            NumColumn::I64(v) => v.get(i).copied(),
        }
    }

    pub fn iter_i64(&self) -> NumIter<'_> {
        NumIter::new(self)
    }

    pub fn sum_i64(&self) -> i64 {
        match self {
            NumColumn::Empty => 0,
            NumColumn::Constant { len, value } => (*len as i64) * (*value),
            NumColumn::I32(v) => v.iter().map(|x| *x as i64).sum(),
            NumColumn::I64(v) => v.iter().sum(),
        }
    }

    /// Append one value (build path).  Forces the column into `I64` storage —
    /// finalisation downcasts later.
    pub fn push(&mut self, v: i64) {
        match self {
            NumColumn::Empty => *self = NumColumn::I64(vec![v]),
            NumColumn::I64(values) => values.push(v),
            NumColumn::I32(_) | NumColumn::Constant { .. } => {
                // Build path appends in I64 mode; compact variants only appear
                // post-finalise.  Promote back to I64 to stay correct if ever
                // hit.
                let mut values: Vec<i64> = (0..self.len()).map(|i| self.get(i).unwrap_or(0)).collect();
                values.push(v);
                *self = NumColumn::I64(values);
            }
        }
    }

    /// Append all values from `other` (build-path merge).  Promotes whichever
    /// side isn't already `I64` so we can extend in-place.
    pub fn extend_from(&mut self, other: NumColumn) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other;
            return;
        }
        // Promote both sides to I64 first; compactification happens at finalize.
        let extra: Vec<i64> = (0..other.len()).map(|i| other.get(i).unwrap_or(0)).collect();
        match self {
            NumColumn::I64(values) => values.extend(extra),
            _ => {
                let mut values: Vec<i64> = (0..self.len()).map(|i| self.get(i).unwrap_or(0)).collect();
                values.extend(extra);
                *self = NumColumn::I64(values);
            }
        }
    }

    /// Compact: detect constant; otherwise downcast to `i32` if safe.
    pub fn compact(&mut self) {
        let n = self.len();
        if n == 0 {
            *self = NumColumn::Empty;
            return;
        }
        // Materialise first value and scan for constancy + range.
        let first = match self.get(0) {
            Some(v) => v,
            None => return,
        };
        let mut all_same = true;
        let mut min_v = first;
        let mut max_v = first;
        for i in 1..n {
            let v = self.get(i).unwrap_or(0);
            if v != first {
                all_same = false;
            }
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        if all_same {
            *self = NumColumn::Constant {
                len: n as u32,
                value: first,
            };
            return;
        }
        if min_v >= i32::MIN as i64 && max_v <= i32::MAX as i64 {
            // Downcast to i32.
            let mut values: Vec<i32> = Vec::with_capacity(n);
            for i in 0..n {
                values.push(self.get(i).unwrap_or(0) as i32);
            }
            *self = NumColumn::I32(values);
        } else if !matches!(self, NumColumn::I64(_)) {
            // Already in i64 form — nothing to do.
            let values: Vec<i64> = (0..n).map(|i| self.get(i).unwrap_or(0)).collect();
            *self = NumColumn::I64(values);
        }
    }
}

/// Iterator over a `NumColumn` yielding `i64` values.
pub struct NumIter<'a> {
    col: &'a NumColumn,
    idx: usize,
    end: usize,
}

impl<'a> NumIter<'a> {
    fn new(col: &'a NumColumn) -> Self {
        let end = col.len();
        Self { col, idx: 0, end }
    }
}

impl<'a> Iterator for NumIter<'a> {
    type Item = i64;
    fn next(&mut self) -> Option<i64> {
        if self.idx >= self.end {
            return None;
        }
        let v = self.col.get(self.idx)?;
        self.idx += 1;
        Some(v)
    }
}

/// Compact string column.  Detects per-template constants and otherwise stores
/// each instance's value.  Used for GPU `stream_tid`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum StringColumn {
    #[default]
    Empty,
    Constant {
        len: u32,
        value: String,
    },
    PerInstance(Vec<String>),
}

impl StringColumn {
    pub fn len(&self) -> usize {
        match self {
            StringColumn::Empty => 0,
            StringColumn::Constant { len, .. } => *len as usize,
            StringColumn::PerInstance(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<&str> {
        match self {
            StringColumn::Empty => None,
            StringColumn::Constant { len, value } => {
                if i < *len as usize {
                    Some(value.as_str())
                } else {
                    None
                }
            }
            StringColumn::PerInstance(v) => v.get(i).map(String::as_str),
        }
    }

    pub fn push(&mut self, v: String) {
        match self {
            StringColumn::Empty => *self = StringColumn::PerInstance(vec![v]),
            StringColumn::PerInstance(values) => values.push(v),
            StringColumn::Constant { len, value } => {
                // Build-path: convert back to PerInstance.
                let mut values: Vec<String> = std::iter::repeat(value.clone()).take(*len as usize).collect();
                values.push(v);
                *self = StringColumn::PerInstance(values);
            }
        }
    }

    pub fn extend_from(&mut self, other: StringColumn) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other;
            return;
        }
        let other_len = other.len();
        let mut other_values: Vec<String> = (0..other_len)
            .map(|i| other.get(i).map(str::to_owned).unwrap_or_default())
            .collect();
        match self {
            StringColumn::PerInstance(v) => v.extend(other_values.drain(..)),
            _ => {
                let self_len = self.len();
                let mut values: Vec<String> = (0..self_len)
                    .map(|i| self.get(i).map(str::to_owned).unwrap_or_default())
                    .collect();
                values.append(&mut other_values);
                *self = StringColumn::PerInstance(values);
            }
        }
    }

    pub fn compact(&mut self) {
        let n = self.len();
        if n == 0 {
            *self = StringColumn::Empty;
            return;
        }
        let first = match self.get(0) {
            Some(s) => s.to_owned(),
            None => return,
        };
        let mut all_same = true;
        for i in 1..n {
            if self.get(i).map(|s| s != first.as_str()).unwrap_or(true) {
                all_same = false;
                break;
            }
        }
        if all_same {
            *self = StringColumn::Constant {
                len: n as u32,
                value: first,
            };
        }
    }
}

/// Compact phase column.  Phases are 1-byte tags that overwhelmingly take a
/// single value across a template (`X` for complete events, `M` for metadata,
/// etc.), so we collapse to a constant when possible.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum PhaseColumn {
    #[default]
    Empty,
    Constant {
        len: u32,
        value: u8,
    },
    PerInstance(Vec<u8>),
}

impl PhaseColumn {
    pub fn len(&self) -> usize {
        match self {
            PhaseColumn::Empty => 0,
            PhaseColumn::Constant { len, .. } => *len as usize,
            PhaseColumn::PerInstance(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<Phase> {
        match self {
            PhaseColumn::Empty => None,
            PhaseColumn::Constant { len, value } => {
                if i < *len as usize {
                    Some(Phase(*value))
                } else {
                    None
                }
            }
            PhaseColumn::PerInstance(v) => v.get(i).copied().map(Phase),
        }
    }

    pub fn push(&mut self, ph: Phase) {
        match self {
            PhaseColumn::Empty => *self = PhaseColumn::PerInstance(vec![ph.0]),
            PhaseColumn::PerInstance(values) => values.push(ph.0),
            PhaseColumn::Constant { len, value } => {
                let mut values: Vec<u8> = std::iter::repeat(*value).take(*len as usize).collect();
                values.push(ph.0);
                *self = PhaseColumn::PerInstance(values);
            }
        }
    }

    pub fn extend_from(&mut self, other: PhaseColumn) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other;
            return;
        }
        let other_len = other.len();
        let mut other_values: Vec<u8> = (0..other_len)
            .map(|i| other.get(i).map(|p| p.0).unwrap_or(b'X'))
            .collect();
        match self {
            PhaseColumn::PerInstance(v) => v.extend(other_values.drain(..)),
            _ => {
                let self_len = self.len();
                let mut values: Vec<u8> = (0..self_len)
                    .map(|i| self.get(i).map(|p| p.0).unwrap_or(b'X'))
                    .collect();
                values.append(&mut other_values);
                *self = PhaseColumn::PerInstance(values);
            }
        }
    }

    pub fn compact(&mut self) {
        let n = self.len();
        if n == 0 {
            *self = PhaseColumn::Empty;
            return;
        }
        let first = match self.get(0) {
            Some(p) => p.0,
            None => return,
        };
        let mut all_same = true;
        for i in 1..n {
            if self.get(i).map(|p| p.0 != first).unwrap_or(true) {
                all_same = false;
                break;
            }
        }
        if all_same {
            *self = PhaseColumn::Constant {
                len: n as u32,
                value: first,
            };
        }
    }
}

/// Compact args column.  At build time every arg lands in `PerInstance`; at
/// finalise time `compact()` detects:
///
/// * a single shared value (`Constant`),
/// * a uniformly-typed numeric column (`I32`/`I64`/`F64`),
/// * a uniformly-typed `Bool` column,
/// * a string column with low cardinality (`StrDict`) — most common shape for
///   chrome-trace string args like `cat`/`detail` strings,
/// * a heterogeneous fallback (`PerInstance`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ArgColumn {
    Constant(ArgValue),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Bool(Vec<u8>),
    Str(Vec<String>),
    StrDict { dict: Vec<String>, ids: Vec<u32> },
    PerInstance(Vec<ArgValue>),
}

impl ArgColumn {
    pub fn len(&self) -> usize {
        match self {
            ArgColumn::Constant(_) => 1,
            ArgColumn::I32(v) => v.len(),
            ArgColumn::I64(v) => v.len(),
            ArgColumn::F64(v) => v.len(),
            ArgColumn::Bool(v) => v.len(),
            ArgColumn::Str(v) => v.len(),
            ArgColumn::StrDict { ids, .. } => ids.len(),
            ArgColumn::PerInstance(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialise the `i`-th instance's value.  Hot paths (analyses) should
    /// avoid this if possible; the typed accessors are cheaper.
    pub fn get_owned(&self, i: usize) -> Option<ArgValue> {
        match self {
            ArgColumn::Constant(v) => Some(v.clone()),
            ArgColumn::I32(v) => v.get(i).map(|x| serde_json::Value::from(*x as i64)),
            ArgColumn::I64(v) => v.get(i).map(|x| serde_json::Value::from(*x)),
            ArgColumn::F64(v) => v.get(i).and_then(|x| {
                serde_json::Number::from_f64(*x).map(serde_json::Value::Number)
            }),
            ArgColumn::Bool(v) => v.get(i).map(|x| serde_json::Value::Bool(*x != 0)),
            ArgColumn::Str(v) => v.get(i).cloned().map(serde_json::Value::String),
            ArgColumn::StrDict { dict, ids } => ids.get(i).and_then(|id| {
                dict.get(*id as usize)
                    .cloned()
                    .map(serde_json::Value::String)
            }),
            ArgColumn::PerInstance(v) => v.get(i).cloned(),
        }
    }

    pub fn push(&mut self, value: ArgValue) {
        if let ArgColumn::PerInstance(vs) = self {
            vs.push(value);
            return;
        }
        debug_assert!(false, "ArgColumn::push on non-PerInstance variant");
        let n = self.len();
        let mut values: Vec<ArgValue> = (0..n)
            .map(|i| self.get_owned(i).unwrap_or(serde_json::Value::Null))
            .collect();
        values.push(value);
        *self = ArgColumn::PerInstance(values);
    }

    /// Compact a `PerInstance` column to the most efficient typed variant.
    /// Other variants are left untouched.
    pub fn compact(&mut self) {
        let values = match self {
            ArgColumn::PerInstance(v) if !v.is_empty() => v,
            _ => return,
        };
        // 1. Constant detection.
        let first = values[0].clone();
        if values.iter().all(|v| v == &first) {
            *self = ArgColumn::Constant(first);
            return;
        }
        // 2. Typed numeric / bool.
        let mut all_int = true;
        let mut all_float = true;
        let mut all_bool = true;
        let mut all_string = true;
        let mut min_i: i64 = i64::MAX;
        let mut max_i: i64 = i64::MIN;
        for v in values.iter() {
            match v {
                serde_json::Value::Bool(_) => {
                    all_int = false;
                    all_float = false;
                    all_string = false;
                }
                serde_json::Value::Number(n) => {
                    all_bool = false;
                    all_string = false;
                    if let Some(i) = n.as_i64() {
                        if i < min_i {
                            min_i = i;
                        }
                        if i > max_i {
                            max_i = i;
                        }
                        // ints are legal floats too; keep all_float possible.
                    } else if n.is_f64() {
                        all_int = false;
                    } else {
                        all_int = false;
                        all_float = false;
                    }
                }
                serde_json::Value::String(_) => {
                    all_int = false;
                    all_float = false;
                    all_bool = false;
                }
                _ => {
                    all_int = false;
                    all_float = false;
                    all_bool = false;
                    all_string = false;
                }
            }
        }
        if all_bool {
            let bools: Vec<u8> = values
                .iter()
                .map(|v| match v {
                    serde_json::Value::Bool(b) => *b as u8,
                    _ => 0,
                })
                .collect();
            *self = ArgColumn::Bool(bools);
            return;
        }
        if all_int {
            if min_i >= i32::MIN as i64 && max_i <= i32::MAX as i64 {
                let ints: Vec<i32> = values
                    .iter()
                    .map(|v| v.as_i64().unwrap_or(0) as i32)
                    .collect();
                *self = ArgColumn::I32(ints);
            } else {
                let ints: Vec<i64> = values.iter().map(|v| v.as_i64().unwrap_or(0)).collect();
                *self = ArgColumn::I64(ints);
            }
            return;
        }
        if all_float {
            let floats: Vec<f64> = values
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            *self = ArgColumn::F64(floats);
            return;
        }
        if all_string {
            // Decide between Str and StrDict based on observed cardinality.
            let mut dict: Vec<String> = Vec::new();
            let mut index: AHashMap<String, u32> = AHashMap::new();
            let mut ids: Vec<u32> = Vec::with_capacity(values.len());
            for v in values.iter() {
                let s = match v {
                    serde_json::Value::String(s) => s.clone(),
                    _ => String::new(),
                };
                let id = if let Some(&id) = index.get(&s) {
                    id
                } else {
                    let id = dict.len() as u32;
                    index.insert(s.clone(), id);
                    dict.push(s);
                    id
                };
                ids.push(id);
            }
            // If dictionary is significantly smaller than total, dedup; else
            // leave as flat strings.
            let total = values.len();
            if dict.len() * 2 <= total {
                *self = ArgColumn::StrDict { dict, ids };
            } else {
                let strs: Vec<String> = ids.iter().map(|i| dict[*i as usize].clone()).collect();
                *self = ArgColumn::Str(strs);
            }
            return;
        }
        // Heterogeneous — keep as PerInstance but the existing storage already
        // represents that, so nothing to do.
    }
}

impl Default for ArgColumn {
    fn default() -> Self {
        ArgColumn::PerInstance(Vec::new())
    }
}

/// One column of digit-fillers that reconstruct an instance's name from the
/// template's `name_pattern`.  Each occurrence of `0` in the pattern consumes
/// one digit column at decode time.
///
/// `width` (when present) is the original character width including leading
/// zeros — e.g. `"040"` → `width = 3`, `value = 40`.  This preserves
/// hex-pointer literals like `0x4387e040` without paying for a `String` per
/// instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DigitColumn {
    /// All instances share the same digit string for this column position.
    Constant(String),
    /// Per-instance integer digits stored as `i32` (with optional leading-zero
    /// width).  `width = 0` means no zero-padding (use plain decimal).
    I32 { width: u8, values: Vec<i32> },
    /// Per-instance integer digits stored as `i64` (overflow fallback).
    I64 { width: u8, values: Vec<i64> },
    /// Free-form string digits (hex-pointer literals, mixed widths, etc.).
    Strings(Vec<String>),
}

impl DigitColumn {
    pub fn len(&self) -> usize {
        match self {
            DigitColumn::Constant(_) => 1,
            DigitColumn::I32 { values, .. } => values.len(),
            DigitColumn::I64 { values, .. } => values.len(),
            DigitColumn::Strings(v) => v.len(),
        }
    }

    pub fn get_string(&self, i: usize) -> String {
        match self {
            DigitColumn::Constant(s) => s.clone(),
            DigitColumn::I32 { width, values } => {
                let v = values.get(i).copied().unwrap_or(0);
                format_int_with_width(v as i64, *width)
            }
            DigitColumn::I64 { width, values } => {
                let v = values.get(i).copied().unwrap_or(0);
                format_int_with_width(v, *width)
            }
            DigitColumn::Strings(v) => v.get(i).cloned().unwrap_or_default(),
        }
    }
}

fn format_int_with_width(v: i64, width: u8) -> String {
    if width == 0 {
        return v.to_string();
    }
    let s = v.to_string();
    if s.len() >= width as usize {
        s
    } else {
        format!("{:0>width$}", s, width = width as usize)
    }
}

/// Columnar representation of per-instance digit fillers.  Empty when the
/// raw name had no digits at all (`name_pattern == name`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum NameNums {
    #[default]
    Empty,
    /// Per-instance digit lists, **as appended** (row-major).  Cleared once
    /// `compress_name_nums()` has folded into `Columnar` form.
    Rows(Vec<Vec<String>>),
    /// Transposed form: outer Vec is one entry per `0` in `name_pattern`;
    /// each entry is a typed [`DigitColumn`].
    Columnar(Vec<DigitColumn>),
}

// ---------------------------------------------------------------------------
// Aggregated templates
// ---------------------------------------------------------------------------

/// Aggregated CPU template.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MergeEvent {
    /// Digit-collapsed name pattern (single string, shared across instances).
    pub name_pattern: String,
    /// Per-instance digit fillers; each row reconstructs one instance's name.
    pub name_nums: NameNums,

    pub cat: Option<String>,
    pub bp: Option<String>,
    pub s: Option<String>,

    /// Sorted args key set; one [`ArgColumn`] per key, parallel to this Vec.
    pub arg_keys: Vec<String>,
    /// **Column-major** args storage.
    pub args_columns: Vec<ArgColumn>,

    pub ts: NumColumn,
    pub dur: NumColumn,
    pub id: NumColumn,
}

impl MergeEvent {
    pub fn instance_count(&self) -> usize {
        self.ts.len()
    }
}

/// GPU-side raw event (pulled from a `correlation` arg).
#[derive(Clone, Debug)]
pub struct KernelEvent {
    pub event: Event,
    /// The pid the originating cpu launch belonged to.
    pub pid: i64,
    /// Stream id, typically the suffix after `"stream "`.
    pub stream_tid: String,
    pub ph: Phase,
}

/// Aggregated GPU template.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MergeKernelEvent {
    pub name_pattern: String,
    pub name_nums: NameNums,
    pub cat: Option<String>,
    pub arg_keys: Vec<String>,
    pub args_columns: Vec<ArgColumn>,
    pub ts: NumColumn,
    pub dur: NumColumn,
    pub pid: NumColumn,
    pub stream_tid: StringColumn,
    pub ph: PhaseColumn,
}

impl MergeKernelEvent {
    pub fn instance_count(&self) -> usize {
        self.ts.len()
    }
}

/// Discriminated template.
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

    /// Sum of every instance's `dur` (zero if column is empty).
    pub fn dur_total(&self) -> i64 {
        match self {
            Template::Cpu(t) => t.dur.sum_i64(),
            Template::Gpu(t) => t.dur.sum_i64(),
        }
    }

    /// `dur` accessor for analyses that need per-instance values.
    pub fn dur_at(&self, i: usize) -> Option<i64> {
        match self {
            Template::Cpu(t) => t.dur.get(i),
            Template::Gpu(t) => t.dur.get(i),
        }
    }

    pub fn ts_at(&self, i: usize) -> Option<i64> {
        match self {
            Template::Cpu(t) => t.ts.get(i),
            Template::Gpu(t) => t.ts.get(i),
        }
    }

    pub fn name_pattern(&self) -> &str {
        match self {
            Template::Cpu(t) => &t.name_pattern,
            Template::Gpu(t) => &t.name_pattern,
        }
    }

    pub fn name_nums(&self) -> &NameNums {
        match self {
            Template::Cpu(t) => &t.name_nums,
            Template::Gpu(t) => &t.name_nums,
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Template::Gpu(_))
    }
}

/// Helper: discover every distinct arg key across a slice of events.
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
