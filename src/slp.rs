//! Segmented Linear Predictor — numeric column compression.
//!
//! Three roles in the original Python implementation:
//!
//! 1. **Numeric segment compression** — split a sequence of monotonic-ish
//!    numbers (timestamps, durations, ids) into `(start, step, length)` runs.
//!    Reconstruction is `start + step * i`.  Sequences that don't fit a
//!    single run get split greedily.
//! 2. **Name compression** — transpose `Vec<Vec<i64>>` (per-instance name
//!    digit fillers) into per-column arrays; collapse columns that are
//!    constant across instances.
//! 3. **Args dedup** — collapse a list of args values into a single value
//!    when every instance shares the same value.
//!
//! This Rust version keeps the same on-disk semantics so a round-trip with
//! the Python format is conceptually possible (we don't go out of our way
//! for byte-exact compatibility because the new repo is a clean break,
//! but the algorithm shapes match).

use serde::{Deserialize, Serialize};

use crate::event::{ArgValue, NameNums};

/// One linear segment.  Reconstruction: `values[i] = start + step * i` for `i in 0..length`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearSegment {
    pub start: i64,
    pub step: i64,
    pub length: u32,
}

/// SLP-encoded numeric column.  When the input was already a single arithmetic
/// progression the result is one segment; arbitrary sequences get split into
/// many.  Empty inputs produce an empty `segments` Vec.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SlpColumn {
    pub segments: Vec<LinearSegment>,
}

impl SlpColumn {
    /// Greedy segmentation: extend each segment as long as the next value
    /// matches the predicted `start + step * i`.
    pub fn encode(values: &[i64]) -> Self {
        let mut segments = Vec::new();
        let mut i = 0;
        while i < values.len() {
            let start = values[i];
            // Single-element segment as default.
            if i + 1 == values.len() {
                segments.push(LinearSegment { start, step: 0, length: 1 });
                break;
            }
            let step = values[i + 1].wrapping_sub(start);
            let mut length: u32 = 2;
            let mut j = i + 2;
            while j < values.len() {
                let predicted = start.wrapping_add(step.wrapping_mul(length as i64));
                if values[j] == predicted {
                    length += 1;
                    j += 1;
                } else {
                    break;
                }
            }
            segments.push(LinearSegment { start, step, length });
            i += length as usize;
        }
        SlpColumn { segments }
    }

    /// Total reconstructed length.
    pub fn len(&self) -> usize {
        self.segments.iter().map(|s| s.length as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decode back to the original sequence.
    pub fn decode(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.len());
        for seg in &self.segments {
            for i in 0..seg.length {
                out.push(seg.start.wrapping_add(seg.step.wrapping_mul(i as i64)));
            }
        }
        out
    }

    /// Random access by global index without materialising the full Vec.
    pub fn get(&self, mut index: usize) -> Option<i64> {
        for seg in &self.segments {
            let len = seg.length as usize;
            if index < len {
                return Some(seg.start.wrapping_add(seg.step.wrapping_mul(index as i64)));
            }
            index -= len;
        }
        None
    }
}

/// Transpose a row-major `Vec<Vec<i64>>` into columnar form, collapsing
/// every column whose values are all equal into a 1-element column.
pub fn compress_name_nums(rows: &NameNums) -> NameNums {
    let rows = match rows {
        NameNums::Empty => return NameNums::Empty,
        NameNums::Rows(r) => r,
        NameNums::Columnar(c) => return NameNums::Columnar(c.clone()),
    };
    if rows.is_empty() {
        return NameNums::Empty;
    }
    let width = rows[0].len();
    if width == 0 {
        return NameNums::Empty;
    }
    let mut columns: Vec<Vec<i64>> = (0..width).map(|_| Vec::with_capacity(rows.len())).collect();
    for row in rows {
        for (i, v) in row.iter().enumerate().take(width) {
            columns[i].push(*v);
        }
    }
    for col in columns.iter_mut() {
        if let Some(first) = col.first().copied() {
            if col.iter().all(|v| *v == first) {
                col.truncate(1);
            }
        }
    }
    NameNums::Columnar(columns)
}

/// Decode the `i`-th instance's digit fillers from `Columnar` form.
pub fn decode_name_nums(nums: &NameNums, instance: usize) -> Vec<i64> {
    match nums {
        NameNums::Empty => Vec::new(),
        NameNums::Rows(rows) => rows.get(instance).cloned().unwrap_or_default(),
        NameNums::Columnar(cols) => cols
            .iter()
            .map(|col| {
                if col.len() == 1 {
                    col[0]
                } else {
                    *col.get(instance).unwrap_or(&0)
                }
            })
            .collect(),
    }
}

/// Args-side dedup: if every value in `values` is identical, return a 1-elem
/// Vec; otherwise return as-is.  Strings, numbers and JSON sub-objects are
/// compared via `serde_json::Value::eq`.
pub fn compress_same_args(values: &mut Vec<ArgValue>) {
    if values.len() <= 1 {
        return;
    }
    let first = values[0].clone();
    if values.iter().all(|v| v == &first) {
        values.truncate(1);
    }
}

/// Args-side decode counterpart: pick `index`-th instance, falling back to
/// the singleton if the column was deduped.
pub fn decode_arg(values: &[ArgValue], index: usize) -> Option<&ArgValue> {
    if values.len() == 1 {
        values.first()
    } else {
        values.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slp_round_trip_arithmetic_progression() {
        let original: Vec<i64> = (0..1000).map(|i| 100 + i * 7).collect();
        let encoded = SlpColumn::encode(&original);
        // Should encode as ONE segment.
        assert_eq!(encoded.segments.len(), 1);
        assert_eq!(encoded.decode(), original);
    }

    #[test]
    fn slp_split_on_jump() {
        let original = vec![1, 2, 3, 100, 101, 102];
        let encoded = SlpColumn::encode(&original);
        assert_eq!(encoded.segments.len(), 2);
        assert_eq!(encoded.decode(), original);
    }

    #[test]
    fn slp_random_access() {
        let original: Vec<i64> = vec![5, 10, 15, 20, 100, 150];
        let encoded = SlpColumn::encode(&original);
        for (i, expected) in original.iter().enumerate() {
            assert_eq!(encoded.get(i), Some(*expected));
        }
        assert_eq!(encoded.get(original.len()), None);
    }

    #[test]
    fn name_nums_columnar_collapses_constants() {
        let rows = NameNums::Rows(vec![vec![1, 99], vec![2, 99], vec![3, 99]]);
        let columnar = compress_name_nums(&rows);
        match columnar {
            NameNums::Columnar(cols) => {
                assert_eq!(cols[0], vec![1, 2, 3]);
                assert_eq!(cols[1], vec![99]); // collapsed
            }
            _ => panic!(),
        }
    }
}
