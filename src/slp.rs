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

use crate::event::{ArgValue, DigitColumn, NameNums};

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

// ---------------------------------------------------------------------------
// Piecewise-linear compression with residuals (i8/i16 per-segment)
// ---------------------------------------------------------------------------

/// Per-segment residual storage: i8 when all residuals fit, otherwise i16.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ResidualStorage {
    I8(Vec<i8>),
    I16(Vec<i16>),
}

/// One segment of the piecewise-linear encoding with residuals.
/// Reconstruction: `values[i] = start + slope * i + residuals[i]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlpSegment {
    pub start: i64,
    pub slope: i64,
    pub residuals: ResidualStorage,
}

impl SlpSegment {
    pub fn len(&self) -> usize {
        match &self.residuals {
            ResidualStorage::I8(v) => v.len(),
            ResidualStorage::I16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Piecewise-linear column with i8/i16 residuals per segment.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SlpCompact {
    pub segments: Vec<SlpSegment>,
}

impl SlpCompact {
    /// Greedy segmentation with residual encoding.
    ///
    /// For each segment: slope = v[1] - v[0] (integer).
    /// Extend while |residual| <= i16::MAX.
    /// After segment is collected, store residuals as i8 if all fit, else i16.
    pub fn encode(values: &[i64]) -> Self {
        if values.is_empty() {
            return SlpCompact { segments: Vec::new() };
        }
        let mut segments = Vec::new();
        let mut i = 0;
        while i < values.len() {
            let start = values[i];
            if i + 1 >= values.len() {
                // Single-element segment: residual is 0
                segments.push(SlpSegment {
                    start,
                    slope: 0,
                    residuals: ResidualStorage::I8(vec![0]),
                });
                break;
            }
            let slope = values[i + 1] - start;
            // Collect residuals for this segment
            let mut residuals: Vec<i16> = Vec::new();
            let mut j = i;
            while j < values.len() {
                let local_idx = (j - i) as i64;
                let predicted = start + slope * local_idx;
                let residual = values[j] - predicted;
                if residual < i16::MIN as i64 || residual > i16::MAX as i64 {
                    break;
                }
                residuals.push(residual as i16);
                j += 1;
            }
            // If we couldn't even fit one value (shouldn't happen since residual at idx 0 is 0),
            // force at least one value.
            if residuals.is_empty() {
                residuals.push(0);
                j = i + 1;
            }
            // Choose i8 or i16 storage
            let all_i8 = residuals.iter().all(|&r| r >= i8::MIN as i16 && r <= i8::MAX as i16);
            let storage = if all_i8 {
                ResidualStorage::I8(residuals.iter().map(|&r| r as i8).collect())
            } else {
                ResidualStorage::I16(residuals)
            };
            segments.push(SlpSegment { start, slope, residuals: storage });
            i = j;
        }
        SlpCompact { segments }
    }

    /// Total number of values.
    pub fn len(&self) -> usize {
        self.segments.iter().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Random access by global index.
    pub fn get(&self, mut index: usize) -> Option<i64> {
        for seg in &self.segments {
            let len = seg.len();
            if index < len {
                let predicted = seg.start + seg.slope * (index as i64);
                let residual = match &seg.residuals {
                    ResidualStorage::I8(v) => v[index] as i64,
                    ResidualStorage::I16(v) => v[index] as i64,
                };
                return Some(predicted + residual);
            }
            index -= len;
        }
        None
    }

    /// Decode back to the original sequence.
    pub fn decode(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.len());
        for seg in &self.segments {
            let n = seg.len();
            for idx in 0..n {
                let predicted = seg.start + seg.slope * (idx as i64);
                let residual = match &seg.residuals {
                    ResidualStorage::I8(v) => v[idx] as i64,
                    ResidualStorage::I16(v) => v[idx] as i64,
                };
                out.push(predicted + residual);
            }
        }
        out
    }

    /// Sum all values without full decode.
    pub fn sum_i64(&self) -> i64 {
        let mut total: i64 = 0;
        for seg in &self.segments {
            let n = seg.len() as i64;
            // Sum of (start + slope*i + residual[i]) for i in 0..n
            // = n*start + slope * n*(n-1)/2 + sum(residuals)
            total += n * seg.start;
            total += seg.slope * (n * (n - 1) / 2);
            total += match &seg.residuals {
                ResidualStorage::I8(v) => v.iter().map(|&r| r as i64).sum::<i64>(),
                ResidualStorage::I16(v) => v.iter().map(|&r| r as i64).sum::<i64>(),
            };
        }
        total
    }

    /// Heap bytes used by this column (for memory accounting).
    pub fn heap_bytes(&self) -> usize {
        let mut bytes = self.segments.capacity() * std::mem::size_of::<SlpSegment>();
        for seg in &self.segments {
            bytes += match &seg.residuals {
                ResidualStorage::I8(v) => v.capacity(),
                ResidualStorage::I16(v) => v.capacity() * 2,
            };
        }
        bytes
    }

    pub fn shrink_to_fit(&mut self) {
        self.segments.shrink_to_fit();
        for seg in &mut self.segments {
            match &mut seg.residuals {
                ResidualStorage::I8(v) => v.shrink_to_fit(),
                ResidualStorage::I16(v) => v.shrink_to_fit(),
            }
        }
    }
}

/// Transpose a row-major `Vec<Vec<String>>` into columnar form, collapsing
/// every column whose values are all equal into a [`DigitColumn::Constant`]
/// and downcasting purely-decimal columns to packed `i32` / `i64` arrays.
///
/// This is the in-memory analogue of the Python implementation's
/// `numpy.asarray(col, dtype=np.int32)` strategy: most digit columns are
/// small layer / iter / rank counters whose values fit in `i32`, so this
/// shaves a 24-byte `String` header *plus* heap allocation per instance down
/// to 4 bytes.
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
    let mut string_columns: Vec<Vec<String>> = (0..width).map(|_| Vec::with_capacity(rows.len())).collect();
    for row in rows {
        for (i, v) in row.iter().enumerate().take(width) {
            string_columns[i].push(v.clone());
        }
    }
    let columns: Vec<DigitColumn> = string_columns.into_iter().map(compact_digit_column).collect();
    NameNums::Columnar(columns)
}

/// Compact one digit column.  Tries (in order):
///
/// 1. constant — every entry identical;
/// 2. fixed-width decimal — every entry is `[0-9]+`, all share one
///    character width, and the parsed values fit in `i32` (otherwise `i64`);
/// 3. fall back to `Strings` (hex pointers, mixed widths, etc.).
fn compact_digit_column(values: Vec<String>) -> DigitColumn {
    if values.is_empty() {
        return DigitColumn::Strings(values);
    }
    let first = values[0].clone();
    if values.iter().all(|v| *v == first) {
        return DigitColumn::Constant(first);
    }

    // Try decimal downcast.
    let mut all_decimal = true;
    let mut max_len: usize = 0;
    let mut min_len: usize = usize::MAX;
    let mut min_v: i128 = i128::MAX;
    let mut max_v: i128 = i128::MIN;
    for v in values.iter() {
        if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
            all_decimal = false;
            break;
        }
        if v.len() > max_len {
            max_len = v.len();
        }
        if v.len() < min_len {
            min_len = v.len();
        }
        match v.parse::<i128>() {
            Ok(parsed) => {
                if parsed < min_v {
                    min_v = parsed;
                }
                if parsed > max_v {
                    max_v = parsed;
                }
            }
            Err(_) => {
                all_decimal = false;
                break;
            }
        }
    }

    if all_decimal {
        // We use a single `width` per column.  Two cases:
        // - all entries already share the same character width => width=N
        //   so leading-zero padding is preserved on decode.
        // - widths differ but no entry has a leading zero => width=0 (plain
        //   decimal) — `format_int_with_width(0)` re-emits without padding.
        let uniform_width = min_len == max_len;
        let any_padding = values
            .iter()
            .any(|v| v.starts_with('0') && v.len() > 1);
        let width: u8 = if uniform_width && any_padding {
            max_len.min(255) as u8
        } else if !any_padding {
            0
        } else {
            // Heterogeneous leading zeros — can't encode with a single width.
            return DigitColumn::Strings(values);
        };
        if min_v >= i32::MIN as i128 && max_v <= i32::MAX as i128 {
            let ints: Vec<i32> = values
                .iter()
                .map(|v| v.parse::<i32>().unwrap_or(0))
                .collect();
            return DigitColumn::I32 { width, values: ints };
        } else if min_v >= i64::MIN as i128 && max_v <= i64::MAX as i128 {
            let ints: Vec<i64> = values
                .iter()
                .map(|v| v.parse::<i64>().unwrap_or(0))
                .collect();
            return DigitColumn::I64 { width, values: ints };
        }
    }

    DigitColumn::Strings(values)
}

/// Decode the `i`-th instance's digit fillers.
pub fn decode_name_nums(nums: &NameNums, instance: usize) -> Vec<String> {
    match nums {
        NameNums::Empty => Vec::new(),
        NameNums::Rows(rows) => rows.get(instance).cloned().unwrap_or_default(),
        NameNums::Columnar(cols) => cols.iter().map(|col| col.get_string(instance)).collect(),
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
        let rows = NameNums::Rows(vec![
            vec!["1".to_string(), "99".to_string()],
            vec!["2".to_string(), "99".to_string()],
            vec!["3".to_string(), "99".to_string()],
        ]);
        let columnar = compress_name_nums(&rows);
        let decoded: Vec<Vec<String>> = (0..3).map(|i| decode_name_nums(&columnar, i)).collect();
        assert_eq!(
            decoded,
            vec![
                vec!["1".to_string(), "99".to_string()],
                vec!["2".to_string(), "99".to_string()],
                vec!["3".to_string(), "99".to_string()],
            ]
        );
        if let NameNums::Columnar(cols) = columnar {
            // Second column is constant; first column is typed I32.
            assert!(matches!(cols[1], crate::event::DigitColumn::Constant(_)));
            assert!(matches!(cols[0], crate::event::DigitColumn::I32 { .. }));
        } else {
            panic!()
        }
    }

    #[test]
    fn name_nums_round_trip_preserves_leading_zeros() {
        let rows = NameNums::Rows(vec![
            vec!["0x".to_string(), "040".to_string()],
            vec!["0x".to_string(), "007".to_string()],
        ]);
        let columnar = compress_name_nums(&rows);
        let inst0 = decode_name_nums(&columnar, 0);
        let inst1 = decode_name_nums(&columnar, 1);
        assert_eq!(inst0, vec!["0x".to_string(), "040".to_string()]);
        assert_eq!(inst1, vec!["0x".to_string(), "007".to_string()]);
    }

    #[test]
    fn name_nums_falls_back_to_strings_for_hex_or_mixed_widths() {
        let rows = NameNums::Rows(vec![
            vec!["a".to_string(), "1".to_string()],
            vec!["bf".to_string(), "12".to_string()],
        ]);
        let columnar = compress_name_nums(&rows);
        let inst0 = decode_name_nums(&columnar, 0);
        let inst1 = decode_name_nums(&columnar, 1);
        assert_eq!(inst0, vec!["a".to_string(), "1".to_string()]);
        assert_eq!(inst1, vec!["bf".to_string(), "12".to_string()]);
    }

    #[test]
    fn slp_compact_round_trip() {
        let original: Vec<i64> = (0..1000).map(|i| 100 + i * 7).collect();
        let encoded = SlpCompact::encode(&original);
        assert_eq!(encoded.decode(), original);
        // Pure arithmetic progression: 1 segment, all i8 residuals (all zero)
        assert_eq!(encoded.segments.len(), 1);
        assert!(matches!(encoded.segments[0].residuals, ResidualStorage::I8(_)));
    }

    #[test]
    fn slp_compact_with_noise() {
        // Values with small deviations from linear — should use i8 residuals
        let original: Vec<i64> = (0..100).map(|i| 1000 + i * 50 + (i % 7) - 3).collect();
        let encoded = SlpCompact::encode(&original);
        assert_eq!(encoded.decode(), original);
        for (i, expected) in original.iter().enumerate() {
            assert_eq!(encoded.get(i), Some(*expected));
        }
    }

    #[test]
    fn slp_compact_i16_residuals() {
        // Values with larger deviations that need i16
        let original: Vec<i64> = (0..100).map(|i| 1000 + i * 50 + (i % 3) * 200 - 200).collect();
        let encoded = SlpCompact::encode(&original);
        assert_eq!(encoded.decode(), original);
    }

    #[test]
    fn slp_compact_sum() {
        let original: Vec<i64> = (0..500).map(|i| 10 + i * 3 + (i % 5)).collect();
        let encoded = SlpCompact::encode(&original);
        let expected_sum: i64 = original.iter().sum();
        assert_eq!(encoded.sum_i64(), expected_sum);
    }

    #[test]
    fn slp_compact_segment_break() {
        // Force a segment break with large jump
        let mut original: Vec<i64> = (0..50).map(|i| i * 10).collect();
        original.extend((0..50).map(|i| 100000 + i * 7));
        let encoded = SlpCompact::encode(&original);
        assert_eq!(encoded.decode(), original);
        assert!(encoded.segments.len() >= 2);
    }
}
