//! Measure accounted resident memory of a loaded .padoc.zst artifact,
//! broken down by column type.

use padoc::event::{ArgColumn, DigitColumn, NameNums, NumColumn, StringColumn, Template};
use padoc::trace::CompressedTrace;
use std::mem::size_of;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: measure_accounted <artifact.padoc.zst>");

    let ct = CompressedTrace::read_from_path(&path)?;

    let mut ts_bytes = 0usize;
    let mut dur_bytes = 0usize;
    let mut id_bytes = 0usize;
    let mut gpu_pid_stream = 0usize;
    let mut args_bytes = 0usize;
    let mut name_nums_bytes = 0usize;

    for tmpl in &ct.templates {
        match tmpl {
            Template::Cpu(t) => {
                ts_bytes += num_bytes(&t.ts);
                dur_bytes += num_bytes(&t.dur);
                id_bytes += num_bytes(&t.id);
                args_bytes += args_total(&t.args_columns);
                name_nums_bytes += name_nums_total(&t.name_nums);
            }
            Template::Gpu(t) => {
                ts_bytes += num_bytes(&t.ts);
                dur_bytes += num_bytes(&t.dur);
                gpu_pid_stream += num_bytes(&t.pid);
                gpu_pid_stream += str_col_bytes(&t.stream_tid);
                args_bytes += args_total(&t.args_columns);
                name_nums_bytes += name_nums_total(&t.name_nums);
            }
        }
    }

    let arena_bytes: usize = ct
        .arenas
        .as_ref()
        .map(|arenas| {
            arenas
                .values()
                .flat_map(|p| p.values().flat_map(|t| t.values().flat_map(|ph| ph.values())))
                .map(|(arena, _)| arena.heap_bytes())
                .sum()
        })
        .unwrap_or(0);

    let total = ts_bytes + dur_bytes + id_bytes + gpu_pid_stream + args_bytes + name_nums_bytes + arena_bytes;

    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        path, ts_bytes, dur_bytes, id_bytes, gpu_pid_stream, args_bytes, name_nums_bytes,
        arena_bytes, total
    );
    eprintln!(
        "ts={:.3} dur={:.3} id={:.3} gpu_pid_stream={:.3} args={:.3} name_nums={:.3} arena={:.3} TOTAL={:.3} GiB",
        ts_bytes as f64 / 1e9 * 0.931,
        dur_bytes as f64 / 1e9 * 0.931,
        id_bytes as f64 / 1e9 * 0.931,
        gpu_pid_stream as f64 / 1e9 * 0.931,
        args_bytes as f64 / 1e9 * 0.931,
        name_nums_bytes as f64 / 1e9 * 0.931,
        arena_bytes as f64 / 1e9 * 0.931,
        total as f64 / 1e9 * 0.931,
    );
    Ok(())
}

fn num_bytes(col: &NumColumn) -> usize {
    match col {
        NumColumn::Empty => 0,
        NumColumn::Constant { .. } => 12,
        NumColumn::I32(v) => v.capacity() * 4,
        NumColumn::I64(v) => v.capacity() * 8,
        NumColumn::Slp(slp) => slp.heap_bytes(),
    }
}

fn str_col_bytes(col: &StringColumn) -> usize {
    match col {
        StringColumn::Empty => 0,
        StringColumn::Constant { value, .. } => size_of::<String>() + value.capacity(),
        StringColumn::PerInstance(v) => {
            v.capacity() * size_of::<String>() + v.iter().map(|s| s.capacity()).sum::<usize>()
        }
    }
}

fn args_total(cols: &[ArgColumn]) -> usize {
    cols.iter()
        .map(|col| match col {
            ArgColumn::Constant(v) => v.to_string().len(),
            ArgColumn::I32(v) => v.capacity() * 4,
            ArgColumn::I64(v) => v.capacity() * 8,
            ArgColumn::F64(v) => v.capacity() * 8,
            ArgColumn::Bool(v) => v.capacity(),
            ArgColumn::Str(v) => {
                v.capacity() * size_of::<String>() + v.iter().map(|s| s.capacity()).sum::<usize>()
            }
            ArgColumn::StrDict { dict, ids } => {
                ids.capacity() * 4
                    + dict.capacity() * size_of::<String>()
                    + dict.iter().map(|s| s.capacity()).sum::<usize>()
            }
            ArgColumn::PerInstance(v) => {
                v.capacity() * size_of::<serde_json::Value>()
                    + v.iter().map(|x| x.to_string().len()).sum::<usize>()
            }
            ArgColumn::SlpI32(slp) => slp.heap_bytes(),
        })
        .sum()
}

fn name_nums_total(nums: &NameNums) -> usize {
    match nums {
        NameNums::Empty => 0,
        NameNums::Rows(rows) => {
            rows.capacity() * size_of::<Vec<String>>()
                + rows
                    .iter()
                    .map(|r| {
                        r.capacity() * size_of::<String>()
                            + r.iter().map(|s| s.capacity()).sum::<usize>()
                    })
                    .sum::<usize>()
        }
        NameNums::Columnar(cols) => {
            cols.capacity() * size_of::<DigitColumn>()
                + cols
                    .iter()
                    .map(|c| match c {
                        DigitColumn::Constant(v) => v.capacity(),
                        DigitColumn::I32 { values, .. } => values.capacity() * 4,
                        DigitColumn::I64 { values, .. } => values.capacity() * 8,
                        DigitColumn::Strings(v) => {
                            v.capacity() * size_of::<String>()
                                + v.iter().map(|s| s.capacity()).sum::<usize>()
                        }
                    })
                    .sum::<usize>()
        }
    }
}
