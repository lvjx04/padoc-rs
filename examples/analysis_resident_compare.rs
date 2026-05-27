use ahash::AHashMap;
use padoc::analysis;
use padoc::baselines;
use padoc::event::{
    ArgColumn, DigitColumn, Event, NameNums, NumColumn, PhaseColumn, StringColumn, Template,
};
use padoc::node::Node;
use padoc::trace::{CompressedTrace, Trace};
use std::env;
use std::mem::size_of;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let compressor_name = args
        .next()
        .expect("usage: analysis_resident_compare <compressor> <artifact> [task]");
    let artifact = PathBuf::from(
        args.next()
            .expect("usage: analysis_resident_compare <compressor> <artifact> [task]"),
    );
    let task_name = args.next();

    let artifact_bytes = std::fs::metadata(&artifact)?.len();
    let read_start = Instant::now();
    let bytes = std::fs::read(&artifact)?;
    let read_secs = read_start.elapsed().as_secs_f64();

    let load_start = Instant::now();
    let loaded = if compressor_name == "padoc" {
        Loaded::Padoc(CompressedTrace::from_bytes(&bytes)?)
    } else {
        let registry = baselines::registry();
        let compressor = registry
            .iter()
            .find(|compressor| compressor.name() == compressor_name)
            .ok_or_else(|| anyhow::anyhow!("unknown compressor `{compressor_name}`"))?;
        Loaded::Raw(compressor.decompress(&bytes)?)
    };
    drop(bytes);
    let load_secs = load_start.elapsed().as_secs_f64();
    let event_count = loaded.event_count();
    let resident_bytes = match &loaded {
        Loaded::Padoc(trace) => estimate_compressed_resident(trace),
        Loaded::Raw(trace) => estimate_raw_trace_resident(trace),
    };

    let (analyze_secs, rows, attributed, total, fraction) = if let Some(task_name) = &task_name {
        let registry = analysis::registry();
        let task = registry
            .iter()
            .find(|task| task.name() == task_name)
            .ok_or_else(|| anyhow::anyhow!("unknown task `{task_name}`"))?;
        let start = Instant::now();
        let result = match &loaded {
            Loaded::Padoc(trace) => task.run_in_situ(trace)?,
            Loaded::Raw(trace) => task.run_raw(trace)?,
        };
        let analyze_secs = start.elapsed().as_secs_f64();
        let (rows, attributed, total, fraction) = summarize_result(&result);
        (analyze_secs, rows, attributed, total, fraction)
    } else {
        (0.0, 0, 0, 0, 0.0)
    };

    println!(
        "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}",
        compressor_name,
        artifact.display(),
        task_name.as_deref().unwrap_or("none"),
        artifact_bytes,
        read_secs,
        load_secs,
        analyze_secs,
        event_count,
        resident_bytes,
        rows,
        attributed,
        total,
        fraction,
        peak_rss_kb(),
    );
    Ok(())
}

enum Loaded {
    Padoc(CompressedTrace),
    Raw(Trace),
}

impl Loaded {
    fn event_count(&self) -> usize {
        match self {
            Loaded::Padoc(trace) => trace
                .templates
                .iter()
                .map(|template| template.instance_count())
                .sum(),
            Loaded::Raw(trace) => trace.event_count(),
        }
    }
}

fn summarize_result(result: &serde_json::Value) -> (usize, u64, u64, f64) {
    let value = result.get("result").unwrap_or(result);
    let coverage = value.get("coverage").or_else(|| {
        value
            .as_array()
            .and_then(|rows| rows.first())
            .and_then(|row| row.get("coverage"))
    });
    let rows = value
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .or_else(|| value.as_array().map(Vec::len))
        .unwrap_or(0);
    let attributed = coverage
        .and_then(|coverage| coverage.get("attributed_gpu_refs"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total = coverage
        .and_then(|coverage| coverage.get("total_gpu_refs"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let fraction = coverage
        .and_then(|coverage| coverage.get("attributed_fraction"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    (rows, attributed, total, fraction)
}

fn estimate_raw_trace_resident(trace: &Trace) -> usize {
    let mut bytes = trace.ranks.len() * size_of::<(String, padoc::trace::StreamMap)>();
    bytes += trace.metadata.len() * size_of::<(String, AHashMap<String, serde_json::Value>)>();
    bytes += trace.start_timestamp.len() * size_of::<(String, i64)>();
    for (rank, processes) in &trace.ranks {
        bytes += rank.capacity();
        bytes += processes.len()
            * size_of::<(
                i64,
                indexmap::IndexMap<String, indexmap::IndexMap<padoc::event::Phase, Vec<Event>>>,
            )>();
        for threads in processes.values() {
            bytes += threads.len()
                * size_of::<(String, indexmap::IndexMap<padoc::event::Phase, Vec<Event>>)>();
            for (tid, phases) in threads {
                bytes += tid.capacity();
                bytes += phases.len() * size_of::<(padoc::event::Phase, Vec<Event>)>();
                for events in phases.values() {
                    bytes += events.capacity() * size_of::<Event>();
                    for event in events {
                        bytes += event.name.capacity();
                        bytes += event.tid.capacity();
                        bytes += event.cat.as_ref().map(|v| v.capacity()).unwrap_or(0);
                        bytes += event.bp.as_ref().map(|v| v.capacity()).unwrap_or(0);
                        bytes += event.s.as_ref().map(|v| v.capacity()).unwrap_or(0);
                        if let Some(args) = &event.args {
                            bytes += args.len() * size_of::<(String, serde_json::Value)>();
                            for (key, value) in args {
                                bytes += key.capacity();
                                bytes += json_value_payload_bytes(value);
                            }
                        }
                    }
                }
            }
        }
    }
    bytes
}

fn json_value_payload_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(value) => value.capacity(),
        serde_json::Value::Array(values) => {
            values.capacity() * size_of::<serde_json::Value>()
                + values.iter().map(json_value_payload_bytes).sum::<usize>()
        }
        serde_json::Value::Object(values) => {
            values.len() * size_of::<(String, serde_json::Value)>()
                + values
                    .iter()
                    .map(|(key, value)| key.capacity() + json_value_payload_bytes(value))
                    .sum::<usize>()
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => 0,
    }
}

fn estimate_compressed_resident(trace: &CompressedTrace) -> usize {
    let mut stats = CompressedStats::default();
    for template in &trace.templates {
        match template {
            Template::Cpu(cpu) => {
                stats.string_payload_bytes += cpu.name_pattern.len();
                stats.string_payload_bytes +=
                    cpu.cat.as_ref().map(|value| value.len()).unwrap_or(0);
                stats.string_payload_bytes += cpu.bp.as_ref().map(|value| value.len()).unwrap_or(0);
                stats.string_payload_bytes += cpu.s.as_ref().map(|value| value.len()).unwrap_or(0);
                stats.string_payload_bytes +=
                    cpu.arg_keys.iter().map(|value| value.len()).sum::<usize>();
                stats.num_bytes += num_column_bytes(&cpu.ts);
                stats.num_bytes += num_column_bytes(&cpu.dur);
                stats.num_bytes += num_column_bytes(&cpu.id);
                count_name_nums(&cpu.name_nums, &mut stats);
                count_arg_columns(&cpu.args_columns, &mut stats);
            }
            Template::Gpu(gpu) => {
                stats.string_payload_bytes += gpu.name_pattern.len();
                stats.string_payload_bytes +=
                    gpu.cat.as_ref().map(|value| value.len()).unwrap_or(0);
                stats.string_payload_bytes +=
                    gpu.arg_keys.iter().map(|value| value.len()).sum::<usize>();
                stats.num_bytes += num_column_bytes(&gpu.ts);
                stats.num_bytes += num_column_bytes(&gpu.dur);
                stats.num_bytes += num_column_bytes(&gpu.pid);
                stats.num_bytes += phase_column_bytes(&gpu.ph);
                stats.num_bytes += string_column_bytes(&gpu.stream_tid);
                count_name_nums(&gpu.name_nums, &mut stats);
                count_arg_columns(&gpu.args_columns, &mut stats);
            }
        }
    }
    for processes in trace.ranks.values() {
        for threads in processes.values() {
            for phases in threads.values() {
                for root in phases.values() {
                    count_node(root, &mut stats);
                }
            }
        }
    }
    stats.num_bytes
        + stats.name_num_vec_bytes
        + stats.name_num_payload_bytes
        + stats.node_vec_bytes
        + stats.node_u32_vec_bytes
        + stats.arg_vec_bytes
        + stats.arg_payload_bytes
        + stats.string_payload_bytes
}

#[derive(Default)]
struct CompressedStats {
    num_bytes: usize,
    name_num_vec_bytes: usize,
    name_num_payload_bytes: usize,
    node_vec_bytes: usize,
    node_u32_vec_bytes: usize,
    arg_vec_bytes: usize,
    arg_payload_bytes: usize,
    string_payload_bytes: usize,
}

fn num_column_bytes(col: &NumColumn) -> usize {
    match col {
        NumColumn::Empty => 0,
        NumColumn::Constant { .. } => size_of::<i64>() + size_of::<u32>(),
        NumColumn::I32(values) => values.capacity() * size_of::<i32>(),
        NumColumn::I64(values) => values.capacity() * size_of::<i64>(),
    }
}

fn phase_column_bytes(col: &PhaseColumn) -> usize {
    match col {
        PhaseColumn::Empty => 0,
        PhaseColumn::Constant { .. } => size_of::<u8>() + size_of::<u32>(),
        PhaseColumn::PerInstance(values) => values.capacity() * size_of::<u8>(),
    }
}

fn string_column_bytes(col: &StringColumn) -> usize {
    match col {
        StringColumn::Empty => 0,
        StringColumn::Constant { value, .. } => size_of::<String>() + value.capacity(),
        StringColumn::PerInstance(values) => {
            values.capacity() * size_of::<String>()
                + values.iter().map(|value| value.capacity()).sum::<usize>()
        }
    }
}

fn count_arg_columns(cols: &[ArgColumn], stats: &mut CompressedStats) {
    for col in cols {
        match col {
            ArgColumn::Constant(value) => {
                stats.arg_payload_bytes += value.to_string().len();
            }
            ArgColumn::I32(values) => stats.arg_vec_bytes += values.capacity() * size_of::<i32>(),
            ArgColumn::I64(values) => stats.arg_vec_bytes += values.capacity() * size_of::<i64>(),
            ArgColumn::F64(values) => stats.arg_vec_bytes += values.capacity() * size_of::<f64>(),
            ArgColumn::Bool(values) => stats.arg_vec_bytes += values.capacity() * size_of::<u8>(),
            ArgColumn::Str(values) => {
                stats.arg_vec_bytes += values.capacity() * size_of::<String>();
                stats.arg_payload_bytes +=
                    values.iter().map(|value| value.capacity()).sum::<usize>();
            }
            ArgColumn::StrDict { dict, ids } => {
                stats.arg_vec_bytes += ids.capacity() * size_of::<u32>();
                stats.arg_vec_bytes += dict.capacity() * size_of::<String>();
                stats.arg_payload_bytes += dict.iter().map(|value| value.capacity()).sum::<usize>();
            }
            ArgColumn::PerInstance(values) => {
                stats.arg_vec_bytes += values.capacity() * size_of::<serde_json::Value>();
                stats.arg_payload_bytes += values
                    .iter()
                    .map(|value| value.to_string().len())
                    .sum::<usize>();
            }
        }
    }
}

fn count_name_nums(nums: &NameNums, stats: &mut CompressedStats) {
    match nums {
        NameNums::Empty => {}
        NameNums::Rows(rows) => {
            stats.name_num_vec_bytes += rows.capacity() * size_of::<Vec<String>>();
            for row in rows {
                stats.name_num_vec_bytes += row.capacity() * size_of::<String>();
                stats.name_num_payload_bytes +=
                    row.iter().map(|value| value.capacity()).sum::<usize>();
            }
        }
        NameNums::Columnar(cols) => {
            stats.name_num_vec_bytes += cols.capacity() * size_of::<DigitColumn>();
            for col in cols {
                match col {
                    DigitColumn::Constant(value) => {
                        stats.name_num_payload_bytes += value.capacity()
                    }
                    DigitColumn::I32 { values, .. } => {
                        stats.name_num_vec_bytes += values.capacity() * size_of::<i32>();
                    }
                    DigitColumn::I64 { values, .. } => {
                        stats.name_num_vec_bytes += values.capacity() * size_of::<i64>();
                    }
                    DigitColumn::Strings(values) => {
                        stats.name_num_vec_bytes += values.capacity() * size_of::<String>();
                        stats.name_num_payload_bytes +=
                            values.iter().map(|value| value.capacity()).sum::<usize>();
                    }
                }
            }
        }
    }
}

fn count_node(node: &Node, stats: &mut CompressedStats) {
    match node {
        Node::Root { children } => {
            stats.node_vec_bytes += children.capacity() * size_of::<Node>();
            for child in children {
                count_node(child, stats);
            }
        }
        Node::Cpu(cpu) => {
            stats.node_vec_bytes += cpu.children.capacity() * size_of::<Node>();
            stats.node_vec_bytes += cpu.slots.capacity() * size_of::<Node>();
            for child in &cpu.children {
                count_node(child, stats);
            }
            for child in &cpu.slots {
                count_node(child, stats);
            }
        }
        Node::SameCpu(same) => {
            stats.node_u32_vec_bytes += same.instances.capacity() * size_of::<u32>();
            stats.node_vec_bytes += same.children.capacity() * size_of::<Node>();
            stats.node_vec_bytes += same.slots.capacity() * size_of::<padoc::node::SameCpuSlot>();
            for child in &same.children {
                count_node(child, stats);
            }
            for slot in same.slots.entries() {
                stats.node_vec_bytes += slot.children.capacity() * size_of::<Node>();
                for child in &slot.children {
                    count_node(child, stats);
                }
            }
        }
        Node::Gpu(gpu) => {
            stats.node_u32_vec_bytes += gpu.templates.capacity() * size_of::<u32>();
            stats.node_u32_vec_bytes += gpu.instances.capacity() * size_of::<u32>();
        }
        Node::KernelLaunch(_) => {}
        Node::KernelsLaunch(kernels) => {
            stats.node_u32_vec_bytes += kernels.cpu_instances.capacity() * size_of::<u32>();
            stats.node_u32_vec_bytes += kernels.gpu_templates.capacity() * size_of::<u32>();
            stats.node_u32_vec_bytes += kernels.gpu_instances.capacity() * size_of::<u32>();
        }
    }
}

fn peak_rss_kb() -> u64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            ru.ru_maxrss as u64
        } else {
            0
        }
    }
}
