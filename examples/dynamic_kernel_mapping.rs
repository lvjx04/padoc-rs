use ahash::{AHashMap, AHashSet};
use once_cell::sync::Lazy;
use padoc::event::{ArgColumn, NameNums, Template};
use padoc::node::{InstanceId, Node, TemplateId};
use padoc::slp::decode_name_nums;
use padoc::trace::CompressedTrace;
use regex::Regex;
use serde_json::Value;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

static LAYER_PATTERN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?:^|[^A-Za-z0-9])layers?[._/-]0(?:[^0-9]|$)|(?:^|[^A-Za-z0-9])(?:[A-Za-z]*Layer|[A-Za-z]*Block|ResBlock|ViTLayer)[_-]?0(?:[^0-9]|$)",
    )
    .unwrap()
});

const REPEATED_SCOPE_MIN_INSTANCES: usize = 2;
const REPEATED_SCOPE_MAX_INSTANCES: usize = 512;

#[derive(Clone, Copy)]
struct GpuKernel {
    tmpl_id: TemplateId,
    inst_id: InstanceId,
}

#[derive(Default, Clone, Copy)]
struct Coverage {
    attributed_gpu_refs: u64,
    total_gpu_refs: u64,
}

#[derive(Default)]
struct KernelAgg {
    count: u64,
    total_dur_us: i64,
}

#[derive(Default)]
struct IntervalAgg {
    compute: Vec<(i64, i64)>,
    comm: Vec<(i64, i64)>,
    compute_total_us: i64,
    comm_total_us: i64,
}

#[derive(Default, Clone)]
struct RankLayerAgg {
    compute_us: i64,
    comm_us: i64,
}

#[derive(Clone, Default)]
enum ActiveLayer {
    #[default]
    None,
    One(String),
    Many(Arc<[Option<String>]>),
}

impl ActiveLayer {
    fn from_option(layer: Option<String>) -> Self {
        match layer {
            Some(layer) => ActiveLayer::One(layer),
            None => ActiveLayer::None,
        }
    }

    fn from_layers(layers: Vec<Option<String>>) -> Self {
        let mut unique = layers.iter().filter_map(|layer| layer.as_deref());
        let Some(first) = unique.next() else {
            return ActiveLayer::None;
        };
        if unique.all(|layer| layer == first) && layers.iter().all(Option::is_some) {
            ActiveLayer::One(first.to_string())
        } else {
            ActiveLayer::Many(layers.into())
        }
    }

    fn or(self, fallback: ActiveLayer) -> Self {
        match self {
            ActiveLayer::None => fallback,
            _ => self,
        }
    }

    fn at(&self, idx: usize) -> Option<String> {
        match self {
            ActiveLayer::None => None,
            ActiveLayer::One(layer) => Some(layer.clone()),
            ActiveLayer::Many(layers) => layers.get(idx).cloned().flatten(),
        }
    }

    fn scalar(&self) -> Option<String> {
        match self {
            ActiveLayer::One(layer) => Some(layer.clone()),
            ActiveLayer::None | ActiveLayer::Many(_) => None,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let artifact = PathBuf::from(
        args.next()
            .expect("usage: dynamic_kernel_mapping <artifact.padoc.zst> <task>"),
    );
    let task = args
        .next()
        .expect("usage: dynamic_kernel_mapping <artifact.padoc.zst> <task>");

    let load_start = Instant::now();
    let trace = CompressedTrace::read_from_path(&artifact)?;
    let load_secs = load_start.elapsed().as_secs_f64();

    let index_start = Instant::now();
    let gpu_by_corr = build_gpu_correlation_index(&trace);
    let index_secs = index_start.elapsed().as_secs_f64();

    let analyze_start = Instant::now();
    let (coverage, rows) = match task.as_str() {
        "layer_kernel_hotspot" => run_hotspot(&trace, &gpu_by_corr),
        "layer_compute_comm_overlap" => run_overlap(&trace, &gpu_by_corr),
        "layer_rank_balance" => run_rank_balance(&trace, &gpu_by_corr),
        other => anyhow::bail!("unknown task `{other}`"),
    };
    let analyze_secs = analyze_start.elapsed().as_secs_f64();
    println!(
        "padoc_dynamic_mapping\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.6}\t{}",
        artifact.display(),
        task,
        load_secs,
        index_secs,
        analyze_secs,
        coverage.attributed_gpu_refs,
        coverage.total_gpu_refs,
        attributed_fraction(coverage),
        rows
    );
    Ok(())
}

fn build_gpu_correlation_index(trace: &CompressedTrace) -> AHashMap<(String, i64), GpuKernel> {
    let mut out = AHashMap::new();
    for (rank, processes) in &trace.ranks {
        for threads in processes.values() {
            for phases in threads.values() {
                for root in phases.values() {
                    collect_gpu_correlation_from_node(trace, rank, root, &mut out);
                }
            }
        }
    }
    out
}

fn collect_gpu_correlation_from_node(
    trace: &CompressedTrace,
    rank: &str,
    node: &Node,
    out: &mut AHashMap<(String, i64), GpuKernel>,
) {
    match node {
        Node::Root { children } => {
            for child in children {
                collect_gpu_correlation_from_node(trace, rank, child, out);
            }
        }
        Node::Cpu(n) => {
            for child in &n.children {
                collect_gpu_correlation_from_node(trace, rank, child, out);
            }
            for child in &n.slots {
                collect_gpu_correlation_from_node(trace, rank, child, out);
            }
        }
        Node::SameCpu(n) => {
            for child in &n.children {
                collect_gpu_correlation_from_node(trace, rank, child, out);
            }
            for slot in n.slots.entries() {
                for child in &slot.children {
                    collect_gpu_correlation_from_node(trace, rank, child, out);
                }
            }
        }
        Node::Gpu(n) => {
            for (tmpl_id, inst_id) in n.templates.iter().zip(n.instances.iter()) {
                insert_gpu_corr(trace, rank, *tmpl_id, *inst_id, out);
            }
        }
        Node::KernelLaunch(n) => {
            insert_gpu_corr(trace, rank, n.gpu_template, n.gpu_instance, out);
        }
        Node::KernelsLaunch(n) => {
            for (tmpl_id, inst_id) in n.gpu_templates.iter().zip(n.gpu_instances.iter()) {
                insert_gpu_corr(trace, rank, *tmpl_id, *inst_id, out);
            }
        }
    }
}

fn insert_gpu_corr(
    trace: &CompressedTrace,
    rank: &str,
    tmpl_id: TemplateId,
    inst_id: InstanceId,
    out: &mut AHashMap<(String, i64), GpuKernel>,
) {
    let Some(Template::Gpu(g)) = trace.templates.get(tmpl_id as usize) else {
        return;
    };
    if g.cat.as_deref() != Some("kernel") {
        return;
    }
    let Some(corr) = arg_i64(
        &g.arg_keys,
        &g.args_columns,
        inst_id as usize,
        "correlation",
    )
    .or_else(|| {
        arg_i64(
            &g.arg_keys,
            &g.args_columns,
            inst_id as usize,
            "External id",
        )
    }) else {
        return;
    };
    out.entry((rank.to_string(), corr))
        .or_insert(GpuKernel { tmpl_id, inst_id });
}

fn run_hotspot(
    trace: &CompressedTrace,
    gpu_by_corr: &AHashMap<(String, i64), GpuKernel>,
) -> (Coverage, usize) {
    let mut tally: AHashMap<(String, String), KernelAgg> = AHashMap::new();
    let mut used_gpu: AHashSet<(String, TemplateId, InstanceId)> = AHashSet::new();
    let mut coverage = Coverage {
        total_gpu_refs: total_gpu_kernel_refs(trace),
        ..Coverage::default()
    };
    walk_rank_layer_dynamic(trace, gpu_by_corr, |rank, layer, gpu| {
        if !used_gpu.insert((rank.to_string(), gpu.tmpl_id, gpu.inst_id)) {
            return;
        }
        let Some(Template::Gpu(g)) = trace.templates.get(gpu.tmpl_id as usize) else {
            return;
        };
        if g.cat.as_deref() != Some("kernel") {
            return;
        }
        coverage.attributed_gpu_refs += 1;
        let key = (layer.to_string(), g.name_pattern.clone());
        let entry = tally.entry(key).or_default();
        entry.count += 1;
        entry.total_dur_us += g.dur.get(gpu.inst_id as usize).unwrap_or(0);
    });
    let mut rows: Vec<_> = tally
        .into_iter()
        .map(|((layer, kernel), agg)| {
            serde_json::json!({
                "layer": layer,
                "kernel": kernel,
                "count": agg.count,
                "total_dur_us": agg.total_dur_us,
                "avg_dur_us": if agg.count > 0 {
                    agg.total_dur_us as f64 / agg.count as f64
                } else {
                    0.0
                },
            })
        })
        .collect();
    rows.sort_by(|a, b| {
        b["total_dur_us"]
            .as_i64()
            .unwrap_or(0)
            .cmp(&a["total_dur_us"].as_i64().unwrap_or(0))
    });
    rows.truncate(20);
    let _result = serde_json::json!({
        "coverage": coverage_json(coverage),
        "rows": rows,
    });
    (
        coverage,
        _result["rows"].as_array().map(Vec::len).unwrap_or(0),
    )
}

fn run_overlap(
    trace: &CompressedTrace,
    gpu_by_corr: &AHashMap<(String, i64), GpuKernel>,
) -> (Coverage, usize) {
    let mut by_rank_layer: AHashMap<(String, String), IntervalAgg> = AHashMap::new();
    let mut used_gpu: AHashSet<(String, TemplateId, InstanceId)> = AHashSet::new();
    let mut coverage = Coverage {
        total_gpu_refs: total_gpu_kernel_refs(trace),
        ..Coverage::default()
    };
    walk_rank_layer_dynamic(trace, gpu_by_corr, |rank, layer, gpu| {
        if !used_gpu.insert((rank.to_string(), gpu.tmpl_id, gpu.inst_id)) {
            return;
        }
        let Some(Template::Gpu(g)) = trace.templates.get(gpu.tmpl_id as usize) else {
            return;
        };
        if g.cat.as_deref() != Some("kernel") {
            return;
        }
        let ts = g.ts.get(gpu.inst_id as usize).unwrap_or(0);
        let dur = g.dur.get(gpu.inst_id as usize).unwrap_or(0);
        if dur <= 0 {
            return;
        }
        coverage.attributed_gpu_refs += 1;
        let entry = by_rank_layer
            .entry((rank.to_string(), layer.to_string()))
            .or_default();
        push_interval(entry, &g.name_pattern, ts, dur);
    });
    let mut rows: Vec<Value> = by_rank_layer
        .into_iter()
        .map(|((rank, layer), agg)| {
            let compute_union = union_len(agg.compute.clone());
            let comm_union = union_len(agg.comm.clone());
            let overlap = overlap_len(agg.compute, agg.comm);
            let denom = compute_union.min(comm_union);
            serde_json::json!({
                "rank": rank,
                "layer": layer,
                "compute_total_us": agg.compute_total_us,
                "comm_total_us": agg.comm_total_us,
                "compute_union_us": compute_union,
                "comm_union_us": comm_union,
                "overlap_us": overlap,
                "overlap_fraction_of_min_union": if denom > 0 {
                    overlap as f64 / denom as f64
                } else {
                    0.0
                },
            })
        })
        .collect();
    rows.sort_by(|a, b| {
        a["rank"]
            .as_str()
            .unwrap_or("")
            .cmp(b["rank"].as_str().unwrap_or(""))
            .then_with(|| {
                a["layer"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["layer"].as_str().unwrap_or(""))
            })
    });
    let _result = serde_json::json!({
        "coverage": coverage_json(coverage),
        "rows": rows,
    });
    let rows = _result["rows"].as_array().map(Vec::len).unwrap_or(0);
    (coverage, rows)
}

fn run_rank_balance(
    trace: &CompressedTrace,
    gpu_by_corr: &AHashMap<(String, i64), GpuKernel>,
) -> (Coverage, usize) {
    let mut by_rank_layer: AHashMap<(String, String), RankLayerAgg> = AHashMap::new();
    let mut used_gpu: AHashSet<(String, TemplateId, InstanceId)> = AHashSet::new();
    let mut coverage = Coverage {
        total_gpu_refs: total_gpu_kernel_refs(trace),
        ..Coverage::default()
    };
    walk_rank_layer_dynamic(trace, gpu_by_corr, |rank, layer, gpu| {
        if !used_gpu.insert((rank.to_string(), gpu.tmpl_id, gpu.inst_id)) {
            return;
        }
        let Some(Template::Gpu(g)) = trace.templates.get(gpu.tmpl_id as usize) else {
            return;
        };
        if g.cat.as_deref() != Some("kernel") {
            return;
        }
        let dur = g.dur.get(gpu.inst_id as usize).unwrap_or(0);
        coverage.attributed_gpu_refs += 1;
        let entry = by_rank_layer
            .entry((rank.to_string(), layer.to_string()))
            .or_default();
        if is_nccl_kernel(&g.name_pattern) {
            entry.comm_us += dur;
        } else {
            entry.compute_us += dur;
        }
    });
    let mut ranks: Vec<String> = by_rank_layer.keys().map(|(rank, _)| rank.clone()).collect();
    ranks.sort();
    ranks.dedup();
    let mut layers: Vec<String> = by_rank_layer
        .keys()
        .map(|(_, layer)| layer.clone())
        .collect();
    layers.sort();
    layers.dedup();
    let mut rows = Vec::with_capacity(layers.len());
    for layer in layers {
        let mut compute_values = Vec::with_capacity(ranks.len());
        let mut comm_values = Vec::with_capacity(ranks.len());
        let mut total_values = Vec::with_capacity(ranks.len());
        for rank in &ranks {
            let agg = by_rank_layer
                .get(&(rank.clone(), layer.clone()))
                .cloned()
                .unwrap_or_default();
            compute_values.push(agg.compute_us);
            comm_values.push(agg.comm_us);
            total_values.push(agg.compute_us + agg.comm_us);
        }
        rows.push(serde_json::json!({
            "layer": layer,
            "compute": metric_summary(&compute_values),
            "comm": metric_summary(&comm_values),
            "total": metric_summary(&total_values),
        }));
    }
    rows.sort_by(|a, b| {
        let ai = a["total"]["imbalance_max_min_over_mean"]
            .as_f64()
            .unwrap_or(0.0);
        let bi = b["total"]["imbalance_max_min_over_mean"]
            .as_f64()
            .unwrap_or(0.0);
        bi.partial_cmp(&ai).unwrap_or(std::cmp::Ordering::Equal)
    });
    let _result = serde_json::json!({
        "coverage": coverage_json(coverage),
        "rank_count": ranks.len(),
        "rows": rows,
    });
    let rows = _result["rows"].as_array().map(Vec::len).unwrap_or(0);
    (coverage, rows)
}

fn walk_rank_layer_dynamic(
    trace: &CompressedTrace,
    gpu_by_corr: &AHashMap<(String, i64), GpuKernel>,
    mut f: impl FnMut(&str, &str, GpuKernel),
) {
    for (rank, processes) in &trace.ranks {
        for threads in processes.values() {
            for phases in threads.values() {
                for root in phases.values() {
                    walk_node_for_layers(trace, rank, root, ActiveLayer::None, gpu_by_corr, &mut f);
                }
            }
        }
    }
}

fn walk_node_for_layers(
    trace: &CompressedTrace,
    rank: &str,
    node: &Node,
    active_layer: ActiveLayer,
    gpu_by_corr: &AHashMap<(String, i64), GpuKernel>,
    f: &mut impl FnMut(&str, &str, GpuKernel),
) {
    match node {
        Node::Root { children } => {
            for child in children {
                walk_node_for_layers(trace, rank, child, active_layer.clone(), gpu_by_corr, f);
            }
        }
        Node::Cpu(n) => {
            let next_layer =
                ActiveLayer::from_option(cpu_instance_layer(trace, n.template, n.instance))
                    .or(active_layer);
            for child in &n.children {
                walk_node_for_layers(trace, rank, child, next_layer.clone(), gpu_by_corr, f);
            }
            for child in &n.slots {
                walk_node_for_layers(trace, rank, child, next_layer.clone(), gpu_by_corr, f);
            }
        }
        Node::SameCpu(n) => {
            let repeated_scope = repeated_scope_layers(trace, n.template, n.instances.len());
            let next_layer = ActiveLayer::from_layers(
                n.instances
                    .iter()
                    .enumerate()
                    .map(|(idx, inst)| {
                        cpu_instance_layer(trace, n.template, *inst)
                            .or_else(|| {
                                repeated_scope
                                    .as_ref()
                                    .map(|scope| format!("{scope}#{idx}"))
                            })
                            .or_else(|| active_layer.at(idx))
                    })
                    .collect(),
            )
            .or(active_layer);
            for child in &n.children {
                walk_node_for_layers(trace, rank, child, next_layer.clone(), gpu_by_corr, f);
            }
            for slot in n.slots.entries() {
                let slot_layer =
                    ActiveLayer::from_option(next_layer.at(slot.instance_index as usize));
                for child in &slot.children {
                    walk_node_for_layers(trace, rank, child, slot_layer.clone(), gpu_by_corr, f);
                }
            }
        }
        Node::Gpu(n) => {
            if let Some(layer) = active_layer.scalar() {
                for (tmpl_id, inst_id) in n.templates.iter().zip(n.instances.iter()) {
                    f(
                        rank,
                        &layer,
                        GpuKernel {
                            tmpl_id: *tmpl_id,
                            inst_id: *inst_id,
                        },
                    );
                }
            }
        }
        Node::KernelLaunch(n) => {
            if let Some(layer) = cpu_instance_layer(trace, n.cpu_template, n.cpu_instance)
                .or_else(|| active_layer.scalar())
            {
                if let Some(gpu) =
                    cpu_dynamic_gpu(trace, rank, n.cpu_template, n.cpu_instance, gpu_by_corr)
                {
                    f(rank, &layer, gpu);
                }
            }
        }
        Node::KernelsLaunch(n) => {
            for (idx, cpu_inst) in n.cpu_instances.iter().enumerate() {
                if let Some(layer) = cpu_instance_layer(trace, n.cpu_template, *cpu_inst)
                    .or_else(|| active_layer.at(idx))
                {
                    if let Some(gpu) =
                        cpu_dynamic_gpu(trace, rank, n.cpu_template, *cpu_inst, gpu_by_corr)
                    {
                        f(rank, &layer, gpu);
                    }
                }
            }
        }
    }
}

fn cpu_dynamic_gpu(
    trace: &CompressedTrace,
    rank: &str,
    tmpl_id: TemplateId,
    inst_id: InstanceId,
    gpu_by_corr: &AHashMap<(String, i64), GpuKernel>,
) -> Option<GpuKernel> {
    let Template::Cpu(t) = trace.templates.get(tmpl_id as usize)? else {
        return None;
    };
    let corr = arg_i64(
        &t.arg_keys,
        &t.args_columns,
        inst_id as usize,
        "correlation",
    )
    .or_else(|| {
        arg_i64(
            &t.arg_keys,
            &t.args_columns,
            inst_id as usize,
            "External id",
        )
    })?;
    gpu_by_corr.get(&(rank.to_string(), corr)).copied()
}

fn arg_i64(keys: &[String], cols: &[ArgColumn], inst_id: usize, key: &str) -> Option<i64> {
    let pos = keys.iter().position(|candidate| candidate == key)?;
    match cols.get(pos)? {
        ArgColumn::Constant(value) => value.as_i64(),
        ArgColumn::I32(values) => values.get(inst_id).map(|value| *value as i64),
        ArgColumn::I64(values) => values.get(inst_id).copied(),
        other => other.get_owned(inst_id).and_then(|value| value.as_i64()),
    }
}

fn cpu_instance_layer(
    trace: &CompressedTrace,
    tmpl_id: TemplateId,
    inst_id: InstanceId,
) -> Option<String> {
    let Template::Cpu(t) = trace.templates.get(tmpl_id as usize)? else {
        return None;
    };
    let zero_idx = layer_zero_index(&t.name_pattern)?;
    let digits = decode_name_nums(&t.name_nums, inst_id as usize);
    let layer = digits.get(zero_idx)?;
    Some(format!("{}#{}", layer_scope_name(&t.name_pattern), layer))
}

fn layer_zero_index(pattern: &str) -> Option<usize> {
    let m = LAYER_PATTERN_RE.find(pattern)?;
    let zero_in_match = pattern[m.start()..m.end()].find('0')?;
    let zero_byte_pos = m.start() + zero_in_match;
    Some(
        pattern.as_bytes()[..zero_byte_pos]
            .iter()
            .filter(|&&b| b == b'0')
            .count(),
    )
}

fn repeated_scope_layers(
    trace: &CompressedTrace,
    tmpl_id: TemplateId,
    instances: usize,
) -> Option<String> {
    if !(REPEATED_SCOPE_MIN_INSTANCES..=REPEATED_SCOPE_MAX_INSTANCES).contains(&instances) {
        return None;
    }
    let Template::Cpu(t) = trace.templates.get(tmpl_id as usize)? else {
        return None;
    };
    let scope = layer_scope_name(&t.name_pattern);
    if is_low_value_repeated_scope(&scope) {
        None
    } else {
        Some(scope)
    }
}

fn layer_scope_name(name: &str) -> String {
    let mut scope = padoc::utils::normalize_name(name);
    if let Some(pos) = scope.rfind(':') {
        let (_, tail) = scope.split_at(pos + 1);
        scope = tail.trim().to_string();
    }
    scope = scope
        .trim()
        .trim_matches(|c: char| c == '<' || c == '>' || c == '"' || c == '\'')
        .to_string();
    if scope.is_empty() {
        "scope".to_string()
    } else {
        scope
    }
}

fn is_low_value_repeated_scope(scope: &str) -> bool {
    matches!(
        scope,
        "suLaunchKernel"
            | "cudaLaunchKernel"
            | "hipLaunchKernel"
            | "aten::empty"
            | "aten::detach"
            | "detach"
            | "aten::zero_"
            | "aten::fill_"
            | "aten::copy_"
            | "aten::to"
            | "aten::_to_copy"
            | "aten::uniform_"
            | "aten::item"
            | "aten::_local_scalar_dense"
            | "AddBackward0"
            | "CloneBackward0"
    )
}

fn is_nccl_kernel(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("nccl") || lower.contains("genericmultishmop") || lower.contains("genericixccl")
}

fn push_interval(entry: &mut IntervalAgg, kernel_name: &str, ts: i64, dur: i64) {
    let interval = (ts, ts + dur);
    if is_nccl_kernel(kernel_name) {
        entry.comm.push(interval);
        entry.comm_total_us += dur;
    } else {
        entry.compute.push(interval);
        entry.compute_total_us += dur;
    }
}

fn union_len(mut intervals: Vec<(i64, i64)>) -> i64 {
    if intervals.is_empty() {
        return 0;
    }
    intervals.sort_unstable();
    let mut total = 0;
    let (mut start, mut end) = intervals[0];
    for (s, e) in intervals.into_iter().skip(1) {
        if s <= end {
            end = end.max(e);
        } else {
            total += end - start;
            start = s;
            end = e;
        }
    }
    total + end - start
}

fn overlap_len(mut a: Vec<(i64, i64)>, mut b: Vec<(i64, i64)>) -> i64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    a.sort_unstable();
    b.sort_unstable();
    let mut i = 0;
    let mut j = 0;
    let mut total = 0;
    while i < a.len() && j < b.len() {
        let start = a[i].0.max(b[j].0);
        let end = a[i].1.min(b[j].1);
        if end > start {
            total += end - start;
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

fn total_gpu_kernel_refs(trace: &CompressedTrace) -> u64 {
    trace
        .templates
        .iter()
        .map(|tmpl| match tmpl {
            Template::Gpu(g) if g.cat.as_deref() == Some("kernel") => g.dur.len() as u64,
            _ => 0,
        })
        .sum()
}

fn coverage_json(coverage: Coverage) -> Value {
    serde_json::json!({
        "attributed_gpu_refs": coverage.attributed_gpu_refs,
        "total_gpu_refs": coverage.total_gpu_refs,
        "attributed_fraction": attributed_fraction(coverage),
    })
}

fn metric_summary(values: &[i64]) -> Value {
    if values.is_empty() {
        return serde_json::json!({
            "max_us": 0,
            "min_us": 0,
            "mean_us": 0.0,
            "stddev_us": 0.0,
            "cv": 0.0,
            "imbalance_max_min_over_mean": 0.0,
        });
    }
    let max_v = *values.iter().max().unwrap();
    let min_v = *values.iter().min().unwrap();
    let n = values.len() as f64;
    let mean = values.iter().sum::<i64>() as f64 / n;
    let var = values
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let stddev = var.sqrt();
    serde_json::json!({
        "max_us": max_v,
        "min_us": min_v,
        "mean_us": mean,
        "stddev_us": stddev,
        "cv": if mean > 0.0 { stddev / mean } else { 0.0 },
        "imbalance_max_min_over_mean": if mean > 0.0 {
            (max_v - min_v) as f64 / mean
        } else {
            0.0
        },
    })
}

fn attributed_fraction(coverage: Coverage) -> f64 {
    if coverage.total_gpu_refs > 0 {
        coverage.attributed_gpu_refs as f64 / coverage.total_gpu_refs as f64
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn _name_nums_len(nums: &NameNums) -> usize {
    match nums {
        NameNums::Empty => 0,
        NameNums::Rows(rows) => rows.len(),
        NameNums::Columnar(cols) => cols.first().map(|col| col.len()).unwrap_or(0),
    }
}
