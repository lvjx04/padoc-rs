//! Layer-aware GPU attribution analyses.
//!
//! These tasks intentionally exercise the structural part of PADOC: they
//! start from CPU nodes whose names encode a model layer, then collect GPU
//! kernels reachable through `KernelLaunch` / `KernelsLaunch` descendants.
//! Without CPU->GPU provenance links these queries lose their semantics.

use ahash::AHashMap;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use std::sync::Arc;

use crate::analysis::kernel_class::is_nccl_kernel;
use crate::analysis::{elapsed_secs, profiled_result, AnalysisTask};
use crate::arena::{ArenaNode, NodeArena, NodeId};
use crate::event::Template;
use crate::node::{InstanceId, Node, TemplateId};
use crate::slp::decode_name_nums;
use crate::trace::{CompressedTrace, StreamMap, Trace};
use crate::Result;

static LAYER_PATTERN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?:^|[^A-Za-z0-9])layers?[._/-]0(?:[^0-9]|$)|(?:^|[^A-Za-z0-9])(?:[A-Za-z]*Layer|[A-Za-z]*Block|ResBlock|ViTLayer)[_-]?0(?:[^0-9]|$)",
    )
    .unwrap()
});
static RAW_LAYER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?:^|[^A-Za-z0-9])layers?[._/-](\d+)(?:[^0-9]|$)|(?:^|[^A-Za-z0-9])(?:[A-Za-z]*Layer|[A-Za-z]*Block|ResBlock|ViTLayer)[_-]?(\d+)(?:[^0-9]|$)",
    )
    .unwrap()
});

const REPEATED_SCOPE_MIN_INSTANCES: usize = 8;
const REPEATED_SCOPE_MAX_INSTANCES: usize = 512;

#[derive(Default)]
pub struct LayerKernelHotspot {
    pub top_k: usize,
}

#[derive(Default)]
pub struct LayerComputeCommOverlap;

#[derive(Default, Clone)]
struct KernelAgg {
    count: u64,
    total_dur_us: i64,
}

/// Streaming overlap state per (rank, layer) — avoids collecting all intervals.
#[derive(Default, Clone)]
struct StreamingOverlapState {
    comp_window: (i64, i64),
    comm_window: (i64, i64),
    overlap_accum: i64,
    last_overlap_end: i64,
    compute_total_us: i64,
    comm_total_us: i64,
}

impl StreamingOverlapState {
    fn push(&mut self, kernel_name: &str, ts: i64, dur: i64) {
        let end = ts + dur;
        if is_nccl_kernel(kernel_name) {
            // Update comm window
            if ts > self.comm_window.1 {
                self.comm_total_us += dur;
                self.comm_window = (ts, end);
            } else if end > self.comm_window.1 {
                self.comm_total_us += end - self.comm_window.1;
                self.comm_window.1 = end;
            } else {
                self.comm_total_us += dur; // fully contained, still count
            }
        } else {
            // Update compute window
            if ts > self.comp_window.1 {
                self.compute_total_us += dur;
                self.comp_window = (ts, end);
            } else if end > self.comp_window.1 {
                self.compute_total_us += end - self.comp_window.1;
                self.comp_window.1 = end;
            } else {
                self.compute_total_us += dur;
            }
        }
        // Compute overlap between current windows
        let overlap_start = self.comp_window.0.max(self.comm_window.0);
        let overlap_end = self.comp_window.1.min(self.comm_window.1);
        if overlap_start < overlap_end {
            if overlap_start >= self.last_overlap_end {
                self.overlap_accum += overlap_end - overlap_start;
                self.last_overlap_end = overlap_end;
            } else if overlap_end > self.last_overlap_end {
                self.overlap_accum += overlap_end - self.last_overlap_end;
                self.last_overlap_end = overlap_end;
            }
        }
    }

    fn to_json(&self, rank: &str, layer: &str) -> Value {
        let denom = self.compute_total_us.min(self.comm_total_us);
        let overlap_fraction = if denom > 0 {
            self.overlap_accum as f64 / denom as f64
        } else {
            0.0
        };
        serde_json::json!({
            "rank": rank,
            "layer": layer,
            "compute_total_us": self.compute_total_us,
            "comm_total_us": self.comm_total_us,
            "overlap_us": self.overlap_accum,
            "overlap_fraction": overlap_fraction,
        })
    }
}

#[derive(Default, Clone)]
struct IntervalAgg {
    compute: Vec<(i64, i64)>,
    comm: Vec<(i64, i64)>,
    compute_total_us: i64,
    comm_total_us: i64,
}

#[derive(Clone, Copy)]
struct GpuKernel {
    tmpl_id: TemplateId,
    inst_id: InstanceId,
}

#[derive(Clone)]
struct RawGpuKernel {
    name: String,
    ts: i64,
    dur: i64,
}

#[derive(Clone)]
struct RawLayerGpu {
    rank: String,
    layer: String,
    gpu: RawGpuKernel,
}

impl AnalysisTask for LayerKernelHotspot {
    fn name(&self) -> &str {
        "layer_kernel_hotspot"
    }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut tally: AHashMap<(String, String), KernelAgg> = AHashMap::new();
        let raw = collect_raw_layer_gpu(trace);
        for item in &raw.items {
            let key = (
                item.layer.clone(),
                crate::utils::normalize_name(&item.gpu.name),
            );
            let entry = tally.entry(key).or_default();
            entry.count += 1;
            entry.total_dur_us += item.gpu.dur;
        }
        let mut result = kernel_hotspot_json(tally, self.top_k.max(20));
        add_coverage(&mut result, raw.coverage);
        Ok(result)
    }

    fn supports_in_situ(&self) -> bool {
        true
    }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let start = std::time::Instant::now();
        let mut tally: AHashMap<(String, String), KernelAgg> = AHashMap::new();
        let mut coverage = Coverage {
            total_gpu_refs: total_gpu_kernel_refs(compressed),
            ..Coverage::default()
        };
        if compressed.arenas.is_some() {
            walk_layer_subtrees_arena(compressed, |layer, gpu| {
                let Some(Template::Gpu(g)) = compressed.templates.get(gpu.tmpl_id as usize) else {
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
        } else {
            walk_layer_subtrees_legacy(compressed, |layer, gpu| {
                let Some(Template::Gpu(g)) = compressed.templates.get(gpu.tmpl_id as usize) else {
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
        }
        let collect_secs = elapsed_secs(start);
        let start = std::time::Instant::now();
        let mut result = kernel_hotspot_json(tally, self.top_k.max(20));
        add_coverage(&mut result, coverage);
        Ok(profiled_result(
            result,
            vec![
                ("layer_gpu_collect", collect_secs),
                ("sort_json", elapsed_secs(start)),
            ],
        ))
    }
}

impl AnalysisTask for LayerComputeCommOverlap {
    fn name(&self) -> &str {
        "layer_compute_comm_overlap"
    }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut by_rank_layer: AHashMap<(String, String), IntervalAgg> = AHashMap::new();
        let raw = collect_raw_layer_gpu(trace);
        for item in &raw.items {
            let entry = by_rank_layer
                .entry((item.rank.clone(), item.layer.clone()))
                .or_default();
            push_interval(entry, &item.gpu.name, item.gpu.ts, item.gpu.dur);
        }
        Ok(overlap_json(by_rank_layer, raw.coverage))
    }

    fn supports_in_situ(&self) -> bool {
        true
    }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let start = std::time::Instant::now();
        let mut by_rank_layer: AHashMap<(String, String), StreamingOverlapState> = AHashMap::new();
        let mut coverage = Coverage {
            total_gpu_refs: total_gpu_kernel_refs(compressed),
            ..Coverage::default()
        };
        if compressed.arenas.is_some() {
            walk_rank_layer_subtrees_arena(compressed, |rank, layer, gpu| {
                let Some(Template::Gpu(g)) = compressed.templates.get(gpu.tmpl_id as usize) else {
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
                entry.push(&g.name_pattern, ts, dur);
            });
        } else {
            walk_rank_layer_subtrees_legacy(compressed, |rank, layer, gpu| {
                let Some(Template::Gpu(g)) = compressed.templates.get(gpu.tmpl_id as usize) else {
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
                entry.push(&g.name_pattern, ts, dur);
            });
        }
        let collect_secs = elapsed_secs(start);
        let start = std::time::Instant::now();
        let mut rows: Vec<Value> = by_rank_layer
            .into_iter()
            .map(|((rank, layer), state)| state.to_json(&rank, &layer))
            .collect();
        rows.sort_by(rank_layer_value_cmp);
        let result = serde_json::json!({
            "coverage": coverage_json(coverage),
            "rows": rows,
        });
        Ok(profiled_result(
            result,
            vec![
                ("layer_streaming_overlap", collect_secs),
                ("json_output", elapsed_secs(start)),
            ],
        ))
    }
}


#[derive(Default, Clone, Copy)]
struct Coverage {
    attributed_gpu_refs: u64,
    total_gpu_refs: u64,
}

#[derive(Default)]
struct RawCollection {
    items: Vec<RawLayerGpu>,
    coverage: Coverage,
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

fn walk_layer_subtrees_legacy(compressed: &CompressedTrace, mut f: impl FnMut(&str, GpuKernel)) {
    walk_rank_layer_subtrees_legacy(compressed, |_rank, layer, gpu| f(layer, gpu));
}

fn walk_rank_layer_subtrees_legacy(
    compressed: &CompressedTrace,
    mut f: impl FnMut(&str, &str, GpuKernel),
) {
    for (rank, processes) in &compressed.ranks {
        for threads in processes.values() {
            for phases in threads.values() {
                for root in phases.values() {
                    walk_node_for_layers_legacy(compressed, rank, root, ActiveLayer::None, &mut f);
                }
            }
        }
    }
}

fn walk_node_for_layers_legacy(
    compressed: &CompressedTrace,
    rank: &str,
    node: &Node,
    active_layer: ActiveLayer,
    f: &mut impl FnMut(&str, &str, GpuKernel),
) {
    match node {
        Node::Root { children } => {
            for child in children {
                walk_node_for_layers_legacy(compressed, rank, child, active_layer.clone(), f);
            }
        }
        Node::Cpu(n) => {
            let next_layer =
                ActiveLayer::from_option(cpu_instance_layer(compressed, n.template, n.instance))
                    .or(active_layer);
            for child in &n.children {
                walk_node_for_layers_legacy(compressed, rank, child, next_layer.clone(), f);
            }
            for child in &n.slots {
                walk_node_for_layers_legacy(compressed, rank, child, next_layer.clone(), f);
            }
        }
        Node::SameCpu(n) => {
            let repeated_scope = repeated_scope_layers(compressed, n.template, n.instances.len());
            let next_layer = ActiveLayer::from_layers(
                n.instances
                    .iter()
                    .enumerate()
                    .map(|(idx, inst)| {
                        cpu_instance_layer(compressed, n.template, *inst)
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
                walk_node_for_layers_legacy(compressed, rank, child, next_layer.clone(), f);
            }
            for slot in n.slots.entries() {
                let slot_layer =
                    ActiveLayer::from_option(next_layer.at(slot.instance_index as usize));
                for child in &slot.children {
                    walk_node_for_layers_legacy(compressed, rank, child, slot_layer.clone(), f);
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
            } else if let ActiveLayer::Many(layers) = &active_layer {
                if layers.len() == n.templates.len() {
                    for ((layer, tmpl_id), inst_id) in layers
                        .iter()
                        .zip(n.templates.iter())
                        .zip(n.instances.iter())
                    {
                        if let Some(layer) = layer {
                            f(
                                rank,
                                layer,
                                GpuKernel {
                                    tmpl_id: *tmpl_id,
                                    inst_id: *inst_id,
                                },
                            );
                        }
                    }
                }
            }
        }
        Node::KernelLaunch(n) => {
            if let Some(layer) = cpu_instance_layer(compressed, n.cpu_template, n.cpu_instance)
                .or_else(|| active_layer.scalar())
            {
                f(
                    rank,
                    &layer,
                    GpuKernel {
                        tmpl_id: n.gpu_template,
                        inst_id: n.gpu_instance,
                    },
                );
            }
        }
        Node::KernelsLaunch(n) => {
            for (idx, ((cpu_inst, tmpl_id), inst_id)) in n
                .cpu_instances
                .iter()
                .zip(n.gpu_templates.iter())
                .zip(n.gpu_instances.iter())
                .enumerate()
            {
                if let Some(layer) = cpu_instance_layer(compressed, n.cpu_template, *cpu_inst)
                    .or_else(|| active_layer.at(idx))
                {
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
    }
}

fn walk_layer_subtrees_arena(compressed: &CompressedTrace, mut f: impl FnMut(&str, GpuKernel)) {
    walk_rank_layer_subtrees_arena(compressed, |_rank, layer, gpu| f(layer, gpu));
}

fn walk_rank_layer_subtrees_arena(
    compressed: &CompressedTrace,
    mut f: impl FnMut(&str, &str, GpuKernel),
) {
    let arenas = compressed.arenas.as_ref().unwrap();
    for (rank, processes) in arenas {
        for threads in processes.values() {
            for phases in threads.values() {
                for (arena, root_id) in phases.values() {
                    walk_node_for_layers_arena(compressed, rank, arena, *root_id, ActiveLayer::None, &mut f);
                }
            }
        }
    }
}

fn walk_node_for_layers_arena(
    compressed: &CompressedTrace,
    rank: &str,
    arena: &NodeArena,
    node_id: NodeId,
    active_layer: ActiveLayer,
    f: &mut impl FnMut(&str, &str, GpuKernel),
) {
    match arena.get(node_id) {
        ArenaNode::Root { children } => {
            let children = *children;
            for &child_id in arena.children(children) {
                walk_node_for_layers_arena(compressed, rank, arena, child_id, active_layer.clone(), f);
            }
        }
        ArenaNode::Cpu { template, instance, children, slots } => {
            let (template, instance, children, slots) = (*template, *instance, *children, *slots);
            let next_layer =
                ActiveLayer::from_option(cpu_instance_layer(compressed, template, instance))
                    .or(active_layer);
            for &child_id in arena.children(children) {
                walk_node_for_layers_arena(compressed, rank, arena, child_id, next_layer.clone(), f);
            }
            for &child_id in arena.children(slots) {
                walk_node_for_layers_arena(compressed, rank, arena, child_id, next_layer.clone(), f);
            }
        }
        ArenaNode::SameCpu { template, instances, children, slots_start, slots_len } => {
            let (template, instances, children, slots_start, slots_len) =
                (*template, *instances, *children, *slots_start, *slots_len);
            let inst_slice = arena.instances(instances);
            let repeated_scope = repeated_scope_layers(compressed, template, inst_slice.len());
            let next_layer = ActiveLayer::from_layers(
                inst_slice
                    .iter()
                    .enumerate()
                    .map(|(idx, inst)| {
                        cpu_instance_layer(compressed, template, *inst)
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
            for &child_id in arena.children(children) {
                walk_node_for_layers_arena(compressed, rank, arena, child_id, next_layer.clone(), f);
            }
            for slot in arena.slots_slice(slots_start, slots_len) {
                let slot_layer =
                    ActiveLayer::from_option(next_layer.at(slot.instance_index as usize));
                for &child_id in arena.children(slot.children) {
                    walk_node_for_layers_arena(compressed, rank, arena, child_id, slot_layer.clone(), f);
                }
            }
        }
        ArenaNode::Gpu { refs_start, refs_len } => {
            let (refs_start, refs_len) = (*refs_start, *refs_len);
            let refs = arena.gpu_refs_slice(refs_start, refs_len);
            if let Some(layer) = active_layer.scalar() {
                for gpu_ref in refs {
                    f(
                        rank,
                        &layer,
                        GpuKernel {
                            tmpl_id: gpu_ref.template,
                            inst_id: gpu_ref.instance,
                        },
                    );
                }
            } else if let ActiveLayer::Many(layers) = &active_layer {
                if layers.len() == refs.len() {
                    for (layer, gpu_ref) in layers.iter().zip(refs.iter()) {
                        if let Some(layer) = layer {
                            f(
                                rank,
                                layer,
                                GpuKernel {
                                    tmpl_id: gpu_ref.template,
                                    inst_id: gpu_ref.instance,
                                },
                            );
                        }
                    }
                }
            }
        }
        ArenaNode::KernelLaunch { cpu_template, cpu_instance, gpu_template, gpu_instance } => {
            let (cpu_template, cpu_instance, gpu_template, gpu_instance) =
                (*cpu_template, *cpu_instance, *gpu_template, *gpu_instance);
            if let Some(layer) = cpu_instance_layer(compressed, cpu_template, cpu_instance)
                .or_else(|| active_layer.scalar())
            {
                f(
                    rank,
                    &layer,
                    GpuKernel {
                        tmpl_id: gpu_template,
                        inst_id: gpu_instance,
                    },
                );
            }
        }
        ArenaNode::KernelsLaunch { cpu_template, refs_start, refs_len } => {
            let (cpu_template, refs_start, refs_len) = (*cpu_template, *refs_start, *refs_len);
            for (idx, r) in arena.kernels_refs_slice(refs_start, refs_len).iter().enumerate() {
                if let Some(layer) = cpu_instance_layer(compressed, cpu_template, r.cpu_instance)
                    .or_else(|| active_layer.at(idx))
                {
                    f(
                        rank,
                        &layer,
                        GpuKernel {
                            tmpl_id: r.gpu_template,
                            inst_id: r.gpu_instance,
                        },
                    );
                }
            }
        }
    }
}

fn cpu_instance_layer(
    compressed: &CompressedTrace,
    tmpl_id: TemplateId,
    inst_id: InstanceId,
) -> Option<String> {
    let Template::Cpu(t) = compressed.templates.get(tmpl_id as usize)? else {
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

fn raw_layer_from_name(name: &str) -> Option<String> {
    RAW_LAYER_RE.captures(name).and_then(|c| {
        c.get(1)
            .or_else(|| c.get(2))
            .map(|m| format!("{}#{}", layer_scope_name(name), m.as_str()))
    })
}

fn repeated_scope_layers(
    compressed: &CompressedTrace,
    tmpl_id: TemplateId,
    instances: usize,
) -> Option<String> {
    if !(REPEATED_SCOPE_MIN_INSTANCES..=REPEATED_SCOPE_MAX_INSTANCES).contains(&instances) {
        return None;
    }
    let Template::Cpu(t) = compressed.templates.get(tmpl_id as usize)? else {
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
    let mut scope = crate::utils::normalize_name(name);
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

fn collect_raw_layer_gpu(trace: &Trace) -> RawCollection {
    let mut out = RawCollection::default();
    for (rank, streams) in &trace.ranks {
        collect_raw_rank_layer_gpu(rank, streams, &mut out);
    }
    out
}

fn collect_raw_rank_layer_gpu(rank: &str, streams: &StreamMap, out: &mut RawCollection) {
    let mut gpu_by_corr: AHashMap<i64, RawGpuKernel> = AHashMap::new();
    for threads in streams.values() {
        for (tid, phases) in threads {
            if !is_gpu_stream(tid) {
                continue;
            }
            for events in phases.values() {
                for ev in events {
                    if ev.cat.as_deref() != Some("kernel") {
                        continue;
                    }
                    out.coverage.total_gpu_refs += 1;
                    let Some(corr) = event_correlation(ev) else {
                        continue;
                    };
                    gpu_by_corr.entry(corr).or_insert_with(|| RawGpuKernel {
                        name: ev.name.clone(),
                        ts: ev.ts,
                        dur: ev.dur.unwrap_or(0),
                    });
                }
            }
        }
    }

    let mut used_corrs = ahash::AHashSet::with_capacity(gpu_by_corr.len());
    for threads in streams.values() {
        for (tid, phases) in threads {
            if is_gpu_stream(tid) {
                continue;
            }
            for events in phases.values() {
                collect_cpu_stream_layer_gpu(rank, events, &gpu_by_corr, &mut used_corrs, out);
            }
        }
    }
}

fn collect_cpu_stream_layer_gpu(
    rank: &str,
    events: &[crate::event::Event],
    gpu_by_corr: &AHashMap<i64, RawGpuKernel>,
    used_corrs: &mut ahash::AHashSet<i64>,
    out: &mut RawCollection,
) {
    let mut sorted: Vec<&crate::event::Event> = events.iter().collect();
    sorted.sort_by(|a, b| {
        a.ts.cmp(&b.ts)
            .then_with(|| b.dur.unwrap_or(0).cmp(&a.dur.unwrap_or(0)))
    });

    let repeated_scopes = raw_repeated_scope_occurrences(&sorted);
    let mut repeated_seen: AHashMap<String, usize> = AHashMap::new();
    let mut stack: Vec<(i64, Option<String>)> = Vec::new();
    for ev in sorted {
        let dur = ev.dur.unwrap_or(0).max(0);
        let end_ts = ev.ts + dur;
        while stack.last().is_some_and(|(end, _)| *end <= ev.ts) {
            stack.pop();
        }
        let inherited = stack.last().and_then(|(_, layer)| layer.clone());
        let repeated_layer = raw_repeated_layer(&ev.name, &repeated_scopes, &mut repeated_seen);
        let active_layer = raw_layer_from_name(&ev.name).or(repeated_layer).or(inherited);
        if let (Some(layer), Some(corr)) = (active_layer.as_ref(), event_correlation(ev)) {
            if used_corrs.insert(corr) {
                if let Some(gpu) = gpu_by_corr.get(&corr) {
                    out.coverage.attributed_gpu_refs += 1;
                    out.items.push(RawLayerGpu {
                        rank: rank.to_string(),
                        layer: layer.clone(),
                        gpu: gpu.clone(),
                    });
                }
            }
        }
        if dur > 0 {
            stack.push((end_ts, active_layer));
        }
    }
}

fn raw_repeated_scope_occurrences(events: &[&crate::event::Event]) -> ahash::AHashSet<String> {
    let mut counts: AHashMap<String, usize> = AHashMap::new();
    for ev in events {
        if event_correlation(ev).is_none() {
            continue;
        }
        let scope = layer_scope_name(&ev.name);
        if is_low_value_repeated_scope(&scope) {
            continue;
        }
        *counts.entry(scope).or_default() += 1;
    }
    counts
        .into_iter()
        .filter_map(|(scope, count)| {
            if (REPEATED_SCOPE_MIN_INSTANCES..=REPEATED_SCOPE_MAX_INSTANCES).contains(&count) {
                Some(scope)
            } else {
                None
            }
        })
        .collect()
}

fn raw_repeated_layer(
    name: &str,
    repeated_scopes: &ahash::AHashSet<String>,
    repeated_seen: &mut AHashMap<String, usize>,
) -> Option<String> {
    let scope = layer_scope_name(name);
    if !repeated_scopes.contains(&scope) {
        return None;
    }
    let idx = repeated_seen.entry(scope.clone()).or_default();
    let layer = format!("{scope}#{idx}");
    *idx += 1;
    Some(layer)
}

fn event_correlation(event: &crate::event::Event) -> Option<i64> {
    event.args.as_ref().and_then(|a| {
        a.get("correlation")
            .or_else(|| a.get("External id"))
            .and_then(|v| v.as_i64())
    })
}

fn is_gpu_stream(tid: &str) -> bool {
    tid.contains("stream")
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

fn kernel_hotspot_json(tally: AHashMap<(String, String), KernelAgg>, top_k: usize) -> Value {
    let mut rows: Vec<(String, String, KernelAgg)> = tally
        .into_iter()
        .map(|((layer, kernel), agg)| (layer, kernel, agg))
        .collect();
    rows.sort_by(|a, b| b.2.total_dur_us.cmp(&a.2.total_dur_us));
    rows.truncate(top_k);
    Value::Array(
        rows.into_iter()
            .map(|(layer, kernel, agg)| {
                let avg = if agg.count > 0 {
                    agg.total_dur_us as f64 / agg.count as f64
                } else {
                    0.0
                };
                serde_json::json!({
                    "layer": layer,
                    "kernel": kernel,
                    "count": agg.count,
                    "total_dur_us": agg.total_dur_us,
                    "avg_dur_us": avg,
                })
            })
            .collect(),
    )
}

fn overlap_json(
    by_rank_layer: AHashMap<(String, String), IntervalAgg>,
    coverage: Coverage,
) -> Value {
    let mut rows: Vec<Value> = by_rank_layer
        .into_iter()
        .map(|((rank, layer), agg)| {
            let compute_union = union_len(agg.compute.clone());
            let comm_union = union_len(agg.comm.clone());
            let overlap = overlap_len(agg.compute, agg.comm);
            let denom = compute_union.min(comm_union);
            let overlap_fraction = if denom > 0 {
                overlap as f64 / denom as f64
            } else {
                0.0
            };
            serde_json::json!({
                "rank": rank,
                "layer": layer,
                "compute_total_us": agg.compute_total_us,
                "comm_total_us": agg.comm_total_us,
                "compute_union_us": compute_union,
                "comm_union_us": comm_union,
                "overlap_us": overlap,
                "overlap_fraction_of_min_union": overlap_fraction,
            })
        })
        .collect();
    rows.sort_by(rank_layer_value_cmp);
    serde_json::json!({
        "coverage": coverage_json(coverage),
        "rows": rows,
    })
}

fn add_coverage(result: &mut Value, coverage: Coverage) {
    let rows = std::mem::take(result);
    *result = serde_json::json!({
        "coverage": coverage_json(coverage),
        "rows": rows,
    });
}

fn coverage_json(coverage: Coverage) -> Value {
    let fraction = if coverage.total_gpu_refs > 0 {
        coverage.attributed_gpu_refs as f64 / coverage.total_gpu_refs as f64
    } else {
        0.0
    };
    serde_json::json!({
        "attributed_gpu_refs": coverage.attributed_gpu_refs,
        "total_gpu_refs": coverage.total_gpu_refs,
        "attributed_fraction": fraction,
    })
}

fn total_gpu_kernel_refs(compressed: &CompressedTrace) -> u64 {
    compressed
        .templates
        .iter()
        .map(|tmpl| match tmpl {
            Template::Gpu(g) if g.cat.as_deref() == Some("kernel") => g.dur.len() as u64,
            _ => 0,
        })
        .sum()
}

fn union_len(mut intervals: Vec<(i64, i64)>) -> i64 {
    if intervals.is_empty() {
        return 0;
    }
    intervals.sort_unstable();
    let mut total = 0;
    let (mut cur_s, mut cur_e) = intervals[0];
    for (s, e) in intervals.into_iter().skip(1) {
        if s <= cur_e {
            cur_e = cur_e.max(e);
        } else {
            total += cur_e - cur_s;
            cur_s = s;
            cur_e = e;
        }
    }
    total + cur_e - cur_s
}

fn overlap_len(compute: Vec<(i64, i64)>, comm: Vec<(i64, i64)>) -> i64 {
    let compute = merge_intervals(compute);
    let comm = merge_intervals(comm);
    let mut i = 0;
    let mut j = 0;
    let mut overlap = 0;
    while i < compute.len() && j < comm.len() {
        let start = compute[i].0.max(comm[j].0);
        let end = compute[i].1.min(comm[j].1);
        if end > start {
            overlap += end - start;
        }
        if compute[i].1 < comm[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    overlap
}

fn merge_intervals(mut intervals: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    if intervals.is_empty() {
        return intervals;
    }
    intervals.sort_unstable();
    let mut out = Vec::new();
    let (mut cur_s, mut cur_e) = intervals[0];
    for (s, e) in intervals.into_iter().skip(1) {
        if s <= cur_e {
            cur_e = cur_e.max(e);
        } else {
            out.push((cur_s, cur_e));
            cur_s = s;
            cur_e = e;
        }
    }
    out.push((cur_s, cur_e));
    out
}

fn rank_layer_value_cmp(a: &Value, b: &Value) -> std::cmp::Ordering {
    let ar = a["rank"].as_str().unwrap_or_default();
    let br = b["rank"].as_str().unwrap_or_default();
    rank_cmp(ar, br).then_with(|| {
        let al = a["layer"].as_str().unwrap_or_default();
        let bl = b["layer"].as_str().unwrap_or_default();
        al.cmp(bl)
    })
}

fn rank_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    match (a.parse::<i64>(), b.parse::<i64>()) {
        (Ok(x), Ok(y)) => x.cmp(&y),
        _ => a.cmp(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_layer_zero_index() {
        assert_eq!(layer_zero_index("model.layers.0.attn"), Some(0));
        assert_eq!(layer_zero_index("x.0.layers.0.y"), Some(1));
        assert_eq!(layer_zero_index("aten::mm"), None);
    }

    #[test]
    fn overlap_math_handles_intersections() {
        let compute = vec![(0, 10), (20, 30)];
        let comm = vec![(5, 25)];
        assert_eq!(overlap_len(compute, comm), 10);
    }
}
