//! Compute/communication overlap by rank — streaming window algorithm.
//!
//! For each rank, this task iterates GPU kernel events sorted by timestamp,
//! maintaining a compute window and a communication window.  The overlap is
//! accumulated incrementally without collecting all intervals into memory.
//!
//! Reference: PerFlow-AI `communication_analysis.py` streaming approach.

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::kernel_class::is_nccl_kernel;
use crate::analysis::{elapsed_secs, profiled_result, AnalysisTask};
use crate::arena::{ArenaNode, NodeArena, NodeId};
use crate::event::Template;
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

#[derive(Default)]
pub struct ComputeCommOverlap;

/// Per-rank streaming overlap state.
#[derive(Default)]
struct OverlapState {
    comp_window: (i64, i64), // (start, end) of current compute window
    comm_window: (i64, i64), // (start, end) of current comm window
    overlap_accum: i64,
    last_overlap_end: i64,
    comm_total: i64,
    compute_total: i64,
}

impl OverlapState {
    fn push_event(&mut self, ts: i64, dur: i64, is_comm: bool) {
        let end = ts + dur;
        if is_comm {
            // Update comm window
            if ts > self.comm_window.1 {
                self.comm_total += dur;
                self.comm_window = (ts, end);
            } else if end > self.comm_window.1 {
                self.comm_total += end - self.comm_window.1;
                self.comm_window.1 = end;
            }
            // Compute overlap between comp and comm windows
            self.add_overlap();
        } else {
            // Update compute window
            if ts > self.comp_window.1 {
                self.compute_total += dur;
                self.comp_window = (ts, end);
            } else if end > self.comp_window.1 {
                self.compute_total += end - self.comp_window.1;
                self.comp_window.1 = end;
            }
            // Compute overlap between comp and comm windows
            self.add_overlap();
        }
    }

    fn add_overlap(&mut self) {
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

    fn to_json(&self, rank: &str) -> Value {
        let denom = self.comm_total.min(self.compute_total);
        let overlap_fraction = if denom > 0 {
            self.overlap_accum as f64 / denom as f64
        } else {
            0.0
        };
        serde_json::json!({
            "rank": rank,
            "compute_total_us": self.compute_total,
            "comm_total_us": self.comm_total,
            "overlap_us": self.overlap_accum,
            "overlap_fraction": overlap_fraction,
        })
    }
}

impl AnalysisTask for ComputeCommOverlap {
    fn name(&self) -> &str {
        "compute_comm_overlap"
    }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        // Collect GPU kernel events per rank, sort by ts, then stream through
        let mut rank_events: AHashMap<String, Vec<(i64, i64, bool)>> = AHashMap::new();
        for (rank, _pid, _tid, _ph, events) in trace.iter_streams() {
            for ev in events {
                if ev.cat.as_deref() != Some("kernel") {
                    continue;
                }
                let dur = ev.dur.unwrap_or(0);
                if dur <= 0 {
                    continue;
                }
                let is_comm = is_nccl_kernel(&ev.name);
                rank_events
                    .entry(rank.to_string())
                    .or_default()
                    .push((ev.ts, dur, is_comm));
            }
        }

        let mut rows: Vec<(String, Value)> = Vec::new();
        for (rank, mut events) in rank_events {
            events.sort_unstable_by_key(|e| e.0);
            let mut state = OverlapState::default();
            for (ts, dur, is_comm) in events {
                state.push_event(ts, dur, is_comm);
            }
            rows.push((rank.clone(), state.to_json(&rank)));
        }
        rows.sort_by(|a, b| rank_cmp(&a.0, &b.0));
        Ok(Value::Array(rows.into_iter().map(|(_, v)| v).collect()))
    }

    fn supports_in_situ(&self) -> bool {
        true
    }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let start = std::time::Instant::now();

        // Collect GPU kernel events per rank from the tree, then sort and stream
        let mut rank_events: AHashMap<String, Vec<(i64, i64, bool)>> = AHashMap::new();

        if let Some(arenas) = &compressed.arenas {
            for (rank, processes) in arenas {
                let entry = rank_events.entry(rank.clone()).or_default();
                for (_pid, threads) in processes {
                    for (_tid, phases) in threads {
                        for (_ph, (arena, root_id)) in phases {
                            walk_gpu_instances_arena(arena, *root_id, compressed, entry);
                        }
                    }
                }
            }
        } else {
            for (rank, processes) in &compressed.ranks {
                let entry = rank_events.entry(rank.clone()).or_default();
                for (_pid, threads) in processes {
                    for (_tid, phases) in threads {
                        for (_ph, root) in phases {
                            walk_gpu_instances_legacy(root, &mut |tmpl_id, inst_id| {
                                let Some(Template::Gpu(g)) =
                                    compressed.templates.get(tmpl_id as usize)
                                else {
                                    return;
                                };
                                if g.cat.as_deref() != Some("kernel") {
                                    return;
                                }
                                let ts = g.ts.get(inst_id as usize).unwrap_or(0);
                                let dur = g.dur.get(inst_id as usize).unwrap_or(0);
                                if dur <= 0 {
                                    return;
                                }
                                let is_comm = is_nccl_kernel(&g.name_pattern);
                                entry.push((ts, dur, is_comm));
                            });
                        }
                    }
                }
            }
        }

        let collect_secs = elapsed_secs(start);

        let start = std::time::Instant::now();
        let mut rows: Vec<(String, Value)> = Vec::new();
        for (rank, mut events) in rank_events {
            events.sort_unstable_by_key(|e| e.0);
            let mut state = OverlapState::default();
            for (ts, dur, is_comm) in events {
                state.push_event(ts, dur, is_comm);
            }
            rows.push((rank.clone(), state.to_json(&rank)));
        }
        rows.sort_by(|a, b| rank_cmp(&a.0, &b.0));
        Ok(profiled_result(
            Value::Array(rows.into_iter().map(|(_, v)| v).collect()),
            vec![
                ("gpu_event_collect", collect_secs),
                ("sort_and_overlap", elapsed_secs(start)),
            ],
        ))
    }
}

fn walk_gpu_instances_legacy(
    node: &crate::node::Node,
    f: &mut impl FnMut(crate::node::TemplateId, crate::node::InstanceId),
) {
    use crate::node::Node;
    match node {
        Node::Root { children } => {
            for child in children {
                walk_gpu_instances_legacy(child, f);
            }
        }
        Node::Cpu(n) => {
            for child in &n.children {
                walk_gpu_instances_legacy(child, f);
            }
            for child in &n.slots {
                walk_gpu_instances_legacy(child, f);
            }
        }
        Node::SameCpu(n) => {
            for child in &n.children {
                walk_gpu_instances_legacy(child, f);
            }
            for slot in n.slots.entries() {
                for child in &slot.children {
                    walk_gpu_instances_legacy(child, f);
                }
            }
        }
        Node::Gpu(n) => {
            for (tmpl, inst) in n.templates.iter().zip(n.instances.iter()) {
                f(*tmpl, *inst);
            }
        }
        Node::KernelLaunch(n) => f(n.gpu_template, n.gpu_instance),
        Node::KernelsLaunch(n) => {
            for (tmpl, inst) in n.gpu_templates.iter().zip(n.gpu_instances.iter()) {
                f(*tmpl, *inst);
            }
        }
    }
}

/// Arena-based version of `walk_gpu_instances_legacy`.
fn walk_gpu_instances_arena(
    arena: &NodeArena,
    node_id: NodeId,
    compressed: &CompressedTrace,
    out: &mut Vec<(i64, i64, bool)>,
) {
    match arena.get(node_id) {
        ArenaNode::Root { children } => {
            for &child_id in arena.children(*children) {
                walk_gpu_instances_arena(arena, child_id, compressed, out);
            }
        }
        ArenaNode::Cpu { children, slots, .. } => {
            let (children, slots) = (*children, *slots);
            for &child_id in arena.children(children) {
                walk_gpu_instances_arena(arena, child_id, compressed, out);
            }
            for &child_id in arena.children(slots) {
                walk_gpu_instances_arena(arena, child_id, compressed, out);
            }
        }
        ArenaNode::SameCpu { children, slots_start, slots_len, .. } => {
            let (children, slots_start, slots_len) = (*children, *slots_start, *slots_len);
            for &child_id in arena.children(children) {
                walk_gpu_instances_arena(arena, child_id, compressed, out);
            }
            for slot in arena.slots_slice(slots_start, slots_len) {
                for &child_id in arena.children(slot.children) {
                    walk_gpu_instances_arena(arena, child_id, compressed, out);
                }
            }
        }
        ArenaNode::Gpu { refs_start, refs_len } => {
            let (refs_start, refs_len) = (*refs_start, *refs_len);
            for gpu_ref in arena.gpu_refs_slice(refs_start, refs_len) {
                if let Some(Template::Gpu(g)) = compressed.templates.get(gpu_ref.template as usize) {
                    if g.cat.as_deref() != Some("kernel") {
                        continue;
                    }
                    let ts = g.ts.get(gpu_ref.instance as usize).unwrap_or(0);
                    let dur = g.dur.get(gpu_ref.instance as usize).unwrap_or(0);
                    if dur > 0 {
                        let is_comm = is_nccl_kernel(&g.name_pattern);
                        out.push((ts, dur, is_comm));
                    }
                }
            }
        }
        ArenaNode::KernelLaunch { gpu_template, gpu_instance, .. } => {
            let (gpu_template, gpu_instance) = (*gpu_template, *gpu_instance);
            if let Some(Template::Gpu(g)) = compressed.templates.get(gpu_template as usize) {
                if g.cat.as_deref() == Some("kernel") {
                    let ts = g.ts.get(gpu_instance as usize).unwrap_or(0);
                    let dur = g.dur.get(gpu_instance as usize).unwrap_or(0);
                    if dur > 0 {
                        let is_comm = is_nccl_kernel(&g.name_pattern);
                        out.push((ts, dur, is_comm));
                    }
                }
            }
        }
        ArenaNode::KernelsLaunch { refs_start, refs_len, .. } => {
            let (refs_start, refs_len) = (*refs_start, *refs_len);
            for r in arena.kernels_refs_slice(refs_start, refs_len) {
                if let Some(Template::Gpu(g)) = compressed.templates.get(r.gpu_template as usize) {
                    if g.cat.as_deref() != Some("kernel") {
                        continue;
                    }
                    let ts = g.ts.get(r.gpu_instance as usize).unwrap_or(0);
                    let dur = g.dur.get(r.gpu_instance as usize).unwrap_or(0);
                    if dur > 0 {
                        let is_comm = is_nccl_kernel(&g.name_pattern);
                        out.push((ts, dur, is_comm));
                    }
                }
            }
        }
    }
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
    fn streaming_overlap_basic() {
        let mut state = OverlapState::default();
        // Compute: [0, 10]
        state.push_event(0, 10, false);
        // Comm: [5, 15] → overlap with compute = [5, 10] = 5
        state.push_event(5, 10, true);
        assert_eq!(state.overlap_accum, 5);
        assert_eq!(state.compute_total, 10);
        assert_eq!(state.comm_total, 10);
    }

    #[test]
    fn streaming_overlap_no_overlap() {
        let mut state = OverlapState::default();
        state.push_event(0, 10, false);
        state.push_event(20, 10, true);
        assert_eq!(state.overlap_accum, 0);
    }
}
