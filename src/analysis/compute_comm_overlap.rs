//! Compute/communication overlap by rank.
//!
//! For each rank, this task builds two interval lists from GPU-side kernel
//! events: compute kernels and communication kernels.  It reports total
//! compute time, total communication time, the union time of each class, and
//! the intersection length between the two unions.

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::kernel_class::is_nccl_kernel;
use crate::analysis::{elapsed_secs, profiled_result, AnalysisTask};
use crate::event::Template;
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

#[derive(Default)]
pub struct ComputeCommOverlap;

#[derive(Default)]
struct RankIntervals {
    compute: Vec<(i64, i64)>,
    comm: Vec<(i64, i64)>,
}

impl AnalysisTask for ComputeCommOverlap {
    fn name(&self) -> &str { "compute_comm_overlap" }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut by_rank: AHashMap<String, RankIntervals> = AHashMap::new();
        for (rank, _pid, _tid, _ph, events) in trace.iter_streams() {
            for ev in events {
                if ev.cat.as_deref() != Some("kernel") {
                    continue;
                }
                let dur = ev.dur.unwrap_or(0);
                if dur <= 0 {
                    continue;
                }
                let interval = (ev.ts, ev.ts + dur);
                let entry = by_rank.entry(rank.to_string()).or_default();
                if is_nccl_kernel(&ev.name) {
                    entry.comm.push(interval);
                } else {
                    entry.compute.push(interval);
                }
            }
        }
        Ok(to_json(by_rank))
    }

    fn supports_in_situ(&self) -> bool { true }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let start = std::time::Instant::now();
        let mut by_rank: AHashMap<String, RankIntervals> = AHashMap::new();

        for (rank, processes) in &compressed.ranks {
            let entry = by_rank.entry(rank.clone()).or_default();
            for (_pid, threads) in processes {
                for (_tid, phases) in threads {
                    for (_ph, root) in phases {
                        walk_gpu_instances(root, &mut |tmpl_id, inst_id| {
                            let Some(Template::Gpu(g)) = compressed.templates.get(tmpl_id as usize) else {
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
                            let interval = (ts, ts + dur);
                            if is_nccl_kernel(&g.name_pattern) {
                                entry.comm.push(interval);
                            } else {
                                entry.compute.push(interval);
                            }
                        });
                    }
                }
            }
        }
        let collect_secs = elapsed_secs(start);

        let start = std::time::Instant::now();
        let result = to_json(by_rank);
        Ok(profiled_result(result, vec![
            ("rank_interval_collect", collect_secs),
            ("interval_merge_and_json", elapsed_secs(start)),
        ]))
    }
}

fn walk_gpu_instances(
    node: &crate::node::Node,
    f: &mut impl FnMut(crate::node::TemplateId, crate::node::InstanceId),
) {
    use crate::node::Node;
    match node {
        Node::Root { children } => {
            for child in children {
                walk_gpu_instances(child, f);
            }
        }
        Node::Cpu(n) => {
            for child in &n.children {
                walk_gpu_instances(child, f);
            }
            for child in &n.slots {
                walk_gpu_instances(child, f);
            }
        }
        Node::SameCpu(n) => {
            for child in &n.children {
                walk_gpu_instances(child, f);
            }
            for slot in &n.slots {
                for child in slot {
                    walk_gpu_instances(child, f);
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

fn to_json(mut by_rank: AHashMap<String, RankIntervals>) -> Value {
    let mut rows: Vec<(String, Value)> = by_rank
        .drain()
        .map(|(rank, intervals)| {
            let compute_total = total_interval_len(&intervals.compute);
            let comm_total = total_interval_len(&intervals.comm);
            let compute_union = union_len(intervals.compute.clone());
            let comm_union = union_len(intervals.comm.clone());
            let overlap = overlap_len(intervals.compute, intervals.comm);
            let denom = comm_union.min(compute_union);
            let overlap_fraction = if denom > 0 {
                overlap as f64 / denom as f64
            } else {
                0.0
            };
            (
                rank.clone(),
                serde_json::json!({
                    "rank": rank,
                    "compute_total_us": compute_total,
                    "comm_total_us": comm_total,
                    "compute_union_us": compute_union,
                    "comm_union_us": comm_union,
                    "overlap_us": overlap,
                    "overlap_fraction_of_min_union": overlap_fraction,
                }),
            )
        })
        .collect();
    rows.sort_by(|a, b| rank_cmp(&a.0, &b.0));
    Value::Array(rows.into_iter().map(|(_, v)| v).collect())
}

fn rank_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    match (a.parse::<i64>(), b.parse::<i64>()) {
        (Ok(x), Ok(y)) => x.cmp(&y),
        _ => a.cmp(b),
    }
}

fn total_interval_len(intervals: &[(i64, i64)]) -> i64 {
    intervals.iter().map(|(s, e)| e.saturating_sub(*s)).sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_counts_intersection_of_unions() {
        let compute = vec![(0, 10), (8, 20), (30, 40)];
        let comm = vec![(5, 12), (15, 35)];
        assert_eq!(union_len(compute.clone()), 30);
        assert_eq!(union_len(comm.clone()), 27);
        assert_eq!(overlap_len(compute, comm), 17);
    }
}
