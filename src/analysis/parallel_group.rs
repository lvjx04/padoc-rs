//! Per-rank load balance — accesses the trace by **rank** dimension.
//!
//! Detecting parallel groups (TP/DP/PP) reliably from chrome traces is
//! brittle: most profilers don't annotate `Process Group Name`, and
//! inferring groups from rank participation requires cross-rank ts
//! alignment that's beyond a single-task analysis.  Instead this task
//! reports the more fundamental measurement that the parallel-group
//! analysis was a means to: **how unbalanced is the per-rank workload**?
//!
//! For each rank we report two device-time totals derived from
//! `cat == "kernel"` events (the unambiguous GPU side, free of the
//! c10d/nccl host-wrapper double-count problem):
//!
//! * `compute_busy_us` — total dur of all GPU kernels that are NOT
//!   NCCL collectives (matmul, layernorm, fused-mlp, etc.)
//! * `comm_busy_us`    — total dur of NCCL kernels (`ncclKernel_*`,
//!   `genericMultiShmOp`, etc.) — the cross-rank communication.
//!
//! Per metric we emit `max/min/mean/stddev/cv/imbalance_max_min_over_mean`
//! plus the per-rank vector so the paper can plot a histogram.
//!
//! In-situ on padoc: pre-classify each GPU template as compute / comm
//! once (a single name match per template, NOT per instance), then walk
//! every rank's call tree summing dur into the right bucket.  Cost is
//! O(rank_tree_nodes) — independent of event count, which is the in-situ
//! advantage over baselines that need to materialise every kernel
//! event.

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::AnalysisTask;
use crate::analysis::kernel_class::is_nccl_kernel;
use crate::event::Template;
use crate::node::{InstanceId, Node, TemplateId};
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

#[derive(Default)]
pub struct ParallelGroup;

impl AnalysisTask for ParallelGroup {
    fn name(&self) -> &str { "rank_load_balance" }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        let mut compute: AHashMap<String, i64> = AHashMap::new();
        let mut comm:    AHashMap<String, i64> = AHashMap::new();
        for (rank, _pid, _tid, _ph, events) in trace.iter_streams() {
            for ev in events {
                if ev.cat.as_deref() != Some("kernel") { continue; }
                let dur = ev.dur.unwrap_or(0);
                if is_nccl_kernel(&ev.name) {
                    *comm.entry(rank.to_string()).or_insert(0) += dur;
                } else {
                    *compute.entry(rank.to_string()).or_insert(0) += dur;
                }
            }
        }
        Ok(load_balance_json(&compute, &comm))
    }

    fn supports_in_situ(&self) -> bool { true }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        // Per template: classify once into compute / comm / ignore.
        // 0 = ignore (not a GPU kernel), 1 = compute, 2 = comm.
        let class: Vec<u8> = compressed
            .templates
            .iter()
            .map(|t| match t {
                Template::Gpu(g) if g.cat.as_deref() == Some("kernel") => {
                    if is_nccl_kernel(&g.name_pattern) { 2 } else { 1 }
                }
                _ => 0,
            })
            .collect();

        let mut compute: AHashMap<String, i64> = AHashMap::new();
        let mut comm:    AHashMap<String, i64> = AHashMap::new();
        for (rank, processes) in &compressed.ranks {
            for (_pid, threads) in processes {
                for (_tid, phases) in threads {
                    for (_ph, root) in phases {
                        walk_kernels(root, &mut |tmpl_id, inst_id| {
                            let c = class[tmpl_id as usize];
                            if c == 0 { return; }
                            let Template::Gpu(g) = &compressed.templates[tmpl_id as usize] else { return; };
                            let dur = g.dur.get(inst_id as usize).unwrap_or(0);
                            let bucket = if c == 1 { &mut compute } else { &mut comm };
                            *bucket.entry(rank.clone()).or_insert(0) += dur;
                        });
                    }
                }
            }
        }
        Ok(load_balance_json(&compute, &comm))
    }
}

/// Walk a `Node` tree, calling `f(template_id, instance_id)` for every
/// **GPU-side** instance encountered (Gpu nodes plus the `gpu_*` half
/// of KernelLaunch / KernelsLaunch).  CPU instances are intentionally
/// skipped — this task is GPU-device-time only.
fn walk_kernels(node: &Node, f: &mut impl FnMut(TemplateId, InstanceId)) {
    match node {
        Node::Root { children } => {
            for c in children { walk_kernels(c, f); }
        }
        Node::Cpu(n) => {
            for c in &n.children { walk_kernels(c, f); }
            for s in &n.slots { walk_kernels(s, f); }
        }
        Node::SameCpu(n) => {
            for c in &n.children { walk_kernels(c, f); }
            for slot in &n.slots {
                for s in slot { walk_kernels(s, f); }
            }
        }
        Node::Gpu(g) => {
            for (t, i) in g.templates.iter().zip(g.instances.iter()) {
                f(*t, *i);
            }
        }
        Node::KernelLaunch(k) => {
            f(k.gpu_template, k.gpu_instance);
        }
        Node::KernelsLaunch(k) => {
            for (t, i) in k.gpu_templates.iter().zip(k.gpu_instances.iter()) {
                f(*t, *i);
            }
        }
    }
}

fn load_balance_json(
    compute: &AHashMap<String, i64>,
    comm:    &AHashMap<String, i64>,
) -> Value {
    // Union of ranks observed in either bucket; rank may have one without
    // the other (e.g. rank 0 broadcasts but doesn't compute much).
    let mut ranks: Vec<String> = compute.keys().chain(comm.keys()).cloned().collect();
    ranks.sort();
    ranks.dedup();
    // Prefer numeric sort for "0", "1", ... "1023".
    ranks.sort_by(|a, b| match (a.parse::<i64>(), b.parse::<i64>()) {
        (Ok(x), Ok(y)) => x.cmp(&y),
        _              => a.cmp(b),
    });

    let pick = |m: &AHashMap<String, i64>| -> Vec<i64> {
        ranks.iter().map(|r| *m.get(r).unwrap_or(&0)).collect()
    };
    let compute_v = pick(compute);
    let comm_v    = pick(comm);

    serde_json::json!({
        "n_ranks": ranks.len(),
        "compute": metric_summary(&ranks, &compute_v),
        "comm":    metric_summary(&ranks, &comm_v),
    })
}

fn metric_summary(ranks: &[String], values: &[i64]) -> Value {
    if values.is_empty() {
        return serde_json::json!({
            "n_ranks": 0, "max_us": 0, "min_us": 0, "mean_us": 0.0,
            "stddev_us": 0.0, "cv": 0.0, "imbalance_max_min_over_mean": 0.0,
            "per_rank_us": []
        });
    }
    let max_v = *values.iter().max().unwrap();
    let min_v = *values.iter().min().unwrap();
    let n     = values.len() as f64;
    let mean  = values.iter().sum::<i64>() as f64 / n;
    let var   = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let cv     = if mean > 0.0 { stddev / mean } else { 0.0 };
    let imbal  = if mean > 0.0 { (max_v as f64 - min_v as f64) / mean } else { 0.0 };

    serde_json::json!({
        "n_ranks":  values.len(),
        "max_us":   max_v,
        "min_us":   min_v,
        "mean_us":  mean,
        "stddev_us": stddev,
        "cv":       cv,
        "imbalance_max_min_over_mean": imbal,
        "per_rank_us": ranks.iter().zip(values.iter()).map(|(r, v)| {
            serde_json::json!({"rank": r, "us": *v})
        }).collect::<Vec<_>>(),
    })
}
