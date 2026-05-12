//! Structural statistics over a `CompressedTrace` — depth / branching /
//! same-cpu multiplier histograms used by the paper's "trace shape" figures.

use crate::node::Node;
use crate::trace::CompressedTrace;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TreeStatistics {
    pub max_depth: usize,
    pub mean_depth: f64,
    pub max_branching: usize,
    pub mean_branching: f64,
    pub same_cpu_node_count: usize,
    pub max_same_cpu_multiplier: usize,
    pub kernel_link_node_count: usize,
}

pub fn measure_tree_statistics(compressed: &CompressedTrace) -> TreeStatistics {
    let mut stats = TreeStatistics::default();
    let mut depth_total = 0u64;
    let mut depth_samples = 0u64;
    let mut branch_total = 0u64;
    let mut branch_samples = 0u64;
    for (_rank, processes) in &compressed.ranks {
        for (_pid, threads) in processes {
            for (_tid, phases) in threads {
                for (_ph, root) in phases {
                    visit(root, 0, &mut stats, &mut depth_total, &mut depth_samples, &mut branch_total, &mut branch_samples);
                }
            }
        }
    }
    if depth_samples > 0 { stats.mean_depth = depth_total as f64 / depth_samples as f64; }
    if branch_samples > 0 { stats.mean_branching = branch_total as f64 / branch_samples as f64; }
    stats
}

fn visit(
    node: &Node,
    depth: usize,
    stats: &mut TreeStatistics,
    depth_total: &mut u64,
    depth_samples: &mut u64,
    branch_total: &mut u64,
    branch_samples: &mut u64,
) {
    *depth_total += depth as u64;
    *depth_samples += 1;
    if depth > stats.max_depth { stats.max_depth = depth; }

    let children: Vec<&Node> = node.children().into_iter().collect();
    let degree = children.len();
    if degree > 0 {
        *branch_total += degree as u64;
        *branch_samples += 1;
        if degree > stats.max_branching { stats.max_branching = degree; }
    }
    if let Node::SameCpu(s) = node {
        stats.same_cpu_node_count += 1;
        if s.instances.len() > stats.max_same_cpu_multiplier {
            stats.max_same_cpu_multiplier = s.instances.len();
        }
    }
    if matches!(node, Node::KernelLaunch(_) | Node::KernelsLaunch(_)) {
        stats.kernel_link_node_count += 1;
    }
    for c in children {
        visit(c, depth + 1, stats, depth_total, depth_samples, branch_total, branch_samples);
    }
}
