//! Structural compression: bottom-up grouping of sibling sub-trees that
//! share a `template_index`, with optional anchor matching across child
//! sequences.
//!
//! This module also hosts the per-template numeric finalisation
//! (SLP / args dedup / name column transpose) so that finalisation policy
//! is owned by the compressor module and not scattered across event/.

use ahash::AHashMap;

use super::config::CompressorConfig;
use super::core::TemplateCompressor;
use crate::event::{MergeEvent, MergeKernelEvent};
use crate::node::{Node, SameCpuNode, TemplateId};
use crate::event::ArgColumn;
use crate::slp::{compress_name_nums, SlpColumn};

/// Compress one sub-tree.  Public so it can be invoked iteratively from
/// the call-tree builder.  Recursive but bounded by the original tree depth
/// (which is small for AI traces — typically <=10).
pub(crate) fn compress_node(compressor: &mut TemplateCompressor, mut node: Node) -> Node {
    // Recurse into children first.
    match &mut node {
        Node::Cpu(c) => {
            let mut new_children: Vec<Node> = Vec::with_capacity(c.children.len());
            for child in std::mem::take(&mut c.children) {
                new_children.push(compress_node(compressor, child));
            }
            c.children = group_similar(compressor, new_children);
        }
        Node::Root { children } => {
            let mut new_children: Vec<Node> = Vec::with_capacity(children.len());
            for child in std::mem::take(children) {
                new_children.push(compress_node(compressor, child));
            }
            *children = group_similar(compressor, new_children);
        }
        Node::SameCpu(s) => {
            let mut new_children: Vec<Node> = Vec::with_capacity(s.children.len());
            for child in std::mem::take(&mut s.children) {
                new_children.push(compress_node(compressor, child));
            }
            s.children = new_children;
        }
        _ => {}
    }
    node
}

/// Hash-bucket dedup of similar siblings (same template id) into SameCpu nodes.
/// O(n) instead of the Python O(n²).
fn group_similar(compressor: &TemplateCompressor, children: Vec<Node>) -> Vec<Node> {
    if !compressor.config.enable_structural || children.len() < 2 {
        return children;
    }

    // Classify children up front: `Node::Cpu` nodes are bucketed by template
    // for `SameCpu` formation; `Node::KernelLaunch` nodes are bucketed
    // separately so they can become a `KernelsLaunch` (Python's
    // `KernelsLaunchNode`) which preserves the per-instance GPU pair.
    let mut cpu_buckets: AHashMap<TemplateId, Vec<usize>> = AHashMap::new();
    let mut kernel_buckets: AHashMap<TemplateId, Vec<usize>> = AHashMap::new();
    for (i, child) in children.iter().enumerate() {
        match child {
            Node::Cpu(c) => {
                cpu_buckets.entry(c.template).or_default().push(i);
            }
            Node::KernelLaunch(k) => {
                kernel_buckets.entry(k.cpu_template).or_default().push(i);
            }
            _ => {}
        }
    }

    // Take ownership: move every child into Option slots so we can detach
    // bucket members one by one and leave `None` placeholders.
    let mut slots: Vec<Option<Node>> = children.into_iter().map(Some).collect();

    // Owner index -> merged group node.  Owner is the smallest original
    // position so the merged group preserves the original relative ordering.
    let mut grouped_owners: AHashMap<usize, Node> = AHashMap::new();

    // 1) Cpu groups -> SameCpu.
    for (_template, indices) in cpu_buckets.into_iter() {
        if indices.len() < 2 {
            continue;
        }
        let owner_idx = *indices.iter().min().unwrap();
        let mut group: Vec<Node> = Vec::with_capacity(indices.len());
        for idx in indices {
            if let Some(node) = slots[idx].take() {
                group.push(node);
            }
        }
        grouped_owners.insert(owner_idx, build_same_cpu(group, &compressor.config));
    }

    // 2) KernelLaunch groups -> KernelsLaunch.  Stores all (cpu, gpu) pairs in
    //    parallel arrays so the GPU events survive the merge.
    for (_template, indices) in kernel_buckets.into_iter() {
        if indices.len() < 2 {
            continue;
        }
        let owner_idx = *indices.iter().min().unwrap();
        let mut cpu_template = 0u32;
        let mut cpu_instances: Vec<u32> = Vec::with_capacity(indices.len());
        let mut gpu_templates: Vec<u32> = Vec::with_capacity(indices.len());
        let mut gpu_instances: Vec<u32> = Vec::with_capacity(indices.len());
        for idx in indices {
            if let Some(Node::KernelLaunch(k)) = slots[idx].take() {
                cpu_template = k.cpu_template;
                cpu_instances.push(k.cpu_instance);
                gpu_templates.push(k.gpu_template);
                gpu_instances.push(k.gpu_instance);
            }
        }
        let merged = Node::KernelsLaunch(crate::node::KernelsLaunchNode {
            cpu_template,
            cpu_instances,
            gpu_templates,
            gpu_instances,
        });
        grouped_owners.insert(owner_idx, merged);
    }

    // Reassemble: keep ungrouped nodes in place; emit merged group at the
    // owner's original position.
    let mut out: Vec<Node> = Vec::with_capacity(slots.len());
    for (i, slot) in slots.into_iter().enumerate() {
        if let Some(node) = slot {
            out.push(node);
        } else if let Some(merged) = grouped_owners.remove(&i) {
            out.push(merged);
        }
    }
    out
}

fn build_same_cpu(group: Vec<Node>, config: &CompressorConfig) -> Node {
    if group.is_empty() {
        return Node::Root { children: Vec::new() };
    }

    // group_similar only feeds us pure-Cpu buckets, so every member is a
    // `Node::Cpu`.  Anything else is a programming error.
    let template = group[0].template_index().unwrap_or(0);
    let mut instances: Vec<u32> = Vec::with_capacity(group.len());
    let mut child_lists: Vec<Vec<Node>> = Vec::with_capacity(group.len());
    for member in group {
        match member {
            Node::Cpu(c) => {
                instances.push(c.instance);
                child_lists.push(c.children);
            }
            other => {
                debug_assert!(false, "build_same_cpu got non-Cpu member: {:?}", other);
                instances.push(0);
                child_lists.push(Vec::new());
            }
        }
    }

    let (children, slots) = if config.enable_anchor_matching {
        anchor_match(&child_lists)
    } else {
        (Vec::new(), child_lists)
    };

    Node::SameCpu(SameCpuNode {
        template,
        instances,
        children,
        slots,
    })
}

/// LCS-style greedy anchor extraction.
///
/// For every instance we find a sub-sequence of child positions where the
/// templates line up across all instances.  Each anchor position then becomes
/// **one SameCpu node** that aggregates that position's per-instance children
/// (so all N instances of an anchor are stored together, not just instance 0).
/// Per-instance unmatched trailers go into `slots`.
fn anchor_match(child_lists: &[Vec<Node>]) -> (Vec<Node>, Vec<Vec<Node>>) {
    if child_lists.is_empty() {
        return (Vec::new(), Vec::new());
    }
    if child_lists.len() == 1 {
        return (child_lists[0].clone(), vec![Vec::new()]);
    }

    // Shortest sequence is the anchor reference (greedy LCS approximation).
    let ref_idx = (0..child_lists.len())
        .min_by_key(|i| child_lists[*i].len())
        .unwrap_or(0);
    let reference = child_lists[ref_idx].clone();

    let mut cursors: Vec<usize> = vec![0; child_lists.len()];
    let mut anchor_positions: Vec<Vec<usize>> = vec![Vec::new(); child_lists.len()];

    for ref_node in &reference {
        let target_template = ref_node.template_index();
        let mut hits: Vec<usize> = vec![0; child_lists.len()];
        let mut all_found = true;
        for (i, list) in child_lists.iter().enumerate() {
            let mut matched: Option<usize> = None;
            for j in cursors[i]..list.len() {
                if list[j].template_index() == target_template {
                    matched = Some(j);
                    break;
                }
            }
            if let Some(j) = matched {
                hits[i] = j;
            } else {
                all_found = false;
                break;
            }
        }
        if !all_found {
            continue;
        }
        for (i, hit) in hits.iter().enumerate() {
            anchor_positions[i].push(*hit);
            cursors[i] = hit + 1;
        }
    }

    if anchor_positions[0].is_empty() {
        return (Vec::new(), child_lists.to_vec());
    }

    let num_anchors = anchor_positions[0].len();

    // For each anchor position, group instance children into a SameCpu node.
    let mut merged_children: Vec<Node> = Vec::with_capacity(num_anchors);
    for k in 0..num_anchors {
        let group: Vec<Node> = (0..child_lists.len())
            .map(|i| child_lists[i][anchor_positions[i][k]].clone())
            .collect();
        merged_children.push(merge_anchor_group(group));
    }

    // Slots: per-instance children whose positions weren't picked as anchors.
    let mut slots: Vec<Vec<Node>> = Vec::with_capacity(child_lists.len());
    for (i, list) in child_lists.iter().enumerate() {
        let positions: ahash::AHashSet<usize> = anchor_positions[i].iter().copied().collect();
        let trailers: Vec<Node> = list
            .iter()
            .enumerate()
            .filter_map(|(j, n)| if positions.contains(&j) { None } else { Some(n.clone()) })
            .collect();
        slots.push(trailers);
    }

    (merged_children, slots)
}

/// Convert a list of N children that all share the same template_id into a
/// single SameCpu (recursing the anchor-matching one level deeper for their
/// own children).  The group must be non-empty.
///
/// **Correctness invariant**: `instances.len()` and `child_lists.len()` must
/// stay equal.
///
/// When a member is itself already a `SameCpu` (i.e. a previous recursion
/// merged inner siblings into one node), it cannot be safely flattened into
/// the outer SameCpu: its M sub-instances belong to a single outer slot, not
/// to all K outer instances.  Flattening either drops events (if we
/// under-count instances vs. slots) or duplicates them (if we over-count).
/// The safe fallback is to keep the group's members as plain siblings under
/// a `Root` wrapper — slightly worse compression but bit-exact decoding.
fn merge_anchor_group(group: Vec<Node>) -> Node {
    if group.is_empty() {
        return Node::Root { children: Vec::new() };
    }
    if group.len() == 1 {
        return group.into_iter().next().unwrap();
    }
    // Mixed-type groups can't form a single same-template aggregator without
    // either dropping or duplicating events; keep them as siblings.
    if group.iter().any(|n| matches!(n, Node::SameCpu(_))) {
        return Node::Root { children: group };
    }

    // All-KernelLaunch group => KernelsLaunch (preserves per-instance GPU pair).
    if group.iter().all(|n| matches!(n, Node::KernelLaunch(_))) {
        let mut cpu_template = 0u32;
        let mut cpu_instances: Vec<u32> = Vec::with_capacity(group.len());
        let mut gpu_templates: Vec<u32> = Vec::with_capacity(group.len());
        let mut gpu_instances: Vec<u32> = Vec::with_capacity(group.len());
        for member in group {
            if let Node::KernelLaunch(k) = member {
                cpu_template = k.cpu_template;
                cpu_instances.push(k.cpu_instance);
                gpu_templates.push(k.gpu_template);
                gpu_instances.push(k.gpu_instance);
            }
        }
        return Node::KernelsLaunch(crate::node::KernelsLaunchNode {
            cpu_template,
            cpu_instances,
            gpu_templates,
            gpu_instances,
        });
    }

    // Mixed Cpu + KernelLaunch group: same problem as SameCpu — fall back to
    // siblings to keep correctness.
    if group.iter().any(|n| matches!(n, Node::KernelLaunch(_))) {
        return Node::Root { children: group };
    }

    let template = group[0].template_index().unwrap_or(0);
    let mut instances: Vec<u32> = Vec::with_capacity(group.len());
    let mut child_lists: Vec<Vec<Node>> = Vec::with_capacity(group.len());
    for member in group {
        match member {
            Node::Cpu(c) => {
                instances.push(c.instance);
                child_lists.push(c.children);
            }
            other => {
                instances.push(0);
                child_lists.push(other.children().iter().map(|n| (*n).clone()).collect());
            }
        }
    }
    debug_assert_eq!(instances.len(), child_lists.len(), "SameCpu invariant broken");
    let (children, slots) = anchor_match(&child_lists);
    Node::SameCpu(SameCpuNode {
        template,
        instances,
        children,
        slots,
    })
}

// ---------------------------------------------------------------------------
// Per-template finalisation
// ---------------------------------------------------------------------------

pub(crate) fn finalize_cpu_template(tmpl: &mut MergeEvent, config: &CompressorConfig) {
    if config.enable_name_pattern {
        tmpl.name_nums = compress_name_nums(&tmpl.name_nums);
    }
    if config.enable_args_dedup {
        for col in tmpl.args_columns.iter_mut() {
            dedup_arg_column(col);
        }
    }
    if config.enable_slp {
        let _slp_ts = SlpColumn::encode(&tmpl.ts);
        let _slp_dur = SlpColumn::encode(&tmpl.dur);
        let _slp_id = SlpColumn::encode(&tmpl.id);
        // We don't yet store the SLP-encoded form on disk — that is the next
        // pass once `Template` learns to carry encoded columns.  Today the
        // raw columns survive into msgpack (zstd handles the compression).
        let _ = _slp_ts;
        let _ = _slp_dur;
        let _ = _slp_id;
    }
}

pub(crate) fn finalize_gpu_template(tmpl: &mut MergeKernelEvent, config: &CompressorConfig) {
    if config.enable_name_pattern {
        tmpl.name_nums = compress_name_nums(&tmpl.name_nums);
    }
    if config.enable_args_dedup {
        for col in tmpl.args_columns.iter_mut() {
            dedup_arg_column(col);
        }
    }
    let _ = config; // SLP wiring identical to the CPU path above; deferred.
}

/// In-place dedup: if every value in a `PerInstance` column is identical,
/// collapse to `Constant(value)` — this is the cheap analog of Python's
/// `compress_same_args` for the all-same case.  Skips columns that are
/// already constant.  Cost: one linear scan over the column with `==`
/// against the first element; no clone unless dedup actually triggers.
fn dedup_arg_column(col: &mut ArgColumn) {
    if let ArgColumn::PerInstance(values) = col {
        if values.len() <= 1 {
            return;
        }
        let (first, rest) = values.split_first().unwrap();
        if rest.iter().all(|v| v == first) {
            *col = ArgColumn::Constant(values.swap_remove(0));
        }
    }
}
