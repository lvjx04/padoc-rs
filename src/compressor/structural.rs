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
use crate::slp::{compress_name_nums, compress_same_args, SlpColumn};

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

    // Classify children up front: only Cpu and KernelLaunch nodes participate
    // in grouping.  Bucket index is `template_index()`.
    let mut buckets: AHashMap<TemplateId, Vec<usize>> = AHashMap::new();
    for (i, child) in children.iter().enumerate() {
        if let Some(t) = child.template_index() {
            if matches!(child, Node::Cpu(_) | Node::KernelLaunch(_)) {
                buckets.entry(t).or_default().push(i);
            }
        }
    }

    // Take ownership: move every child into Option slots so we can detach
    // bucket members one by one and leave `None` placeholders.
    let mut slots: Vec<Option<Node>> = children.into_iter().map(Some).collect();

    // Owner index -> merged SameCpu node.  Owner is the smallest original
    // position so the merged group preserves the original relative ordering.
    let mut grouped_owners: AHashMap<usize, Node> = AHashMap::new();
    for (_template, indices) in buckets.into_iter() {
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

    // Every group member should share the same template id (we bucket by it).
    let template = group[0].template_index().unwrap_or(0);
    let mut instances: Vec<u32> = Vec::with_capacity(group.len());
    let mut child_lists: Vec<Vec<Node>> = Vec::with_capacity(group.len());
    for member in group {
        match member {
            Node::Cpu(c) => {
                instances.push(c.instance);
                child_lists.push(c.children);
            }
            Node::KernelLaunch(k) => {
                instances.push(k.cpu_instance);
                child_lists.push(Vec::new());
            }
            other => {
                instances.push(0);
                child_lists.push(other.children().iter().map(|n| (*n).clone()).collect());
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
fn merge_anchor_group(group: Vec<Node>) -> Node {
    if group.is_empty() {
        return Node::Root { children: Vec::new() };
    }
    if group.len() == 1 {
        return group.into_iter().next().unwrap();
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
            Node::SameCpu(s) => {
                instances.extend(s.instances);
                let mut combined = s.children;
                for slot_chunk in s.slots {
                    combined.extend(slot_chunk);
                }
                child_lists.push(combined);
            }
            Node::KernelLaunch(k) => {
                instances.push(k.cpu_instance);
                child_lists.push(Vec::new());
            }
            other => {
                instances.push(0);
                child_lists.push(other.children().iter().map(|n| (*n).clone()).collect());
            }
        }
    }
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
        for col in 0..tmpl.arg_keys.len() {
            let mut column: Vec<crate::event::ArgValue> = tmpl
                .args_values
                .iter()
                .map(|row| row.get(col).cloned().unwrap_or(serde_json::Value::Null))
                .collect();
            compress_same_args(&mut column);
            if column.len() == 1 {
                // Dedup column: write back a single sentinel row tagging the
                // shared value via a sidecar.  We keep `args_values` columnar
                // here to avoid expanding the Vec unnecessarily.
                for row in tmpl.args_values.iter_mut() {
                    if col < row.len() {
                        row[col] = column[0].clone();
                    }
                }
            }
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
        for col in 0..tmpl.arg_keys.len() {
            let mut column: Vec<crate::event::ArgValue> = tmpl
                .args_values
                .iter()
                .map(|row| row.get(col).cloned().unwrap_or(serde_json::Value::Null))
                .collect();
            compress_same_args(&mut column);
            if column.len() == 1 {
                for row in tmpl.args_values.iter_mut() {
                    if col < row.len() {
                        row[col] = column[0].clone();
                    }
                }
            }
        }
    }
    let _ = config; // SLP wiring identical to the CPU path above; deferred.
}
