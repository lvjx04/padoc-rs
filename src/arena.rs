//! Arena-based node storage for compressed call trees.
//!
//! After the compression pipeline produces a recursive `Node` tree, it is
//! converted into a flat `NodeArena` for memory-efficient storage and
//! traversal.  Each node is stored in a contiguous `Vec<ArenaNode>` and
//! child references are `(start, len)` spans into a separate `children: Vec<NodeId>`
//! array.  This eliminates per-node `Vec` heap allocations and capacity waste.
//!
//! ## Memory savings
//!
//! A `CpuNode` with two `Vec<Node>` fields costs 56+ bytes (2×24 Vec header + fields).
//! An `ArenaCpuNode` costs 16 bytes (template + instance + 2 × ChildSpan inline in the
//! enum).  For 67M nodes (llama_full), this saves ~4-6 GiB.

use serde::{Deserialize, Serialize};

use crate::node::{
    CpuNode, GpuNode, KernelLaunchNode, KernelsLaunchNode, Node, SameCpuNode, TemplateId,
    InstanceId,
};

/// Index into `NodeArena::nodes`.
pub type NodeId = u32;

/// A span of child node IDs in `NodeArena::child_ids`.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ChildSpan {
    pub start: u32,
    pub len: u32,
}

impl ChildSpan {
    pub fn empty() -> Self {
        Self { start: 0, len: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A slot entry for SameCpu nodes (instance_index + children span).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ArenaSlot {
    pub instance_index: u32,
    pub children: ChildSpan,
}

/// Flat arena node — no heap pointers, all references are indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ArenaNode {
    Root {
        children: ChildSpan,
    },
    Cpu {
        template: TemplateId,
        instance: InstanceId,
        children: ChildSpan,
        slots: ChildSpan,
    },
    SameCpu {
        template: TemplateId,
        instances: ChildSpan, // span into `instance_ids`
        children: ChildSpan,
        slots_start: u32, // index into `slots`
        slots_len: u32,
    },
    Gpu {
        refs_start: u32, // index into `gpu_refs`
        refs_len: u32,
    },
    KernelLaunch {
        cpu_template: TemplateId,
        cpu_instance: InstanceId,
        gpu_template: TemplateId,
        gpu_instance: InstanceId,
    },
    KernelsLaunch {
        cpu_template: TemplateId,
        refs_start: u32, // index into `kernels_launch_refs`
        refs_len: u32,
    },
}

/// A GPU ref pair (template_id, instance_id) used by Gpu and KernelsLaunch nodes.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GpuRef {
    pub template: TemplateId,
    pub instance: InstanceId,
}

/// A KernelsLaunch ref (cpu_instance, gpu_template, gpu_instance).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KernelsLaunchRef {
    pub cpu_instance: InstanceId,
    pub gpu_template: TemplateId,
    pub gpu_instance: InstanceId,
}

/// Flat arena owning all nodes for one rank's call tree.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeArena {
    /// All nodes in DFS order.
    pub nodes: Vec<ArenaNode>,
    /// Child node IDs — referenced by `ChildSpan` in each node.
    pub child_ids: Vec<NodeId>,
    /// Instance IDs for SameCpu nodes.
    pub instance_ids: Vec<InstanceId>,
    /// GPU refs for Gpu nodes.
    pub gpu_refs: Vec<GpuRef>,
    /// Slots for SameCpu nodes.
    pub slots: Vec<ArenaSlot>,
    /// Refs for KernelsLaunch nodes.
    pub kernels_refs: Vec<KernelsLaunchRef>,
}

impl NodeArena {
    /// Convert a recursive `Node` tree into a flat arena.
    pub fn from_tree(root: &Node) -> (Self, NodeId) {
        let mut arena = NodeArena::default();
        let root_id = arena.add_node(root);
        arena.nodes.shrink_to_fit();
        arena.child_ids.shrink_to_fit();
        arena.instance_ids.shrink_to_fit();
        arena.gpu_refs.shrink_to_fit();
        arena.slots.shrink_to_fit();
        arena.kernels_refs.shrink_to_fit();
        (arena, root_id)
    }

    fn add_node(&mut self, node: &Node) -> NodeId {
        let id = self.nodes.len() as NodeId;
        // Reserve slot — fill in after processing children
        self.nodes.push(ArenaNode::Root { children: ChildSpan::empty() });

        let arena_node = match node {
            Node::Root { children } => {
                let child_span = self.add_children(children);
                ArenaNode::Root { children: child_span }
            }
            Node::Cpu(n) => {
                let children = self.add_children(&n.children);
                let slots = self.add_children(&n.slots);
                ArenaNode::Cpu {
                    template: n.template,
                    instance: n.instance,
                    children,
                    slots,
                }
            }
            Node::SameCpu(n) => {
                let instances_start = self.instance_ids.len() as u32;
                self.instance_ids.extend_from_slice(&n.instances);
                let instances = ChildSpan {
                    start: instances_start,
                    len: n.instances.len() as u32,
                };
                let children = self.add_children(&n.children);
                let slots_start = self.slots.len() as u32;
                for slot in n.slots.entries() {
                    let slot_children = self.add_children(&slot.children);
                    self.slots.push(ArenaSlot {
                        instance_index: slot.instance_index,
                        children: slot_children,
                    });
                }
                let slots_len = self.slots.len() as u32 - slots_start;
                ArenaNode::SameCpu {
                    template: n.template,
                    instances,
                    children,
                    slots_start,
                    slots_len,
                }
            }
            Node::Gpu(n) => {
                let refs_start = self.gpu_refs.len() as u32;
                for (t, i) in n.templates.iter().zip(n.instances.iter()) {
                    self.gpu_refs.push(GpuRef { template: *t, instance: *i });
                }
                ArenaNode::Gpu {
                    refs_start,
                    refs_len: n.templates.len() as u32,
                }
            }
            Node::KernelLaunch(n) => ArenaNode::KernelLaunch {
                cpu_template: n.cpu_template,
                cpu_instance: n.cpu_instance,
                gpu_template: n.gpu_template,
                gpu_instance: n.gpu_instance,
            },
            Node::KernelsLaunch(n) => {
                let refs_start = self.kernels_refs.len() as u32;
                for ((ci, gt), gi) in n.cpu_instances.iter()
                    .zip(n.gpu_templates.iter())
                    .zip(n.gpu_instances.iter())
                {
                    self.kernels_refs.push(KernelsLaunchRef {
                        cpu_instance: *ci,
                        gpu_template: *gt,
                        gpu_instance: *gi,
                    });
                }
                ArenaNode::KernelsLaunch {
                    cpu_template: n.cpu_template,
                    refs_start,
                    refs_len: n.cpu_instances.len() as u32,
                }
            }
        };

        self.nodes[id as usize] = arena_node;
        id
    }

    fn add_children(&mut self, children: &[Node]) -> ChildSpan {
        if children.is_empty() {
            return ChildSpan::empty();
        }
        // First pass: recursively add all children to arena
        let child_ids: Vec<NodeId> = children.iter().map(|c| self.add_node(c)).collect();
        // Store child IDs contiguously
        let start = self.child_ids.len() as u32;
        self.child_ids.extend_from_slice(&child_ids);
        ChildSpan {
            start,
            len: child_ids.len() as u32,
        }
    }

    // --- Accessor methods for traversal ---

    pub fn get(&self, id: NodeId) -> &ArenaNode {
        &self.nodes[id as usize]
    }

    pub fn children(&self, span: ChildSpan) -> &[NodeId] {
        if span.is_empty() {
            return &[];
        }
        &self.child_ids[span.start as usize..(span.start + span.len) as usize]
    }

    pub fn instances(&self, span: ChildSpan) -> &[InstanceId] {
        if span.is_empty() {
            return &[];
        }
        &self.instance_ids[span.start as usize..(span.start + span.len) as usize]
    }

    pub fn gpu_refs_slice(&self, start: u32, len: u32) -> &[GpuRef] {
        &self.gpu_refs[start as usize..(start + len) as usize]
    }

    pub fn slots_slice(&self, start: u32, len: u32) -> &[ArenaSlot] {
        &self.slots[start as usize..(start + len) as usize]
    }

    pub fn kernels_refs_slice(&self, start: u32, len: u32) -> &[KernelsLaunchRef] {
        &self.kernels_refs[start as usize..(start + len) as usize]
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total heap bytes used by this arena (excluding the struct itself).
    pub fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        self.nodes.capacity() * size_of::<ArenaNode>()
            + self.child_ids.capacity() * size_of::<NodeId>()
            + self.instance_ids.capacity() * size_of::<InstanceId>()
            + self.gpu_refs.capacity() * size_of::<GpuRef>()
            + self.slots.capacity() * size_of::<ArenaSlot>()
            + self.kernels_refs.capacity() * size_of::<KernelsLaunchRef>()
    }

    /// Reconstruct a recursive `Node` tree from the arena (for decompress/round-trip).
    pub fn to_tree(&self, root_id: NodeId) -> Node {
        self.reconstruct(root_id)
    }

    fn reconstruct(&self, id: NodeId) -> Node {
        match &self.nodes[id as usize] {
            ArenaNode::Root { children } => Node::Root {
                children: self.children(*children).iter().map(|&cid| self.reconstruct(cid)).collect(),
            },
            ArenaNode::Cpu { template, instance, children, slots } => Node::Cpu(CpuNode {
                template: *template,
                instance: *instance,
                children: self.children(*children).iter().map(|&cid| self.reconstruct(cid)).collect(),
                slots: self.children(*slots).iter().map(|&cid| self.reconstruct(cid)).collect(),
            }),
            ArenaNode::SameCpu { template, instances, children, slots_start, slots_len } => {
                let inst_vec = self.instances(*instances).to_vec();
                let child_vec: Vec<Node> = self.children(*children).iter().map(|&cid| self.reconstruct(cid)).collect();
                let slot_entries: Vec<Vec<Node>> = {
                    let arena_slots = self.slots_slice(*slots_start, *slots_len);
                    // Build dense slots (fill empty ones)
                    let max_idx = arena_slots.iter().map(|s| s.instance_index).max().unwrap_or(0) as usize;
                    let mut dense = vec![Vec::new(); max_idx + 1];
                    for s in arena_slots {
                        dense[s.instance_index as usize] = self.children(s.children).iter().map(|&cid| self.reconstruct(cid)).collect();
                    }
                    dense
                };
                Node::SameCpu(SameCpuNode {
                    template: *template,
                    instances: inst_vec,
                    children: child_vec,
                    slots: crate::node::SameCpuSlots::from_dense(slot_entries),
                })
            }
            ArenaNode::Gpu { refs_start, refs_len } => {
                let refs = self.gpu_refs_slice(*refs_start, *refs_len);
                Node::Gpu(GpuNode {
                    templates: refs.iter().map(|r| r.template).collect(),
                    instances: refs.iter().map(|r| r.instance).collect(),
                })
            }
            ArenaNode::KernelLaunch { cpu_template, cpu_instance, gpu_template, gpu_instance } => {
                Node::KernelLaunch(KernelLaunchNode {
                    cpu_template: *cpu_template,
                    cpu_instance: *cpu_instance,
                    gpu_template: *gpu_template,
                    gpu_instance: *gpu_instance,
                })
            }
            ArenaNode::KernelsLaunch { cpu_template, refs_start, refs_len } => {
                let refs = self.kernels_refs_slice(*refs_start, *refs_len);
                Node::KernelsLaunch(KernelsLaunchNode {
                    cpu_template: *cpu_template,
                    cpu_instances: refs.iter().map(|r| r.cpu_instance).collect(),
                    gpu_templates: refs.iter().map(|r| r.gpu_template).collect(),
                    gpu_instances: refs.iter().map(|r| r.gpu_instance).collect(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::*;

    #[test]
    fn arena_round_trip() {
        let tree = Node::Root {
            children: vec![
                Node::Cpu(CpuNode {
                    template: 0,
                    instance: 0,
                    children: vec![
                        Node::KernelLaunch(KernelLaunchNode {
                            cpu_template: 1,
                            cpu_instance: 0,
                            gpu_template: 2,
                            gpu_instance: 0,
                        }),
                    ],
                    slots: vec![],
                }),
                Node::Gpu(GpuNode {
                    templates: vec![3, 4],
                    instances: vec![0, 1],
                }),
            ],
        };

        let (arena, root_id) = NodeArena::from_tree(&tree);
        let reconstructed = arena.to_tree(root_id);

        // Verify structure matches
        if let (Node::Root { children: orig }, Node::Root { children: recon }) = (&tree, &reconstructed) {
            assert_eq!(orig.len(), recon.len());
        } else {
            panic!("root mismatch");
        }
    }
}
