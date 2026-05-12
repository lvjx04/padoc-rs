//! Compressed call-tree nodes.
//!
//! In the Python implementation each node was its own class
//! (`CPUNode`, `SameCPUNode`, `KernelLaunchNode`, `KernelsLaunchNode`,
//! `GPUNode`).  Rust collapses them into one [`Node`] enum so the tree
//! is uniform and traversal is one `match`.
//!
//! ## Node semantics
//!
//! * [`Node::Cpu`] — single instance reference: `(template_id, instance_id)`,
//!   plus optional ordered children (anchor-matched) and per-instance unmatched slots.
//! * [`Node::SameCpu`] — N instances all sharing the same template, structurally
//!   identical down to the children they share.  Anchor-matched children are
//!   in `children`; unmatched per-instance trailers in `slots`.
//! * [`Node::Gpu`] — leaf carrying a flat list of (template_id, instance_id)
//!   pairs for events that live on a GPU stream and were not paired to any
//!   CPU launch.
//! * [`Node::KernelLaunch`] — paired CPU launch + GPU kernel correlation.
//! * [`Node::KernelsLaunch`] — N such pairs sharing the same launch template.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Index into `CompressedTrace::templates` — a global template table.
pub type TemplateId = u32;
/// Per-template position; together with `TemplateId` identifies one event instance.
pub type InstanceId = u32;

/// A compressed call-tree node.  See module docs for variant semantics.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "k", content = "v", rename_all = "snake_case")]
pub enum Node {
    /// Sentinel root with no template, used when a forest of independent root
    /// nodes is needed.
    Root { children: Vec<Node> },

    Cpu(CpuNode),
    SameCpu(SameCpuNode),
    Gpu(GpuNode),
    KernelLaunch(KernelLaunchNode),
    KernelsLaunch(KernelsLaunchNode),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuNode {
    pub template: TemplateId,
    pub instance: InstanceId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Node>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slots: Vec<Node>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SameCpuNode {
    pub template: TemplateId,
    /// One entry per instance — the actual position inside the template's
    /// per-instance arrays (ts/dur/id/...).
    pub instances: Vec<InstanceId>,
    /// Anchor-matched children: each child node itself usually carries
    /// `instances.len()` instance ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Node>,
    /// Per-instance unmatched trailers.  `slots[i]` is the trailing nodes that
    /// belonged to the `i`-th instance only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slots: Vec<Vec<Node>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuNode {
    /// Parallel arrays — `templates[i]` and `instances[i]` describe the i-th GPU event.
    pub templates: Vec<TemplateId>,
    pub instances: Vec<InstanceId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelLaunchNode {
    /// CPU launch event.
    pub cpu_template: TemplateId,
    pub cpu_instance: InstanceId,
    /// GPU kernel event correlated by `correlation` arg.
    pub gpu_template: TemplateId,
    pub gpu_instance: InstanceId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelsLaunchNode {
    pub cpu_template: TemplateId,
    pub cpu_instances: Vec<InstanceId>,
    pub gpu_templates: Vec<TemplateId>,
    pub gpu_instances: Vec<InstanceId>,
}

impl Node {
    /// Iterate immediate child nodes (regardless of variant).
    pub fn children(&self) -> SmallVec<[&Node; 4]> {
        let mut out: SmallVec<[&Node; 4]> = SmallVec::new();
        match self {
            Node::Root { children } => out.extend(children.iter()),
            Node::Cpu(n) => {
                out.extend(n.children.iter());
                out.extend(n.slots.iter());
            }
            Node::SameCpu(n) => {
                out.extend(n.children.iter());
                for slot in &n.slots {
                    out.extend(slot.iter());
                }
            }
            Node::Gpu(_) | Node::KernelLaunch(_) | Node::KernelsLaunch(_) => {}
        }
        out
    }

    pub fn template_index(&self) -> Option<TemplateId> {
        match self {
            Node::Cpu(n) => Some(n.template),
            Node::SameCpu(n) => Some(n.template),
            Node::KernelLaunch(n) => Some(n.cpu_template),
            Node::KernelsLaunch(n) => Some(n.cpu_template),
            Node::Gpu(_) | Node::Root { .. } => None,
        }
    }

    /// True if all children of a SameCpuNode share *this* template structurally.
    pub fn is_kernel(&self) -> bool {
        matches!(self, Node::KernelLaunch(_) | Node::KernelsLaunch(_))
    }
}
