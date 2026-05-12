//! Storage profile of a `CompressedTrace` — what fraction of bytes goes to
//! templates vs structure vs metadata.

use crate::trace::CompressedTrace;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StorageBreakdown {
    pub total_bytes: u64,
    pub template_bytes: u64,
    pub structure_bytes: u64,
    pub metadata_bytes: u64,
    pub template_count: usize,
    pub node_count: usize,
}

pub fn measure_storage(compressed: &CompressedTrace) -> StorageBreakdown {
    let templates_bytes = rmp_serde::to_vec_named(&compressed.templates).map(|b| b.len() as u64).unwrap_or(0);
    let structure_bytes = rmp_serde::to_vec_named(&compressed.ranks).map(|b| b.len() as u64).unwrap_or(0);
    let metadata_bytes = rmp_serde::to_vec_named(&compressed.metadata).map(|b| b.len() as u64).unwrap_or(0);
    StorageBreakdown {
        total_bytes: templates_bytes + structure_bytes + metadata_bytes,
        template_bytes: templates_bytes,
        structure_bytes,
        metadata_bytes,
        template_count: compressed.templates.len(),
        node_count: count_nodes(compressed),
    }
}

fn count_nodes(compressed: &CompressedTrace) -> usize {
    let mut n = 0;
    for (_rank, processes) in &compressed.ranks {
        for (_pid, threads) in processes {
            for (_tid, phases) in threads {
                for (_ph, root) in phases {
                    n += count_subtree(root);
                }
            }
        }
    }
    n
}

fn count_subtree(node: &crate::node::Node) -> usize {
    let mut n = 1;
    for c in node.children() { n += count_subtree(c); }
    n
}
