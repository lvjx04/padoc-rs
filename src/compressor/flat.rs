//! Stable flat stream encoding.
//!
//! The research implementation built recursive call trees and then applied
//! structural sibling/anchor merging. Real profiler traces can contain
//! thousands of nested intervals, making that representation expensive to
//! build and impossible to decode with bounded MessagePack recursion.
//!
//! The public format keeps the useful part—template and column compression—
//! while storing each stream as parallel template/instance arrays. This is
//! deterministic, bounded-depth, and sufficient for both lossless
//! reconstruction and the supported in-situ analyses.

use std::collections::BTreeMap;

use super::core::TemplateCompressor;
use crate::node::{CpuBatchNode, GpuNode, InstanceId, Node, TemplateId};
use crate::trace::StreamMap;

pub(crate) fn build_rank(
    compressor: &mut TemplateCompressor,
    streams: &StreamMap,
) -> BTreeMap<i64, BTreeMap<String, BTreeMap<u8, Node>>> {
    let mut out = BTreeMap::new();

    for (pid, threads) in streams {
        for (tid, phases) in threads {
            for (phase, events) in phases {
                let mut templates: Vec<TemplateId> = Vec::with_capacity(events.len());
                let mut instances: Vec<InstanceId> = Vec::with_capacity(events.len());

                let node = if tid.contains("stream") {
                    for event in events {
                        let (template, instance) = compressor.intern_kernel_template(event, tid);
                        templates.push(template);
                        instances.push(instance);
                    }
                    Node::Gpu(GpuNode {
                        templates,
                        instances,
                    })
                } else {
                    for event in events {
                        let (template, instance) = compressor.intern_event_template(event);
                        templates.push(template);
                        instances.push(instance);
                    }
                    Node::CpuBatch(CpuBatchNode {
                        templates,
                        instances,
                    })
                };

                out.entry(*pid)
                    .or_insert_with(BTreeMap::new)
                    .entry(tid.clone())
                    .or_insert_with(BTreeMap::new)
                    .insert(phase.0, node);
            }
        }
    }

    out
}
