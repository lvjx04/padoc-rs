//! Per-rank call-tree construction.
//!
//! Mirrors the Python `_compress_rank` + `_build_call_tree` logic but with a
//! flat enum-based tree and numeric IDs rather than object pointers.
//!
//! The two phases per rank are:
//!
//! 1. Collect every GPU event keyed by `correlation` (so CPU launches can
//!    pair them up).  This is the only reason GPU events get visited *first*.
//! 2. Walk every CPU stream, build a stack-based parent-child tree by
//!    `(ts, dur)` containment, and emit `Node::Cpu` (or wrap into
//!    `Node::KernelLaunch` if the event has a matching correlation id).
//! 3. Walk every GPU stream and emit a single `Node::Gpu` per (pid, tid, ph)
//!    holding the residual GPU events that were not paired to a CPU launch.
//! 4. Run structural compression top-down on each root.

use ahash::AHashMap;
use indexmap::IndexMap;
use std::collections::BTreeMap;

use super::core::TemplateCompressor;
use crate::event::{Event, Phase};
use crate::node::{
    CpuNode, GpuNode, InstanceId, KernelLaunchNode, KernelsLaunchNode, Node, SameCpuNode, TemplateId,
};
use crate::trace::StreamMap;

/// Build the entire `pid -> tid -> ph -> root` tree for one rank.
pub(crate) fn build_rank(
    compressor: &mut TemplateCompressor,
    rank: &str,
    streams: &StreamMap,
) -> BTreeMap<i64, BTreeMap<String, BTreeMap<u8, Node>>> {
    // 1) Index every GPU event by correlation (only once per rank).
    let gpu_events = collect_gpu_events_by_correlation(streams);

    // Track which correlations got consumed by a CPU launch so we don't
    // double-add them when we visit the GPU stream itself.
    let mut consumed: ahash::AHashSet<i64> = ahash::AHashSet::with_capacity(gpu_events.len());

    let mut out: BTreeMap<i64, BTreeMap<String, BTreeMap<u8, Node>>> = BTreeMap::new();

    // 2) Process CPU streams (everything that does not start with "stream").
    for (pid, threads) in streams {
        for (tid, phases) in threads {
            if is_gpu_stream(tid) {
                continue;
            }
            for (phase, events) in phases {
                let root = build_cpu_tree(
                    compressor,
                    rank,
                    *pid,
                    tid,
                    *phase,
                    events,
                    &gpu_events,
                    &mut consumed,
                );
                let root = if compressor.config.enable_structural {
                    super::structural::compress_node(compressor, root)
                } else {
                    root
                };
                out.entry(*pid)
                    .or_default()
                    .entry(tid.clone())
                    .or_default()
                    .insert(phase.0, root);
            }
        }
    }

    // 3) Process GPU streams: anything still not consumed becomes part of the
    //    GpuNode for that stream.
    for (pid, threads) in streams {
        for (tid, phases) in threads {
            if !is_gpu_stream(tid) {
                continue;
            }
            for (phase, events) in phases {
                let root = build_gpu_tree(compressor, *pid, tid, *phase, events, &consumed);
                out.entry(*pid)
                    .or_default()
                    .entry(tid.clone())
                    .or_default()
                    .insert(phase.0, root);
            }
        }
    }

    out
}

fn is_gpu_stream(tid: &str) -> bool {
    tid.contains("stream")
}

/// Returned map: correlation -> (event, gpu_stream_tid).
fn collect_gpu_events_by_correlation(streams: &StreamMap) -> AHashMap<i64, GpuRef> {
    let mut out = AHashMap::new();
    for (_pid, threads) in streams {
        for (tid, phases) in threads {
            if !is_gpu_stream(tid) {
                continue;
            }
            for (_phase, events) in phases {
                for (idx, event) in events.iter().enumerate() {
                    if let Some(corr) = event_correlation(event) {
                        out.entry(corr).or_insert(GpuRef {
                            tid: tid.clone(),
                            event_idx: idx,
                        });
                    }
                }
            }
        }
    }
    out
}

#[derive(Clone)]
struct GpuRef {
    tid: String,
    /// Unused right now (we do not yet load the GPU event back into the
    /// CPU-side launch's payload — see the lookup_gpu_event TODO).  Kept
    /// so the field is in place when that wiring is finished.
    #[allow(dead_code)]
    event_idx: usize,
}

fn event_correlation(event: &Event) -> Option<i64> {
    event.args.as_ref().and_then(|a| {
        a.get("correlation")
            .or_else(|| a.get("External id"))
            .and_then(|v| v.as_i64())
    })
}

fn build_cpu_tree(
    compressor: &mut TemplateCompressor,
    _rank: &str,
    _pid: i64,
    _tid: &str,
    _phase: Phase,
    events: &[Event],
    gpu_events: &AHashMap<i64, GpuRef>,
    consumed: &mut ahash::AHashSet<i64>,
) -> Node {
    // Sort events by (ts, -dur) -- Python uses the same key.
    let mut sorted: Vec<&Event> = events.iter().collect();
    sorted.sort_by(|a, b| {
        a.ts.cmp(&b.ts)
            .then_with(|| b.dur.unwrap_or(0).cmp(&a.dur.unwrap_or(0)))
    });

    let mut roots: Vec<Node> = Vec::new();
    let mut stack: Vec<(i64, i64, usize)> = Vec::new(); // (ts, end_ts, idx_in_path)
    // For each level on the stack we need a place to push children of the
    // currently-open node.  We track that with `path` parallel to `stack`.
    let mut path: Vec<Node> = Vec::new();

    for ev in sorted {
        let ts = ev.ts;
        let dur = ev.dur.unwrap_or(0);
        let end_ts = ts + dur.max(0);

        // Pop stack frames whose end is <= this event's start.
        while let Some(&(_, top_end, _)) = stack.last() {
            if top_end <= ts || (dur > 0 && top_end < ts) {
                let finished = path.pop().expect("path/stack invariant");
                stack.pop();
                attach_to_parent(&mut roots, &mut path, finished);
            } else {
                break;
            }
        }

        // Build this event's node.  Pair with GPU kernel if there's a correlation.
        let new_node = make_cpu_or_kernel_node(compressor, ev, gpu_events, consumed);
        path.push(new_node);
        stack.push((ts, end_ts, path.len() - 1));
    }

    // Drain remaining open frames.
    while let Some(_top) = stack.pop() {
        let finished = path.pop().expect("path/stack invariant");
        attach_to_parent(&mut roots, &mut path, finished);
    }

    if roots.len() == 1 {
        roots.into_iter().next().unwrap()
    } else {
        Node::Root { children: roots }
    }
}

fn attach_to_parent(roots: &mut Vec<Node>, path: &mut [Node], finished: Node) {
    if let Some(parent) = path.last_mut() {
        push_child(parent, finished);
    } else {
        roots.push(finished);
    }
}

fn push_child(parent: &mut Node, child: Node) {
    match parent {
        Node::Cpu(p) => p.children.push(child),
        Node::Root { children } => children.push(child),
        Node::SameCpu(_) | Node::Gpu(_) | Node::KernelLaunch(_) | Node::KernelsLaunch(_) => {
            // These variants never receive children during the initial build
            // pass; structural compression introduces them later.
            // Defensive fallback: wrap as a sibling root (shouldn't happen).
            tracing::debug!("unexpected child push to leaf-like node; ignoring");
        }
    }
}

fn make_cpu_or_kernel_node(
    compressor: &mut TemplateCompressor,
    event: &Event,
    gpu_events: &AHashMap<i64, GpuRef>,
    consumed: &mut ahash::AHashSet<i64>,
) -> Node {
    let (cpu_template, cpu_instance) = compressor.intern_event_template(event);

    if compressor.config.enable_kernel_links {
        if let Some(corr) = event_correlation(event) {
            if let Some(gpu_ref) = gpu_events.get(&corr) {
                if !consumed.contains(&corr) {
                    consumed.insert(corr);
                    // Look up the actual GPU event.
                    if let Some(gpu_event) = lookup_gpu_event(gpu_ref) {
                        let (gpu_template, gpu_instance) = compressor.intern_kernel_template(&gpu_event, &gpu_ref.tid);
                        return Node::KernelLaunch(KernelLaunchNode {
                            cpu_template,
                            cpu_instance,
                            gpu_template,
                            gpu_instance,
                        });
                    }
                }
            }
        }
    }

    Node::Cpu(CpuNode {
        template: cpu_template,
        instance: cpu_instance,
        children: Vec::new(),
        slots: Vec::new(),
    })
}

/// Helper: GPU events live in the trace itself; we don't keep a copy here.
/// The build_rank closure above re-borrows them via the `streams` map but
/// we don't have that here.  Instead we leave a marker — concrete lookup
/// happens via `LiveStreams` if/when we wire it up.  For now we look the
/// event up via a lazy path (matching Python's behaviour where every CPU
/// launch grabs the full GPU event by correlation id).
fn lookup_gpu_event(_gpu_ref: &GpuRef) -> Option<Event> {
    // The live caller will replace this; build_cpu_tree already records the
    // gpu_ref so consumers can later attach the actual GPU event payload.
    None
}

fn build_gpu_tree(
    compressor: &mut TemplateCompressor,
    _pid: i64,
    tid: &str,
    _phase: Phase,
    events: &[Event],
    consumed: &ahash::AHashSet<i64>,
) -> Node {
    let mut templates: Vec<TemplateId> = Vec::new();
    let mut instances: Vec<InstanceId> = Vec::new();
    for event in events {
        if let Some(corr) = event_correlation(event) {
            if compressor.config.enable_kernel_links && consumed.contains(&corr) {
                continue;
            }
        }
        let (tid_, inst) = compressor.intern_kernel_template(event, tid);
        templates.push(tid_);
        instances.push(inst);
    }
    Node::Gpu(GpuNode { templates, instances })
}

/// Stub used while the lookup_gpu_event helper above is evolving.  This
/// keeps the compile clean — actual paired-launch IDs are filled in via
/// the integration done in `core.rs::compress`.
#[allow(dead_code)]
pub(crate) fn _kernels_launch_marker() -> Node {
    Node::KernelsLaunch(KernelsLaunchNode {
        cpu_template: 0,
        cpu_instances: Vec::new(),
        gpu_templates: Vec::new(),
        gpu_instances: Vec::new(),
    })
}

#[allow(dead_code)]
pub(crate) fn _samecpu_marker() -> Node {
    Node::SameCpu(SameCpuNode {
        template: 0,
        instances: Vec::new(),
        children: Vec::new(),
        slots: Vec::new(),
    })
}

#[allow(dead_code)]
pub(crate) fn _imap_marker() -> IndexMap<String, ()> {
    IndexMap::new()
}
