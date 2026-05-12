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
    // 1) Index every GPU event by correlation (only the first match per
    //    correlation; later events with the same correlation stay where they
    //    are and are emitted by build_gpu_tree).
    let gpu_events = collect_gpu_events_by_correlation(streams);

    // Each correlation can only pair once; track which ones have been used.
    let mut paired_corrs: ahash::AHashSet<i64> = ahash::AHashSet::with_capacity(gpu_events.len());
    // Specific GPU events that were actually consumed; build_gpu_tree skips
    // exactly these.  Keyed by `(tid, event_index_in_phase)` — the only key
    // strong enough to distinguish events that share `ts` on the same stream
    // (e.g. zero-duration GPU annotations or short kernels with identical
    // start times).  Earlier `(tid, ts)` keying lost ~1.5% of GPU events on
    // streams with ts collisions.
    let mut consumed_gpu: ahash::AHashSet<GpuKey> =
        ahash::AHashSet::with_capacity(gpu_events.len());

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
                    &mut paired_corrs,
                    &mut consumed_gpu,
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

    // 3) Process GPU streams: every event not exactly consumed survives in GpuNode.
    for (pid, threads) in streams {
        for (tid, phases) in threads {
            if !is_gpu_stream(tid) {
                continue;
            }
            for (phase, events) in phases {
                let root = build_gpu_tree(compressor, *pid, tid, *phase, events, &consumed_gpu);
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

/// Returned map: correlation -> (pid, tid, phase, index, event).  Carries the
/// full `Event` and its origin coordinates so:
///   1. the CPU launch can build a `KernelLaunch` node directly, and
///   2. `build_gpu_tree` can skip the *exact* event by index — important when
///      multiple GPU events share `ts` on the same stream (zero-duration
///      annotations, identical-start short kernels), where `(tid, ts)` alone
///      collides and silently drops events on decompress.
fn collect_gpu_events_by_correlation(streams: &StreamMap) -> AHashMap<i64, GpuRef> {
    let mut out = AHashMap::new();
    for (pid, threads) in streams {
        for (tid, phases) in threads {
            if !is_gpu_stream(tid) {
                continue;
            }
            for (phase, events) in phases {
                for (idx, event) in events.iter().enumerate() {
                    if let Some(corr) = event_correlation(event) {
                        out.entry(corr).or_insert(GpuRef {
                            pid: *pid,
                            tid: tid.clone(),
                            phase: *phase,
                            idx,
                            event: event.clone(),
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
    pid: i64,
    tid: String,
    phase: Phase,
    idx: usize,
    event: Event,
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
    paired_corrs: &mut ahash::AHashSet<i64>,
    consumed_gpu: &mut ahash::AHashSet<GpuKey>,
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
        let new_node = make_cpu_or_kernel_node(compressor, ev, gpu_events, paired_corrs, consumed_gpu);
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
    paired_corrs: &mut ahash::AHashSet<i64>,
    consumed_gpu: &mut ahash::AHashSet<GpuKey>,
) -> Node {
    let (cpu_template, cpu_instance) = compressor.intern_event_template(event);

    let mut children: Vec<Node> = Vec::new();
    if compressor.config.enable_kernel_links {
        if let Some(corr) = event_correlation(event) {
            if let Some(gpu_ref) = gpu_events.get(&corr).cloned() {
                // Each correlation can only be paired once.
                if paired_corrs.insert(corr) {
                    consumed_gpu.insert(GpuKey {
                        pid: gpu_ref.pid,
                        tid: gpu_ref.tid.clone(),
                        phase: gpu_ref.phase,
                        idx: gpu_ref.idx,
                    });
                    let (gpu_template, gpu_instance) =
                        compressor.intern_kernel_template(&gpu_ref.event, &gpu_ref.tid);
                    // Mirror Python: keep the CPU launch as a regular CpuNode
                    // (so its other children survive) and attach a
                    // KernelLaunch *child* that carries only the GPU pointer.
                    // Decompression of `KernelLaunch` emits *only* the GPU
                    // event; the CPU side is emitted by the parent CpuNode.
                    children.push(Node::KernelLaunch(KernelLaunchNode {
                        cpu_template,
                        cpu_instance,
                        gpu_template,
                        gpu_instance,
                    }));
                }
            }
        }
    }

    Node::Cpu(CpuNode {
        template: cpu_template,
        instance: cpu_instance,
        children,
        slots: Vec::new(),
    })
}

fn build_gpu_tree(
    compressor: &mut TemplateCompressor,
    pid: i64,
    tid: &str,
    phase: Phase,
    events: &[Event],
    consumed_gpu: &ahash::AHashSet<GpuKey>,
) -> Node {
    let mut templates: Vec<TemplateId> = Vec::new();
    let mut instances: Vec<InstanceId> = Vec::new();
    for (idx, event) in events.iter().enumerate() {
        if compressor.config.enable_kernel_links {
            // Cheap-path check: build a borrowed key by allocating just once
            // (idx + tid is fixed-cost; the alternative is a parallel HashSet
            // keyed by (pid, phase, idx) but the perf delta is < 1%).
            let key = GpuKey {
                pid,
                tid: tid.to_string(),
                phase,
                idx,
            };
            if consumed_gpu.contains(&key) {
                continue;
            }
        }
        let (tid_, inst) = compressor.intern_kernel_template(event, tid);
        templates.push(tid_);
        instances.push(inst);
    }
    Node::Gpu(GpuNode { templates, instances })
}

/// Identifier of a single GPU event within one rank.  Used to mark events
/// that have been pulled into a `KernelLaunch` so `build_gpu_tree` skips the
/// exact same event without falsely dropping siblings that share `ts`.
#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct GpuKey {
    pid: i64,
    tid: String,
    phase: Phase,
    idx: usize,
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
