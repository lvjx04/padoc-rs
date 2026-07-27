//! Lossless reconstruction of a [`Trace`] from a [`CompressedTrace`].
//!
//! The compressed form keeps every original field — we just have to walk the
//! call-tree and re-emit one [`Event`] per `(template_id, instance_id)` pair,
//! placing it back under the right `(rank, pid, tid, phase)` stream.
//!
//! Notes on placement:
//!
//! * `Node::Cpu` / `Node::SameCpu` events live at the tree's stream coordinate
//!   (the `(rank, pid, tid, ph)` key under which the root was stored).
//! * `Node::Gpu` events live at the template's per-instance `(pid, stream_tid, ph)`
//!   triple — that's why GPU templates carry those columns.
//! * `Node::KernelLaunch` / `KernelsLaunch` emits **two** events: the CPU launch
//!   under the tree coordinate, and the GPU kernel under the GPU template's
//!   per-instance coordinate.
//!
//! After reconstruction the per-rank `ts` is shifted back to absolute by adding
//! [`CompressedTrace::start_timestamp`].

use ahash::AHashMap;

use crate::event::{ArgColumn, Args, Event, MergeEvent, MergeKernelEvent, Phase, Template};
use crate::node::Node;
use crate::slp::decode_name_nums;
use crate::trace::{CompressedTrace, Trace};
use crate::utils;

/// Reconstruct a fully-materialised `Trace` from a `CompressedTrace`.
pub fn decompress(compressed: &CompressedTrace) -> Trace {
    let mut trace = Trace::empty();

    for (rank, pid_map) in &compressed.ranks {
        let start_ts = compressed.start_timestamp.get(rank).copied().unwrap_or(0);

        for (pid, tid_map) in pid_map {
            for (tid, ph_map) in tid_map {
                for (ph_byte, node) in ph_map {
                    let context = DecodeContext {
                        rank,
                        cpu_pid: *pid,
                        cpu_tid: tid,
                        cpu_phase: Phase(*ph_byte),
                        templates: &compressed.templates,
                    };
                    visit(node, &context, &mut trace);
                }
            }
        }
        // Restore start_timestamp for downstream tools that care.
        trace.start_timestamp.insert(rank.clone(), start_ts);
    }
    trace.metadata = compressed.metadata.clone();
    trace
}

struct DecodeContext<'a> {
    rank: &'a str,
    cpu_pid: i64,
    cpu_tid: &'a str,
    cpu_phase: Phase,
    templates: &'a [Template],
}

fn visit(node: &Node, context: &DecodeContext<'_>, trace: &mut Trace) {
    match node {
        Node::Root { children } => {
            for c in children {
                visit(c, context, trace);
            }
        }
        Node::Cpu(n) => {
            emit_cpu(context, n.template, n.instance, trace);
            for c in &n.children {
                visit(c, context, trace);
            }
            for c in &n.slots {
                visit(c, context, trace);
            }
        }
        Node::SameCpu(n) => {
            for inst in &n.instances {
                emit_cpu(context, n.template, *inst, trace);
            }
            for c in &n.children {
                visit(c, context, trace);
            }
            for slot in &n.slots {
                for c in slot {
                    visit(c, context, trace);
                }
            }
        }
        Node::Gpu(n) => {
            for (tmpl_id, inst) in n.templates.iter().zip(n.instances.iter()) {
                emit_gpu(context, *tmpl_id, *inst, trace);
            }
        }
        Node::KernelLaunch(n) => {
            // CPU launch event is emitted by the enclosing CpuNode/SameCpuNode;
            // KernelLaunch is just a GPU pointer in the call tree.
            emit_gpu(context, n.gpu_template, n.gpu_instance, trace);
        }
        Node::KernelsLaunch(n) => {
            for (tmpl_id, inst) in n.gpu_templates.iter().zip(n.gpu_instances.iter()) {
                emit_gpu(context, *tmpl_id, *inst, trace);
            }
        }
    }
}

fn emit_cpu(context: &DecodeContext<'_>, tmpl_id: u32, instance: u32, trace: &mut Trace) {
    let tmpl = match context.templates.get(tmpl_id as usize) {
        Some(Template::Cpu(t)) => t,
        _ => return,
    };
    let event = build_cpu_event(
        tmpl,
        instance as usize,
        context.cpu_pid,
        context.cpu_tid,
        context.cpu_phase,
    );
    push(
        trace,
        context.rank,
        context.cpu_pid,
        context.cpu_tid,
        context.cpu_phase,
        event,
    );
}

fn emit_gpu(context: &DecodeContext<'_>, tmpl_id: u32, instance: u32, trace: &mut Trace) {
    let tmpl = match context.templates.get(tmpl_id as usize) {
        Some(Template::Gpu(t)) => t,
        _ => return,
    };
    let i = instance as usize;
    let pid = tmpl.pid.get(i).unwrap_or(0);
    let tid = tmpl.stream_tid.get(i).unwrap_or_default().to_string();
    let phase = tmpl.ph.get(i).unwrap_or(Phase::COMPLETE);
    let event = build_gpu_event(tmpl, i, pid, &tid, phase);
    push(trace, context.rank, pid, &tid, phase, event);
}

fn build_cpu_event(tmpl: &MergeEvent, i: usize, pid: i64, tid: &str, phase: Phase) -> Event {
    let nums = decode_name_nums(&tmpl.name_nums, i);
    let name = utils::restore_digits(&tmpl.name_pattern, &nums);
    // ts in CompressedTrace is already in per-rank relative form (matches the
    // loaded Trace).  Callers that need absolute ts should add `start_timestamp[rank]`.
    let ts = tmpl.ts.get(i).unwrap_or(0);
    let dur = tmpl.dur.get(i);
    let id = tmpl.id.get(i);
    let args = decode_args(&tmpl.arg_keys, &tmpl.args_columns, i);
    Event {
        name,
        ts,
        dur,
        cat: tmpl.cat.clone(),
        ph: phase,
        pid,
        tid: tid.to_string(),
        args,
        id,
        bp: tmpl.bp.clone(),
        s: tmpl.s.clone(),
    }
}

fn build_gpu_event(tmpl: &MergeKernelEvent, i: usize, pid: i64, tid: &str, phase: Phase) -> Event {
    let nums = decode_name_nums(&tmpl.name_nums, i);
    let name = utils::restore_digits(&tmpl.name_pattern, &nums);
    let ts = tmpl.ts.get(i).unwrap_or(0);
    let dur = tmpl.dur.get(i);
    let id = tmpl.id.get(i);
    let args = decode_args(&tmpl.arg_keys, &tmpl.args_columns, i);
    Event {
        name,
        ts,
        dur,
        cat: tmpl.cat.clone(),
        ph: phase,
        pid,
        tid: tid.to_string(),
        args,
        id,
        bp: tmpl.bp.clone(),
        s: tmpl.s.clone(),
    }
}

fn decode_args(arg_keys: &[String], args_columns: &[ArgColumn], i: usize) -> Option<Args> {
    if arg_keys.is_empty() {
        return None;
    }
    // `EventSignature` partitions events by their exact key set, so every
    // instance of a template has the same set of keys.  A stored `Null` is
    // therefore a real `null` from the source JSON, not a "missing" sentinel,
    // and must round-trip into the rebuilt args map.
    let mut map = AHashMap::with_capacity(arg_keys.len());
    for (k, col) in arg_keys.iter().zip(args_columns.iter()) {
        if let Some(v) = col.get_owned(i) {
            map.insert(k.clone(), v);
        }
    }
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

fn push(trace: &mut Trace, rank: &str, pid: i64, tid: &str, phase: Phase, event: Event) {
    let rank_streams = trace.ranks.entry(rank.to_string()).or_default();
    let tid_layer = rank_streams
        .entry(pid)
        .or_default()
        .entry(tid.to_string())
        .or_default();
    tid_layer.entry(phase).or_default().push(event);
}
