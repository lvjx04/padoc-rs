//! Cross-rank template-table merging.
//!
//! Each parallel worker compresses a single rank with its own private
//! [`TemplateCompressor`], producing a [`RankShard`] containing the rank's
//! call tree and **local** template table.  After every shard is built we
//! fold them together: identical signatures from different ranks collapse
//! to a single global template, every node's `(template_id, instance_id)`
//! pair is rewritten into the global namespace, and `finalize` runs once
//! over the global table.
//!
//! This is the classic "compress locally, merge globally" pattern — Phase A
//! is fully parallel with no cross-thread state, Phase B is a cheap
//! single-threaded hash dedup, Phase C re-walks every shard's tree (also
//! parallelisable) to apply the rewrite.
//!
//! Compared to running a single `TemplateCompressor` over an in-memory
//! merged `Trace`:
//!
//! * Output size is **identical** — same global template set, same call
//!   trees.  Only the construction path differs.
//! * Memory peaks at one rank per worker, never the full multi-rank trace.
//! * Throughput scales linearly with `--workers` (compression is
//!   CPU-bound on JSON parse + signature hashing).
//!
//! Caveats:
//!
//! * Shard templates must not have been finalised — SLP and args dedup
//!   need the full cross-rank instance set.  `TemplateCompressor::
//!   compress_rank` deliberately stops before `finalize`.
//! * `EventSignature` is reconstructed from a template's stored fields
//!   (`name_pattern`, `cat`, `bp`, `s`, `arg_keys`).  Those fields exactly
//!   match what `Event::template_signature` produced when the template was
//!   first interned, so the dedup is consistent with single-pass
//!   compression.

use ahash::AHashMap;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;

use super::config::CompressorConfig;
use super::core::TemplateCompressor;
use crate::event::{ArgColumn, EventSignature, MergeEvent, MergeKernelEvent, NameNums, Template};
use crate::node::{InstanceId, Node, TemplateId};
use crate::trace::CompressedTrace;
use crate::Result;

use smallvec::SmallVec;

/// One rank's compressor output before global merging.
pub struct RankShard {
    pub rank: String,
    /// Templates produced by this rank's private interner.  Indices in
    /// `root` are local to this Vec.
    pub templates: Vec<Template>,
    /// `pid -> tid -> ph -> root_node` for this rank.
    pub root: BTreeMap<i64, BTreeMap<String, BTreeMap<u8, Node>>>,
    /// Per-rank metadata payload (process-name etc.).  Cloned from
    /// `Trace::metadata[rank]`.
    pub metadata: Option<ahash::AHashMap<String, Value>>,
    /// Per-rank wall-clock origin used by the analysis layer.
    pub start_timestamp: Option<i64>,
}

/// Collapse a vector of shards into a single [`CompressedTrace`].
///
/// The returned trace is bit-equivalent (modulo template ordering) to the
/// output of running [`TemplateCompressor::compress`] on the union of all
/// the shards' input ranks.
pub fn merge_shards(
    config: &CompressorConfig,
    shards: Vec<RankShard>,
) -> Result<CompressedTrace> {
    // -----------------------------------------------------------------------
    // Phase 1 (sequential): dedup templates into a global table.
    //
    // Touching `global_templates` and `global_index` requires exclusive
    // access — running this in parallel would mean lock contention on every
    // template, which on 1024-rank profiler with ~5–10 k unique templates
    // is far below the cost of the sequential pass anyway.  What's
    // expensive is _appending_ instance columns; that's deferred to the
    // tree-rewrite phase too, so this loop now only touches the matching
    // template's metadata, not its big per-instance Vecs.  See
    // `extend_template` for the column extension itself.
    // -----------------------------------------------------------------------
    let mut global_templates: Vec<Template> = Vec::new();
    let mut global_index: AHashMap<EventSignature, TemplateId> = AHashMap::new();

    let n_shards = shards.len();
    let mut shard_remaps: Vec<Vec<(TemplateId, u32)>> = Vec::with_capacity(n_shards);
    let mut shards = shards;
    for shard in shards.iter_mut() {
        let mut remap: Vec<(TemplateId, u32)> = Vec::with_capacity(shard.templates.len());
        for local_tmpl in shard.templates.drain(..) {
            let signature = template_signature(&local_tmpl);
            match global_index.get(&signature).copied() {
                Some(gtid) => {
                    let offset = global_templates[gtid as usize].instance_count() as u32;
                    extend_template(&mut global_templates[gtid as usize], local_tmpl)?;
                    remap.push((gtid, offset));
                }
                None => {
                    let new_id = global_templates.len() as TemplateId;
                    global_templates.push(local_tmpl);
                    global_index.insert(signature, new_id);
                    remap.push((new_id, 0));
                }
            }
        }
        shard_remaps.push(remap);
    }

    // -----------------------------------------------------------------------
    // Phase 2 (parallel): rewrite every shard's call tree using its remap.
    //
    // No two shards share node ownership, so this fans out cleanly across
    // rayon's pool.  Per-shard cost is O(nodes), which on profiler dominates
    // post-shard wall-clock; running it in parallel turns 30+ s into a few
    // seconds at 32 threads.
    // -----------------------------------------------------------------------
    let rewritten_ranks: Vec<(
        String,
        BTreeMap<i64, BTreeMap<String, BTreeMap<u8, Node>>>,
        Option<ahash::AHashMap<String, Value>>,
        Option<i64>,
    )> = shards
        .into_par_iter()
        .zip(shard_remaps.into_par_iter())
        .map(|(shard, remap)| {
            let RankShard {
                rank,
                templates: _,
                root,
                metadata,
                start_timestamp,
            } = shard;
            let mut rewritten = BTreeMap::new();
            for (pid, threads) in root {
                let mut new_threads = BTreeMap::new();
                for (tid, phases) in threads {
                    let mut new_phases = BTreeMap::new();
                    for (ph, mut node) in phases {
                        rewrite_node(&mut node, &remap);
                        new_phases.insert(ph, node);
                    }
                    new_threads.insert(tid, new_phases);
                }
                rewritten.insert(pid, new_threads);
            }
            (rank, rewritten, metadata, start_timestamp)
        })
        .collect();

    // -----------------------------------------------------------------------
    // Phase 3 (sequential, fast): assemble the output struct and run the
    // (parallel) finaliser over the global template table.
    // -----------------------------------------------------------------------
    let mut compressed = CompressedTrace::default();
    for (rank, rewritten, metadata, start_timestamp) in rewritten_ranks {
        compressed.ranks.insert(rank.clone(), rewritten);
        if let Some(meta) = metadata {
            compressed.metadata.insert(rank.clone(), meta);
        }
        if let Some(ts) = start_timestamp {
            compressed.start_timestamp.insert(rank, ts);
        }
    }

    let mut compressor = TemplateCompressor::with_config(config.clone());
    compressor.set_templates_for_finalize(global_templates);
    compressor.finalize_in_place();
    compressed.templates = compressor.take_templates();

    Ok(compressed)
}

/// Rebuild a stable signature from a stored template.  Equivalent to the
/// signature originally produced when the template was first interned via
/// `Event::template_signature` — it uses the same fields.
fn template_signature(t: &Template) -> EventSignature {
    let (name_pattern, cat, bp, s, arg_keys) = match t {
        Template::Cpu(m) => (
            m.name_pattern.clone(),
            m.cat.clone(),
            m.bp.clone(),
            m.s.clone(),
            m.arg_keys.clone(),
        ),
        Template::Gpu(m) => (
            m.name_pattern.clone(),
            m.cat.clone(),
            None,
            None,
            m.arg_keys.clone(),
        ),
    };
    let mut keys: SmallVec<[String; 8]> = SmallVec::from_vec(arg_keys);
    keys.sort();
    EventSignature {
        normalized_name: name_pattern,
        cat,
        bp,
        s,
        arg_keys: keys,
    }
}

/// Append every instance of `src` to the parallel arrays of `dst`.  Both
/// templates must have the same `arg_keys` — guaranteed by signature
/// equality.  Templates here are still un-finalised, so `args_columns` are
/// `PerInstance` Vec<Value> (Constant only appears post-finalise).
fn extend_template(dst: &mut Template, src: Template) -> Result<()> {
    match (dst, src) {
        (Template::Cpu(d), Template::Cpu(s)) => extend_cpu(d, s),
        (Template::Gpu(d), Template::Gpu(s)) => extend_gpu(d, s),
        _ => Err(crate::Error::Other(
            "template kind mismatch during merge (cpu vs gpu)".into(),
        )),
    }
}

fn extend_cpu(dst: &mut MergeEvent, src: MergeEvent) -> Result<()> {
    if dst.arg_keys != src.arg_keys {
        return Err(crate::Error::Other(format!(
            "arg_keys mismatch when merging template '{}': {:?} vs {:?}",
            dst.name_pattern, dst.arg_keys, src.arg_keys
        )));
    }
    dst.ts.extend_from(src.ts);
    dst.dur.extend_from(src.dur);
    dst.id.extend_from(src.id);
    extend_args_columns(&mut dst.args_columns, src.args_columns)?;
    extend_name_nums(&mut dst.name_nums, src.name_nums)?;
    Ok(())
}

fn extend_gpu(dst: &mut MergeKernelEvent, src: MergeKernelEvent) -> Result<()> {
    if dst.arg_keys != src.arg_keys {
        return Err(crate::Error::Other(format!(
            "arg_keys mismatch when merging gpu template '{}'",
            dst.name_pattern
        )));
    }
    dst.ts.extend_from(src.ts);
    dst.dur.extend_from(src.dur);
    dst.pid.extend_from(src.pid);
    dst.stream_tid.extend_from(src.stream_tid);
    dst.ph.extend_from(src.ph);
    extend_args_columns(&mut dst.args_columns, src.args_columns)?;
    extend_name_nums(&mut dst.name_nums, src.name_nums)?;
    Ok(())
}

fn extend_args_columns(dst: &mut Vec<ArgColumn>, src: Vec<ArgColumn>) -> Result<()> {
    if dst.len() != src.len() {
        return Err(crate::Error::Other(
            "args_columns length mismatch on merge".into(),
        ));
    }
    for (d, s) in dst.iter_mut().zip(src.into_iter()) {
        match (d, s) {
            (ArgColumn::PerInstance(dv), ArgColumn::PerInstance(sv)) => dv.extend(sv),
            _ => {
                return Err(crate::Error::Other(
                    "merge_shards saw a Constant args column — finalise must run after merge"
                        .into(),
                ))
            }
        }
    }
    Ok(())
}

fn extend_name_nums(dst: &mut NameNums, src: NameNums) -> Result<()> {
    use NameNums::*;
    let merged = match (std::mem::take(dst), src) {
        (Empty, Empty) => Empty,
        (Empty, Rows(r)) => Rows(r),
        (Rows(r), Empty) => Rows(r),
        (Rows(mut a), Rows(b)) => {
            a.extend(b);
            Rows(a)
        }
        _ => {
            return Err(crate::Error::Other(
                "name_nums in non-Rows form during merge — finalise ran too early".into(),
            ))
        }
    };
    *dst = merged;
    Ok(())
}

fn rewrite_node(node: &mut Node, remap: &[(TemplateId, u32)]) {
    match node {
        Node::Root { children } => {
            for child in children {
                rewrite_node(child, remap);
            }
        }
        Node::Cpu(n) => {
            let (gtid, offset) = remap[n.template as usize];
            n.template = gtid;
            n.instance = (n.instance + offset) as InstanceId;
            for c in &mut n.children {
                rewrite_node(c, remap);
            }
            for c in &mut n.slots {
                rewrite_node(c, remap);
            }
        }
        Node::SameCpu(n) => {
            let (gtid, offset) = remap[n.template as usize];
            n.template = gtid;
            for inst in &mut n.instances {
                *inst = *inst + offset;
            }
            for c in &mut n.children {
                rewrite_node(c, remap);
            }
            for slot in &mut n.slots {
                for c in slot {
                    rewrite_node(c, remap);
                }
            }
        }
        Node::Gpu(n) => {
            for (tmpl, inst) in n.templates.iter_mut().zip(n.instances.iter_mut()) {
                let (gtid, offset) = remap[*tmpl as usize];
                *tmpl = gtid;
                *inst = *inst + offset;
            }
        }
        Node::KernelLaunch(n) => {
            let (cgtid, coffset) = remap[n.cpu_template as usize];
            n.cpu_template = cgtid;
            n.cpu_instance = n.cpu_instance + coffset;
            let (ggtid, goffset) = remap[n.gpu_template as usize];
            n.gpu_template = ggtid;
            n.gpu_instance = n.gpu_instance + goffset;
        }
        Node::KernelsLaunch(n) => {
            let (cgtid, coffset) = remap[n.cpu_template as usize];
            n.cpu_template = cgtid;
            for inst in &mut n.cpu_instances {
                *inst = *inst + coffset;
            }
            for (tmpl, inst) in n.gpu_templates.iter_mut().zip(n.gpu_instances.iter_mut()) {
                let (gtid, offset) = remap[*tmpl as usize];
                *tmpl = gtid;
                *inst = *inst + offset;
            }
        }
    }
}
