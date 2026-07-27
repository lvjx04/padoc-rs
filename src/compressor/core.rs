//! `TemplateCompressor` driver—coordinates template formation, flat stream
//! references, and column finalization.
//!
use ahash::AHashMap;

use super::config::CompressorConfig;
use crate::event::{
    ArgColumn, Event, EventSignature, MergeEvent, MergeKernelEvent, NameNums, Template,
};
use crate::node::TemplateId;
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

/// Stateful compressor for independent trace artifacts.
pub struct TemplateCompressor {
    pub(crate) config: CompressorConfig,
    pub(crate) templates: Vec<Template>,
    pub(crate) cpu_signature_index: AHashMap<EventSignature, TemplateId>,
    pub(crate) gpu_signature_index: AHashMap<EventSignature, TemplateId>,
}

impl TemplateCompressor {
    pub fn new() -> Self {
        Self::with_config(CompressorConfig::default())
    }

    pub(crate) fn with_config(config: CompressorConfig) -> Self {
        Self {
            config,
            templates: Vec::new(),
            cpu_signature_index: AHashMap::new(),
            gpu_signature_index: AHashMap::new(),
        }
    }

    /// Compress every rank in `trace`.  This is the main entry point.
    pub fn compress(&mut self, trace: &Trace) -> Result<CompressedTrace> {
        self.templates.clear();
        self.cpu_signature_index.clear();
        self.gpu_signature_index.clear();

        let mut compressed = CompressedTrace {
            metadata: trace.metadata.clone(),
            start_timestamp: trace.start_timestamp.clone(),
            ..CompressedTrace::default()
        };

        for rank in trace.rank_ids() {
            let rank_streams = match trace.ranks.get(&rank) {
                Some(s) => s,
                None => continue,
            };
            let rank_root = super::flat::build_rank(self, rank_streams);
            compressed.ranks.insert(rank, rank_root);
        }

        // Numeric / args / name finalisation.
        self.finalize_templates();
        compressed.templates = std::mem::take(&mut self.templates);
        Ok(compressed)
    }

    /// Look up an existing template id by signature, or create a fresh one.
    pub(crate) fn intern_event_template(&mut self, event: &Event) -> (TemplateId, u32) {
        let signature = event.template_signature();
        if let Some(&tid) = self.cpu_signature_index.get(&signature) {
            let inst = append_event_to_template(&mut self.templates[tid as usize], event);
            return (tid, inst);
        }
        let tid = self.templates.len() as TemplateId;
        let arg_keys: Vec<String> = signature.arg_keys.iter().cloned().collect();
        let args_columns: Vec<ArgColumn> = arg_keys
            .iter()
            .map(|_| ArgColumn::PerInstance(Vec::new()))
            .collect();
        let mut tmpl = MergeEvent {
            name_pattern: signature.normalized_name.clone(),
            cat: event.cat.clone(),
            bp: event.bp.clone(),
            s: event.s.clone(),
            arg_keys,
            args_columns,
            ..Default::default()
        };
        let inst = append_to_merge(&mut tmpl, event);
        self.templates.push(Template::Cpu(tmpl));
        self.cpu_signature_index.insert(signature, tid);
        (tid, inst)
    }

    /// GPU-side template lookup; mirrors the CPU path but stores per-instance
    /// pid/stream_tid/ph because GPU events have to be reattached on decompression.
    pub(crate) fn intern_kernel_template(
        &mut self,
        event: &Event,
        gpu_stream_tid: &str,
    ) -> (TemplateId, u32) {
        let signature = event.template_signature();
        if let Some(&tid) = self.gpu_signature_index.get(&signature) {
            let inst =
                append_kernel_to_template(&mut self.templates[tid as usize], event, gpu_stream_tid);
            return (tid, inst);
        }
        let tid = self.templates.len() as TemplateId;
        let arg_keys: Vec<String> = signature.arg_keys.iter().cloned().collect();
        let args_columns: Vec<ArgColumn> = arg_keys
            .iter()
            .map(|_| ArgColumn::PerInstance(Vec::new()))
            .collect();
        let mut tmpl = MergeKernelEvent {
            name_pattern: signature.normalized_name.clone(),
            cat: event.cat.clone(),
            bp: event.bp.clone(),
            s: event.s.clone(),
            arg_keys,
            args_columns,
            ..Default::default()
        };
        let inst = append_to_kernel(&mut tmpl, event, gpu_stream_tid);
        self.templates.push(Template::Gpu(tmpl));
        self.gpu_signature_index.insert(signature, tid);
        (tid, inst)
    }

    /// Apply SLP, name-column transpose, args dedup according to `config`.
    ///
    /// Each template is independently transformed (no shared state), so we
    /// fan out across rayon's global thread pool.  On 1024-rank profiler
    /// traces the global table holds 5–10 k templates, each running SLP on
    /// hundreds of thousands of instances — sequentially that's a ~30 s
    /// tail at the end of the merge phase, which becomes ~1 s with the
    /// thread pool active.
    fn finalize_templates(&mut self) {
        use rayon::prelude::*;
        let config = &self.config;
        self.templates.par_iter_mut().for_each(|tmpl| match tmpl {
            Template::Cpu(t) => super::finalize::cpu_template(t, config),
            Template::Gpu(t) => super::finalize::gpu_template(t, config),
        });
    }
}

impl Default for TemplateCompressor {
    fn default() -> Self {
        Self::new()
    }
}

// --- internal helpers ------------------------------------------------------

fn append_event_to_template(tmpl: &mut Template, event: &Event) -> u32 {
    match tmpl {
        Template::Cpu(t) => append_to_merge(t, event),
        Template::Gpu(_) => {
            // Should not happen in practice; defensive path mirrors Python behaviour.
            panic!("event signature collision: cpu event hashing into gpu template");
        }
    }
}

fn append_kernel_to_template(tmpl: &mut Template, event: &Event, gpu_stream_tid: &str) -> u32 {
    match tmpl {
        Template::Gpu(t) => append_to_kernel(t, event, gpu_stream_tid),
        Template::Cpu(_) => {
            panic!("event signature collision: gpu event hashing into cpu template")
        }
    }
}

pub(crate) fn append_to_merge(tmpl: &mut MergeEvent, event: &Event) -> u32 {
    let nums = crate::utils::extract_digit_runs(&event.name);
    let inst = tmpl.ts.len() as u32;
    tmpl.ts.push(event.ts);
    if let Some(d) = event.dur {
        tmpl.dur.push(d);
    }
    if let Some(i) = event.id {
        tmpl.id.push(i);
    }
    append_args_columns(&mut tmpl.args_columns, &tmpl.arg_keys, event);
    push_name_nums(&mut tmpl.name_nums, nums);
    inst
}

pub(crate) fn append_to_kernel(
    tmpl: &mut MergeKernelEvent,
    event: &Event,
    gpu_stream_tid: &str,
) -> u32 {
    let nums = crate::utils::extract_digit_runs(&event.name);
    let inst = tmpl.ts.len() as u32;
    tmpl.ts.push(event.ts);
    if let Some(d) = event.dur {
        tmpl.dur.push(d);
    }
    if let Some(i) = event.id {
        tmpl.id.push(i);
    }
    tmpl.pid.push(event.pid);
    tmpl.stream_tid.push(gpu_stream_tid.to_string());
    tmpl.ph.push(event.ph);
    append_args_columns(&mut tmpl.args_columns, &tmpl.arg_keys, event);
    push_name_nums(&mut tmpl.name_nums, nums);
    inst
}

fn append_args_columns(args_columns: &mut [ArgColumn], arg_keys: &[String], event: &Event) {
    for (col, key) in args_columns.iter_mut().zip(arg_keys.iter()) {
        let v = event
            .args
            .as_ref()
            .and_then(|a| a.get(key.as_str()))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        col.push(v);
    }
}

fn push_name_nums(slot: &mut NameNums, nums: Vec<String>) {
    match slot {
        NameNums::Empty => {
            if !nums.is_empty() {
                *slot = NameNums::Rows(vec![nums]);
            }
        }
        NameNums::Rows(rows) => rows.push(nums),
        NameNums::Columnar(_) => panic!("cannot append after compress_name_nums"),
    }
}
