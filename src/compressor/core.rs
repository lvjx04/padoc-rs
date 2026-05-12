//! `TemplateCompressor` driver — coordinates call-tree build, template
//! formation, structural compression and numeric finalisation.
//!
//! Currently a thin scaffold; full implementation is filled in module-by-module.

use ahash::AHashMap;

use super::config::CompressorConfig;
use crate::event::{Event, EventSignature, MergeEvent, MergeKernelEvent, NameNums, Template};
use crate::node::TemplateId;
use crate::trace::{CompressedTrace, Trace};
use crate::{Error, Result};

/// Stateful compressor.  Reset between runs via [`Self::with_config`].
pub struct TemplateCompressor {
    pub config: CompressorConfig,
    pub(crate) templates: Vec<Template>,
    pub(crate) signature_index: AHashMap<EventSignature, TemplateId>,
}

impl TemplateCompressor {
    pub fn new() -> Self {
        Self::with_config(CompressorConfig::default())
    }

    pub fn with_config(config: CompressorConfig) -> Self {
        Self {
            config,
            templates: Vec::new(),
            signature_index: AHashMap::new(),
        }
    }

    /// Compress every rank in `trace`.  This is the main entry point.
    pub fn compress(&mut self, trace: &Trace) -> Result<CompressedTrace> {
        let mut compressed = CompressedTrace::default();
        compressed.metadata = trace.metadata.clone();
        compressed.start_timestamp = trace.start_timestamp.clone();

        for rank in trace.rank_ids() {
            let rank_streams = match trace.ranks.get(&rank) {
                Some(s) => s,
                None => continue,
            };
            let rank_root = super::call_tree::build_rank(self, rank.as_str(), rank_streams);
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
        if let Some(&tid) = self.signature_index.get(&signature) {
            let inst = append_event_to_template(&mut self.templates[tid as usize], event);
            return (tid, inst);
        }
        let tid = self.templates.len() as TemplateId;
        let mut tmpl = MergeEvent {
            name_pattern: signature.normalized_name.clone(),
            cat: event.cat.clone(),
            bp: event.bp.clone(),
            s: event.s.clone(),
            arg_keys: signature.arg_keys.iter().cloned().collect(),
            ..Default::default()
        };
        let inst = append_to_merge(&mut tmpl, event);
        self.templates.push(Template::Cpu(tmpl));
        self.signature_index.insert(signature, tid);
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
        if let Some(&tid) = self.signature_index.get(&signature) {
            let inst = append_kernel_to_template(&mut self.templates[tid as usize], event, gpu_stream_tid);
            return (tid, inst);
        }
        let tid = self.templates.len() as TemplateId;
        let mut tmpl = MergeKernelEvent {
            name_pattern: signature.normalized_name.clone(),
            cat: event.cat.clone(),
            arg_keys: signature.arg_keys.iter().cloned().collect(),
            ..Default::default()
        };
        let inst = append_to_kernel(&mut tmpl, event, gpu_stream_tid);
        self.templates.push(Template::Gpu(tmpl));
        self.signature_index.insert(signature, tid);
        (tid, inst)
    }

    /// Apply SLP, name-column transpose, args dedup according to `config`.
    fn finalize_templates(&mut self) {
        for tmpl in self.templates.iter_mut() {
            match tmpl {
                Template::Cpu(t) => super::structural::finalize_cpu_template(t, &self.config),
                Template::Gpu(t) => super::structural::finalize_gpu_template(t, &self.config),
            }
        }
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
        Template::Cpu(_) => panic!("event signature collision: gpu event hashing into cpu template"),
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
    let row: Vec<crate::event::ArgValue> = tmpl
        .arg_keys
        .iter()
        .map(|k| {
            event
                .args
                .as_ref()
                .and_then(|a| a.get(k.as_str()))
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        })
        .collect();
    tmpl.args_values.push(row);
    push_name_nums(&mut tmpl.name_nums, nums);
    inst
}

pub(crate) fn append_to_kernel(tmpl: &mut MergeKernelEvent, event: &Event, gpu_stream_tid: &str) -> u32 {
    let nums = crate::utils::extract_digit_runs(&event.name);
    let inst = tmpl.ts.len() as u32;
    tmpl.ts.push(event.ts);
    if let Some(d) = event.dur {
        tmpl.dur.push(d);
    }
    tmpl.pid.push(event.pid);
    tmpl.stream_tid.push(gpu_stream_tid.to_string());
    tmpl.ph.push(event.ph);
    let row: Vec<crate::event::ArgValue> = tmpl
        .arg_keys
        .iter()
        .map(|k| {
            event
                .args
                .as_ref()
                .and_then(|a| a.get(k.as_str()))
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        })
        .collect();
    tmpl.args_values.push(row);
    push_name_nums(&mut tmpl.name_nums, nums);
    inst
}

fn push_name_nums(slot: &mut NameNums, nums: Vec<i64>) {
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

#[allow(dead_code)]
pub(crate) fn placeholder_decompress(_ct: &CompressedTrace) -> Result<Trace> {
    Err(Error::Other("decompression not yet implemented".into()))
}
