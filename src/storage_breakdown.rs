//! Storage profile of a `CompressedTrace` — what fraction of bytes goes to
//! templates vs structure vs metadata.

use crate::event::{ArgColumn, NameNums, NumColumn, PhaseColumn, StringColumn, Template};
use crate::node::{InstanceId, Node, TemplateId};
use crate::trace::CompressedTrace;
use crate::Result;
use serde::{Deserialize, Serialize};
use serde::ser::{SerializeSeq, Serializer};
use std::io::Write;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StorageBreakdown {
    pub total_bytes: u64,
    pub template_bytes: u64,
    pub structure_bytes: u64,
    pub metadata_bytes: u64,
    pub template_count: usize,
    pub node_count: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OnDiskRegion {
    pub name: String,
    pub msgpack_bytes: u64,
    pub zstd_bytes: u64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OnDiskBreakdown {
    /// Actual artifact size if the caller passed it in.  The region sizes are
    /// independent projections encoded one-at-a-time, so they are not expected
    /// to sum exactly to this value.
    pub artifact_bytes: Option<u64>,
    pub zstd_level: i32,
    pub regions: Vec<OnDiskRegion>,
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

/// Encode field-level projections through the same msgpack + zstd stack used
/// by PADOC artifacts.  Each row is encoded independently; this avoids
/// materialising a giant raw msgpack blob and gives the paper a stable
/// "which logical fields compress to how many bytes?" breakdown.
pub fn measure_on_disk_regions(
    compressed: &CompressedTrace,
    artifact_bytes: Option<u64>,
    zstd_level: i32,
) -> Result<OnDiskBreakdown> {
    let projections = TemplateProjections::new(&compressed.templates);
    let mut regions = Vec::new();

    push_region(&mut regions, "template_headers", &projections.headers, zstd_level)?;
    push_region(&mut regions, "ts_columns", &projections.ts_columns, zstd_level)?;
    push_region(&mut regions, "dur_columns", &projections.dur_columns, zstd_level)?;
    push_region(&mut regions, "ids_pids_phases_streams", &projections.identity_columns, zstd_level)?;
    push_region(&mut regions, "name_nums", &projections.name_nums, zstd_level)?;
    push_region(&mut regions, "args_columns", &projections.args_columns, zstd_level)?;
    push_region(&mut regions, "rank_node_tree", &compressed.ranks, zstd_level)?;
    push_region(
        &mut regions,
        "node_soft_links",
        &NodeRefsProjection { compressed },
        zstd_level,
    )?;
    push_region(&mut regions, "metadata", &compressed.metadata, zstd_level)?;
    push_region(&mut regions, "start_timestamp", &compressed.start_timestamp, zstd_level)?;

    Ok(OnDiskBreakdown {
        artifact_bytes,
        zstd_level,
        regions,
    })
}

fn push_region<T: Serialize>(
    regions: &mut Vec<OnDiskRegion>,
    name: &str,
    value: &T,
    zstd_level: i32,
) -> Result<()> {
    let (msgpack_bytes, zstd_bytes) = encoded_sizes(value, zstd_level)?;
    regions.push(OnDiskRegion {
        name: name.to_string(),
        msgpack_bytes,
        zstd_bytes,
    });
    Ok(())
}

fn encoded_sizes<T: Serialize>(value: &T, zstd_level: i32) -> Result<(u64, u64)> {
    let mut raw_counter = CountingWriter::default();
    rmp_serde::encode::write_named(&mut raw_counter, value)?;

    let encoder = zstd::stream::Encoder::new(CountingWriter::default(), zstd_level)?;
    let mut buf_enc = std::io::BufWriter::with_capacity(1 << 20, encoder);
    rmp_serde::encode::write_named(&mut buf_enc, value)?;
    buf_enc.flush()?;
    let encoder = buf_enc
        .into_inner()
        .map_err(|e| crate::Error::Other(format!("flush BufWriter: {}", e.error())))?;
    let zstd_counter = encoder.finish()?;
    Ok((raw_counter.bytes, zstd_counter.bytes))
}

#[derive(Default)]
struct CountingWriter {
    bytes: u64,
}

impl Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Serialize)]
struct TemplateHeader<'a> {
    kind: &'static str,
    name_pattern: &'a str,
    cat: Option<&'a str>,
    bp: Option<&'a str>,
    s: Option<&'a str>,
    arg_keys: &'a [String],
    instance_count: usize,
}

#[derive(Serialize)]
struct TemplateProjections<'a> {
    headers: Vec<TemplateHeader<'a>>,
    ts_columns: Vec<&'a NumColumn>,
    dur_columns: Vec<&'a NumColumn>,
    identity_columns: IdentityColumns<'a>,
    name_nums: Vec<&'a NameNums>,
    args_columns: Vec<ArgsColumns<'a>>,
}

#[derive(Default, Serialize)]
struct IdentityColumns<'a> {
    cpu_id: Vec<&'a NumColumn>,
    gpu_pid: Vec<&'a NumColumn>,
    gpu_ph: Vec<&'a PhaseColumn>,
    gpu_stream_tid: Vec<&'a StringColumn>,
}

#[derive(Serialize)]
struct ArgsColumns<'a> {
    keys: &'a [String],
    columns: &'a [ArgColumn],
}

impl<'a> TemplateProjections<'a> {
    fn new(templates: &'a [Template]) -> Self {
        let mut out = Self {
            headers: Vec::with_capacity(templates.len()),
            ts_columns: Vec::new(),
            dur_columns: Vec::new(),
            identity_columns: IdentityColumns::default(),
            name_nums: Vec::new(),
            args_columns: Vec::new(),
        };

        for template in templates {
            match template {
                Template::Cpu(t) => {
                    out.headers.push(TemplateHeader {
                        kind: "cpu",
                        name_pattern: &t.name_pattern,
                        cat: t.cat.as_deref(),
                        bp: t.bp.as_deref(),
                        s: t.s.as_deref(),
                        arg_keys: &t.arg_keys,
                        instance_count: t.instance_count(),
                    });
                    out.ts_columns.push(&t.ts);
                    out.dur_columns.push(&t.dur);
                    out.identity_columns.cpu_id.push(&t.id);
                    out.name_nums.push(&t.name_nums);
                    out.args_columns.push(ArgsColumns {
                        keys: &t.arg_keys,
                        columns: &t.args_columns,
                    });
                }
                Template::Gpu(t) => {
                    out.headers.push(TemplateHeader {
                        kind: "gpu",
                        name_pattern: &t.name_pattern,
                        cat: t.cat.as_deref(),
                        bp: None,
                        s: None,
                        arg_keys: &t.arg_keys,
                        instance_count: t.instance_count(),
                    });
                    out.ts_columns.push(&t.ts);
                    out.dur_columns.push(&t.dur);
                    out.identity_columns.gpu_pid.push(&t.pid);
                    out.identity_columns.gpu_ph.push(&t.ph);
                    out.identity_columns.gpu_stream_tid.push(&t.stream_tid);
                    out.name_nums.push(&t.name_nums);
                    out.args_columns.push(ArgsColumns {
                        keys: &t.arg_keys,
                        columns: &t.args_columns,
                    });
                }
            }
        }

        out
    }
}

struct NodeRefsProjection<'a> {
    compressed: &'a CompressedTrace,
}

impl Serialize for NodeRefsProjection<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(None)?;
        for processes in self.compressed.ranks.values() {
            for threads in processes.values() {
                for phases in threads.values() {
                    for root in phases.values() {
                        serialize_node_refs(root, &mut seq)?;
                    }
                }
            }
        }
        seq.end()
    }
}

#[derive(Serialize)]
struct NodeRefRecord<'a> {
    kind: &'static str,
    template: Option<TemplateId>,
    instance: Option<InstanceId>,
    templates: &'a [TemplateId],
    instances: &'a [InstanceId],
    cpu_instances: &'a [InstanceId],
    gpu_templates: &'a [TemplateId],
    gpu_instances: &'a [InstanceId],
}

fn serialize_node_refs<S: SerializeSeq>(
    node: &Node,
    seq: &mut S,
) -> std::result::Result<(), S::Error> {
    const EMPTY: &[u32] = &[];

    match node {
        Node::Root { children } => {
            seq.serialize_element(&NodeRefRecord {
                kind: "root",
                template: None,
                instance: None,
                templates: EMPTY,
                instances: EMPTY,
                cpu_instances: EMPTY,
                gpu_templates: EMPTY,
                gpu_instances: EMPTY,
            })?;
            for child in children {
                serialize_node_refs(child, seq)?;
            }
        }
        Node::Cpu(n) => {
            seq.serialize_element(&NodeRefRecord {
                kind: "cpu",
                template: Some(n.template),
                instance: Some(n.instance),
                templates: EMPTY,
                instances: EMPTY,
                cpu_instances: EMPTY,
                gpu_templates: EMPTY,
                gpu_instances: EMPTY,
            })?;
            for child in &n.children {
                serialize_node_refs(child, seq)?;
            }
            for child in &n.slots {
                serialize_node_refs(child, seq)?;
            }
        }
        Node::SameCpu(n) => {
            seq.serialize_element(&NodeRefRecord {
                kind: "same_cpu",
                template: Some(n.template),
                instance: None,
                templates: EMPTY,
                instances: &n.instances,
                cpu_instances: EMPTY,
                gpu_templates: EMPTY,
                gpu_instances: EMPTY,
            })?;
            for child in &n.children {
                serialize_node_refs(child, seq)?;
            }
            for slot in &n.slots {
                for child in slot {
                    serialize_node_refs(child, seq)?;
                }
            }
        }
        Node::Gpu(n) => {
            seq.serialize_element(&NodeRefRecord {
                kind: "gpu",
                template: None,
                instance: None,
                templates: &n.templates,
                instances: &n.instances,
                cpu_instances: EMPTY,
                gpu_templates: EMPTY,
                gpu_instances: EMPTY,
            })?;
        }
        Node::KernelLaunch(n) => {
            seq.serialize_element(&NodeRefRecord {
                kind: "kernel_launch",
                template: Some(n.cpu_template),
                instance: Some(n.cpu_instance),
                templates: EMPTY,
                instances: EMPTY,
                cpu_instances: EMPTY,
                gpu_templates: std::slice::from_ref(&n.gpu_template),
                gpu_instances: std::slice::from_ref(&n.gpu_instance),
            })?;
        }
        Node::KernelsLaunch(n) => {
            seq.serialize_element(&NodeRefRecord {
                kind: "kernels_launch",
                template: Some(n.cpu_template),
                instance: None,
                templates: EMPTY,
                instances: EMPTY,
                cpu_instances: &n.cpu_instances,
                gpu_templates: &n.gpu_templates,
                gpu_instances: &n.gpu_instances,
            })?;
        }
    }
    Ok(())
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
