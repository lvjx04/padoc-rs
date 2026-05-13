use padoc::event::{ArgColumn, DigitColumn, NameNums, NumColumn, PhaseColumn, StringColumn, Template};
use padoc::node::Node;
use padoc::trace::CompressedTrace;
use std::env;
use std::mem::size_of;
use std::path::PathBuf;

#[derive(Default)]
struct Stats {
    templates: usize,
    cpu_templates: usize,
    gpu_templates: usize,
    cpu_instances: usize,
    gpu_instances: usize,
    cpu_ts_bytes: usize,
    cpu_dur_bytes: usize,
    cpu_id_bytes: usize,
    gpu_ts_bytes: usize,
    gpu_dur_bytes: usize,
    gpu_pid_bytes: usize,
    gpu_ph_bytes: usize,
    gpu_stream_bytes: usize,
    name_num_vec_bytes: usize,
    name_num_payload_bytes: usize,
    arg_value_count: usize,
    arg_vec_bytes: usize,
    arg_payload_bytes: usize,
    string_payload_bytes: usize,
    nodes: usize,
    root_nodes: usize,
    cpu_nodes: usize,
    same_cpu_nodes: usize,
    gpu_nodes: usize,
    kernel_launch_nodes: usize,
    kernels_launch_nodes: usize,
    node_vec_bytes: usize,
    node_u32_vec_bytes: usize,
    constant_num_cols: usize,
    i32_num_cols: usize,
    i64_num_cols: usize,
}

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).map(PathBuf::from).expect("usage: inspect_artifact <artifact>");
    let on_disk = std::fs::metadata(&path)?.len();
    eprintln!("loading {}", path.display());
    let trace = CompressedTrace::read_from_path(&path)?;
    let mut s = Stats { templates: trace.templates.len(), ..Stats::default() };

    for t in &trace.templates {
        match t {
            Template::Cpu(c) => {
                s.cpu_templates += 1;
                s.cpu_instances += c.instance_count();
                s.string_payload_bytes += c.name_pattern.len();
                s.string_payload_bytes += c.cat.as_ref().map(|x| x.len()).unwrap_or(0);
                s.string_payload_bytes += c.bp.as_ref().map(|x| x.len()).unwrap_or(0);
                s.string_payload_bytes += c.s.as_ref().map(|x| x.len()).unwrap_or(0);
                s.string_payload_bytes += c.arg_keys.iter().map(|x| x.len()).sum::<usize>();
                s.cpu_ts_bytes += num_column_bytes(&c.ts, &mut s);
                s.cpu_dur_bytes += num_column_bytes(&c.dur, &mut s);
                s.cpu_id_bytes += num_column_bytes(&c.id, &mut s);
                count_name_nums(&c.name_nums, &mut s);
                count_args(&c.args_columns, &mut s);
            }
            Template::Gpu(g) => {
                s.gpu_templates += 1;
                s.gpu_instances += g.instance_count();
                s.string_payload_bytes += g.name_pattern.len();
                s.string_payload_bytes += g.cat.as_ref().map(|x| x.len()).unwrap_or(0);
                s.string_payload_bytes += g.arg_keys.iter().map(|x| x.len()).sum::<usize>();
                s.gpu_ts_bytes += num_column_bytes(&g.ts, &mut s);
                s.gpu_dur_bytes += num_column_bytes(&g.dur, &mut s);
                s.gpu_pid_bytes += num_column_bytes(&g.pid, &mut s);
                s.gpu_ph_bytes += phase_column_bytes(&g.ph);
                s.gpu_stream_bytes += string_column_bytes(&g.stream_tid);
                count_name_nums(&g.name_nums, &mut s);
                count_args(&g.args_columns, &mut s);
            }
        }
    }

    for processes in trace.ranks.values() {
        for threads in processes.values() {
            for phases in threads.values() {
                for root in phases.values() {
                    count_node(root, &mut s);
                }
            }
        }
    }

    println!("on_disk_bytes\t{}", on_disk);
    println!("templates\t{}", s.templates);
    println!("cpu_templates\t{}", s.cpu_templates);
    println!("gpu_templates\t{}", s.gpu_templates);
    println!("cpu_instances\t{}", s.cpu_instances);
    println!("gpu_instances\t{}", s.gpu_instances);
    println!("nodes\t{}", s.nodes);
    println!(
        "node_breakdown\troot={} cpu={} same_cpu={} gpu={} kernel_launch={} kernels_launch={}",
        s.root_nodes, s.cpu_nodes, s.same_cpu_nodes, s.gpu_nodes, s.kernel_launch_nodes, s.kernels_launch_nodes
    );
    println!(
        "num_column_breakdown\tconstant={} i32={} i64={}",
        s.constant_num_cols, s.i32_num_cols, s.i64_num_cols
    );
    print_bytes("cpu_ts", s.cpu_ts_bytes);
    print_bytes("cpu_dur", s.cpu_dur_bytes);
    print_bytes("cpu_id", s.cpu_id_bytes);
    print_bytes("gpu_ts", s.gpu_ts_bytes);
    print_bytes("gpu_dur", s.gpu_dur_bytes);
    print_bytes("gpu_pid", s.gpu_pid_bytes);
    print_bytes("gpu_ph", s.gpu_ph_bytes);
    print_bytes("gpu_stream", s.gpu_stream_bytes);
    print_bytes("name_num_vecs", s.name_num_vec_bytes);
    print_bytes("name_num_payload", s.name_num_payload_bytes);
    print_bytes("node_vec_storage", s.node_vec_bytes);
    print_bytes("node_u32_vec_storage", s.node_u32_vec_bytes);
    println!("arg_value_count\t{}", s.arg_value_count);
    print_bytes("arg_vec_storage", s.arg_vec_bytes);
    print_bytes("arg_payload", s.arg_payload_bytes);
    print_bytes("string_payload_other", s.string_payload_bytes);
    let accounted = s.cpu_ts_bytes
        + s.cpu_dur_bytes
        + s.cpu_id_bytes
        + s.gpu_ts_bytes
        + s.gpu_dur_bytes
        + s.gpu_pid_bytes
        + s.gpu_ph_bytes
        + s.gpu_stream_bytes
        + s.name_num_vec_bytes
        + s.name_num_payload_bytes
        + s.node_vec_bytes
        + s.node_u32_vec_bytes
        + s.arg_vec_bytes
        + s.arg_payload_bytes
        + s.string_payload_bytes;
    print_bytes("accounted_selected_total", accounted);
    Ok(())
}

fn num_column_bytes(col: &NumColumn, s: &mut Stats) -> usize {
    match col {
        NumColumn::Empty => 0,
        NumColumn::Constant { .. } => {
            s.constant_num_cols += 1;
            size_of::<i64>() + size_of::<u32>()
        }
        NumColumn::I32(v) => {
            s.i32_num_cols += 1;
            v.capacity() * size_of::<i32>()
        }
        NumColumn::I64(v) => {
            s.i64_num_cols += 1;
            v.capacity() * size_of::<i64>()
        }
    }
}

fn phase_column_bytes(col: &PhaseColumn) -> usize {
    match col {
        PhaseColumn::Empty => 0,
        PhaseColumn::Constant { .. } => size_of::<u8>() + size_of::<u32>(),
        PhaseColumn::PerInstance(v) => v.capacity() * size_of::<u8>(),
    }
}

fn string_column_bytes(col: &StringColumn) -> usize {
    match col {
        StringColumn::Empty => 0,
        StringColumn::Constant { value, .. } => size_of::<String>() + value.capacity(),
        StringColumn::PerInstance(v) => {
            v.capacity() * size_of::<String>()
                + v.iter().map(|x| x.capacity()).sum::<usize>()
        }
    }
}

fn count_args(cols: &[ArgColumn], s: &mut Stats) {
    for col in cols {
        match col {
            ArgColumn::Constant(v) => {
                s.arg_value_count += 1;
                s.arg_payload_bytes += v.to_string().len();
            }
            ArgColumn::I32(v) => {
                s.arg_value_count += v.len();
                s.arg_vec_bytes += v.capacity() * size_of::<i32>();
            }
            ArgColumn::I64(v) => {
                s.arg_value_count += v.len();
                s.arg_vec_bytes += v.capacity() * size_of::<i64>();
            }
            ArgColumn::F64(v) => {
                s.arg_value_count += v.len();
                s.arg_vec_bytes += v.capacity() * size_of::<f64>();
            }
            ArgColumn::Bool(v) => {
                s.arg_value_count += v.len();
                s.arg_vec_bytes += v.capacity() * size_of::<u8>();
            }
            ArgColumn::Str(v) => {
                s.arg_value_count += v.len();
                s.arg_vec_bytes += v.capacity() * size_of::<String>();
                s.arg_payload_bytes += v.iter().map(|x| x.capacity()).sum::<usize>();
            }
            ArgColumn::StrDict { dict, ids } => {
                s.arg_value_count += ids.len();
                s.arg_vec_bytes += ids.capacity() * size_of::<u32>();
                s.arg_vec_bytes += dict.capacity() * size_of::<String>();
                s.arg_payload_bytes += dict.iter().map(|x| x.capacity()).sum::<usize>();
            }
            ArgColumn::PerInstance(values) => {
                s.arg_value_count += values.len();
                s.arg_vec_bytes += values.capacity() * size_of::<serde_json::Value>();
                s.arg_payload_bytes += values.iter().map(|v| v.to_string().len()).sum::<usize>();
            }
        }
    }
}

fn count_name_nums(nums: &NameNums, s: &mut Stats) {
    match nums {
        NameNums::Empty => {}
        NameNums::Rows(rows) => {
            s.name_num_vec_bytes += rows.capacity() * size_of::<Vec<String>>();
            for row in rows {
                s.name_num_vec_bytes += row.capacity() * size_of::<String>();
                s.name_num_payload_bytes += row.iter().map(|x| x.capacity()).sum::<usize>();
            }
        }
        NameNums::Columnar(cols) => {
            s.name_num_vec_bytes += cols.capacity() * size_of::<DigitColumn>();
            for col in cols {
                match col {
                    DigitColumn::Constant(v) => {
                        s.name_num_payload_bytes += v.capacity();
                    }
                    DigitColumn::I32 { values, .. } => {
                        s.name_num_vec_bytes += values.capacity() * size_of::<i32>();
                    }
                    DigitColumn::I64 { values, .. } => {
                        s.name_num_vec_bytes += values.capacity() * size_of::<i64>();
                    }
                    DigitColumn::Strings(v) => {
                        s.name_num_vec_bytes += v.capacity() * size_of::<String>();
                        s.name_num_payload_bytes += v.iter().map(|x| x.capacity()).sum::<usize>();
                    }
                }
            }
        }
    }
}

fn count_node(node: &Node, s: &mut Stats) {
    s.nodes += 1;
    match node {
        Node::Root { children } => {
            s.root_nodes += 1;
            s.node_vec_bytes += children.capacity() * size_of::<Node>();
            for child in children {
                count_node(child, s);
            }
        }
        Node::Cpu(n) => {
            s.cpu_nodes += 1;
            s.node_vec_bytes += n.children.capacity() * size_of::<Node>();
            s.node_vec_bytes += n.slots.capacity() * size_of::<Node>();
            for child in &n.children {
                count_node(child, s);
            }
            for child in &n.slots {
                count_node(child, s);
            }
        }
        Node::SameCpu(n) => {
            s.same_cpu_nodes += 1;
            s.node_u32_vec_bytes += n.instances.capacity() * size_of::<u32>();
            s.node_vec_bytes += n.children.capacity() * size_of::<Node>();
            s.node_vec_bytes += n.slots.capacity() * size_of::<Vec<Node>>();
            for child in &n.children {
                count_node(child, s);
            }
            for slot in &n.slots {
                s.node_vec_bytes += slot.capacity() * size_of::<Node>();
                for child in slot {
                    count_node(child, s);
                }
            }
        }
        Node::Gpu(n) => {
            s.gpu_nodes += 1;
            s.node_u32_vec_bytes += n.templates.capacity() * size_of::<u32>();
            s.node_u32_vec_bytes += n.instances.capacity() * size_of::<u32>();
        }
        Node::KernelLaunch(_) => {
            s.kernel_launch_nodes += 1;
        }
        Node::KernelsLaunch(n) => {
            s.kernels_launch_nodes += 1;
            s.node_u32_vec_bytes += n.cpu_instances.capacity() * size_of::<u32>();
            s.node_u32_vec_bytes += n.gpu_templates.capacity() * size_of::<u32>();
            s.node_u32_vec_bytes += n.gpu_instances.capacity() * size_of::<u32>();
        }
    }
}

fn print_bytes(label: &str, bytes: usize) {
    println!("{label}_bytes\t{bytes}\t{:.3} GiB", bytes as f64 / 1024.0 / 1024.0 / 1024.0);
}
