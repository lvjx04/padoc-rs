//! GPU bubble rate by rank — streaming window algorithm.
//!
//! For each rank, iterates GPU kernel events sorted by timestamp, maintaining
//! a busy window.  The bubble rate is the fraction of time the GPU is idle:
//!   `bubble_rate = 1 - busy_time / total_span`
//!
//! where `total_span = last_event_end - first_event_start` and `busy_time` is
//! the union of all kernel durations (handling overlaps).
//!
//! This task exercises the "temporal access" dimension: events must be visited
//! in timestamp order to compute the idle gaps.

use ahash::AHashMap;
use serde_json::Value;

use crate::analysis::{elapsed_secs, profiled_result, AnalysisTask};
use crate::event::Template;
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

#[derive(Default)]
pub struct GpuBubbleRate;

/// Per-rank streaming busy-window state.
#[derive(Default)]
struct BubbleState {
    first_ts: i64,
    last_end: i64,
    busy_window_end: i64, // end of current busy window
    busy_total: i64,      // accumulated busy time (union of kernels)
    event_count: u64,
}

impl BubbleState {
    fn push_event(&mut self, ts: i64, dur: i64) {
        let end = ts + dur;
        if self.event_count == 0 {
            self.first_ts = ts;
            self.last_end = end;
            self.busy_window_end = end;
            self.busy_total = dur;
            self.event_count = 1;
            return;
        }
        self.event_count += 1;
        if end > self.last_end {
            self.last_end = end;
        }
        // Update busy window (union)
        if ts > self.busy_window_end {
            // Gap detected — start new busy window
            self.busy_total += dur;
            self.busy_window_end = end;
        } else if end > self.busy_window_end {
            // Extend current busy window
            self.busy_total += end - self.busy_window_end;
            self.busy_window_end = end;
        }
    }

    fn to_json(&self, rank: &str) -> Value {
        let total_span = self.last_end - self.first_ts;
        let bubble_time = total_span - self.busy_total;
        let bubble_rate = if total_span > 0 {
            bubble_time as f64 / total_span as f64
        } else {
            0.0
        };
        serde_json::json!({
            "rank": rank,
            "total_span_us": total_span,
            "busy_us": self.busy_total,
            "bubble_us": bubble_time,
            "bubble_rate": bubble_rate,
            "kernel_count": self.event_count,
        })
    }
}

impl AnalysisTask for GpuBubbleRate {
    fn name(&self) -> &str {
        "gpu_bubble_rate"
    }

    fn run_raw(&self, trace: &Trace) -> Result<Value> {
        // Collect GPU kernel events per rank, sort by ts, stream through
        let mut rank_events: AHashMap<String, Vec<(i64, i64)>> = AHashMap::new();
        for (rank, _pid, _tid, _ph, events) in trace.iter_streams() {
            for ev in events {
                if ev.cat.as_deref() != Some("kernel") {
                    continue;
                }
                let dur = ev.dur.unwrap_or(0);
                if dur <= 0 {
                    continue;
                }
                rank_events
                    .entry(rank.to_string())
                    .or_default()
                    .push((ev.ts, dur));
            }
        }

        let mut rows: Vec<(String, Value)> = Vec::new();
        for (rank, mut events) in rank_events {
            events.sort_unstable_by_key(|e| e.0);
            let mut state = BubbleState::default();
            for (ts, dur) in events {
                state.push_event(ts, dur);
            }
            rows.push((rank.clone(), state.to_json(&rank)));
        }
        rows.sort_by(|a, b| rank_cmp(&a.0, &b.0));
        Ok(Value::Array(rows.into_iter().map(|(_, v)| v).collect()))
    }

    fn supports_in_situ(&self) -> bool {
        true
    }

    fn run_in_situ(&self, compressed: &CompressedTrace) -> Result<Value> {
        let start = std::time::Instant::now();

        // Collect GPU kernel (ts, dur) per rank
        let mut rank_events: AHashMap<String, Vec<(i64, i64)>> = AHashMap::new();
        for (rank, processes) in &compressed.ranks {
            let entry = rank_events.entry(rank.clone()).or_default();
            for (_pid, threads) in processes {
                for (_tid, phases) in threads {
                    for (_ph, root) in phases {
                        walk_gpu_instances(root, compressed, entry);
                    }
                }
            }
        }
        let collect_secs = elapsed_secs(start);

        let start = std::time::Instant::now();
        let mut rows: Vec<(String, Value)> = Vec::new();
        for (rank, mut events) in rank_events {
            events.sort_unstable_by_key(|e| e.0);
            let mut state = BubbleState::default();
            for (ts, dur) in events {
                state.push_event(ts, dur);
            }
            rows.push((rank.clone(), state.to_json(&rank)));
        }
        rows.sort_by(|a, b| rank_cmp(&a.0, &b.0));
        Ok(profiled_result(
            Value::Array(rows.into_iter().map(|(_, v)| v).collect()),
            vec![
                ("gpu_event_collect", collect_secs),
                ("sort_and_bubble", elapsed_secs(start)),
            ],
        ))
    }
}

fn walk_gpu_instances(
    node: &crate::node::Node,
    compressed: &CompressedTrace,
    out: &mut Vec<(i64, i64)>,
) {
    use crate::node::Node;
    match node {
        Node::Root { children } => {
            for child in children {
                walk_gpu_instances(child, compressed, out);
            }
        }
        Node::Cpu(n) => {
            for child in &n.children {
                walk_gpu_instances(child, compressed, out);
            }
            for child in &n.slots {
                walk_gpu_instances(child, compressed, out);
            }
        }
        Node::SameCpu(n) => {
            for child in &n.children {
                walk_gpu_instances(child, compressed, out);
            }
            for slot in n.slots.entries() {
                for child in &slot.children {
                    walk_gpu_instances(child, compressed, out);
                }
            }
        }
        Node::Gpu(n) => {
            for (tmpl, inst) in n.templates.iter().zip(n.instances.iter()) {
                if let Some(Template::Gpu(g)) = compressed.templates.get(*tmpl as usize) {
                    if g.cat.as_deref() != Some("kernel") {
                        continue;
                    }
                    let ts = g.ts.get(*inst as usize).unwrap_or(0);
                    let dur = g.dur.get(*inst as usize).unwrap_or(0);
                    if dur > 0 {
                        out.push((ts, dur));
                    }
                }
            }
        }
        Node::KernelLaunch(n) => {
            if let Some(Template::Gpu(g)) = compressed.templates.get(n.gpu_template as usize) {
                if g.cat.as_deref() == Some("kernel") {
                    let ts = g.ts.get(n.gpu_instance as usize).unwrap_or(0);
                    let dur = g.dur.get(n.gpu_instance as usize).unwrap_or(0);
                    if dur > 0 {
                        out.push((ts, dur));
                    }
                }
            }
        }
        Node::KernelsLaunch(n) => {
            for (tmpl, inst) in n.gpu_templates.iter().zip(n.gpu_instances.iter()) {
                if let Some(Template::Gpu(g)) = compressed.templates.get(*tmpl as usize) {
                    if g.cat.as_deref() != Some("kernel") {
                        continue;
                    }
                    let ts = g.ts.get(*inst as usize).unwrap_or(0);
                    let dur = g.dur.get(*inst as usize).unwrap_or(0);
                    if dur > 0 {
                        out.push((ts, dur));
                    }
                }
            }
        }
    }
}

fn rank_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    match (a.parse::<i64>(), b.parse::<i64>()) {
        (Ok(x), Ok(y)) => x.cmp(&y),
        _ => a.cmp(b),
    }
}
