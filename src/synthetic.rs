//! Deterministic synthetic AI trace generator — used by scalability sweeps
//! when real traces from the cluster are not available locally.
//!
//! Each spec produces a transformer-style training trace: per-rank,
//! per-iteration, per-layer events for forward + backward + collective ops.

use indexmap::IndexMap;

use crate::event::{Event, Phase};
use crate::trace::{StreamMap, Trace};
use ahash::AHashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyntheticTraceSpec {
    pub gpu_count: usize,
    pub layer_count: usize,
    pub iteration_count: usize,
    /// Number of operators inside each layer's forward pass.
    pub ops_per_layer: usize,
    /// Microseconds per operator (pre-jitter).
    pub op_dur_us: i64,
    /// Random seed; same seed -> identical trace.
    pub seed: u64,
}

impl Default for SyntheticTraceSpec {
    fn default() -> Self {
        Self {
            gpu_count: 4,
            layer_count: 4,
            iteration_count: 2,
            ops_per_layer: 6,
            op_dur_us: 50,
            seed: 0xCAFEBABE,
        }
    }
}

pub fn generate_trace(spec: &SyntheticTraceSpec) -> Trace {
    let mut trace = Trace::default();
    let mut rng = SplitMix64::new(spec.seed);

    for rank in 0..spec.gpu_count {
        let mut streams: StreamMap = IndexMap::new();
        let pid = 100 + rank as i64;

        let mut cpu_events: Vec<Event> = Vec::new();
        let mut gpu_events: Vec<Event> = Vec::new();
        let mut comm_events: Vec<Event> = Vec::new();

        let mut now: i64 = 0;
        let mut correlation: i64 = 1;

        for _iter in 0..spec.iteration_count {
            // Forward pass.
            for layer in 0..spec.layer_count {
                for op in 0..spec.ops_per_layer {
                    let name = format!("layers.{}.attn.op_{}", layer, op);
                    let dur = spec.op_dur_us + (rng.next_i64().rem_euclid(8));
                    let mut args = AHashMap::new();
                    args.insert("correlation".to_string(), serde_json::json!(correlation));
                    cpu_events.push(Event {
                        name: name.clone(),
                        ts: now,
                        dur: Some(dur),
                        cat: Some("cpu_op".into()),
                        ph: Phase::COMPLETE,
                        pid,
                        tid: "cpu_thread".into(),
                        args: Some(args.clone()),
                        id: None,
                        bp: None,
                        s: None,
                    });
                    gpu_events.push(Event {
                        name: format!("kernel_{}", op),
                        ts: now + 1,
                        dur: Some(dur - 2),
                        cat: Some("kernel".into()),
                        ph: Phase::COMPLETE,
                        pid,
                        tid: "stream 7".into(),
                        args: Some(args),
                        id: None,
                        bp: None,
                        s: None,
                    });
                    correlation += 1;
                    now += dur + 1;
                }
                // Inter-layer comm.
                let mut args = AHashMap::new();
                args.insert("Process Group Name".to_string(), serde_json::json!(format!("tp_group")));
                comm_events.push(Event {
                    name: format!("nccl:all_reduce"),
                    ts: now,
                    dur: Some(20),
                    cat: Some("kernel".into()),
                    ph: Phase::COMPLETE,
                    pid,
                    tid: "stream 12".into(),
                    args: Some(args),
                    id: None,
                    bp: None,
                    s: None,
                });
                now += 25;
            }
            // Backward pass mirrors forward.
            for layer in (0..spec.layer_count).rev() {
                for op in (0..spec.ops_per_layer).rev() {
                    let name = format!("layers.{}.attn.op_{}.backward", layer, op);
                    let dur = spec.op_dur_us + (rng.next_i64().rem_euclid(8));
                    cpu_events.push(Event {
                        name,
                        ts: now,
                        dur: Some(dur),
                        cat: Some("cpu_op".into()),
                        ph: Phase::COMPLETE,
                        pid,
                        tid: "cpu_thread".into(),
                        args: None,
                        id: None,
                        bp: None,
                        s: None,
                    });
                    now += dur + 1;
                }
            }
        }

        // CPU thread.
        let mut cpu_phases: IndexMap<String, IndexMap<Phase, Vec<Event>>> = IndexMap::new();
        let mut cpu_ph_map: IndexMap<Phase, Vec<Event>> = IndexMap::new();
        cpu_ph_map.insert(Phase::COMPLETE, cpu_events);
        cpu_phases.insert("cpu_thread".into(), cpu_ph_map);

        let mut gpu_ph_map: IndexMap<Phase, Vec<Event>> = IndexMap::new();
        gpu_ph_map.insert(Phase::COMPLETE, gpu_events);
        cpu_phases.insert("stream 7".into(), gpu_ph_map);

        let mut comm_ph_map: IndexMap<Phase, Vec<Event>> = IndexMap::new();
        comm_ph_map.insert(Phase::COMPLETE, comm_events);
        cpu_phases.insert("stream 12".into(), comm_ph_map);

        streams.insert(pid, cpu_phases);
        trace.ranks.insert(rank.to_string(), streams);
    }

    trace
}

/// Tiny splitmix64; deterministic and dependency-free.
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_i64(&mut self) -> i64 { self.next_u64() as i64 }
}
