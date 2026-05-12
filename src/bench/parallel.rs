//! Multi-thread compression speedup benchmark — uses rayon to parallelise
//! per-rank compression and reports the wall-clock time per worker count.

use serde::{Deserialize, Serialize};

use crate::baselines::BaselineCompressor;
use crate::trace::Trace;
use crate::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParallelRecord {
    pub compressor: String,
    pub workers: usize,
    pub trace_count: usize,
    pub total_event_count: usize,
    pub wall_secs: f64,
    pub speedup_vs_single: f64,
}

pub fn run_parallel_compression(
    compressor: &dyn BaselineCompressor,
    traces: &[Trace],
    workers: &[usize],
) -> Result<Vec<ParallelRecord>> {
    use rayon::prelude::*;
    let total_events: usize = traces.iter().map(|t| t.event_count()).sum();
    let mut out = Vec::new();
    let mut single_secs: Option<f64> = None;
    for &w in workers {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(w.max(1)).build().map_err(|e| crate::Error::Other(e.to_string()))?;
        let start = std::time::Instant::now();
        pool.install(|| -> Result<()> {
            traces.par_iter().try_for_each(|trace| -> Result<()> {
                compressor.compress(trace)?;
                Ok(())
            })
        })?;
        let wall = start.elapsed().as_secs_f64();
        if w == 1 { single_secs = Some(wall); }
        let speedup = single_secs.map(|s| if wall > 0.0 { s / wall } else { 0.0 }).unwrap_or(1.0);
        out.push(ParallelRecord {
            compressor: compressor.name().to_string(),
            workers: w,
            trace_count: traces.len(),
            total_event_count: total_events,
            wall_secs: wall,
            speedup_vs_single: speedup,
        });
    }
    Ok(out)
}
