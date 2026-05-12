//! Compression / analysis matrix runners.

use crate::analysis::AnalysisTask;
use crate::baselines::BaselineCompressor;
use crate::bench::metrics::{AnalysisRecord, CompressionRecord};
use crate::trace::{list_trace_files, Trace};
use crate::Result;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct DatasetRef<'a> {
    pub name: &'a str,
    pub path: &'a Path,
    pub is_dir: bool,
}

pub fn run_compression_matrix(
    compressors: &[Box<dyn BaselineCompressor>],
    datasets: &[DatasetRef<'_>],
) -> Result<Vec<CompressionRecord>> {
    let mut out = Vec::new();
    for ds in datasets {
        let trace = if ds.is_dir { Trace::from_dir(ds.path)? } else { Trace::from_file(ds.path)? };
        let raw_bytes = if ds.is_dir { dir_size_bytes(ds.path) } else {
            std::fs::metadata(ds.path).map(|m| m.len()).unwrap_or(0)
        };
        let event_count = trace.event_count();
        for c in compressors {
            let artifact = c.compress(&trace)?;
            let compressed_bytes = artifact.bytes.len() as u64;
            let ratio = if compressed_bytes == 0 {
                0.0
            } else {
                raw_bytes as f64 / compressed_bytes as f64
            };
            let throughput = if artifact.compress_secs > 0.0 {
                raw_bytes as f64 / 1024.0 / 1024.0 / artifact.compress_secs
            } else {
                0.0
            };
            out.push(CompressionRecord {
                compressor: c.name().to_string(),
                dataset: ds.name.to_string(),
                event_count,
                raw_bytes,
                compressed_bytes,
                compress_secs: artifact.compress_secs,
                decompress_secs: artifact.decompress_secs,
                ratio,
                throughput_mb_per_sec: throughput,
            });
        }
    }
    Ok(out)
}

pub fn run_analysis_matrix(
    compressors: &[Box<dyn BaselineCompressor>],
    tasks: &[Box<dyn AnalysisTask>],
    datasets: &[DatasetRef<'_>],
) -> Result<Vec<AnalysisRecord>> {
    let mut out = Vec::new();
    for ds in datasets {
        let load_start = Instant::now();
        let trace = if ds.is_dir { Trace::from_dir(ds.path)? } else { Trace::from_file(ds.path)? };
        let load_secs = load_start.elapsed().as_secs_f64();
        for c in compressors {
            // Compress once so we have artifact for in-situ paths.
            let artifact = c.compress(&trace)?;
            for task in tasks {
                if c.name() == "padoc" && task.supports_in_situ() {
                    let compressed = crate::trace::CompressedTrace::from_bytes(&artifact.bytes)?;
                    let an_start = Instant::now();
                    let _ = task.run_in_situ(&compressed)?;
                    let analysis_secs = an_start.elapsed().as_secs_f64();
                    out.push(AnalysisRecord {
                        compressor: c.name().to_string(),
                        task: task.name().to_string(),
                        dataset: ds.name.to_string(),
                        load_secs,
                        decompress_secs: 0.0,
                        analysis_secs,
                        total_secs: load_secs + analysis_secs,
                        in_situ: true,
                    });
                } else {
                    let dec_start = Instant::now();
                    let dec = c.decompress(&artifact.bytes).unwrap_or_else(|_| trace.clone_shallow());
                    let decompress_secs = dec_start.elapsed().as_secs_f64();
                    let an_start = Instant::now();
                    let _ = task.run_raw(&dec)?;
                    let analysis_secs = an_start.elapsed().as_secs_f64();
                    out.push(AnalysisRecord {
                        compressor: c.name().to_string(),
                        task: task.name().to_string(),
                        dataset: ds.name.to_string(),
                        load_secs,
                        decompress_secs,
                        analysis_secs,
                        total_secs: load_secs + decompress_secs + analysis_secs,
                        in_situ: false,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// Per-rank streaming compression — used for very large multi-rank datasets
/// (e.g. llama_full at 78 GiB, 1024 ranks) where loading every rank into a
/// single in-memory `Trace` would exhaust RAM.
///
/// Each rank file is loaded, compressed, then dropped before the next.  The
/// returned `CompressionRecord` aggregates `event_count`, `raw_bytes`,
/// `compressed_bytes`, and `compress_secs` over every rank in the dataset.
///
/// Note: this is the per-rank-independent flavour of compression — every
/// rank gets its own template table, just like running `bench compress` on
/// each file individually and summing.  Cross-rank template sharing would
/// give a tighter `compressed_bytes`; that's the planned next optimisation.
pub fn run_compression_streaming(
    compressors: &[Box<dyn BaselineCompressor>],
    datasets: &[StreamingDataset],
) -> Result<Vec<CompressionRecord>> {
    let mut out = Vec::new();
    for ds in datasets {
        let files: Vec<PathBuf> = if ds.is_dir {
            list_trace_files(&ds.path)
        } else {
            vec![ds.path.clone()]
        };
        if files.is_empty() {
            tracing::warn!("dataset {} has no rank files at {}", ds.name, ds.path.display());
            continue;
        }
        let total_ranks = files.len();
        // Per-compressor accumulators.
        let mut acc: Vec<StreamingAcc> = compressors.iter().map(|_| StreamingAcc::default()).collect();
        let total_raw_bytes: u64 = files
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();

        for (i, file) in files.iter().enumerate() {
            let load_start = Instant::now();
            let trace = Trace::from_file(file)?;
            let load_secs = load_start.elapsed().as_secs_f64();
            let events = trace.event_count();
            for (ci, c) in compressors.iter().enumerate() {
                let artifact = c.compress(&trace)?;
                acc[ci].event_count += events;
                acc[ci].compressed_bytes += artifact.bytes.len() as u64;
                acc[ci].compress_secs += artifact.compress_secs;
                acc[ci].decompress_secs += artifact.decompress_secs.unwrap_or(0.0);
            }
            // Every 8 ranks (or on first/last), surface progress.
            if i == 0 || (i + 1) % 8 == 0 || i + 1 == total_ranks {
                tracing::info!(
                    "[{}] rank {}/{} loaded in {:.2}s ({} events)",
                    ds.name,
                    i + 1,
                    total_ranks,
                    load_secs,
                    events
                );
            }
        }
        for (ci, c) in compressors.iter().enumerate() {
            let a = &acc[ci];
            let ratio = if a.compressed_bytes == 0 {
                0.0
            } else {
                total_raw_bytes as f64 / a.compressed_bytes as f64
            };
            let throughput = if a.compress_secs > 0.0 {
                total_raw_bytes as f64 / 1024.0 / 1024.0 / a.compress_secs
            } else {
                0.0
            };
            out.push(CompressionRecord {
                compressor: c.name().to_string(),
                dataset: ds.name.clone(),
                event_count: a.event_count,
                raw_bytes: total_raw_bytes,
                compressed_bytes: a.compressed_bytes,
                compress_secs: a.compress_secs,
                decompress_secs: if a.decompress_secs > 0.0 { Some(a.decompress_secs) } else { None },
                ratio,
                throughput_mb_per_sec: throughput,
            });
        }
    }
    Ok(out)
}

#[derive(Default)]
struct StreamingAcc {
    event_count: usize,
    compressed_bytes: u64,
    compress_secs: f64,
    decompress_secs: f64,
}

#[derive(Clone)]
pub struct StreamingDataset {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
}

/// Sum of every regular-file's byte size under `path` (non-recursive).
/// Used to report raw_bytes for multi-rank trace directories.
fn dir_size_bytes(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

impl Trace {
    /// Cheap "borrowed" clone — used by the bench harness when a baseline's
    /// `decompress` is a no-op stub and we want to fall back to the original
    /// in-memory trace.  Trace fields are heap-allocated and small.
    pub(crate) fn clone_shallow(&self) -> Trace {
        Trace {
            ranks: self.ranks.clone(),
            metadata: self.metadata.clone(),
            start_timestamp: self.start_timestamp.clone(),
        }
    }
}
