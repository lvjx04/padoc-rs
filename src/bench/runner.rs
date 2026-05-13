//! Compression / analysis matrix runners.

use crate::analysis::AnalysisTask;
use crate::baselines::BaselineCompressor;
use crate::bench::metrics::{AnalysisRecord, CompressionRecord};
use crate::compressor::{merge_shards, CompressorConfig, RankShard, TemplateCompressor};
use crate::trace::{list_trace_files, Trace};
use crate::Result;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

pub struct DatasetRef<'a> {
    pub name: &'a str,
    pub path: &'a Path,
    pub is_dir: bool,
}

pub fn run_compression_matrix(
    compressors: &[Box<dyn BaselineCompressor>],
    datasets: &[DatasetRef<'_>],
    out_dir: Option<&Path>,
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
            if let Some(d) = out_dir {
                let path = artifact_path(d, &ds.name, c.name());
                std::fs::write(&path, &artifact.bytes)?;
                tracing::info!(
                    "[{}] saved {} artifact -> {} ({} bytes)",
                    ds.name,
                    c.name(),
                    path.display(),
                    compressed_bytes
                );
            }
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

/// `<dir>/<dataset>.<compressor>.bin` for non-padoc compressors,
/// `<dir>/<dataset>.padoc.zst` for padoc.
fn artifact_path(dir: &Path, dataset: &str, compressor: &str) -> PathBuf {
    let ext = if compressor == "padoc" { "zst" } else { "bin" };
    dir.join(format!("{dataset}.{compressor}.{ext}"))
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
    out_dir: Option<&Path>,
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
        let ds_out = out_dir.map(|d| d.join(&ds.name));
        if let Some(d) = &ds_out {
            std::fs::create_dir_all(d)?;
        }

        for (i, file) in files.iter().enumerate() {
            let load_start = Instant::now();
            let trace = Trace::from_file(file)?;
            let load_secs = load_start.elapsed().as_secs_f64();
            let events = trace.event_count();
            let stem = file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("rank");
            for (ci, c) in compressors.iter().enumerate() {
                let artifact = c.compress(&trace)?;
                acc[ci].event_count += events;
                acc[ci].compressed_bytes += artifact.bytes.len() as u64;
                acc[ci].compress_secs += artifact.compress_secs;
                acc[ci].decompress_secs += artifact.decompress_secs.unwrap_or(0.0);
                if let Some(d) = &ds_out {
                    let path = artifact_path(d, stem, c.name());
                    std::fs::write(&path, &artifact.bytes)?;
                }
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

/// Padoc-specific cross-rank parallel compression.
///
/// Spawns `workers` rayon threads, each loading one rank file at a time
/// and producing a [`RankShard`] (private template table + call tree).
/// Once every shard is built, [`merge_shards`] folds them into a single
/// global template table — equivalent in size to running padoc on the
/// merged in-memory `Trace` but with bounded RAM and parallel
/// throughput.
///
/// `decompress_secs` is left unset because we don't perform a round-trip
/// here; use `padoc roundtrip` for losslessness checks.
pub fn run_padoc_parallel(
    config: &CompressorConfig,
    datasets: &[StreamingDataset],
    workers: usize,
    zstd_level: i32,
    out_dir: Option<&Path>,
) -> Result<Vec<CompressionRecord>> {
    run_padoc_parallel_with_name(config, "padoc", datasets, workers, zstd_level, out_dir)
}

fn run_padoc_parallel_with_name(
    config: &CompressorConfig,
    compressor_name: &str,
    datasets: &[StreamingDataset],
    workers: usize,
    zstd_level: i32,
    out_dir: Option<&Path>,
) -> Result<Vec<CompressionRecord>> {
    use rayon::prelude::*;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()
        .map_err(|e| crate::Error::Other(format!("rayon pool: {e}")))?;

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
        let total_raw_bytes: u64 = files
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();
        let total_ranks = files.len();
        tracing::info!(
            "[{}] padoc-parallel: {} ranks, {} workers",
            ds.name,
            total_ranks,
            workers
        );

        let progress = Mutex::new(0usize);
        let label = ds.name.clone();

        let start = Instant::now();
        let shards: Vec<RankShard> = pool.install(|| -> Result<Vec<RankShard>> {
            files
                .par_iter()
                .map(|file| -> Result<RankShard> {
                    let trace = Trace::from_file(file)?;
                    let rank_id = trace
                        .rank_ids()
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| {
                            file.file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("unknown")
                                .to_string()
                        });
                    let shard = TemplateCompressor::compress_rank(config, &rank_id, &trace);
                    let mut p = progress.lock().unwrap();
                    *p += 1;
                    if *p == 1 || *p % 8 == 0 || *p == total_ranks {
                        tracing::info!("[{}] {}/{} ranks compressed", label, *p, total_ranks);
                    }
                    Ok(shard)
                })
                .collect()
        })?;
        let parallel_secs = start.elapsed().as_secs_f64();

        let merge_start = Instant::now();
        let event_count: usize = shards
            .iter()
            .map(|s| count_events_in_shard(s))
            .sum();
        let compressed = merge_shards(config, shards)?;
        let merge_secs = merge_start.elapsed().as_secs_f64();

        let serialize_start = Instant::now();
        // When out_dir is set, stream straight into the file (skips a giant
        // intermediate Vec<u8>); otherwise build the in-memory blob so the
        // record's compressed_bytes is meaningful.
        let compressed_bytes = if let Some(d) = out_dir {
            let path = d.join(format!("{}.{}.zst", ds.name, compressor_name));
            compressed.write_to_path(&path, zstd_level)?;
            let n = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            tracing::info!(
                "[{}] saved padoc artifact -> {} ({} bytes)",
                ds.name,
                path.display(),
                n
            );
            n
        } else {
            let bytes = compressed.to_bytes(zstd_level)?;
            bytes.len() as u64
        };
        let serialize_secs = serialize_start.elapsed().as_secs_f64();

        let total_secs = parallel_secs + merge_secs + serialize_secs;
        let ratio = if compressed_bytes == 0 {
            0.0
        } else {
            total_raw_bytes as f64 / compressed_bytes as f64
        };
        let throughput = if total_secs > 0.0 {
            total_raw_bytes as f64 / 1024.0 / 1024.0 / total_secs
        } else {
            0.0
        };
        tracing::info!(
            "[{}] padoc-parallel done: parallel={:.2}s merge={:.2}s serialize={:.2}s total={:.2}s ratio={:.2}x",
            ds.name,
            parallel_secs,
            merge_secs,
            serialize_secs,
            total_secs,
            ratio
        );

        out.push(CompressionRecord {
            compressor: compressor_name.to_string(),
            dataset: ds.name.clone(),
            event_count,
            raw_bytes: total_raw_bytes,
            compressed_bytes,
            compress_secs: total_secs,
            decompress_secs: None,
            ratio,
            throughput_mb_per_sec: throughput,
        });
    }

    Ok(out)
}

/// Sequential PADOC compression for one or more explicit configs.  This is
/// primarily for ablation rows on datasets that fit in memory.
pub fn run_padoc_config_matrix(
    configs: &[CompressorConfig],
    datasets: &[StreamingDataset],
    zstd_level: i32,
    out_dir: Option<&Path>,
) -> Result<Vec<CompressionRecord>> {
    let mut out = Vec::new();
    for ds in datasets {
        let trace = if ds.is_dir {
            Trace::from_dir(&ds.path)?
        } else {
            Trace::from_file(&ds.path)?
        };
        let raw_bytes = if ds.is_dir {
            dir_size_bytes(&ds.path)
        } else {
            std::fs::metadata(&ds.path).map(|m| m.len()).unwrap_or(0)
        };
        let event_count = trace.event_count();

        for config in configs {
            let start = Instant::now();
            let mut compressor = TemplateCompressor::with_config(config.clone());
            let compressed = compressor.compress(&trace)?;
            let compressed_bytes = if let Some(d) = out_dir {
                let path = padoc_config_artifact_path(d, &ds.name, config);
                compressed.write_to_path(&path, zstd_level)?;
                std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
            } else {
                compressed.to_bytes(zstd_level)?.len() as u64
            };
            let secs = start.elapsed().as_secs_f64();
            let ratio = if compressed_bytes == 0 {
                0.0
            } else {
                raw_bytes as f64 / compressed_bytes as f64
            };
            let throughput = if secs > 0.0 {
                raw_bytes as f64 / 1024.0 / 1024.0 / secs
            } else {
                0.0
            };
            out.push(CompressionRecord {
                compressor: padoc_config_name(config),
                dataset: ds.name.clone(),
                event_count,
                raw_bytes,
                compressed_bytes,
                compress_secs: secs,
                decompress_secs: None,
                ratio,
                throughput_mb_per_sec: throughput,
            });
        }
    }
    Ok(out)
}

/// Parallel PADOC compression for one or more explicit configs.  Used for
/// large ablation rows where loading all ranks into one `Trace` is not viable.
pub fn run_padoc_parallel_config_matrix(
    configs: &[CompressorConfig],
    datasets: &[StreamingDataset],
    workers: usize,
    zstd_level: i32,
    out_dir: Option<&Path>,
) -> Result<Vec<CompressionRecord>> {
    let mut out = Vec::new();
    for config in configs {
        let records = run_padoc_parallel_with_name(
            config,
            &padoc_config_name(config),
            datasets,
            workers,
            zstd_level,
            out_dir,
        )?;
        out.extend(records);
    }
    Ok(out)
}

fn padoc_config_name(config: &CompressorConfig) -> String {
    if config.label == "default" {
        "padoc".to_string()
    } else {
        format!("padoc_{}", config.label)
    }
}

fn padoc_config_artifact_path(dir: &Path, dataset: &str, config: &CompressorConfig) -> PathBuf {
    let name = padoc_config_name(config);
    dir.join(format!("{dataset}.{name}.zst"))
}

fn count_events_in_shard(shard: &RankShard) -> usize {
    shard
        .templates
        .iter()
        .map(|t| t.instance_count())
        .sum()
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
