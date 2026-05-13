//! `padoc` — CLI driver.
//!
//! Subcommands:
//!
//! * `compress`     — read chrome-trace JSON / dir → write `.pdc`
//! * `decompress`   — read `.pdc` → write chrome-trace JSON
//! * `analyze`      — run an analysis task on a trace (raw or compressed)
//! * `bench compress` / `bench analyze` / `bench scalability` / `bench parallel`
//! * `list`         — show available compressors / tasks

use anyhow::Context;
use clap::{Parser, Subcommand};
use padoc::analysis;
use padoc::baselines::{self, BaselineCompressor};
use padoc::bench;
use padoc::compressor::{all_ablation_presets, CompressorConfig, TemplateCompressor};
use padoc::synthetic::SyntheticTraceSpec;
use padoc::trace::Trace;
use padoc::utils;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "padoc", version, about = "PADOC — AI trace compression in Rust", arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compress a chrome-trace JSON (file or directory).
    Compress {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 3)]
        zstd_level: i32,
    },
    /// Decompress a `.pdc` back to chrome-trace-style JSON (lossy until full pipeline lands).
    Decompress {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compress a trace, decompress it, and verify lossless round-trip.
    Roundtrip {
        input: PathBuf,
        /// One of: padoc, scalatrace, tracezip, gzip_json, gzip_msgpack
        #[arg(long, default_value = "padoc")]
        compressor: String,
        /// Treat `input` as a directory of per-rank JSONs.
        #[arg(long)]
        dir: bool,
        /// Use the parallel padoc path (`compress_rank` + `merge_shards`) with
        /// `N` rayon workers.  `1` keeps the existing single-threaded path.
        /// Only valid for `--compressor padoc` and a directory input.
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Run an analysis task on a trace.
    Analyze {
        trace: PathBuf,
        #[arg(long)]
        task: String,
        /// Run via PADOC compress + in-situ instead of raw chrome-trace.
        #[arg(long)]
        in_situ: bool,
    },
    /// Show compressors / tasks available.
    List,
    /// Bench harness.
    Bench {
        #[command(subcommand)]
        sub: BenchCmd,
    },
}

#[derive(Subcommand)]
enum BenchCmd {
    /// Compression matrix.
    ///
    /// Two ways to feed datasets:
    ///   --datasets <PATH>...  : direct paths (file or directory)
    ///   --manifest <PATH>     : JSON manifest with `{datasets:[{name, path,
    ///                           is_directory, gpus}]}`.
    /// Pass `--per-rank` to stream every rank file individually instead of
    /// loading the whole directory into RAM at once — required for 1024-rank
    /// llama-style datasets that don't fit otherwise.
    Compress {
        #[arg(long, value_delimiter = ',')]
        datasets: Vec<PathBuf>,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        compressors: Option<Vec<String>>,
        /// PADOC config labels to run instead of the normal compressor
        /// registry path.  Use `all` for every ablation preset.
        #[arg(long, value_delimiter = ',')]
        padoc_presets: Option<Vec<String>>,
        /// Process each rank file independently (lower RAM, no cross-rank
        /// template sharing).
        #[arg(long, default_value_t = false)]
        per_rank: bool,
        /// Worker thread count for cross-rank parallel padoc compression.
        /// `1` keeps the existing single-threaded path; `>1` engages
        /// `run_padoc_parallel` (rayon, per-rank streaming, global merge).
        /// Only honoured when the compressor set is exactly `padoc`.
        #[arg(long, default_value_t = 1)]
        workers: usize,
        /// zstd level used when serialising the merged compressed trace.
        #[arg(long, default_value_t = 3)]
        zstd_level: i32,
        /// If set, write each `<dataset>.<compressor>` artifact to this
        /// directory (created if missing).  Filenames:
        ///   `<name>.padoc.zst`           — padoc / merged padoc parallel
        ///   `<name>.<compressor>.bin`    — every other baseline (raw, gzip,
        ///                                  scalatrace, tracezip, …).
        /// Skipping this argument keeps the historic in-memory-only behaviour.
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },
    /// Analysis matrix.
    Analyze {
        #[arg(long, value_delimiter = ',')]
        datasets: Vec<PathBuf>,
    },
    /// One-cell analysis bench: load a single artifact, run a single
    /// task, report load_secs / analyze_secs / peak_rss_kb.  Designed
    /// to be invoked by a driver script (see scripts/analyze_bench.sh)
    /// once per (compressor, dataset, task) cell so each measurement
    /// has its own process and the peak-RSS reading isn't contaminated
    /// by a previous cell's still-allocated heap.
    AnalyzeOne {
        /// Compressor name (raw_json, gzip_json, scalatrace, tracezip, padoc).
        #[arg(long)]
        compressor: String,
        /// Path to the compressed artifact file.
        #[arg(long)]
        artifact: PathBuf,
        /// Analysis task name.  See `padoc list`.
        #[arg(long)]
        task: String,
        /// Number of repetitions (median is reported for timings;
        /// peak_rss is taken from the highest-RSS run).
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Batched analysis bench: load+decompress ONCE, then run multiple
    /// tasks against the in-memory representation, timing each task
    /// separately.  Peak RSS is reported on the final row only — it
    /// covers the whole process and reflects the union of load +
    /// every task's transient allocation.  For padoc this amortises
    /// the (often dominant) deserialise cost across many analyses,
    /// which is the realistic deployment pattern: you load the
    /// compressed trace once and answer many questions about it.
    AnalyzeBatch {
        /// Compressor name (raw_json, gzip_json, scalatrace, tracezip, padoc).
        #[arg(long)]
        compressor: String,
        /// Path to the compressed artifact file.
        #[arg(long)]
        artifact: PathBuf,
        /// Comma-separated analysis task names (in execution order).
        #[arg(long, value_delimiter = ',')]
        tasks: Vec<String>,
        /// Number of repetitions for each task's `analyze` step
        /// (load+decompress always run once).  Median analyze_secs
        /// is reported per task.
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Run analysis tasks on pre-saved compressed artifacts.
    /// For padoc artifacts the analysis runs in-situ on `CompressedTrace`;
    /// for every other compressor we decompress to a `Trace` first, then
    /// run `task.run_raw`.  Reports per-(compressor, dataset, task) timings:
    /// load (read+decompress for baselines / read for padoc), analyze,
    /// total.
    AnalyzeArtifacts {
        /// Directories holding artifacts named `<dataset>.<compressor>.{zst|bin}`.
        /// Multiple `--artifact-dir` may be passed and they're searched in order.
        #[arg(long, value_delimiter = ',')]
        artifact_dir: Vec<PathBuf>,
        /// Comma-separated compressor names (raw_json, gzip_json, scalatrace,
        /// tracezip, padoc, …).  Only artifacts for these compressors are loaded.
        #[arg(long, value_delimiter = ',')]
        compressors: Vec<String>,
        /// Comma-separated dataset stems (e.g. `qwen3,llama_full`).  These
        /// must match the prefix used by `bench compress --out-dir`.
        #[arg(long, value_delimiter = ',')]
        datasets: Vec<String>,
        /// Comma-separated analysis task names.  See `padoc list`.
        #[arg(long, value_delimiter = ',')]
        tasks: Vec<String>,
        /// Repeat each (compressor, dataset, task) measurement N times and
        /// report median timings.  Useful for noisy disks or page-cache
        /// warm-up effects.
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Scalability sweep on synthetic traces.
    Scalability {
        #[arg(long, default_value = "gpus")]
        dimension: String,
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
        values: Vec<usize>,
        #[arg(long, default_value = "padoc")]
        compressor: String,
    },
    /// Parallel compression.
    Parallel {
        dataset_dir: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
        workers: Vec<usize>,
        #[arg(long, default_value = "padoc")]
        compressor: String,
    },
}

fn main() -> anyhow::Result<()> {
    utils::init_logging();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compress { input, output, zstd_level } => cmd_compress(&input, &output, zstd_level),
        Cmd::Decompress { input, output } => cmd_decompress(&input, &output),
        Cmd::Roundtrip { input, compressor, dir, workers } => cmd_roundtrip(&input, &compressor, dir, workers),
        Cmd::Analyze { trace, task, in_situ } => cmd_analyze(&trace, &task, in_situ),
        Cmd::List => cmd_list(),
        Cmd::Bench { sub } => match sub {
            BenchCmd::Compress { datasets, manifest, compressors, padoc_presets, per_rank, workers, zstd_level, out_dir } => {
                cmd_bench_compress(
                    &datasets,
                    manifest.as_deref(),
                    compressors.as_deref(),
                    padoc_presets.as_deref(),
                    per_rank,
                    workers,
                    zstd_level,
                    out_dir.as_deref(),
                )
            }
            BenchCmd::Analyze { datasets } => cmd_bench_analyze(&datasets),
            BenchCmd::AnalyzeOne { compressor, artifact, task, repeat } =>
                cmd_bench_analyze_one(&compressor, &artifact, &task, repeat),
            BenchCmd::AnalyzeBatch { compressor, artifact, tasks, repeat } =>
                cmd_bench_analyze_batch(&compressor, &artifact, &tasks, repeat),
            BenchCmd::AnalyzeArtifacts { artifact_dir, compressors, datasets, tasks, repeat } =>
                cmd_bench_analyze_artifacts(&artifact_dir, &compressors, &datasets, &tasks, repeat),
            BenchCmd::Scalability { dimension, values, compressor } => cmd_bench_scalability(&dimension, &values, &compressor),
            BenchCmd::Parallel { dataset_dir, workers, compressor } => cmd_bench_parallel(&dataset_dir, &workers, &compressor),
        },
    }
}

fn cmd_compress(input: &Path, output: &Path, zstd_level: i32) -> anyhow::Result<()> {
    let trace = load_trace(input)?;
    let mut compressor = TemplateCompressor::with_config(CompressorConfig::default());
    let compressed = compressor.compress(&trace)?;
    compressed.write_to_path(output, zstd_level)?;
    println!("compressed {} events -> {} bytes", trace.event_count(), std::fs::metadata(output)?.len());
    Ok(())
}

fn cmd_decompress(input: &Path, output: &Path) -> anyhow::Result<()> {
    let compressed = padoc::trace::CompressedTrace::read_from_path(input)?;
    // Decompression to a chrome-trace JSON is wired up in the next step; for
    // now we emit the templates table so the file is non-empty.
    let preview = serde_json::json!({
        "templates": compressed.templates.len(),
        "ranks": compressed.ranks.len(),
    });
    std::fs::write(output, serde_json::to_vec_pretty(&preview)?)?;
    println!("decoded {} templates", compressed.templates.len());
    Ok(())
}

fn cmd_roundtrip(input: &Path, compressor_name: &str, force_dir: bool, workers: usize) -> anyhow::Result<()> {
    use std::time::Instant;

    let is_dir = force_dir || input.is_dir();
    let load_start = Instant::now();
    let trace = if is_dir { Trace::from_dir(input)? } else { Trace::from_file(input)? };
    let load_secs = load_start.elapsed().as_secs_f64();
    let original_event_count = trace.event_count();

    let raw_bytes = std::fs::metadata(input)
        .map(|m| m.len())
        .unwrap_or(0);

    let registry = baselines::registry();
    let compressor = registry
        .iter()
        .find(|c| c.name() == compressor_name)
        .context("unknown compressor")?;

    // When `--workers > 1` and the user picked padoc on a directory input, run
    // the parallel `compress_rank` + `merge_shards` path so we round-trip
    // exactly what `bench compress --workers N` produces.  This is how we
    // catch any merge-introduced lossiness before going to the cluster.
    let use_parallel = workers > 1 && compressor_name == "padoc" && is_dir;

    let compress_start = Instant::now();
    let artifact = if use_parallel {
        run_parallel_for_roundtrip(input, workers)?
    } else {
        compressor.compress(&trace)?
    };
    let compress_secs = compress_start.elapsed().as_secs_f64();
    let compressed_bytes = artifact.bytes.len() as u64;

    let decompress_start = Instant::now();
    let recovered = compressor.decompress(&artifact.bytes)?;
    let decompress_secs = decompress_start.elapsed().as_secs_f64();

    let verify_start = Instant::now();
    let report = padoc::verify::compare_traces(&trace, &recovered);
    let verify_secs = verify_start.elapsed().as_secs_f64();

    let ratio = if compressed_bytes > 0 {
        raw_bytes as f64 / compressed_bytes as f64
    } else {
        0.0
    };
    let throughput_mb_s = if compress_secs > 0.0 {
        raw_bytes as f64 / 1024.0 / 1024.0 / compress_secs
    } else {
        0.0
    };

    println!("input              : {}", input.display());
    println!("compressor         : {}", compressor_name);
    println!("input_size         : {}", humansize::format_size(raw_bytes, humansize::BINARY));
    println!("event_count        : {}", original_event_count);
    println!("load_secs          : {:>8.3}", load_secs);
    println!("compress_secs      : {:>8.3}", compress_secs);
    println!("decompress_secs    : {:>8.3}", decompress_secs);
    println!("verify_secs        : {:>8.3}", verify_secs);
    println!("compressed_bytes   : {}", humansize::format_size(compressed_bytes, humansize::BINARY));
    println!("ratio              : {:>8.2}x", ratio);
    println!("compress_throughput: {:>8.1} MB/s", throughput_mb_s);
    println!("--- verify report ---");
    println!("original_events    : {}", report.original_event_count);
    println!("reconstructed      : {}", report.reconstructed_event_count);
    println!("matching           : {}", report.matching_events);
    println!("mismatched         : {}", report.mismatched_events);
    println!("missing_streams    : {}", report.missing_streams.len());
    println!("extra_streams      : {}", report.extra_streams.len());
    if !report.stream_count_diffs.is_empty() {
        println!("stream_count_diffs (first 10):");
        for d in &report.stream_count_diffs {
            println!("  - {}", d);
        }
    }
    if !report.first_mismatches.is_empty() {
        println!("first_mismatches:");
        for m in &report.first_mismatches {
            println!("  - {}", m);
        }
    }
    println!("LOSSLESS           : {}", if report.is_ok() { "YES" } else { "NO" });
    if !report.is_ok() {
        anyhow::bail!("round-trip is NOT lossless");
    }
    Ok(())
}

fn run_parallel_for_roundtrip(input: &Path, workers: usize) -> anyhow::Result<padoc::baselines::CompressArtifact> {
    use padoc::baselines::CompressArtifact;
    use padoc::trace::list_trace_files;
    use rayon::prelude::*;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()?;
    let files = list_trace_files(input);
    if files.is_empty() {
        anyhow::bail!("no rank files in {}", input.display());
    }
    let cfg = CompressorConfig::default();
    let start = std::time::Instant::now();
    let shards = pool.install(|| -> Result<Vec<padoc::compressor::RankShard>, padoc::Error> {
        files
            .par_iter()
            .map(|file| -> Result<padoc::compressor::RankShard, padoc::Error> {
                let trace = Trace::from_file(file)?;
                let rank = trace
                    .rank_ids()
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| {
                        file.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string()
                    });
                Ok(TemplateCompressor::compress_rank(&cfg, &rank, &trace))
            })
            .collect()
    })?;
    let compressed = padoc::compressor::merge_shards(&cfg, shards)?;
    let bytes = compressed.to_bytes(3)?;
    Ok(CompressArtifact::new(bytes, start.elapsed().as_secs_f64()))
}

fn cmd_analyze(trace_path: &Path, task: &str, in_situ: bool) -> anyhow::Result<()> {
    let registry = analysis::registry();
    let task = registry.iter().find(|t| t.name() == task).context("unknown task")?;
    let trace = load_trace(trace_path)?;
    let result = if in_situ {
        let mut compressor = TemplateCompressor::new();
        let compressed = compressor.compress(&trace)?;
        task.run_in_situ(&compressed)?
    } else {
        task.run_raw(&trace)?
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn cmd_list() -> anyhow::Result<()> {
    println!("compressors:");
    for c in baselines::registry() {
        println!("  - {}", c.name());
    }
    println!("\nanalysis tasks:");
    for t in analysis::registry() {
        println!("  - {}{}", t.name(), if t.supports_in_situ() { " [in-situ]" } else { "" });
    }
    Ok(())
}

fn cmd_bench_compress(
    datasets: &[PathBuf],
    manifest: Option<&Path>,
    filter: Option<&[String]>,
    padoc_presets: Option<&[String]>,
    per_rank: bool,
    workers: usize,
    zstd_level: i32,
    out_dir: Option<&Path>,
) -> anyhow::Result<()> {
    // Build the working set from CLI `--datasets` and/or `--manifest`.
    let mut streaming: Vec<bench::StreamingDataset> = Vec::new();
    for p in datasets {
        let is_dir = p.is_dir();
        streaming.push(bench::StreamingDataset {
            name: p.file_name().and_then(|n| n.to_str()).unwrap_or("trace").to_string(),
            path: p.clone(),
            is_dir,
        });
    }
    if let Some(mpath) = manifest {
        let m = bench::Manifest::load(mpath)
            .with_context(|| format!("loading manifest {}", mpath.display()))?;
        for entry in m.datasets {
            streaming.push(bench::StreamingDataset {
                name: entry.name,
                path: entry.path,
                is_dir: entry.is_directory,
            });
        }
    }

    if streaming.is_empty() {
        anyhow::bail!("no datasets — pass --datasets or --manifest");
    }

    if let Some(d) = out_dir {
        std::fs::create_dir_all(d)
            .with_context(|| format!("create_dir_all({})", d.display()))?;
        tracing::info!("artifacts will be written under {}", d.display());
    }

    if let Some(labels) = padoc_presets {
        if filter.is_some() {
            anyhow::bail!("--padoc-presets is mutually exclusive with --compressors");
        }
        let configs = resolve_padoc_presets(labels)?;
        let has_dir = streaming.iter().any(|d| d.is_dir);
        let records = if has_dir || workers > 1 {
            bench::run_padoc_parallel_config_matrix(&configs, &streaming, workers, zstd_level, out_dir)?
        } else if per_rank {
            anyhow::bail!("--padoc-presets does not support --per-rank; use --workers N for large directories");
        } else {
            bench::run_padoc_config_matrix(&configs, &streaming, zstd_level, out_dir)?
        };
        print!("{}", bench::render_compression_table(&records));
        return Ok(());
    }

    let all = baselines::registry();
    let compressors: Vec<Box<dyn BaselineCompressor>> = match filter {
        Some(names) => all.into_iter().filter(|c| names.iter().any(|n| n == c.name())).collect(),
        None => all,
    };
    let only_padoc =
        compressors.len() == 1 && compressors.iter().any(|c| c.name() == "padoc");

    let records = if workers > 1 && only_padoc {
        let cfg = CompressorConfig::default();
        bench::run_padoc_parallel(&cfg, &streaming, workers, zstd_level, out_dir)?
    } else if per_rank {
        bench::run_compression_streaming(&compressors, &streaming, out_dir)?
    } else {
        let refs: Vec<bench::runner::DatasetRef> = streaming
            .iter()
            .map(|d| bench::runner::DatasetRef {
                name: &d.name,
                path: &d.path,
                is_dir: d.is_dir,
            })
            .collect();
        bench::run_compression_matrix(&compressors, &refs, out_dir)?
    };
    print!("{}", bench::render_compression_table(&records));
    Ok(())
}

fn resolve_padoc_presets(labels: &[String]) -> anyhow::Result<Vec<CompressorConfig>> {
    let presets = all_ablation_presets();
    let expanded: Vec<String> = if labels.iter().any(|x| x == "all") {
        presets.keys().cloned().collect()
    } else {
        labels.to_vec()
    };
    let mut configs = Vec::new();
    for label in expanded {
        let cfg = presets
            .get(&label)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown PADOC preset `{label}`"))?;
        configs.push(cfg);
    }
    Ok(configs)
}

fn cmd_bench_analyze(datasets: &[PathBuf]) -> anyhow::Result<()> {
    let compressors = baselines::registry();
    let tasks = analysis::registry();
    let refs: Vec<bench::runner::DatasetRef> = datasets
        .iter()
        .map(|p| bench::runner::DatasetRef {
            name: p.file_name().and_then(|n| n.to_str()).unwrap_or("trace"),
            path: p,
            is_dir: p.is_dir(),
        })
        .collect();
    let records = bench::run_analysis_matrix(&compressors, &tasks, &refs)?;
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}

fn cmd_bench_analyze_artifacts(
    artifact_dirs: &[PathBuf],
    compressor_names: &[String],
    dataset_names: &[String],
    task_names: &[String],
    repeat: usize,
) -> anyhow::Result<()> {
    if artifact_dirs.is_empty() {
        anyhow::bail!("pass at least one --artifact-dir");
    }
    let registry = baselines::registry();
    let compressors: Vec<&dyn padoc::baselines::BaselineCompressor> = compressor_names
        .iter()
        .map(|n| {
            registry
                .iter()
                .find(|c| c.name() == n)
                .map(|b| &**b)
                .ok_or_else(|| anyhow::anyhow!("unknown compressor `{n}`"))
        })
        .collect::<anyhow::Result<_>>()?;
    let task_registry = analysis::registry();
    let tasks: Vec<&dyn padoc::analysis::AnalysisTask> = task_names
        .iter()
        .map(|n| {
            task_registry
                .iter()
                .find(|t| t.name() == n)
                .map(|t| &**t)
                .ok_or_else(|| anyhow::anyhow!("unknown task `{n}`"))
        })
        .collect::<anyhow::Result<_>>()?;
    let repeat = repeat.max(1);

    println!("compressor\tdataset\ttask\tload_secs\tanalyze_secs\ttotal_secs\tartifact_bytes");
    for c in &compressors {
        for ds in dataset_names {
            // Find the artifact file: try .padoc.zst first, then .{compressor}.bin.
            let candidate_names: Vec<String> = if c.name() == "padoc" {
                vec![format!("{ds}.padoc.zst")]
            } else {
                vec![format!("{ds}.{}.bin", c.name())]
            };
            let path = candidate_names.iter().find_map(|cand| {
                artifact_dirs.iter().find_map(|d| {
                    let p = d.join(cand);
                    if p.exists() { Some(p) } else { None }
                })
            });
            let Some(path) = path else {
                eprintln!(
                    "warn: no artifact for compressor={} dataset={} (looked under {:?})",
                    c.name(),
                    ds,
                    artifact_dirs
                );
                continue;
            };
            let bytes_on_disk = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

            for task in &tasks {
                let mut load_samples: Vec<f64> = Vec::with_capacity(repeat);
                let mut analyze_samples: Vec<f64> = Vec::with_capacity(repeat);
                for _ in 0..repeat {
                    let load_start = std::time::Instant::now();
                    if c.name() == "padoc" {
                        let ct = padoc::trace::CompressedTrace::read_from_path(&path)?;
                        let load = load_start.elapsed().as_secs_f64();
                        load_samples.push(load);
                        let an_start = std::time::Instant::now();
                        let _ = task.run_in_situ(&ct)?;
                        analyze_samples.push(an_start.elapsed().as_secs_f64());
                    } else {
                        let bytes = std::fs::read(&path)?;
                        let trace = c.decompress(&bytes)?;
                        let load = load_start.elapsed().as_secs_f64();
                        load_samples.push(load);
                        let an_start = std::time::Instant::now();
                        let _ = task.run_raw(&trace)?;
                        analyze_samples.push(an_start.elapsed().as_secs_f64());
                    }
                }
                let load_med = median(&mut load_samples);
                let an_med = median(&mut analyze_samples);
                println!(
                    "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}",
                    c.name(), ds, task.name(),
                    load_med, an_med, load_med + an_med,
                    bytes_on_disk
                );
            }
        }
    }
    Ok(())
}

/// Linux peak RSS in kilobytes since process start.  Returned by the
/// kernel via `getrusage(RUSAGE_SELF).ru_maxrss`.  `man 2 getrusage`
/// notes the unit is kilobytes on Linux (despite POSIX saying bytes).
fn peak_rss_kb() -> u64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            ru.ru_maxrss as u64
        } else {
            0
        }
    }
}

fn cmd_bench_analyze_one(
    compressor_name: &str,
    artifact_path: &Path,
    task_name: &str,
    repeat: usize,
) -> anyhow::Result<()> {
    let registry = baselines::registry();
    let compressor = registry
        .iter()
        .find(|c| c.name() == compressor_name)
        .ok_or_else(|| anyhow::anyhow!("unknown compressor `{compressor_name}`"))?;
    let task_registry = analysis::registry();
    let task = task_registry
        .iter()
        .find(|t| t.name() == task_name)
        .ok_or_else(|| anyhow::anyhow!("unknown task `{task_name}`"))?;
    let repeat = repeat.max(1);
    let bytes_on_disk = std::fs::metadata(artifact_path).map(|m| m.len()).unwrap_or(0);

    // Per-stage timings:
    //   read     - fs::read or read_from_path's IO portion
    //   decompress - bytes -> Trace (or CompressedTrace::from_bytes for padoc)
    //   analyze  - run_in_situ for padoc, run_raw for everything else
    //
    // We collect samples then report median to filter out one-off page-cache
    // effects.  Peak RSS is taken at the END of the *last* repeat so it
    // covers the whole process lifetime including all repeats.
    let mut load_samples: Vec<f64> = Vec::with_capacity(repeat);
    let mut decompress_samples: Vec<f64> = Vec::with_capacity(repeat);
    let mut analyze_samples: Vec<f64> = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let read_start = std::time::Instant::now();
        let bytes = std::fs::read(artifact_path)?;
        let read_secs = read_start.elapsed().as_secs_f64();
        load_samples.push(read_secs);

        if compressor.name() == "padoc" {
            // padoc artifact = msgpack(CompressedTrace) inside zstd.  The
            // load step here unwraps zstd + msgpack but keeps the templates
            // and node tree in their compressed form (no per-event Trace
            // expansion).
            let dec_start = std::time::Instant::now();
            let ct = padoc::trace::CompressedTrace::from_bytes(&bytes)?;
            decompress_samples.push(dec_start.elapsed().as_secs_f64());
            let an_start = std::time::Instant::now();
            let _ = task.run_in_situ(&ct)?;
            analyze_samples.push(an_start.elapsed().as_secs_f64());
        } else {
            let dec_start = std::time::Instant::now();
            let trace = compressor.decompress(&bytes)?;
            decompress_samples.push(dec_start.elapsed().as_secs_f64());
            let an_start = std::time::Instant::now();
            let _ = task.run_raw(&trace)?;
            analyze_samples.push(an_start.elapsed().as_secs_f64());
        }
    }

    let read_med = median(&mut load_samples);
    let dec_med = median(&mut decompress_samples);
    let an_med = median(&mut analyze_samples);
    let rss = peak_rss_kb();

    // TSV row: easy to aggregate from a driver script.
    println!(
        "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}",
        compressor_name,
        task_name,
        artifact_path.display(),
        read_med,
        dec_med,
        an_med,
        read_med + dec_med + an_med,
        bytes_on_disk,
        rss,
    );
    Ok(())
}

fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() { return f64::NAN; }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// Batched analysis: read+decompress once, then run every task in
/// `task_names` against the in-memory representation.  Each task is
/// re-run `repeat` times so we can report a stable median analyse
/// time.  Output is one TSV row per task (compressor \t task \t
/// artifact \t load_secs \t decompress_secs \t analyze_secs \t
/// total_secs \t bytes \t peak_rss_kb).  Load and decompress columns
/// are the same across rows (single load); peak_rss_kb is the rusage
/// peak as of the LAST printed row, capturing the whole process.
fn cmd_bench_analyze_batch(
    compressor_name: &str,
    artifact_path: &Path,
    task_names: &[String],
    repeat: usize,
) -> anyhow::Result<()> {
    if task_names.is_empty() {
        anyhow::bail!("--tasks must list at least one task");
    }
    let registry = baselines::registry();
    let compressor = registry
        .iter()
        .find(|c| c.name() == compressor_name)
        .ok_or_else(|| anyhow::anyhow!("unknown compressor `{compressor_name}`"))?;
    let task_registry = analysis::registry();
    // Resolve every task name up front; bail early on typos.
    let tasks: Vec<&Box<dyn analysis::AnalysisTask>> = task_names
        .iter()
        .map(|n| task_registry.iter().find(|t| t.name() == n)
            .ok_or_else(|| anyhow::anyhow!("unknown task `{n}`")))
        .collect::<anyhow::Result<_>>()?;
    let repeat = repeat.max(1);
    let bytes_on_disk = std::fs::metadata(artifact_path).map(|m| m.len()).unwrap_or(0);

    // Step 1: read bytes from disk.
    let read_start = std::time::Instant::now();
    let bytes = std::fs::read(artifact_path)?;
    let read_secs = read_start.elapsed().as_secs_f64();

    // Step 2: decompress / deserialise into in-memory representation.
    // For padoc this is `CompressedTrace::from_bytes` (zstd + msgpack);
    // for everything else it's the baseline's full Trace materialisation.
    enum Loaded {
        Padoc(padoc::trace::CompressedTrace),
        Raw(padoc::trace::Trace),
    }
    let dec_start = std::time::Instant::now();
    let loaded = if compressor.name() == "padoc" {
        Loaded::Padoc(padoc::trace::CompressedTrace::from_bytes(&bytes)?)
    } else {
        Loaded::Raw(compressor.decompress(&bytes)?)
    };
    let decompress_secs = dec_start.elapsed().as_secs_f64();
    drop(bytes);

    // Step 3: run each task `repeat` times, collect median analyze time.
    for (task_name, task) in task_names.iter().zip(tasks.iter()) {
        let mut samples: Vec<f64> = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            let an_start = std::time::Instant::now();
            match &loaded {
                Loaded::Padoc(ct)  => { let _ = task.run_in_situ(ct)?; }
                Loaded::Raw(trace) => { let _ = task.run_raw(trace)?; }
            }
            samples.push(an_start.elapsed().as_secs_f64());
        }
        let an_med = median(&mut samples);
        let rss = peak_rss_kb();
        // Same TSV layout as cmd_bench_analyze_one so the driver
        // script can append rows from either subcommand.
        println!(
            "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}",
            compressor_name,
            task_name,
            artifact_path.display(),
            read_secs,
            decompress_secs,
            an_med,
            read_secs + decompress_secs + an_med,
            bytes_on_disk,
            rss,
        );
    }
    Ok(())
}

fn cmd_bench_scalability(dimension: &str, values: &[usize], compressor_name: &str) -> anyhow::Result<()> {
    let compressors = baselines::registry();
    let compressor = compressors.iter().find(|c| c.name() == compressor_name).context("unknown compressor")?;
    let spec = SyntheticTraceSpec::default();
    let points = bench::run_scalability(compressor.as_ref(), &spec, dimension, values)?;
    print!("{}", bench::render_scalability_table(&points));
    Ok(())
}

fn cmd_bench_parallel(dataset_dir: &Path, workers: &[usize], compressor_name: &str) -> anyhow::Result<()> {
    let compressors = baselines::registry();
    let compressor = compressors.iter().find(|c| c.name() == compressor_name).context("unknown compressor")?;
    // One trace per file in `dataset_dir`.
    let mut traces: Vec<Trace> = Vec::new();
    for entry in std::fs::read_dir(dataset_dir)? {
        let entry = entry?;
        if entry.path().is_file() {
            traces.push(Trace::from_file(entry.path())?);
        }
    }
    if traces.is_empty() {
        anyhow::bail!("no traces found in {}", dataset_dir.display());
    }
    let records = bench::run_parallel_compression(compressor.as_ref(), &traces, workers)?;
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}

fn load_trace(path: &Path) -> anyhow::Result<Trace> {
    Ok(if path.is_dir() {
        Trace::from_dir(path)?
    } else {
        Trace::from_file(path)?
    })
}
