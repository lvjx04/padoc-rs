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
use padoc::compressor::{CompressorConfig, TemplateCompressor};
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
    Compress {
        #[arg(long, value_delimiter = ',')]
        datasets: Vec<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        compressors: Option<Vec<String>>,
    },
    /// Analysis matrix.
    Analyze {
        #[arg(long, value_delimiter = ',')]
        datasets: Vec<PathBuf>,
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
        Cmd::Analyze { trace, task, in_situ } => cmd_analyze(&trace, &task, in_situ),
        Cmd::List => cmd_list(),
        Cmd::Bench { sub } => match sub {
            BenchCmd::Compress { datasets, compressors } => cmd_bench_compress(&datasets, compressors.as_deref()),
            BenchCmd::Analyze { datasets } => cmd_bench_analyze(&datasets),
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

fn cmd_bench_compress(datasets: &[PathBuf], filter: Option<&[String]>) -> anyhow::Result<()> {
    let all = baselines::registry();
    let compressors: Vec<Box<dyn BaselineCompressor>> = match filter {
        Some(names) => all.into_iter().filter(|c| names.iter().any(|n| n == c.name())).collect(),
        None => all,
    };
    let refs: Vec<bench::runner::DatasetRef> = datasets
        .iter()
        .map(|p| bench::runner::DatasetRef {
            name: p.file_name().and_then(|n| n.to_str()).unwrap_or("trace"),
            path: p,
            is_dir: p.is_dir(),
        })
        .collect();
    let records = bench::run_compression_matrix(&compressors, &refs)?;
    print!("{}", bench::render_compression_table(&records));
    Ok(())
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

