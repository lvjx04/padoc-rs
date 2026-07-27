//! Command-line interface for PADOC.

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use padoc::analysis;
use padoc::trace::{list_trace_files, CompressedTrace, Trace};
use padoc::TemplateCompressor;
use rayon::prelude::*;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "padoc",
    version,
    about = "Compress and analyze AI profiler traces",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compress one Chrome trace JSON file.
    Compress {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(i32).range(-7..=22))]
        zstd_level: i32,
    },
    /// Compress every trace file in a directory into independent artifacts.
    CompressDir {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Maximum number of files processed concurrently.
        #[arg(long, default_value = "1")]
        workers: std::num::NonZeroUsize,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(i32).range(-7..=22))]
        zstd_level: i32,
    },
    /// Reconstruct a Chrome trace JSON file from one PADOC artifact.
    Decompress {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Verify PADOC's event-level round trip for one trace.
    Verify {
        input: PathBuf,
        /// Verify an existing artifact instead of compressing in memory.
        #[arg(long)]
        artifact: Option<PathBuf>,
    },
    /// Run an in-situ analysis task on a PADOC artifact.
    Analyze {
        input: PathBuf,
        #[arg(long)]
        task: String,
    },
    /// Print artifact metadata as JSON.
    Inspect { input: PathBuf },
    /// List stable analysis task names.
    List,
}

#[derive(Debug, Serialize)]
struct DirectoryManifest {
    schema_version: u32,
    padoc_version: &'static str,
    entries: Vec<DirectoryEntry>,
}

#[derive(Debug, Serialize)]
struct DirectoryEntry {
    source: String,
    artifact: String,
    ranks: Vec<String>,
    events: usize,
    input_bytes: u64,
    artifact_bytes: u64,
}

#[derive(Serialize)]
struct ArtifactInfo {
    path: String,
    format_version: u16,
    artifact_bytes: u64,
    ranks: Vec<String>,
    templates: usize,
    event_instances: usize,
}

fn main() -> anyhow::Result<()> {
    padoc::utils::init_logging();
    let cli = Cli::parse();
    match cli.command {
        Command::Compress {
            input,
            output,
            zstd_level,
        } => {
            let entry = compress_file(&input, &output, zstd_level)?;
            println!(
                "compressed {} events into {} bytes",
                entry.events, entry.artifact_bytes
            );
            Ok(())
        }
        Command::CompressDir {
            input,
            output,
            workers,
            zstd_level,
        } => compress_directory(&input, &output, workers.get(), zstd_level),
        Command::Decompress { input, output } => decompress_file(&input, &output),
        Command::Verify { input, artifact } => verify_file(&input, artifact.as_deref()),
        Command::Analyze { input, task } => analyze_artifact(&input, &task),
        Command::Inspect { input } => inspect_artifact(&input),
        Command::List => list_tasks(),
    }
}

fn compress_file(input: &Path, output: &Path, zstd_level: i32) -> anyhow::Result<DirectoryEntry> {
    ensure_regular_file(input, "input trace")?;
    ensure_output_is_new(output)?;

    let input_bytes = std::fs::metadata(input)?.len();
    let trace = Trace::from_file(input)
        .with_context(|| format!("read Chrome trace {}", input.display()))?;
    if trace.ranks.len() != 1 {
        bail!(
            "{} contains {} ranks; public artifacts contain one input trace each",
            input.display(),
            trace.ranks.len()
        );
    }

    let events = trace.event_count();
    let ranks = trace.rank_ids();
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor
        .compress(&trace)
        .with_context(|| format!("compress {}", input.display()))?;
    compressed
        .write_to_path(output, zstd_level)
        .with_context(|| format!("write artifact {}", output.display()))?;

    Ok(DirectoryEntry {
        source: file_name(input)?,
        artifact: file_name(output)?,
        ranks,
        events,
        input_bytes,
        artifact_bytes: std::fs::metadata(output)?.len(),
    })
}

fn compress_directory(
    input: &Path,
    output: &Path,
    workers: usize,
    zstd_level: i32,
) -> anyhow::Result<()> {
    if !input.is_dir() {
        bail!("input directory does not exist: {}", input.display());
    }
    ensure_output_is_new(output)?;
    let output_parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(output_parent)
        .with_context(|| format!("create output parent {}", output_parent.display()))?;
    let staging = tempfile::Builder::new()
        .prefix(".padoc-staging-")
        .tempdir_in(output_parent)
        .with_context(|| format!("create staging directory in {}", output_parent.display()))?;
    let staging_path = staging.path();
    let manifest_path = staging_path.join("manifest.json");

    let files = list_trace_files(input);
    if files.is_empty() {
        bail!(
            "no .json or .json.gz trace files found in {}",
            input.display()
        );
    }
    let mut artifact_names = std::collections::BTreeSet::new();
    for source in &files {
        let name = artifact_file_name(source)?;
        if !artifact_names.insert(name.clone()) {
            bail!("multiple inputs map to the same artifact name: {name}");
        }
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("create compression worker pool")?;

    let mut entries = pool.install(|| {
        files
            .par_iter()
            .map(|source| {
                let artifact = staging_path.join(artifact_file_name(source)?);
                compress_file(source, &artifact, zstd_level)
            })
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    entries.sort_by(|a, b| a.source.cmp(&b.source));

    let manifest = DirectoryManifest {
        schema_version: 1,
        padoc_version: env!("CARGO_PKG_VERSION"),
        entries,
    };
    write_new_json(&manifest_path, &manifest)?;
    ensure_output_is_new(output)?;
    std::fs::rename(staging_path, output)
        .with_context(|| format!("publish output directory {}", output.display()))?;
    println!(
        "wrote {} independent artifacts and {}",
        manifest.entries.len(),
        output.join("manifest.json").display()
    );
    Ok(())
}

fn decompress_file(input: &Path, output: &Path) -> anyhow::Result<()> {
    ensure_regular_file(input, "PADOC artifact")?;
    ensure_output_is_new(output)?;
    let compressed = CompressedTrace::read_from_path(input)
        .with_context(|| format!("read artifact {}", input.display()))?;
    let trace = padoc::decompress(&compressed);
    trace
        .write_chrome_json(output)
        .with_context(|| format!("write Chrome trace {}", output.display()))?;
    println!("reconstructed {} events", trace.event_count());
    Ok(())
}

fn verify_file(input: &Path, artifact: Option<&Path>) -> anyhow::Result<()> {
    ensure_regular_file(input, "input trace")?;
    let original =
        Trace::from_file(input).with_context(|| format!("read trace {}", input.display()))?;
    let compressed = if let Some(path) = artifact {
        ensure_regular_file(path, "PADOC artifact")?;
        CompressedTrace::read_from_path(path)
            .with_context(|| format!("read artifact {}", path.display()))?
    } else {
        let mut compressor = TemplateCompressor::new();
        compressor.compress(&original)?
    };
    let recovered = padoc::decompress(&compressed);
    let report = padoc::verify::compare_traces(&original, &recovered);

    println!(
        "original={} reconstructed={} matching={} mismatched={}",
        report.original_event_count,
        report.reconstructed_event_count,
        report.matching_events,
        report.mismatched_events
    );
    if !report.is_ok() {
        for mismatch in report.first_mismatches.iter().take(10) {
            eprintln!("mismatch: {mismatch}");
        }
        for rank in &report.metadata_mismatches {
            eprintln!("metadata mismatch: rank {rank}");
        }
        bail!("round-trip verification failed");
    }
    println!("lossless event round-trip: yes");
    Ok(())
}

fn analyze_artifact(input: &Path, task_name: &str) -> anyhow::Result<()> {
    ensure_regular_file(input, "PADOC artifact")?;
    let compressed = CompressedTrace::read_from_path(input)
        .with_context(|| format!("read artifact {}", input.display()))?;
    let tasks = analysis::registry();
    let task = tasks
        .iter()
        .find(|task| task.name() == task_name)
        .with_context(|| format!("unknown analysis task: {task_name}"))?;
    if !task.supports_in_situ() {
        bail!("analysis task {task_name} is not available in situ");
    }
    let result = task.run_in_situ(&compressed)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn inspect_artifact(input: &Path) -> anyhow::Result<()> {
    ensure_regular_file(input, "PADOC artifact")?;
    let compressed = CompressedTrace::read_from_path(input)
        .with_context(|| format!("read artifact {}", input.display()))?;
    let info = ArtifactInfo {
        path: input.display().to_string(),
        format_version: padoc::trace::ARTIFACT_FORMAT_VERSION,
        artifact_bytes: std::fs::metadata(input)?.len(),
        ranks: compressed.ranks.keys().cloned().collect(),
        templates: compressed.templates.len(),
        event_instances: compressed
            .templates
            .iter()
            .map(|template| template.instance_count())
            .sum(),
    };
    println!("{}", serde_json::to_string_pretty(&info)?);
    Ok(())
}

fn list_tasks() -> anyhow::Result<()> {
    for task in analysis::registry() {
        if task.supports_in_situ() {
            println!("{}", task.name());
        }
    }
    Ok(())
}

fn artifact_file_name(source: &Path) -> anyhow::Result<String> {
    let name = file_name(source)?;
    let stem = name
        .strip_suffix(".json.gz")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(&name);
    Ok(format!("{stem}.padoc"))
}

fn file_name(path: &Path) -> anyhow::Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .with_context(|| format!("path has no UTF-8 file name: {}", path.display()))
}

fn ensure_regular_file(path: &Path, label: &str) -> anyhow::Result<()> {
    if !path.is_file() {
        bail!(
            "{label} does not exist or is not a file: {}",
            path.display()
        );
    }
    Ok(())
}

fn ensure_output_is_new(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        bail!("refusing to overwrite existing output: {}", path.display());
    }
    Ok(())
}

fn write_new_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    use std::io::Write;

    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
