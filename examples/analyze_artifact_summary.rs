use padoc::analysis;
use padoc::baselines;
use padoc::trace::CompressedTrace;
use std::env;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let compressor_name = args
        .next()
        .expect("usage: analyze_artifact_summary <compressor> <artifact> <task>");
    let artifact = PathBuf::from(
        args.next()
            .expect("usage: analyze_artifact_summary <compressor> <artifact> <task>"),
    );
    let task_name = args
        .next()
        .expect("usage: analyze_artifact_summary <compressor> <artifact> <task>");

    let task_registry = analysis::registry();
    let task = task_registry
        .iter()
        .find(|task| task.name() == task_name)
        .ok_or_else(|| anyhow::anyhow!("unknown task `{task_name}`"))?;

    let bytes = std::fs::read(&artifact)?;
    let result = if compressor_name == "padoc" {
        let trace = CompressedTrace::from_bytes(&bytes)?;
        task.run_in_situ(&trace)?
    } else {
        let registry = baselines::registry();
        let compressor = registry
            .iter()
            .find(|compressor| compressor.name() == compressor_name)
            .ok_or_else(|| anyhow::anyhow!("unknown compressor `{compressor_name}`"))?;
        let trace = compressor.decompress(&bytes)?;
        task.run_raw(&trace)?
    };

    let (coverage, rows) = summarize(&result);
    let attributed = coverage
        .and_then(|c| c.get("attributed_gpu_refs"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total = coverage
        .and_then(|c| c.get("total_gpu_refs"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let fraction = coverage
        .and_then(|c| c.get("attributed_fraction"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    println!(
        "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}",
        compressor_name,
        artifact.display(),
        task_name,
        attributed,
        total,
        fraction,
        rows
    );
    Ok(())
}

fn summarize(result: &serde_json::Value) -> (Option<&serde_json::Value>, usize) {
    let value = result.get("result").unwrap_or(result);
    if let Some(array) = value.as_array() {
        let coverage = value
            .as_array()
            .and_then(|rows| rows.first())
            .and_then(|row| row.get("coverage"));
        return (coverage, array.len());
    }
    let coverage = value.get("coverage");
    let rows = value
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .or_else(|| {
            value
                .get("hotspots")
                .and_then(serde_json::Value::as_array)
                .map(Vec::len)
        })
        .or_else(|| value.as_array().map(Vec::len))
        .unwrap_or(0);
    (coverage, rows)
}
