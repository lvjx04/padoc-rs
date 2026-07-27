use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn padoc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_padoc"))
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny_trace.json")
}

#[test]
fn single_file_cli_workflow() {
    let dir = tempfile::tempdir().expect("tempdir");
    let artifact = dir.path().join("tiny.padoc");
    let restored = dir.path().join("restored.json");

    assert!(padoc()
        .args(["compress", fixture().to_str().unwrap(), "--output"])
        .arg(&artifact)
        .status()
        .expect("run compress")
        .success());

    let inspect = padoc()
        .arg("inspect")
        .arg(&artifact)
        .output()
        .expect("run inspect");
    assert!(inspect.status.success());
    let info: serde_json::Value = serde_json::from_slice(&inspect.stdout).expect("inspect JSON");
    assert_eq!(info["ranks"], serde_json::json!(["0"]));
    assert_eq!(info["event_instances"], 3);

    assert!(padoc()
        .arg("verify")
        .arg(fixture())
        .arg("--artifact")
        .arg(&artifact)
        .status()
        .expect("run verify")
        .success());

    assert!(padoc()
        .arg("decompress")
        .arg(&artifact)
        .arg("--output")
        .arg(&restored)
        .status()
        .expect("run decompress")
        .success());
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(restored).expect("read restored"))
            .expect("restored JSON");
    let events = value["traceEvents"].as_array().unwrap();
    assert_eq!(events.len(), 5);
    let metadata: Vec<_> = events.iter().filter(|event| event["ph"] == "M").collect();
    assert_eq!(metadata.len(), 2);
    assert_eq!(metadata[0]["pid"], 7);
    assert_eq!(metadata[0]["tid"], "cpu");
    assert_eq!(metadata[1]["pid"], 8);
    assert_eq!(metadata[1]["tid"], "worker");
}

#[test]
fn directory_cli_writes_independent_artifacts_and_manifest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = dir.path().join("artifacts");

    assert!(padoc()
        .arg("compress-dir")
        .arg(fixture().parent().unwrap())
        .arg("--output")
        .arg(&output)
        .arg("--workers")
        .arg("2")
        .status()
        .expect("run compress-dir")
        .success());

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("manifest.json")).expect("manifest"))
            .expect("manifest JSON");
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["entries"].as_array().unwrap().len(), 1);
    assert!(output.join("tiny_trace.padoc").is_file());
}

#[test]
fn failed_directory_compression_leaves_no_partial_output() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = dir.path().join("input");
    let output = dir.path().join("artifacts");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::copy(fixture(), input.join("valid.json")).expect("copy fixture");
    std::fs::write(input.join("invalid.json"), b"{not json").expect("write invalid trace");

    let status = padoc()
        .arg("compress-dir")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .arg("--workers")
        .arg("2")
        .status()
        .expect("run compress-dir");

    assert!(!status.success());
    assert!(!output.exists());
}

#[test]
fn cli_refuses_to_overwrite_artifacts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let artifact = dir.path().join("tiny.padoc");

    let first = padoc()
        .args(["compress", fixture().to_str().unwrap(), "--output"])
        .arg(&artifact)
        .status()
        .expect("first compress");
    let second = padoc()
        .args(["compress", fixture().to_str().unwrap(), "--output"])
        .arg(&artifact)
        .status()
        .expect("second compress");

    assert!(first.success());
    assert!(!second.success());
}

#[test]
fn gzip_trace_input_is_supported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = dir.path().join("tiny.json.gz");
    let artifact = dir.path().join("tiny.padoc");
    let file = std::fs::File::create(&input).expect("create gzip input");
    let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
    encoder
        .write_all(&std::fs::read(fixture()).expect("read fixture"))
        .expect("write gzip input");
    encoder.finish().expect("finish gzip input");

    assert!(padoc()
        .arg("compress")
        .arg(&input)
        .arg("--output")
        .arg(&artifact)
        .status()
        .expect("compress gzip trace")
        .success());
    assert!(padoc()
        .arg("verify")
        .arg(&input)
        .arg("--artifact")
        .arg(&artifact)
        .status()
        .expect("verify gzip trace")
        .success());
}
