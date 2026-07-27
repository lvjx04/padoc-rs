use padoc::{TemplateCompressor, Trace};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .expect("usage: cargo run --example compress -- <trace.json> <trace.padoc>");
    let output = args
        .next()
        .map(PathBuf::from)
        .expect("usage: cargo run --example compress -- <trace.json> <trace.padoc>");

    let trace = Trace::from_file(&input)?;
    let mut compressor = TemplateCompressor::new();
    let compressed = compressor.compress(&trace)?;
    compressed.write_to_path(&output, 3)?;

    println!(
        "compressed {} events into {}",
        trace.event_count(),
        output.display()
    );
    Ok(())
}
