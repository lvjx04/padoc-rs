use padoc::trace::CompressedTrace;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .expect("usage: rewrite_artifact <input.padoc.zst> <output.padoc.zst> [zstd_level]"),
    );
    let output = PathBuf::from(
        args.next()
            .expect("usage: rewrite_artifact <input.padoc.zst> <output.padoc.zst> [zstd_level]"),
    );
    let zstd_level = args
        .next()
        .map(|s| s.parse::<i32>())
        .transpose()?
        .unwrap_or(3);

    let start = Instant::now();
    let trace = CompressedTrace::read_from_path(&input)?;
    let read_secs = start.elapsed().as_secs_f64();

    let start = Instant::now();
    trace.write_to_path(&output, zstd_level)?;
    let write_secs = start.elapsed().as_secs_f64();

    let input_bytes = std::fs::metadata(&input)?.len();
    let output_bytes = std::fs::metadata(&output)?.len();
    println!(
        "{}\t{}\t{}\t{}\t{:.4}\t{:.4}",
        input.display(),
        output.display(),
        input_bytes,
        output_bytes,
        read_secs,
        write_secs
    );
    Ok(())
}
