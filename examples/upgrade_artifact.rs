//! Upgrade an existing .padoc.zst artifact to use SLP compression on all
//! eligible numeric and arg columns.

use padoc::event::Template;
use padoc::trace::CompressedTrace;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: upgrade_artifact <input.padoc.zst> <output.padoc.zst>");
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];

    // Deserialize manually: decode zstd -> msgpack without arena conversion
    // or dropping ranks, so we can write back with ranks intact.
    eprintln!("Loading: {}", input);
    let t0 = Instant::now();
    let bytes = std::fs::read(input)?;
    let raw = zstd::stream::decode_all(bytes.as_slice())?;
    drop(bytes);
    let mut trace: CompressedTrace = rmp_serde::from_slice(&raw)?;
    drop(raw);
    eprintln!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // Apply SLP to all templates
    let mut slp_applied = 0usize;
    for tmpl in &mut trace.templates {
        match tmpl {
            Template::Cpu(t) => {
                t.ts.encode_slp();
                t.dur.encode_slp();
                t.id.encode_slp();
                for col in t.args_columns.iter_mut() {
                    col.encode_slp();
                }
                slp_applied += 1;
            }
            Template::Gpu(t) => {
                t.ts.encode_slp();
                t.dur.encode_slp();
                t.pid.encode_slp();
                for col in t.args_columns.iter_mut() {
                    col.encode_slp();
                }
                slp_applied += 1;
            }
        }
    }
    eprintln!("Applied SLP to {} templates", slp_applied);

    // Write upgraded artifact (ranks are preserved since we didn't drop them)
    let t0 = Instant::now();
    trace.write_to_path(output, 3)?;
    eprintln!("Written: {} in {:.2}s", output, t0.elapsed().as_secs_f64());

    Ok(())
}
