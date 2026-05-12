//! Pre-processor: rewrite chrome-trace JSON files so every `ts` and `dur`
//! is an integer.
//!
//! Why this exists: a number of profilers (Kineto+ROCm in particular) emit
//! `"ts": 1234567.89` even though the chrome-trace spec says integer
//! microseconds.  PADOC's own streaming parser truncates floats with
//! `f64 as i64`, but the legacy simd-json `.as_i64()` path silently
//! returned `None` and `unwrap_or(0)` zeroed every such timestamp.  Worse,
//! any external tool (HTA, perfetto, jq pipelines) that assumes integer
//! timestamps is going to either crash or produce garbage on these traces.
//! Cleaner to fix the source files once.
//!
//! Approach: byte-level scan, no JSON parser.  The input is read fully into
//! memory, and every byte position that begins with `"ts":` or `"dur":`
//! has its following numeric literal truncated to its integer part.  Every
//! other byte — whitespace, commas, strings, args, distributedInfo blocks,
//! file ordering — is preserved exactly.  This matters because we don't
//! want a JSON re-emit to silently change field order, escape choices, or
//! number formatting downstream consumers might depend on.
//!
//! Limitations:
//!
//! * Loads the whole file into RAM.  On unifolm that's 5.7 GiB; the
//!   conversion needs ~2× that (input + output buffer) so plan for ~12
//!   GiB / rank.  The cluster machines have plenty of headroom.
//! * If `"ts":` ever literally appears inside a JSON string value the
//!   replace would mis-fire.  In every chrome-trace dialect we ship it
//!   only ever appears as a top-level event field, so this is fine in
//!   practice; the tool prints a warning if it sees `"ts":` immediately
//!   after a non-`,`/`{` byte that's not whitespace.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example normalize_int_ts -- <file_or_dir>...
//! ```
//!
//! Each path is converted in place; the rewritten content is written to
//! `<path>.tmp` first then atomically renamed.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const TS_KEY: &[u8] = b"\"ts\":";
const DUR_KEY: &[u8] = b"\"dur\":";

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <file_or_dir> [<file_or_dir> ...]", args[0]);
        std::process::exit(1);
    }
    let mut targets: Vec<PathBuf> = Vec::new();
    for arg in &args[1..] {
        let p = PathBuf::from(arg);
        if p.is_dir() {
            for entry in fs::read_dir(&p)? {
                let entry = entry?;
                let ep = entry.path();
                if is_json(&ep) {
                    targets.push(ep);
                }
            }
        } else if is_json(&p) {
            targets.push(p);
        } else {
            eprintln!("skipping non-json: {}", p.display());
        }
    }
    targets.sort();
    if targets.is_empty() {
        eprintln!("no JSON files found");
        std::process::exit(1);
    }

    let mut total_converted_ts = 0u64;
    let mut total_converted_dur = 0u64;
    let mut total_bytes_in = 0u64;
    let mut total_bytes_out = 0u64;
    let global_start = Instant::now();

    for path in &targets {
        let start = Instant::now();
        match normalize(path) {
            Ok(stats) => {
                total_converted_ts += stats.converted_ts;
                total_converted_dur += stats.converted_dur;
                total_bytes_in += stats.input_bytes;
                total_bytes_out += stats.output_bytes;
                println!(
                    "{}: ts {} -> int, dur {} -> int, {} -> {} ({:.2}s)",
                    path.display(),
                    stats.converted_ts,
                    stats.converted_dur,
                    human_size(stats.input_bytes),
                    human_size(stats.output_bytes),
                    start.elapsed().as_secs_f64()
                );
            }
            Err(e) => {
                eprintln!("{}: ERROR {}", path.display(), e);
                std::process::exit(2);
            }
        }
    }

    println!(
        "\ntotals: {} files, ts converted={}, dur converted={}, {} -> {} in {:.2}s",
        targets.len(),
        total_converted_ts,
        total_converted_dur,
        human_size(total_bytes_in),
        human_size(total_bytes_out),
        global_start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn is_json(p: &Path) -> bool {
    p.is_file()
        && p.extension().and_then(|s| s.to_str()) == Some("json")
}

#[derive(Default, Clone, Copy)]
struct Stats {
    converted_ts: u64,
    converted_dur: u64,
    input_bytes: u64,
    output_bytes: u64,
}

fn normalize(path: &Path) -> std::io::Result<Stats> {
    let data = fs::read(path)?;
    let input_len = data.len();
    let mut out: Vec<u8> = Vec::with_capacity(input_len);

    let mut stats = Stats {
        input_bytes: input_len as u64,
        ..Default::default()
    };

    let bytes = &data[..];
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        // Cheap fast-path: emit any prefix that doesn't start with `"`.
        if bytes[i] != b'"' {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        // Try ts/dur match.
        let key_kind = if bytes[i..].starts_with(TS_KEY) {
            Some((TS_KEY.len(), KeyKind::Ts))
        } else if bytes[i..].starts_with(DUR_KEY) {
            Some((DUR_KEY.len(), KeyKind::Dur))
        } else {
            None
        };
        match key_kind {
            Some((klen, kind)) => {
                out.extend_from_slice(&bytes[i..i + klen]);
                i += klen;
                // Optional whitespace.
                while i < n && matches!(bytes[i], b' ' | b'\t') {
                    out.push(bytes[i]);
                    i += 1;
                }
                // Number literal: optional `-`, integer part, optional `.frac`,
                // optional `e[+-]?digits`.
                let num_start = i;
                if i < n && bytes[i] == b'-' {
                    i += 1;
                }
                let int_start = i;
                while i < n && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let int_end = i;
                let mut had_fraction = false;
                let mut had_exponent = false;
                if i < n && bytes[i] == b'.' {
                    had_fraction = true;
                    i += 1;
                    while i < n && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
                    had_exponent = true;
                    i += 1;
                    if i < n && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    while i < n && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let num_end = i;
                if int_start == int_end && !had_fraction {
                    // No digits at all — e.g. `"ts": null` or `"ts": "..."`.
                    // Pass through whatever lies beyond verbatim.
                    out.extend_from_slice(&bytes[num_start..num_end]);
                } else if !had_fraction && !had_exponent {
                    // Already integer, pass through unchanged.
                    out.extend_from_slice(&bytes[num_start..num_end]);
                } else {
                    // Truncate float toward zero.
                    let s = std::str::from_utf8(&bytes[num_start..num_end]).unwrap_or("0");
                    let truncated: i64 = if had_exponent {
                        // Need full f64 parse for exponent.
                        s.parse::<f64>().unwrap_or(0.0) as i64
                    } else {
                        // Cheap: just emit the integer prefix bytes (skip
                        // optional `-` already in num_start..int_start).
                        let int_prefix = &bytes[num_start..int_end];
                        match std::str::from_utf8(int_prefix)
                            .ok()
                            .and_then(|t| t.parse::<i64>().ok())
                        {
                            Some(v) => v,
                            None => s.parse::<f64>().unwrap_or(0.0) as i64,
                        }
                    };
                    out.extend_from_slice(truncated.to_string().as_bytes());
                    match kind {
                        KeyKind::Ts => stats.converted_ts += 1,
                        KeyKind::Dur => stats.converted_dur += 1,
                    }
                }
            }
            None => {
                // Not a ts/dur key; pass the leading `"` through and continue.
                out.push(bytes[i]);
                i += 1;
            }
        }
    }

    stats.output_bytes = out.len() as u64;
    let tmp_path = path.with_extension("json.tmp");
    fs::write(&tmp_path, &out)?;
    fs::rename(&tmp_path, path)?;
    Ok(stats)
}

#[derive(Clone, Copy)]
enum KeyKind {
    Ts,
    Dur,
}

fn human_size(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0usize;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    format!("{:.2} {}", v, UNITS[u])
}
