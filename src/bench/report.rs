//! Markdown rendering for bench tables.

use crate::bench::metrics::CompressionRecord;
use crate::bench::scalability::ScalabilityPoint;

pub fn render_compression_table(records: &[CompressionRecord]) -> String {
    let mut out = String::new();
    out.push_str("| dataset | compressor | events | raw_bytes | compressed_bytes | ratio | compress_secs | mb/s |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|\n");
    for r in records {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.2} | {:.3} | {:.1} |\n",
            r.dataset,
            r.compressor,
            r.event_count,
            humansize::format_size(r.raw_bytes, humansize::BINARY),
            humansize::format_size(r.compressed_bytes, humansize::BINARY),
            r.ratio,
            r.compress_secs,
            r.throughput_mb_per_sec,
        ));
    }
    out
}

pub fn render_scalability_table(points: &[ScalabilityPoint]) -> String {
    let mut out = String::new();
    out.push_str("| dimension | value | events | compressed_bytes | ratio | compress_secs |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for p in points {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.2} | {:.3} |\n",
            p.dimension,
            p.value,
            p.event_count,
            humansize::format_size(p.compressed_bytes, humansize::BINARY),
            p.ratio,
            p.compress_secs,
        ));
    }
    out
}
