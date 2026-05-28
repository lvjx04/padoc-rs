use padoc::trace::CompressedTrace;
use padoc::event::{NumColumn, Template};

fn main() {
    let ct = CompressedTrace::read_from_path(
        "/mnt/treasure/ljx/artifacts_v7_sparse/llama_full.padoc.zst"
    ).unwrap();

    let mut total_values: u64 = 0;
    let mut current_bytes: u64 = 0;
    
    // Strategy: for each i32 column, try piecewise linear fit with i8/i16 residuals
    let mut fit_i8_values: u64 = 0;  // values that fit in i8 residual
    let mut fit_i16_values: u64 = 0; // values that fit in i16 residual
    let mut overflow_values: u64 = 0; // values that don't fit
    let mut segments_count: u64 = 0;
    let mut columns_checked: u64 = 0;
    
    for tmpl in &ct.templates {
        let ts_col = match tmpl {
            Template::Cpu(t) => &t.ts,
            Template::Gpu(t) => &t.ts,
        };
        
        let values = match ts_col {
            NumColumn::I32(v) => v,
            NumColumn::Constant { .. } => continue, // already optimal
            NumColumn::Slp(_) => continue, // already SLP encoded
            _ => continue,
        };
        
        let n = values.len();
        if n < 2 { continue; }
        columns_checked += 1;
        total_values += n as u64;
        current_bytes += n as u64 * 4;
        
        // Greedy segmentation: extend segment while residuals fit in i8
        let mut i = 0;
        while i < n {
            segments_count += 1;
            let start_val = values[i] as i64;
            
            if i + 1 >= n {
                fit_i8_values += 1; // single value, residual = 0
                i += 1;
                continue;
            }
            
            // Slope from first two points (integer)
            let slope = (values[i + 1] as i64) - start_val;
            
            // Try to extend with i8 residuals first
            let mut j = i;
            let mut seg_i8 = 0u64;
            let mut seg_i16 = 0u64;
            
            while j < n {
                let local_idx = (j - i) as i64;
                let predicted = start_val + slope * local_idx;
                let residual = values[j] as i64 - predicted;
                
                if residual >= -128 && residual <= 127 {
                    seg_i8 += 1;
                    j += 1;
                } else if residual >= -32768 && residual <= 32767 {
                    seg_i16 += 1;
                    j += 1;
                } else {
                    break;
                }
            }
            
            // If segment is too short (< 4 values), just count as i16
            if j - i < 4 && j < n {
                // Try with just i16 threshold
                j = i;
                seg_i8 = 0;
                seg_i16 = 0;
                while j < n {
                    let local_idx = (j - i) as i64;
                    let predicted = start_val + slope * local_idx;
                    let residual = values[j] as i64 - predicted;
                    if residual >= -32768 && residual <= 32767 {
                        seg_i16 += 1;
                        j += 1;
                    } else {
                        break;
                    }
                }
            }
            
            fit_i8_values += seg_i8;
            fit_i16_values += seg_i16;
            
            if j == i {
                // Can't fit even with i16, skip one value
                overflow_values += 1;
                i += 1;
            } else {
                i = j;
            }
        }
    }
    
    // Also check dur columns
    let mut dur_total: u64 = 0;
    let mut dur_i8: u64 = 0;
    let mut dur_i16: u64 = 0;
    let mut dur_overflow: u64 = 0;
    let mut dur_segments: u64 = 0;
    
    for tmpl in &ct.templates {
        let dur_col = match tmpl {
            Template::Cpu(t) => &t.dur,
            Template::Gpu(t) => &t.dur,
        };
        let values = match dur_col {
            NumColumn::I32(v) => v,
            NumColumn::Slp(_) => continue,
            _ => continue,
        };
        let n = values.len();
        if n < 2 { continue; }
        dur_total += n as u64;
        
        let mut i = 0;
        while i < n {
            dur_segments += 1;
            let start_val = values[i] as i64;
            if i + 1 >= n { dur_i8 += 1; i += 1; continue; }
            let slope = (values[i+1] as i64) - start_val;
            let mut j = i;
            while j < n {
                let predicted = start_val + slope * (j - i) as i64;
                let residual = values[j] as i64 - predicted;
                if residual >= -128 && residual <= 127 { dur_i8 += 1; j += 1; }
                else if residual >= -32768 && residual <= 32767 { dur_i16 += 1; j += 1; }
                else { break; }
            }
            if j == i { dur_overflow += 1; i += 1; } else { i = j; }
        }
    }

    println!("=== ts columns (llama_full) ===");
    println!("columns checked: {}", columns_checked);
    println!("total values: {} ({:.3} GiB as i32)", total_values, total_values as f64 * 4.0 / 1024.0/1024.0/1024.0);
    println!("segments: {}", segments_count);
    println!("avg segment length: {:.1}", total_values as f64 / segments_count as f64);
    println!("");
    println!("fit in i8 residual: {} ({:.1}%)", fit_i8_values, fit_i8_values as f64 / total_values as f64 * 100.0);
    println!("fit in i16 residual: {} ({:.1}%)", fit_i16_values, fit_i16_values as f64 / total_values as f64 * 100.0);
    println!("overflow (need i32): {} ({:.1}%)", overflow_values, overflow_values as f64 / total_values as f64 * 100.0);
    println!("");
    let compressed_bytes = segments_count * 12 + fit_i8_values * 1 + fit_i16_values * 2 + overflow_values * 4;
    println!("estimated compressed: {:.3} GiB", compressed_bytes as f64 / 1024.0/1024.0/1024.0);
    println!("current i32: {:.3} GiB", current_bytes as f64 / 1024.0/1024.0/1024.0);
    println!("savings: {:.3} GiB ({:.0}%)", (current_bytes - compressed_bytes) as f64 / 1024.0/1024.0/1024.0, (1.0 - compressed_bytes as f64 / current_bytes as f64) * 100.0);
    
    println!("\n=== dur columns (llama_full) ===");
    println!("total values: {} ({:.3} GiB as i32)", dur_total, dur_total as f64 * 4.0 / 1024.0/1024.0/1024.0);
    println!("segments: {}", dur_segments);
    println!("fit i8: {} ({:.1}%)", dur_i8, dur_i8 as f64 / dur_total as f64 * 100.0);
    println!("fit i16: {} ({:.1}%)", dur_i16, dur_i16 as f64 / dur_total as f64 * 100.0);
    println!("overflow: {} ({:.1}%)", dur_overflow, dur_overflow as f64 / dur_total as f64 * 100.0);
    let dur_compressed = dur_segments * 12 + dur_i8 * 1 + dur_i16 * 2 + dur_overflow * 4;
    println!("estimated: {:.3} GiB (from {:.3} GiB, save {:.0}%)", dur_compressed as f64 / 1024.0/1024.0/1024.0, dur_total as f64 * 4.0 / 1024.0/1024.0/1024.0, (1.0 - dur_compressed as f64 / (dur_total as f64 * 4.0)) * 100.0);
}
