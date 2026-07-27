//! Per-template column finalization.

use super::config::CompressorConfig;
use crate::event::{MergeEvent, MergeKernelEvent};
use crate::slp::compress_name_nums;

pub(crate) fn cpu_template(template: &mut MergeEvent, config: &CompressorConfig) {
    if config.enable_name_pattern {
        template.name_nums = compress_name_nums(&template.name_nums);
    }
    if config.enable_args_dedup {
        for column in &mut template.args_columns {
            column.compact();
        }
    }
    template.ts.compact();
    template.dur.compact();
    template.id.compact();
}

pub(crate) fn gpu_template(template: &mut MergeKernelEvent, config: &CompressorConfig) {
    if config.enable_name_pattern {
        template.name_nums = compress_name_nums(&template.name_nums);
    }
    if config.enable_args_dedup {
        for column in &mut template.args_columns {
            column.compact();
        }
    }
    template.ts.compact();
    template.dur.compact();
    template.id.compact();
    template.pid.compact();
    template.stream_tid.compact();
    template.ph.compact();
}
