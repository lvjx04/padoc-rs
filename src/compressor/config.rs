//! Compressor ablation switches.
//!
//! Each flag corresponds to one technique the paper ablates.  Defaults are
//! the production behaviour; flipping a flag turns *off* that technique so
//! a row of the ablation table can be produced without code changes.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressorConfig {
    /// SameCpuNode formation (sibling sub-tree dedup).
    pub enable_structural: bool,
    /// LCS-style anchor extraction inside SameCpuNode.
    pub enable_anchor_matching: bool,
    /// SLP encoding on ts/dur/id columns.
    pub enable_slp: bool,
    /// Per-template args dedup (drop redundant rows).
    pub enable_args_dedup: bool,
    /// Pair CPU launches with GPU kernels via `correlation` arg.
    pub enable_kernel_links: bool,
    /// Digit-collapsing in event names ("layer.12" + "layer.13" -> one template).
    pub enable_name_pattern: bool,
    /// Free-form label used by the bench report to name the row.
    pub label: String,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            enable_structural: true,
            enable_anchor_matching: true,
            enable_slp: true,
            enable_args_dedup: true,
            enable_kernel_links: true,
            enable_name_pattern: true,
            label: "default".into(),
        }
    }
}

impl CompressorConfig {
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn is_default(&self) -> bool {
        self.enable_structural
            && self.enable_anchor_matching
            && self.enable_slp
            && self.enable_args_dedup
            && self.enable_kernel_links
            && self.enable_name_pattern
    }
}

/// Standard ablation rows used by the paper.
pub fn all_ablation_presets() -> BTreeMap<String, CompressorConfig> {
    let presets: Vec<CompressorConfig> = vec![
        CompressorConfig::default(),
        CompressorConfig {
            enable_structural: false,
            enable_anchor_matching: false,
            label: "no_structural".into(),
            ..Default::default()
        },
        CompressorConfig {
            enable_anchor_matching: false,
            label: "no_anchor".into(),
            ..Default::default()
        },
        CompressorConfig {
            enable_slp: false,
            label: "no_slp".into(),
            ..Default::default()
        },
        CompressorConfig {
            enable_args_dedup: false,
            label: "no_args_dedup".into(),
            ..Default::default()
        },
        CompressorConfig {
            enable_kernel_links: false,
            label: "no_kernel_links".into(),
            ..Default::default()
        },
        CompressorConfig {
            enable_name_pattern: false,
            label: "no_name_pattern".into(),
            ..Default::default()
        },
        CompressorConfig {
            enable_structural: false,
            enable_anchor_matching: false,
            enable_slp: false,
            enable_args_dedup: false,
            enable_kernel_links: false,
            enable_name_pattern: false,
            label: "minimal".into(),
        },
    ];
    let mut map = BTreeMap::new();
    for cfg in presets {
        map.insert(cfg.label.clone(), cfg);
    }
    map
}
