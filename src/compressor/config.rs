//! Internal compression policy.
//!
//! Research-only ablation switches live on the research branch. The public
//! implementation deliberately exposes one supported encoding policy.

#[derive(Clone, Debug)]
pub(crate) struct CompressorConfig {
    /// SameCpuNode formation (sibling sub-tree dedup).
    pub enable_structural: bool,
    /// LCS-style anchor extraction inside SameCpuNode.
    pub enable_anchor_matching: bool,
    /// Per-template args dedup (drop redundant rows).
    pub enable_args_dedup: bool,
    /// Pair CPU launches with GPU kernels via `correlation` arg.
    pub enable_kernel_links: bool,
    /// Digit-collapsing in event names ("layer.12" + "layer.13" -> one template).
    pub enable_name_pattern: bool,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            enable_structural: true,
            enable_anchor_matching: true,
            enable_args_dedup: true,
            enable_kernel_links: true,
            enable_name_pattern: true,
        }
    }
}
