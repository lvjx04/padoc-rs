//! Internal compression policy.
//!
//! Research-only ablation switches live on the research branch. The public
//! implementation deliberately exposes one supported encoding policy.

#[derive(Clone, Debug)]
pub(crate) struct CompressorConfig {
    /// Per-template args dedup (drop redundant rows).
    pub enable_args_dedup: bool,
    /// Digit-collapsing in event names ("layer.12" + "layer.13" -> one template).
    pub enable_name_pattern: bool,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            enable_args_dedup: true,
            enable_name_pattern: true,
        }
    }
}
