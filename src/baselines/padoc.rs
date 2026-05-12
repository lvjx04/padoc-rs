//! `padoc` baseline — wraps `TemplateCompressor` so the bench harness can
//! treat it the same as ScalaTrace / TraceZip.

use crate::baselines::{BaselineCompressor, CompressArtifact};
use crate::compressor::{CompressorConfig, TemplateCompressor};
use crate::trace::{CompressedTrace, Trace};
use crate::Result;

#[derive(Default)]
pub struct PadocCompressor {
    pub config: CompressorConfig,
}

impl PadocCompressor {
    pub fn with_config(config: CompressorConfig) -> Self {
        Self { config }
    }
}

impl BaselineCompressor for PadocCompressor {
    fn name(&self) -> &str { "padoc" }

    fn compress(&self, trace: &Trace) -> Result<CompressArtifact> {
        let start = std::time::Instant::now();
        let mut compressor = TemplateCompressor::with_config(self.config.clone());
        let compressed = compressor.compress(trace)?;
        let bytes = compressed.to_bytes(3)?;
        let mut artifact = CompressArtifact::new(bytes, start.elapsed().as_secs_f64());
        artifact.annotations.insert(
            "config".to_string(),
            serde_json::to_value(&self.config).unwrap_or(serde_json::Value::Null),
        );
        Ok(artifact)
    }

    fn decompress(&self, bytes: &[u8]) -> Result<Trace> {
        let compressed = CompressedTrace::from_bytes(bytes)?;
        Ok(crate::compressor::decompress(&compressed))
    }
}
