//! Bench manifest parsing.
//!
//! Manifests describe a list of datasets to run the bench over.  The schema is
//! identical to the one used by the legacy Python pipeline so the same files
//! work for both:
//!
//! ```json
//! {
//!   "datasets": [
//!     {"name": "llama_full", "path": "/mnt/.../llama/profiler",
//!      "is_directory": true, "gpus": 1024}
//!   ]
//! }
//! ```
//!
//! `gpus` is informational; the actual rank file count is derived from the
//! filesystem.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub path: PathBuf,
    #[serde(default)]
    pub is_directory: bool,
    /// Optional advisory GPU count (informational, not enforced).
    #[serde(default)]
    pub gpus: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub datasets: Vec<ManifestEntry>,
}

impl Manifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| {
            crate::Error::Other(format!("manifest {} parse error: {}", path.display(), e))
        })?;
        Ok(manifest)
    }
}
