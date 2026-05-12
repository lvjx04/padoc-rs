//! Shared utilities: logging setup, regex caches, small helpers.

use once_cell::sync::Lazy;
use regex::Regex;

/// Compiled once: collapse every digit run in a name to ``0``.
///
/// Matches the Python implementation's `_normalize_name` semantics so that
/// `wgrad_defer_check-12-3` and `wgrad_defer_check-7-15` map to the same
/// template signature.
pub static DIGIT_RUN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d+").unwrap());

/// Replace every digit run in `name` with `"0"`.
#[inline]
pub fn normalize_name(name: &str) -> String {
    DIGIT_RUN_RE.replace_all(name, "0").into_owned()
}

/// Extract every digit run from `name` as raw lexical strings (preserving
/// leading zeros).  Used together with `name_pattern` to losslessly
/// reconstruct names like `0x4387e040` whose `040` digit-run would otherwise
/// be mangled to `40` if parsed as an integer.
pub fn extract_digit_runs(name: &str) -> Vec<String> {
    DIGIT_RUN_RE
        .find_iter(name)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Re-insert digit runs into a normalized pattern: each single `'0'` placeholder
/// in `pattern` (every digit-run was collapsed to one `'0'` by `normalize_name`)
/// is replaced by the corresponding entry from `nums`.
///
/// **Important**: do **not** collapse adjacent `'0'`s in `pattern` — each one
/// represents an independent digit-run slot.  E.g. for the input
/// `0x4387e040` the pattern is `0x0e0` (3 separate `'0'` placeholders), and
/// `nums = ["0", "4387", "040"]`.
pub fn restore_digits(pattern: &str, nums: &[String]) -> String {
    let mut out = String::with_capacity(pattern.len() + nums.len() * 2);
    let mut iter = nums.iter();
    for c in pattern.chars() {
        if c == '0' {
            if let Some(n) = iter.next() {
                out.push_str(n);
            } else {
                out.push('0');
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Initialise a sensible default tracing subscriber.  No-op if already set.
pub fn init_logging() {
    use tracing_subscriber::{fmt, EnvFilter};
    let _ = fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_collapses_digit_runs() {
        assert_eq!(normalize_name("wgrad-12-3"), "wgrad-0-0");
    }

    #[test]
    fn extract_then_restore_is_lossless() {
        let original = "transformer.layers.42.attention.qkv";
        let pattern = normalize_name(original);
        let nums = extract_digit_runs(original);
        assert_eq!(restore_digits(&pattern, &nums), original);
    }

    #[test]
    fn restore_preserves_leading_zeros() {
        let original = "succl:broadcast Graph (0x4387e040)";
        let pattern = normalize_name(original);
        let nums = extract_digit_runs(original);
        assert_eq!(restore_digits(&pattern, &nums), original);
    }

    #[test]
    fn restore_preserves_adjacent_zero_placeholders() {
        // The tag `0x0` becomes pattern `0x0` (the `0x` part has *one* digit
        // run, and the trailing `0` is a *second* digit run); `restore_digits`
        // must not collapse them into one slot.
        let original = "ptr=0x0";
        let pattern = normalize_name(original);
        let nums = extract_digit_runs(original);
        assert_eq!(pattern, "ptr=0x0");
        assert_eq!(nums, vec!["0".to_string(), "0".to_string()]);
        assert_eq!(restore_digits(&pattern, &nums), original);
    }
}
