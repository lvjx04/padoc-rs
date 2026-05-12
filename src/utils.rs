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

/// Extract every digit run from `name` as integers (used to reconstruct
/// `name_pattern + name_nums`).
pub fn extract_digit_runs(name: &str) -> Vec<i64> {
    DIGIT_RUN_RE
        .find_iter(name)
        .filter_map(|m| m.as_str().parse::<i64>().ok())
        .collect()
}

/// Re-insert digit runs into a normalized pattern: replaces each `0` placeholder
/// in `pattern` with the corresponding entry from `nums`.
pub fn restore_digits(pattern: &str, nums: &[i64]) -> String {
    let mut out = String::with_capacity(pattern.len() + nums.len() * 2);
    let mut iter = nums.iter();
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '0' {
            // Consume run of zeros and replace whole run with one number.
            while let Some('0') = chars.peek().copied() {
                chars.next();
            }
            if let Some(n) = iter.next() {
                out.push_str(&n.to_string());
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
}
