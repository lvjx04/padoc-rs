/// True iff `name` looks like an NCCL / collective device kernel.
///
/// Callers should already have filtered to GPU-side `cat == "kernel"` events.
/// The broad `"nccl"` match intentionally catches both real profiler names
/// (`ncclKernel_*`, `ncclDevKernel_*`) and synthetic/open-trace variants such
/// as `nccl:all_reduce` without accidentally classifying CPU c10d wrappers.
pub(super) fn is_nccl_kernel(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("nccl")
        || lower.contains("genericmultishmop")
        || lower.contains("genericixccl")
}
