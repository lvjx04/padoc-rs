# Contributing

Contributions should keep PADOC's public scope small and predictable.

Before opening a pull request:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Changes to compression or serialization must include an event-level round-trip
test. Changes to the artifact schema must update the format version and
`docs/artifact-format.md`.

Baseline implementations, paper experiments, and machine-specific scripts
belong on the `research/baselines` branch rather than `main`.

Commit messages use a short Conventional Commits prefix such as `fix:`,
`feat:`, `refactor:`, `docs:`, `test:`, or `chore:`.
