# PADOC Trace Viewer prototype

An interaction prototype for exploring traces that are too large to materialize
in a browser all at once.

The full trace remains visible as indexed call-tree summaries. Only the focused
tree is expanded into interactive CPU events and correlated GPU streams.

This directory is intentionally isolated from PADOC's stable Rust CLI. It uses
synthetic data shaped after distributed training traces; the memory numbers in
the interface are estimates for product exploration, not benchmark results.

## Development

```bash
npm install
npm run dev
```

Use `npm test` for a production build and a server-rendering smoke test.
