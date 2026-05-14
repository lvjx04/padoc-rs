# PADOC — Paper Experiment Handoff

This document is the single point of truth for the next agent working on the
PADOC paper. It captures (1) the original experimental plan that the human
PI signed off on, (2) the current status of every item in that plan,
(3) the *why* behind PADOC's analysis speed-up so the next agent can write
about it without re-deriving the argument, and (4) the concrete remaining
work, ordered by impact and effort.

The supporting numbers are in `EXPERIMENTS.md` (paper-style tables) and the
raw TSVs / inspect dumps under `results/main/`. Drivers are in `scripts/`.

Repo HEAD at the time of writing: commit `af3db6c` on
[`lvjx04/padoc-rs`](https://github.com/lvjx04/padoc-rs).

---

## 1. The original plan (verbatim, with the agreed renames)

### Traces

1. dense, 70B, 1024 GPU, training, parallel
2. MoE, 671B, 1024 GPU, training
3. ViT
4. LeWorldModel
5. (open — can pick a few more)

### Analysis tasks (covering different access dimensions)

1. operator hotspot
2. compute / communication overlap
3. per-layer operator balance
4. **per-process load balance** *(originally "parallel group", renamed
   on the PI's instruction — we now report compute-busy and comm-busy
   time per rank instead of trying to identify parallel-group structure)*
5. additional / open

### Baselines

1. uncompressed (`raw_json`)
2. ScalaTrace
3. TraceZip
4. *(extra)* gzip(JSON) — added because reviewers always ask

### Compression ratio

1. vs every baseline
2. on every trace

### Analysis performance

1. every analysis task
2. every baseline
3. every trace

### Compression info (memory / storage profile)

1. how many templates / what does the template tree look like
2. share of storage: template / timestamp / soft-link edges

### Analysis-performance breakdown (profile)

1. where the time goes inside the analyse step

### Model scalability

1. with #GPUs
2. with #layers
3. with #iterations

### Parallel-analysis scalability

1. compress speed vs #threads / #processes
2. analyse speed vs #threads / #processes

### Ablation

1. storage ablation of the key techniques
2. analysis-performance ablation of the key techniques

---

## 2. Status against the plan

Legend: ✅ done; ⚠️ partial — extra work needed; ❌ not started.

| # | Item | Status | Where it lives | Gap |
|---|---|---|---|---|
| T1 | dense 70B 1024-rank | ✅ | `llama_full` (300 M events, 70 GiB) | – |
| T2 | MoE 671B 1024-rank | ❌ | – | **need a real DeepSeek-V3 / Mixtral profile**; don't have a trace today |
| T3 | ViT | ❌ | – | **need a ViT profile** |
| T4 | LeWorldModel | ✅ | `leworldmodel_full` (3.5 M events) | – |
| T5 | extra | ✅ | `qwen3_full` (256-rank Qwen3), `unifolm_full` (UniFolm world-model) | – |
| A1 | operator hotspot | ✅ | `analysis::operator_hotspot` | – |
| A2 | compute/comm overlap | ✅ | `analysis::compute_comm_overlap`; results in `results/remaining/padoc_5task_analysis.tsv` and `results/remaining/ablation_analyze.tsv` | reports compute_us / comm_us / overlap_us per rank in situ |
| A3 | per-layer operator balance | ✅ | `analysis::layer_operator_balance` | – |
| A4 | per-process load balance | ✅ | `analysis::rank_load_balance` (renamed from `parallel_group`); reports `compute_busy_us` and `comm_busy_us` per rank | – |
| A5 | additional | open | – | nice-to-have: stragglers / slowest-N report |
| B1 | uncompressed | ✅ | `baselines::raw_json` | – |
| B2 | ScalaTrace | ✅ | `baselines::scalatrace` (cross-rank + per-rank fallback for llama-1024) | cross-rank OOMs at 1024 rank scale; per-rank fallback is what we report |
| B3 | TraceZip | ✅ | `baselines::tracezip` (same arrangement as ScalaTrace) | same OOM caveat |
| B4 | gzip(JSON) | ✅ | `baselines::gzip_json` | – |
| C1 | compression ratio matrix | ✅ | `EXPERIMENTS.md` § 2 | – |
| D1 | analysis performance matrix | ✅ | `EXPERIMENTS.md` § 4 | – |
| E1a | template count + tree shape | ✅ | `EXPERIMENTS.md` § 3 (`templates`, `cpu_templates`, `gpu_templates`, `nodes`, `node_breakdown`) | – |
| E1b | storage share — template / ts / soft-link | ✅ | `results/remaining/on_disk_breakdown.txt`; `examples/inspect_artifact.rs --on-disk` | reports zstd bytes per encoded region for the main artifacts |
| F1 | analyse-time breakdown (profile) | ✅ | `PADOC_ANALYSIS_PROFILE=1`; `results/remaining/analysis_profile_padoc.{jsonl,tsv}` | profiles inner in-situ phases per task |
| G1 | scalability vs #GPUs | ✅ | `scripts/scalability_gpus.sh`; `results/remaining/gpu_scalability.md` | llama subsets for 1/8/64/256 GPUs |
| G2 | scalability vs #layers | ✅ | `padoc bench scalability --dimension layers`; `results/remaining/synthetic_scalability.md` | synthetic sweep at 8/16/32/64/128 layers |
| G3 | scalability vs #iterations | ✅ | `padoc bench scalability --dimension iterations`; `results/remaining/synthetic_scalability.md` | synthetic sweep at 1/2/4/8/16 iterations |
| H1 | compress speed vs threads | ✅ | `scripts/scalability_compress.sh`; `results/remaining/compress_scalability_full.md` | v6 sweep on small datasets plus full llama, workers 1/2/4/8/16/32/64 |
| H2 | analyse speed vs threads | ❌ | – | All four analyses are currently single-threaded. Decision needed: parallelise `rank_load_balance` and `operator_hotspot` over templates / ranks (rayon) or argue single-thread is already 100×–100 000× faster than baselines and skip. |
| I1 | storage ablation | ✅ | `results/remaining/ablation_storage_from_artifacts.tsv` | full PADOC preset storage ablation on leworldmodel/qwen3/unifolm; llama-scale preset ablation remains optional because of cost |
| I2 | analyse-perf ablation | ✅ | `results/remaining/ablation_analyze.tsv` | 3 datasets x 8 presets x 5 tasks |

---

## 3. Why PADOC analyses are faster — the principle

The next agent should be able to read this section and write the *why*
paragraph of the paper without re-running anything. Three ideas, in
order of how often they fire.

### 3.1 Templates instead of events

A baseline (raw / gzip / ScalaTrace / TraceZip) decompresses to a
`Trace` whose central object is `Vec<Event>` — one entry per recorded
event. Every analysis must therefore touch every event:
`O(events)` work, where `events` is 3.5 M for `leworldmodel_full`,
33.8 M for `qwen3_full`, 80.2 M for `unifolm_full`, 301 M for
`llama_full`.

PADOC's `CompressedTrace` keeps a `Vec<Template>` (one entry per
distinct event class) plus a per-rank `Node` tree that points at
templates. Every per-instance numeric / argument / digit value lives
in a column inside the template, not as a standalone event. So an
analysis that wants to *summarise* events touches templates instead:
`O(templates)`, where `templates` is 4 K – 17 K on the small datasets
and **312 on llama**. The structural ratio events / templates is
965 K× on llama; that's why `operator_hotspot` and
`layer_operator_balance` are 10³ – 10⁵ times faster on PADOC.

### 3.2 Constant detection on numeric columns

Every per-instance numeric column on a `MergeEvent` /
`MergeKernelEvent` is one of:

```
NumColumn = Empty
          | Constant { len, value }
          | I32(Vec<i32>)
          | I64(Vec<i64>)
```

`compact()` on each template detects three cases at finalize time:

1. all values equal → `Constant { len, value }` — `sum_i64()` is one
   integer multiply, `get(i)` is `O(1)`.
2. range fits in `i32` after subtracting the start-of-trace timestamp
   → `I32(Vec<i32>)` — half the bytes, half the cache misses.
3. otherwise → `I64(Vec<i64>)` — never happens in practice on the four
   datasets (see § 3 of `EXPERIMENTS.md`: `i64 cols = 0` everywhere).

For analyses that already iterate templates, the per-template work is
either constant-time (case 1) or a tight `i32` sum (case 2). For
analyses that have to look at per-instance values (e.g. instance-level
`pid` or `stream` for `stream_load_balance`), the `Constant` case
makes the inner loop disappear — a homogeneous GPU template
contributes its whole duration sum to one (pid, stream) bucket in
`O(1)`.

### 3.3 Pre-computed structure for layer / rank queries

Two specific tasks would otherwise be expensive even on a template
representation:

* **Layer detection** — every CPU operator name has a trailing layer
  index like `"Linear_13"`. PADOC's `name_nums` columns store those
  trailing digits as a **digit column** (i32 with a width attribute,
  decoded back into the original padded string at read time). Layer
  detection is therefore one column lookup per template, not one
  regex per event. On `llama_full` this collapses
  `layer_operator_balance` from 51.8 s of analyse-only time on
  ScalaTrace to **0.0005 s** on PADOC — a five-order-of-magnitude
  gap.
* **Per-rank busy time** — every rank has its own root node in the
  template tree, with child offsets that group GPU and comm kernels
  together. `rank_load_balance` walks each rank's subtree and adds
  `tmpl.dur_total()` (which is `NumColumn::sum_i64`) into the rank's
  compute or comm bucket, depending on the template's stream
  category. Baselines have to globally sort kernel events by
  `(pid, stream, ts)` before they can produce the same per-rank
  totals, which is what the 19–22 s of analyse time on the baselines
  pays for.

### 3.4 What does **not** speed up

`stream_load_balance` is the lone task whose analyse step is per-event
on both PADOC and the baselines (every kernel event contributes to
exactly one (pid, stream) bucket, no template-level shortcut). Even
there PADOC is 1.6×–3.0× faster on total time because its
load+decompress is 3×–13× faster.

---

## 4. The repo as it is

```
padoc-rs/
├── Cargo.{toml,lock}
├── README.md
├── EXPERIMENTS.md          ← paper tables (§ 1 datasets, § 2 compress, § 3 mem, § 4 analyse, § 5 why, § 6 lossless, § 7 reproduce)
├── HANDOFF.md              ← this file
├── examples/               ← inspect_artifact, load_breakdown, normalize_int_ts, roundtrip_minimal, serialize_bench
├── src/
│   ├── analysis/           ← operator_hotspot, stream_load_balance, layer_operator_balance, parallel_group (= rank_load_balance)
│   ├── baselines/          ← raw_json, gzip_json, scalatrace, tracezip
│   ├── compressor/         ← core, merge, structural, decompress
│   ├── bench/              ← runner, scalability
│   ├── event.rs            ← MergeEvent / MergeKernelEvent / NumColumn / StringColumn / PhaseColumn / ArgColumn / DigitColumn
│   ├── slp.rs              ← name_nums digit packing
│   └── trace.rs            ← (de)serialise CompressedTrace
├── tests/
├── scripts/
│   ├── compress_all.sh     ← wraps `bench compress` for both manifests
│   ├── analyze_small.sh    ← single-process analyse on lewm/qwen3/unifolm
│   ├── analyze_llama.sh    ← single-process padoc + per-rank baseline aggregator on llama
│   ├── inspect_all.sh      ← runs examples/inspect_artifact on every padoc artifact
│   ├── manifest_small.json
│   └── manifest_llama.json
└── results/main/           ← raw TSV / inspect dumps that back EXPERIMENTS.md
    ├── analyze_small.tsv
    ├── analyze_llama_baselines.tsv
    ├── inspect_small.txt
    └── inspect_llama.txt
```

`.gitignore` now ignores root-level `*.log`, `*.tsv`, `*.bin`, `*.zst`, so
ad-hoc experiments don't pollute the tracked tree any more.

Cluster: both `sc1` (32 cores, 256 GiB) and `sc4` (64 cores, 256 GiB)
mount the same NFS, so artifacts under `/mnt/treasure/ljx/artifacts/`
are visible from either host. Use sc4 for `llama_full` and sc1 for
the other three.

---

## 5. Suggested ordering for what's left

Estimated effort + the section it unlocks in the paper.

| Order | Task | Effort | Unlocks |
|---|---|---|---|
| 1 | T2 / T3 MoE / ViT traces | depends on whether we can find one | extra rows for the compression-ratio table |
| 2 | H2 multi-threaded analyse | open design | only if reviewers complain |
| 3 | A5 additional analysis | open | nice-to-have |

The core experiment backlog is now done. The remaining items are either
contingent on missing traces (T2, T3) or optional reviewer-defence /
nice-to-have work (H2, A5).

---

## 6. Notes for the next agent

* **Don't ask for confirmation on routine compute tasks** — the PI
  explicitly said `执行命令不需要我的确认`. Just run things on the
  cluster.
* **Do experiments on the cluster, not locally** — the PI said
  `尽量在集群上做实验，不然我的电脑会变卡`. Use `ssh sc1` / `ssh sc4`
  and `numactl --interleave=all`.
* **Compress speed is not the headline** — the PI said
  `我们不是很关注压缩的性能提升，所以关于压缩的性能，只需要汇报下最快的那个耗时就行了`.
  The headline is **compression+analysis co-design**: smaller in-memory
  representation → faster random-access analysis. Storage breakdown,
  ablation, and analyse speed-up are what carry the paper.
* **Always verify lossless** — every baseline (raw / gzip / ScalaTrace
  / TraceZip) was rewritten to be bit-exact. Before adding any new
  compressor, baseline, or compaction toggle, re-run
  `padoc roundtrip <dir>` on `leworldmodel_full` to confirm
  `LOSSLESS = YES`.
* **The `parallel_group` source file still has its old name** —
  `src/analysis/parallel_group.rs` — but the registered task name is
  `rank_load_balance`. Don't be confused; that's intentional so we
  don't have to rewrite git history.
* **Per-rank fallback for ScalaTrace / TraceZip on llama-1024 is the
  honest baseline** — cross-rank construction OOMs there even with
  256 GiB. Mention this in the paper as a finding ("PADOC scales to
  trace sizes where cross-rank ScalaTrace/TraceZip can't run"), don't
  hide it.

Have fun.
