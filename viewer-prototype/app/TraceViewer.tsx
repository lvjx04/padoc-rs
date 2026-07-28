"use client";

import { useEffect, useMemo, useState } from "react";

type TreeSummary = {
  id: string;
  step: number;
  start: number;
  duration: number;
  events: number;
  depth: number;
  cpu: number;
  gpu: number;
  hottest: string;
  density: number[];
};

type TraceEvent = {
  id: string;
  name: string;
  start: number;
  duration: number;
  lane: number;
  kind: "framework" | "compute" | "communication" | "runtime";
  detail: string;
};

type GpuEvent = {
  id: string;
  name: string;
  start: number;
  duration: number;
  lane: number;
  kind: "kernel" | "collective" | "memory";
  detail: string;
};

const trees: TreeSummary[] = [
  {
    id: "step-18420",
    step: 18420,
    start: 0,
    duration: 612.4,
    events: 78241,
    depth: 13,
    cpu: 86,
    gpu: 91,
    hottest: "aten::mm",
    density: [3, 5, 7, 9, 10, 7, 11, 8, 9, 6, 4, 3],
  },
  {
    id: "step-18421",
    step: 18421,
    start: 617.9,
    duration: 608.7,
    events: 77618,
    depth: 13,
    cpu: 84,
    gpu: 93,
    hottest: "flash_attn",
    density: [4, 5, 8, 10, 9, 7, 11, 9, 8, 7, 5, 3],
  },
  {
    id: "step-18422",
    step: 18422,
    start: 1232.1,
    duration: 641.9,
    events: 83107,
    depth: 14,
    cpu: 89,
    gpu: 88,
    hottest: "ncclAllReduce",
    density: [3, 6, 9, 11, 8, 8, 10, 7, 12, 8, 4, 3],
  },
  {
    id: "step-18423",
    step: 18423,
    start: 1879.5,
    duration: 614.2,
    events: 78934,
    depth: 13,
    cpu: 85,
    gpu: 92,
    hottest: "aten::mm",
    density: [3, 5, 8, 10, 9, 7, 11, 8, 9, 6, 4, 3],
  },
  {
    id: "step-18424",
    step: 18424,
    start: 2499.2,
    duration: 606.8,
    events: 76902,
    depth: 12,
    cpu: 82,
    gpu: 94,
    hottest: "flash_attn",
    density: [3, 5, 7, 9, 9, 6, 10, 8, 8, 6, 4, 2],
  },
  {
    id: "step-18425",
    step: 18425,
    start: 3112.4,
    duration: 632.5,
    events: 81476,
    depth: 14,
    cpu: 88,
    gpu: 89,
    hottest: "ncclAllReduce",
    density: [4, 6, 8, 11, 9, 7, 10, 8, 12, 7, 5, 3],
  },
];

const eventTemplate: Omit<TraceEvent, "id">[] = [
  { name: "ProfilerStep", start: 0, duration: 100, lane: 0, kind: "framework", detail: "PyTorch profiler iteration boundary" },
  { name: "train_step", start: 1.2, duration: 97.1, lane: 1, kind: "framework", detail: "Forward, backward, optimizer and scheduler" },
  { name: "model.forward", start: 3.4, duration: 38.7, lane: 2, kind: "framework", detail: "Distributed transformer forward pass" },
  { name: "embedding", start: 4.5, duration: 4.8, lane: 3, kind: "compute", detail: "Token and position embedding lookup" },
  { name: "decoder.layers.0–31", start: 10.1, duration: 30.4, lane: 3, kind: "framework", detail: "Collapsed repeated decoder layers" },
  { name: "attention", start: 11.3, duration: 12.8, lane: 4, kind: "compute", detail: "Scaled dot-product attention" },
  { name: "qkv_projection", start: 11.8, duration: 4.1, lane: 5, kind: "compute", detail: "aten::linear → cublasGemmEx" },
  { name: "flash_attn", start: 16.4, duration: 7.0, lane: 5, kind: "compute", detail: "Fused attention kernel" },
  { name: "mlp", start: 25.2, duration: 13.2, lane: 4, kind: "compute", detail: "Gated feed-forward network" },
  { name: "gate_up_proj", start: 25.9, duration: 6.2, lane: 5, kind: "compute", detail: "Fused projection GEMM" },
  { name: "down_proj", start: 32.8, duration: 4.8, lane: 5, kind: "compute", detail: "Tensor-parallel down projection" },
  { name: "loss", start: 39.4, duration: 2.1, lane: 3, kind: "compute", detail: "Cross entropy loss" },
  { name: "autograd::backward", start: 43.5, duration: 42.8, lane: 2, kind: "framework", detail: "Autograd engine backward pass" },
  { name: "AccumulateGrad", start: 44.7, duration: 7.3, lane: 3, kind: "runtime", detail: "Gradient accumulation" },
  { name: "MmBackward", start: 52.8, duration: 13.8, lane: 3, kind: "compute", detail: "Matrix multiplication gradient" },
  { name: "ncclAllReduce", start: 59.1, duration: 6.8, lane: 4, kind: "communication", detail: "Gradient synchronization across ranks" },
  { name: "FlashAttnBackward", start: 67.2, duration: 11.6, lane: 3, kind: "compute", detail: "Fused attention backward kernel" },
  { name: "ncclReduceScatter", start: 79.4, duration: 5.7, lane: 3, kind: "communication", detail: "ZeRO gradient partition" },
  { name: "optimizer.step", start: 87.5, duration: 8.9, lane: 2, kind: "framework", detail: "Fused AdamW parameter update" },
  { name: "cudaLaunchKernel", start: 89.2, duration: 4.3, lane: 3, kind: "runtime", detail: "CUDA runtime dispatch" },
];

const gpuTemplate: Omit<GpuEvent, "id">[] = [
  { name: "embedding_kernel", start: 4.9, duration: 3.8, lane: 0, kind: "kernel", detail: "Embedding lookup dispatched from aten::embedding" },
  { name: "cublasGemmEx", start: 12.1, duration: 3.4, lane: 0, kind: "kernel", detail: "QKV projection matrix multiplication" },
  { name: "flash_fwd", start: 16.8, duration: 6.2, lane: 1, kind: "kernel", detail: "Fused FlashAttention forward kernel" },
  { name: "silu_mul", start: 26.4, duration: 2.9, lane: 0, kind: "kernel", detail: "Fused SiLU gate and elementwise multiply" },
  { name: "cublasLtMatmul", start: 29.7, duration: 6.9, lane: 0, kind: "kernel", detail: "Tensor Core MLP projection" },
  { name: "all_reduce", start: 59.4, duration: 6.1, lane: 2, kind: "collective", detail: "NCCL gradient synchronization across ranks" },
  { name: "flash_bwd", start: 67.6, duration: 10.8, lane: 1, kind: "kernel", detail: "Fused FlashAttention backward kernel" },
  { name: "reduce_scatter", start: 79.7, duration: 5.1, lane: 2, kind: "collective", detail: "NCCL reduce-scatter for sharded gradients" },
  { name: "multi_tensor_adam", start: 89.6, duration: 5.8, lane: 0, kind: "kernel", detail: "Fused AdamW optimizer update" },
  { name: "D2D memcpy", start: 95.8, duration: 1.6, lane: 1, kind: "memory", detail: "Device-to-device parameter copy" },
];

const laneNames = ["root", "framework", "module", "operator", "collective", "kernel"];
const gpuLaneNames = ["GPU 0 · compute", "GPU 0 · stream 7", "GPU 0 · NCCL"];
const number = new Intl.NumberFormat("en-US");

function buildEvents(treeIndex: number): TraceEvent[] {
  const drift = ((treeIndex * 17) % 7) * 0.12;
  return eventTemplate.map((event, index) => ({
    ...event,
    id: `${treeIndex}-cpu-${index}`,
    start: Math.min(98, event.start + (index % 3 === 0 ? drift : 0)),
    duration: Math.max(1, event.duration * (1 + ((treeIndex + index) % 5 - 2) * 0.012)),
  }));
}

function buildGpuEvents(treeIndex: number): GpuEvent[] {
  const drift = ((treeIndex * 11) % 5) * 0.16;
  return gpuTemplate.map((event, index) => ({
    ...event,
    id: `${treeIndex}-gpu-${index}`,
    start: event.start + (index % 2 ? drift : 0),
    duration: event.duration * (1 + ((treeIndex + index) % 3 - 1) * 0.025),
  }));
}

export default function TraceViewer() {
  const [selectedTree, setSelectedTree] = useState(2);
  const [selectedEventId, setSelectedEventId] = useState("2-cpu-15");
  const [zoom, setZoom] = useState(1);
  const [showGpu, setShowGpu] = useState(true);

  const tree = trees[selectedTree];
  const events = useMemo(() => buildEvents(selectedTree), [selectedTree]);
  const gpuEvents = useMemo(() => buildGpuEvents(selectedTree), [selectedTree]);
  const selectedCpuEvent = events.find((event) => event.id === selectedEventId);
  const selectedGpuEvent = gpuEvents.find((event) => event.id === selectedEventId);
  const selectedEvent = selectedCpuEvent ?? selectedGpuEvent ?? events[0];
  const selectedSource = selectedGpuEvent ? "GPU" : "CPU";

  const focusTree = (index: number) => {
    setSelectedTree(index);
    setSelectedEventId(`${index}-cpu-${trees[index].hottest === "ncclAllReduce" ? 15 : 6}`);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === "ArrowLeft") {
        setSelectedTree((current) => {
          const next = Math.max(0, current - 1);
          setSelectedEventId(`${next}-cpu-0`);
          return next;
        });
      }
      if (event.key === "ArrowRight") {
        setSelectedTree((current) => {
          const next = Math.min(trees.length - 1, current + 1);
          setSelectedEventId(`${next}-cpu-0`);
          return next;
        });
      }
      if (event.key === "+" || event.key === "=") {
        setZoom((current) => Math.min(4, current * 2));
      }
      if (event.key === "-") {
        setZoom((current) => Math.max(1, current / 2));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const viewportWidth = `${zoom * 100}%`;
  const eventLeft = (start: number) => `${start}%`;
  const eventWidth = (duration: number) => `${Math.min(duration, 100)}%`;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">PADOC</span>
          <span className="brand-separator">/</span>
          <span className="brand-section">Trace Viewer</span>
          <span className="prototype-pill">interactive prototype</span>
        </div>
        <div className="topbar-actions">
          <span className="index-state"><i /> Index ready</span>
          <button className="quiet-button" type="button">Export view</button>
        </div>
      </header>

      <section className="dataset-header">
        <div>
          <p className="eyebrow">OPEN TRACE</p>
          <h1>qwen3-8b · training_rank109.json</h1>
          <p className="dataset-copy">
            1.04 GB source trace · 14.7M events · 6 indexed call trees
          </p>
        </div>
        <dl className="dataset-stats">
          <div><dt>Visible span</dt><dd>3.74 s</dd></div>
          <div><dt>Indexed in</dt><dd>8.2 s</dd></div>
          <div><dt>Source</dt><dd>Perfetto JSON</dd></div>
        </dl>
      </section>

      <section className="overview panel" aria-label="Full trace overview">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">TRACE MAP</p>
            <h2>One tree in detail. Everything else stays in context.</h2>
          </div>
          <div className="legend">
            <span><i className="legend-detail" /> materialized</span>
            <span><i className="legend-summary" /> indexed summary</span>
          </div>
        </div>

        <div className="ruler ruler-overview" aria-hidden="true">
          {[0, 0.75, 1.5, 2.25, 3, 3.74].map((tick) => <span key={tick}>{tick.toFixed(2)} s</span>)}
        </div>
        <div className="tree-map">
          {trees.map((item, index) => (
            <button
              key={item.id}
              type="button"
              className={`tree-summary ${index === selectedTree ? "is-selected" : ""}`}
              onClick={() => focusTree(index)}
              aria-pressed={index === selectedTree}
              title={`Focus step ${item.step}`}
            >
              <span className="tree-summary-top">
                <strong>step {item.step}</strong>
                <small>{item.duration.toFixed(1)} ms</small>
              </span>
              <span className="density-bars" aria-hidden="true">
                {item.density.map((height, densityIndex) => (
                  <i key={densityIndex} style={{ height: `${height * 2.4}px` }} />
                ))}
              </span>
              <span className="tree-summary-bottom">
                <small>{number.format(item.events)} events</small>
                {index === selectedTree && <em>FOCUS</em>}
              </span>
            </button>
          ))}
        </div>
      </section>

      <section className="viewer panel">
        <div className="viewer-toolbar">
          <div className="focus-title">
            <button type="button" onClick={() => focusTree(Math.max(0, selectedTree - 1))} disabled={selectedTree === 0} aria-label="Previous call tree">←</button>
            <div>
              <p className="eyebrow">MATERIALIZED TREE</p>
              <h2>step {tree.step}</h2>
            </div>
            <button type="button" onClick={() => focusTree(Math.min(trees.length - 1, selectedTree + 1))} disabled={selectedTree === trees.length - 1} aria-label="Next call tree">→</button>
          </div>
          <div className="toolbar-controls">
            <div className="segmented" aria-label="Timeline zoom">
              {[1, 2, 4].map((value) => (
                <button
                  type="button"
                  className={zoom === value ? "active" : ""}
                  key={value}
                  onClick={() => setZoom(value)}
                >
                  {value}×
                </button>
              ))}
            </div>
            <label className="gpu-toggle">
              <input type="checkbox" checked={showGpu} onChange={(event) => setShowGpu(event.target.checked)} />
              <span>GPU streams</span>
            </label>
          </div>
        </div>

        <div className="viewer-grid">
          <aside className="lane-labels" aria-hidden="true">
            <div className="lane-ruler-spacer">CPU call stack</div>
            {laneNames.map((name, index) => <div className="lane-label" key={name}><span>{index}</span>{name}</div>)}
            {showGpu && <>
              <div className="lane-group-label">GPU correlation</div>
              {gpuLaneNames.map((name) => <div className="lane-label gpu-label" key={name}>{name}</div>)}
            </>}
          </aside>

          <div className="timeline-scroll">
            <div className="timeline-content" style={{ width: viewportWidth }}>
              <div className="ruler detail-ruler">
                {[0, 20, 40, 60, 80, 100].map((tick) => (
                  <span key={tick} style={{ left: `${tick}%` }}>{((tick / 100) * tree.duration).toFixed(1)} ms</span>
                ))}
              </div>
              <div className="cpu-lanes">
                {laneNames.map((_, lane) => (
                  <div className="event-lane" key={lane}>
                    {events.filter((event) => event.lane === lane).map((event) => (
                      <button
                        type="button"
                        key={event.id}
                        className={`trace-event ${event.kind} ${selectedEventId === event.id ? "is-active" : ""}`}
                        style={{ left: eventLeft(event.start), width: eventWidth(event.duration) }}
                        onClick={() => setSelectedEventId(event.id)}
                        title={`${event.name} · ${(event.duration / 100 * tree.duration).toFixed(2)} ms`}
                      >
                        <span>{event.name}</span>
                      </button>
                    ))}
                  </div>
                ))}
              </div>
              {showGpu && (
                <div className="gpu-lanes">
                  <div className="correlation-lines" aria-hidden="true">
                    <i style={{ left: "16.8%" }} /><i style={{ left: "59.4%" }} /><i style={{ left: "79.7%" }} />
                  </div>
                  {gpuLaneNames.map((_, lane) => (
                    <div className="event-lane gpu-event-lane" key={lane}>
                      {gpuEvents.filter((event) => event.lane === lane).map((event) => (
                        <button
                          type="button"
                          key={event.id}
                          className={`gpu-event ${event.kind} ${selectedEventId === event.id ? "is-active" : ""}`}
                          style={{ left: eventLeft(event.start), width: eventWidth(event.duration) }}
                          onClick={() => setSelectedEventId(event.id)}
                          title={`${event.name} · ${(event.duration / 100 * tree.duration).toFixed(2)} ms`}
                        >
                          <span>{event.name}</span>
                        </button>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <aside className="inspector">
            <div className="inspector-heading">
              <p className="eyebrow">{selectedSource} EVENT</p>
              <span className={`kind-dot ${selectedEvent.kind}`} />
            </div>
            <h3>{selectedEvent.name}</h3>
            <p>{selectedEvent.detail}</p>
            <dl className="event-stats">
              <div><dt>Start</dt><dd>{(selectedEvent.start / 100 * tree.duration).toFixed(2)} ms</dd></div>
              <div><dt>Duration</dt><dd>{(selectedEvent.duration / 100 * tree.duration).toFixed(2)} ms</dd></div>
              <div><dt>{selectedGpuEvent ? "Stream" : "Self time"}</dt><dd>{selectedGpuEvent ? gpuLaneNames[selectedEvent.lane].replace("GPU 0 · ", "") : `${(selectedEvent.duration / 100 * tree.duration * 0.18).toFixed(2)} ms`}</dd></div>
              <div><dt>{selectedGpuEvent ? "Device" : "Depth"}</dt><dd>{selectedGpuEvent ? "cuda:0" : selectedEvent.lane}</dd></div>
            </dl>
            <div className="arg-block">
              <span>source</span><code>{selectedSource.toLowerCase()}</code>
              <span>rank</span><code>109</code>
              <span>correlation</span><code>#{88210 + selectedEvent.lane * 29}</code>
            </div>
          </aside>
        </div>
      </section>

      <section className="memory-strip">
        <div>
          <p className="eyebrow">BROWSER WORKING SET · PROTOTYPE ESTIMATE</p>
          <p className="memory-lead">Only the focused tree becomes interactive event objects.</p>
        </div>
        <div className="memory-comparison">
          <div className="memory-item baseline">
            <span>Load all events</span>
            <strong>3.8 GiB</strong>
            <i><b style={{ width: "100%" }} /></i>
          </div>
          <div className="memory-item padoc">
            <span>PADOC focus view</span>
            <strong>86 MiB</strong>
            <i><b style={{ width: "8%" }} /></i>
          </div>
        </div>
        <dl className="materialization-stats">
          <div><dt>Materialized</dt><dd>{number.format(tree.events)} events</dd></div>
          <div><dt>Of total</dt><dd>0.56%</dd></div>
          <div><dt>Context retained</dt><dd>6 / 6 trees</dd></div>
        </dl>
      </section>

      <footer>
        <span>Use ← → to switch trees · + − to zoom</span>
        <span>Synthetic data shaped after a distributed training trace</span>
      </footer>
    </main>
  );
}
