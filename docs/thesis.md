# 清华大学本科生毕业论文

# 面向大规模 AI 性能剖析数据的结构化压缩与原位分析

英文题目：Structural Compression and In-situ Analysis for Large-scale AI Profiling Traces

院系：`（请填写）`

专业：`（请填写）`

姓名：`（请填写）`

学号：`（请填写）`

指导教师：`（请填写）`

完成日期：2026 年 5 月

密级：公开

---

## 关于学位论文使用授权的说明

本人完全了解清华大学有关保留、使用学位论文的规定，即：学校有权保留学位论文的复印件，允许该论文被查阅和借阅；学校可以公布该论文的全部或部分内容，可以采用影印、缩印或其他复制手段保存该论文。（涉密的学位论文在解密后应遵守此规定）

作者签名：`（请填写）`

导师签名：`（请填写）`

日期：`（请填写）`

---

# 中文摘要

大规模人工智能模型训练和推理过程会产生海量性能剖析轨迹。以 Chrome Trace 或 PyTorch Profiler 导出的 JSON 文件为例，一次千卡训练迭代的轨迹可达到数百 GB 的文本规模，并包含 CPU 算子、GPU kernel、通信 kernel、线程、rank、stream 与运行时参数等多维信息。传统压缩器能够降低存储成本，但通常需要在分析前完整解压并重建事件序列；面向高性能计算通信轨迹的专用压缩器虽然可以获得较高压缩比，却往往弱化了 AI profiler 中对模型层次、CPU-GPU 关联和 rank 级负载均衡分析至关重要的结构信息。本文研究的问题是：如何在保持较高压缩率的同时，将大规模 AI 性能轨迹表示为可直接查询的结构化压缩对象，从而支持不重建原始轨迹的原位分析。

本文设计并实现了 PADOC，一个面向 AI profiler trace 的结构化压缩与原位分析系统。PADOC 将原始事件归并为模板，将每个模板的时间戳、持续时间、参数和名称数字部分存储为类型化列，并构建以 rank 为根的调用树。系统在树中显式保留 CPU launch 与 GPU kernel 之间的关联边，同时使用重复 CPU 子树合并、锚点匹配、参数去重、名称模式归一化和整数列紧凑化等技术减少冗余。基于这一表示，PADOC 支持五类核心分析任务：算子热点、rank 负载均衡、层级 GPU kernel 热点、层级计算通信重叠，以及层级 rank 负载均衡。这三类层级 GPU 任务直接依赖 CPU 树和 CPU-GPU 关联边，因此能够验证结构化压缩对分析语义的作用。

本文在四个真实 AI 工作负载上评估系统：LeWorldModel 推理、Qwen3 稠密训练、UniFolm world-model 训练和 1024 rank 的 LLaMA-70B 训练，事件数从 346.9 万到 3.01 亿，原始轨迹规模从 884.37 MiB 到 69.95 GiB。实验结果表明，PADOC 将这些轨迹压缩到 37.52 MiB 至 2.40 GiB，对应 23.57 倍至 31.00 倍压缩比。虽然 ScalaTrace 在部分数据集上获得更高字节压缩比，但 PADOC 保留可查询结构，并能在最大数据集上以单个合并 artifact 完成五个核心任务，端到端总时间为 109.86 s 至 168.83 s，峰值内存为 34.32 GiB。消融实验显示，在 Qwen3 数据集上，默认 PADOC 可将 1,592,830 个 GPU kernel 引用归因到层级或重复 scope，覆盖率为 88.19%；关闭 CPU-GPU kernel link 后，三类层级 GPU 分析的可归因引用和结果行均降为 0。这说明关联边不是单纯的存储开销，而是层级 GPU 分析的必要语义结构。综合来看，PADOC 支持的核心结论是：面向分析共同设计的结构化压缩表示可以在保持竞争性压缩率的同时，为大规模 AI 性能轨迹提供可扩展的原位分析能力。

关键词：AI 性能剖析；轨迹压缩；原位分析；结构化压缩；GPU kernel；负载均衡

---

# ABSTRACT

Large-scale artificial intelligence training and inference workloads generate massive profiling traces. Chrome Trace and PyTorch Profiler JSON files contain CPU operators, GPU kernels, communication kernels, ranks, streams, threads and runtime arguments, and a single thousand-GPU training iteration may produce tens or hundreds of gigabytes of trace data. General-purpose compressors reduce storage overhead but usually require full decompression and reconstruction before analysis. Existing trace compressors for high-performance computing can achieve strong byte-level compression, but they often regularize or discard structures that are essential for AI performance analysis, such as model scopes, CPU-GPU launch provenance and rank-level execution hierarchy. This thesis studies how to compress AI profiling traces into an analysis-ready representation that preserves structure and supports in-situ queries without materializing the original event stream.

This thesis presents PADOC, a structural compression and in-situ analysis system for AI profiler traces. PADOC groups raw events into templates, stores per-instance timestamps, durations, arguments and numeric name components in typed columns, and builds a rank-rooted call tree. The tree explicitly preserves CPU launch to GPU kernel provenance links. PADOC further reduces redundancy with repeated CPU subtree compression, anchor matching, argument deduplication, name-pattern normalization and compact integer columns. Based on this representation, PADOC supports five core analysis tasks: operator hotspot, rank load balance, layer-aware GPU kernel hotspot, layer-aware compute-communication overlap, and layer-aware rank balance. The three layer-aware GPU tasks directly rely on the CPU tree and CPU-GPU provenance links, which makes them suitable for validating the semantic value of structural compression.

The system is evaluated on four real AI workloads: LeWorldModel inference, Qwen3 dense training, UniFolm world-model training and a 1024-rank LLaMA-70B training trace. The traces contain 3.47 million to 301.29 million events, with raw sizes from 884.37 MiB to 69.95 GiB. PADOC compresses these traces to 37.52 MiB to 2.40 GiB, corresponding to compression ratios from 23.57x to 31.00x. Although ScalaTrace achieves smaller byte streams on some datasets, PADOC preserves queryable structure and can analyze the largest trace as a single merged artifact. On the LLaMA-70B trace, the five core tasks finish in 109.86 s to 168.83 s end-to-end with 34.32 GiB peak memory. A kernel-link ablation shows that on Qwen3, PADOC attributes 1,592,830 GPU kernel references to layer or repeated scopes with 88.19% coverage, while disabling CPU-GPU links reduces the attributed references and result rows of all three layer-aware GPU analyses to zero. These results support the main conclusion that analysis-aware structural compression can provide competitive compression ratio and scalable in-situ analysis for large AI profiling traces.

Keywords: AI profiling; trace compression; in-situ analysis; structural compression; GPU kernel; load balance

---

# 目录

[TOC]

---

# 主要符号和缩略语说明

**表 0-1 主要符号和缩略语**

| 符号或缩略语 | 含义 |
|---|---|
| AI | Artificial Intelligence，人工智能 |
| CPU | Central Processing Unit，中央处理器 |
| GPU | Graphics Processing Unit，图形处理器 |
| NCCL | NVIDIA Collective Communications Library，GPU 集合通信库；本文也用该类 kernel 名称泛指通信 kernel |
| Rank | 分布式训练中的进程或设备编号，通常对应一个 GPU 或一个 worker |
| Stream | GPU 上的异步执行队列 |
| Kernel | 在 GPU 设备上执行的函数调用 |
| Trace | 性能剖析轨迹，由事件、时间戳、持续时间和参数组成 |
| Template | PADOC 中的事件模板，表示一组归一化后结构相同的事件 |
| Artifact | PADOC 压缩后生成的持久化文件 |
| 原位分析 | 直接在压缩表示上执行分析，而不完整解压为原始事件序列 |
| RSS | Resident Set Size，进程常驻内存峰值 |

---

# 第 1 章 绪论

## 1.1 研究背景

近年来，大规模 AI 模型的训练和推理系统不断扩展。以大语言模型和世界模型为代表的工作负载通常运行在多 GPU、多机和多并行维度环境中。为了定位性能瓶颈，开发者会使用 PyTorch Profiler、Chrome Trace、系统采样器或厂商 profiler 收集执行轨迹。这类轨迹记录 CPU 侧算子调用、运行时 launch、GPU kernel、通信 kernel、线程和 stream 等事件，并附带时间戳、持续时间、参数、相关 ID 等元数据。轨迹数据可以支持热点分析、计算通信重叠分析、rank 间负载均衡分析和模型层级瓶颈定位，是优化训练吞吐和推理延迟的重要依据。

然而，AI profiler 轨迹的规模增长很快。一个单卡或少量 GPU 的调试 trace 已可能达到数百 MB；当系统扩展到数百或上千 GPU 时，单次剖析就可能产生数十 GB 甚至更大规模的 JSON 文件。本文实验中的 LLaMA-70B 训练轨迹包含 301,288,116 个事件，原始大小为 69.95 GiB。如此规模的数据带来三个直接问题。第一，轨迹存储和传输成本高，原始 JSON 不适合作为长期保存格式。第二，分析前完整加载和解压会产生远高于磁盘文件的内存占用，导致普通工作站无法处理。第三，许多分析任务并不需要逐条扫描所有原始事件，而是需要沿模型层、rank、stream 或 CPU-GPU launch 关系进行结构化访问；如果压缩格式只面向字节流而不保留这些结构，分析仍然必须重建全部事件。

因此，AI profiler trace 压缩不能只追求最小文件大小。更实用的目标是压缩与分析共同设计：压缩表示既要尽可能减少冗余，又要保留能够支撑常见分析的结构索引，使分析任务可以直接在压缩表示上运行。本文围绕这一目标设计和实现 PADOC 系统。

## 1.2 问题定义

设原始性能轨迹为事件集合 $E=\{e_1,e_2,\ldots,e_n\}$。每个事件包含名称、类别、阶段、进程、线程、时间戳、持续时间和参数。普通压缩器将 $E$ 编码为字节流 $B$，分析时需要先恢复 $E$。本文希望构造一个压缩表示 $C$，满足以下性质：

1. 信息保持：从 $C$ 能够恢复原始事件的关键字段，满足 lossless 或实验所需的 bit-exact round-trip 验证。
2. 空间有效：$|C|$ 相比原始 JSON 显著降低，并与专用轨迹压缩器保持竞争性。
3. 结构可查询：$C$ 显式保留事件模板、rank 树、CPU-GPU launch 关联和重复 scope 等结构，使常用分析不必完整物化 $E$。
4. 扩展性：系统可以处理数百到上千 rank 的真实训练 trace，且压缩和分析的时间、内存开销可控。

在这一问题下，本文重点回答三个研究问题：

1. 如何将 AI profiler trace 中重复的事件名称、参数、时间列和调用结构压缩为统一表示。
2. 如何在压缩表示上实现热点、rank 负载和层级 GPU 分析任务。
3. 结构化信息对分析是否必要，以及它与压缩率、加载时间和内存占用之间存在什么权衡。

## 1.3 研究挑战

AI profiler trace 与传统通信 trace 或通用日志相比有不同特点。

第一，事件类型多且字段异构。CPU 侧可能包含 Python、ATen、runtime launch、autograd、框架辅助操作等事件；GPU 侧包含计算 kernel、NCCL 通信 kernel 和设备侧 runtime 事件。不同事件的参数键、id 字段和名称模式不一致。

第二，事件数量远大于模板数量。大量事件在不同 iteration、layer、rank 和 stream 上重复出现，名称中仅层号、编号或参数发生变化。若压缩器不能识别这种模式，会重复存储大量冗余字符串和参数。

第三，CPU 与 GPU 的语义关系不等同于时间包含关系。一个 CPU launch 事件通过 correlation id 对应一个或多个 GPU kernel；GPU kernel 运行在独立 stream 上，时间上可能与 CPU 调用错开。层级分析需要知道“某个模型 layer 下 launch 了哪些 GPU kernel”，仅靠 GPU 时间戳排序无法稳定回答这一问题。

第四，分析任务的访问维度不同。算子热点按模板聚合，rank 负载按 rank 树聚合，层级 overlap 需要从 layer scope 收集 kernel 并进行区间合并。这些任务对压缩表示的要求不同，单一的顺序编码难以兼顾。

## 1.4 本文贡献

本文的主要贡献如下。

1. 提出并实现一种面向 AI profiler trace 的结构化压缩表示。该表示以模板列为基本单位，以 rank-rooted node tree 保存调用结构，并显式保留 CPU launch 到 GPU kernel 的 provenance link。
2. 实现完整 Rust 系统 PADOC，包括 Chrome Trace 读取、模板构建、结构合并、列压缩、zstd 包装持久化、baseline 压缩器和 benchmark harness。
3. 设计五个核心原位分析任务，覆盖模板聚合、rank 级聚合和层级 GPU 访问三个维度，其中层级 GPU 任务直接使用 CPU-GPU link 验证结构信息的必要性。
4. 在四个真实 AI 工作负载上完成系统评估。实验覆盖 346.9 万到 3.01 亿事件、2 到 1024 rank、884.37 MiB 到 69.95 GiB 原始数据规模。
5. 通过消融实验、存储拆解和扩展性实验分析系统权衡。实验表明，PADOC 并不总是产生最小字节流，但保留的结构能够支持原位分析，并在关闭关键 link 后导致层级 GPU 分析语义失效。

## 1.5 论文结构

第 2 章介绍相关工作。第 3 章描述 PADOC 的数据模型和压缩流程。第 4 章介绍原位分析任务及其复杂度。第 5 章给出实验设计。第 6 章展示实验结果与分析。第 7 章讨论局限和未来工作。第 8 章总结全文。

---

# 第 2 章 相关工作

## 2.1 AI 性能剖析和 Chrome Trace

AI 框架通常提供 profiler 用于记录模型执行过程。PyTorch Profiler 可以导出 Chrome Trace 兼容 JSON，包含 CPU operator、runtime 调用、GPU kernel、通信 kernel 和 metadata<sup>[1]</sup>。Chrome Trace Event Format 使用事件数组表达不同 phase、pid、tid、timestamp 和 args，格式通用且易于可视化<sup>[2]</sup>，但并不面向大规模离线存储优化。JSON 文本格式在可读性和工具兼容性方面有优势，但字段名、字符串和参数对象重复度高，直接保存会产生较大空间开销。

对于 AI 系统优化，单纯的 timeline 可视化不足以解决全部问题。研究者和工程师通常还需要回答更结构化的问题，例如最耗时的 operator 是什么，某些 rank 是否存在 straggler，某个模型 layer 内的计算与通信是否重叠，以及某类 GPU kernel 是否集中出现在特定 scope。本文的出发点是将这些分析需求前移到压缩表示设计中。

## 2.2 通用压缩与序列化

gzip、zstd 等通用压缩器在日志和 trace 存储中广泛使用<sup>[3][4]</sup>。它们对重复字符串和局部相似字节序列有较好效果，部署简单，且可以无损恢复原始文件。然而，通用压缩器输出的是字节流，不包含针对 trace 语义的索引。分析任务通常需要先解压整个文件，再将 JSON 解析为事件对象。对于数十 GB 轨迹，这一步会消耗大量时间和内存。

MessagePack 等二进制序列化格式可以减少 JSON 字段名和数字文本表示带来的开销<sup>[5]</sup>，但如果仍然逐事件保存对象，分析时仍需处理完整事件向量。PADOC 采用 msgpack 加 zstd 的持久化方式，但核心压缩收益并非来自序列化格式本身，而来自模板化、列式存储和结构树。

## 2.3 性能轨迹压缩

高性能计算领域长期关注通信 trace 压缩。ScalaTrace 等方法利用大规模并行程序中通信模式的重复性，将事件序列归纳为 regular section descriptor 或类似结构，从而压缩跨 rank 重复行为<sup>[6]</sup>。Score-P 等性能测量基础设施也说明了统一 trace 采集和分析接口在高性能计算中的重要性<sup>[7]</sup>。TraceZip 类方法也通过识别重复序列和结构模式减少 trace 大小。这些方法证明了结构化压缩在性能轨迹中的价值。

但 AI profiler trace 与 MPI 通信 trace 的分析对象不同。AI trace 中 GPU kernel 与 CPU launch 的 correlation、模型 layer 的重复 scope、stream 级并发和 rank 级负载均衡都具有重要语义。若压缩器过度合并或只保留可回放序列，就可能无法直接支持“某个 layer 下有哪些 GPU kernel”这类问题。本文并不否定现有 trace 压缩器的字节压缩能力，而是在此基础上强调面向 AI 分析的结构保留。

## 2.4 原位分析与近数据处理

原位分析的核心思想是在数据产生或数据压缩后的本地表示上直接计算分析结果，避免昂贵的数据搬移和全量重建。在数据库和科学计算中，列式存储、压缩域计算和近数据处理都体现了类似思想。对 profiler trace 而言，如果分析任务可以在模板列、树节点和索引上完成，就无需将每个事件恢复为独立对象。

PADOC 将这一思想应用于 AI profiler trace。对于算子热点，系统直接遍历模板并调用每个模板的持续时间列求和。对于 rank 负载，系统沿 rank 树聚合 GPU kernel。对于层级 GPU 分析，系统沿 CPU scope 和 CPU-GPU link 收集 kernel，然后做热点、overlap 或 rank balance 计算。其共同点是分析对象从原始事件序列转为压缩结构。

## 2.5 本文工作的定位

本文工作位于通用压缩、trace 专用压缩和性能分析工具之间。与 gzip/zstd 相比，PADOC 不是黑盒字节流压缩，而是显式建模 trace 语义。与 ScalaTrace/TraceZip 相比，PADOC 不单纯追求最小字节数，而是保留 AI profiler 中对原位分析有价值的结构。与可视化 profiler 相比，PADOC 关注离线压缩和批量分析，目标是让大规模 trace 可以被长期保存和快速查询。

---

# 第 3 章 PADOC 系统设计

## 3.1 设计目标

PADOC 的设计目标包括四点。

第一，保持无损。压缩 artifact 应能够恢复原始事件关键字段，包括名称、时间戳、持续时间、进程、线程、phase、参数和 id 等。实验中所有 compressor 均通过 round-trip 验证。

第二，减少冗余。AI trace 中大量事件共享名称模式、参数键和结构位置。PADOC 应将这些重复信息归并到模板和结构节点中，只为每个实例保存必要的变化列。

第三，支持原位分析。压缩表示应提供直接访问模板、rank 树、GPU kernel 和 CPU-GPU provenance 的接口，使分析任务无需重建完整 `Trace`。

第四，工程可扩展。系统需要处理多 rank 目录、超大 JSON 文件和数百 GB 中间数据，避免单次读取全部 JSON 树导致内存爆炸。

## 3.2 输入数据模型

PADOC 输入为 Chrome Trace 风格事件。一个事件主要包含以下字段：`name` 表示事件名称，`ts` 为时间戳，`dur` 为持续时间，`cat` 为类别，`ph` 为 phase，`pid` 和 `tid` 表示进程和线程或 stream，`args` 保存任意参数，`id`、`bp`、`s` 保存异步事件或其他 trace 字段。系统将每个 trace 文件视作一个 rank，或从 `distributedInfo.rank` 中读取 rank id。每个 rank 内部按 `(pid, tid, ph)` 分组，得到原始流集合。

GPU stream 事件通常通过 `args.stream` 或 profiler 的 GPU 线程命名识别。CPU launch 与 GPU kernel 的关系主要依赖 correlation id。PADOC 在构建调用树时读取这些 id，并把 launch 事件与对应 kernel 关联为结构节点。

## 3.3 模板与列式存储

PADOC 将共享相同归一化签名的事件归并为模板。签名包含归一化后的名称、类别、`bp`、`s` 和参数键集合。名称归一化会将名称中的数字部分抽出，例如不同 layer 编号或算子实例编号可以共享同一 `name_pattern`，具体数字保存在 `name_nums` 列中。

每个模板保存不随实例变化的字段，以及随实例变化的列。数值列使用如下四种表示：

```text
NumColumn = Empty
          | Constant { len, value }
          | I32(Vec<i32>)
          | I64(Vec<i64>)
```

构建阶段先以可追加形式收集数值，finalize 阶段执行紧凑化。如果一列所有值相同，则转为 `Constant`；否则若数值范围在时间戳归一化后可放入 32 位整数，则转为 `I32`；只有不满足条件时才保留 `I64`。在本文四个主数据集上，最终 artifact 中 `i64` 列数量均为 0。持续时间求和可以直接在列上执行：常量列为一次乘法，`I32` 列为紧凑顺序求和。

字符串和参数也采用列式表示。参数键由模板保存，每个参数值按实例列存储并进行去重。名称中的数字部分使用 digit column 记录，使不同 layer 或编号事件可以共享模板名。

## 3.4 结构树

PADOC 的核心结构是 rank-rooted node tree。Rust 实现中使用统一的 `Node` 枚举表示不同节点：

1. `Root`：rank、pid、tid、phase 下的根节点。
2. `Cpu`：单个 CPU 事件实例，保存模板 id、实例 id 和子节点。
3. `SameCpu`：多个共享相同模板且结构相似的 CPU 实例，用一个节点表示重复 scope。
4. `Gpu`：未与 CPU launch 配对的 GPU 事件集合。
5. `KernelLaunch`：一个 CPU launch 与一个 GPU kernel 的关联。
6. `KernelsLaunch`：多个共享 launch 模板的 CPU-GPU kernel 关联集合。

**图 3-1 PADOC 压缩表示示意**

```text
CompressedTrace
  templates:
    T0: CPU template, columns(ts, dur, args, name_nums)
    T1: GPU template, columns(ts, dur, pid, stream, args)
  ranks:
    rank 0
      Root
        SameCpu(layer/block scope)
          Cpu(runtime launch)
            KernelLaunch(cpu instance -> gpu instance)
```

这棵树同时服务压缩和分析。压缩方面，重复 CPU 子树可以合并为 `SameCpu`，共享子结构通过 anchor matching 减少重复节点。分析方面，rank 负载任务可以从每个 rank root 出发遍历 GPU kernel；层级 GPU 分析可以从 CPU layer scope 出发，沿 `KernelLaunch` 或 `KernelsLaunch` 找到对应 GPU kernel。

## 3.5 CPU-GPU 关联边

在 AI profiler 中，CPU 事件和 GPU kernel 并不位于同一时间线。CPU runtime launch 事件通常较短，GPU kernel 在设备 stream 上异步执行。如果分析任务要回答某个 CPU model layer 产生了哪些 GPU kernel，仅靠时间范围包含关系是不可靠的，因为 GPU 执行可能延迟、重叠或跨 stream 排队。

PADOC 使用 profiler 提供的 correlation 信息建立 CPU-GPU provenance link。若一个 CPU launch 对应一个 GPU kernel，系统生成 `KernelLaunch`；若一组 launch 和 kernel 具有相同模板结构，则合并为 `KernelsLaunch`。这些节点在存储上会增加实例引用和结构树开销，但在语义上使层级 GPU 分析成为可能。

需要澄清的是，on-disk breakdown 中的历史字段名容易被误解为仅表示“软连接边”。实际上，`node_instance_refs` 和 `rank_node_tree` 代表节点实例引用和树结构，不只是 CPU-GPU link。对最大 `llama_full` artifact，`rank_node_tree` 和 `node_instance_refs` 分别贡献约 946.11 MB 和 925.15 MB 的 zstd 后区域大小，但它们对应整棵树和实例引用，而不是仅有 kernel link。真正的关键问题不是 link 能否减少字节，而是删除 link 后层级 GPU 分析是否还成立。第 6 章的消融实验表明，关闭 kernel link 后层级 GPU 分析结果行消失，说明该结构具有必要语义。

## 3.6 压缩流程

PADOC 压缩流程包括以下阶段。

1. 读取 trace。对于小文件使用 SIMD JSON 路径，对于大文件使用 streaming parser，避免将完整 JSON 树一次性展开。
2. 构建模板。系统对每个事件计算模板签名，归并到 CPU 或 GPU 模板，并追加实例列。
3. 构建调用树。CPU 事件按时间戳和持续时间组织为栈式调用树；GPU stream 事件按 correlation id 与 CPU launch 配对。
4. 结构压缩。系统识别重复 CPU 子树，形成 `SameCpu`；对共享子节点执行 anchor matching；对不能共享的尾部节点保留 slot。
5. 列紧凑化。对数值列执行常量检测和 `i32` downcast，对参数和名称数字列执行去重或紧凑表示。
6. 序列化。最终 `CompressedTrace` 使用 msgpack 序列化，并通过 zstd 压缩输出 artifact。

压缩率定义为：

$$
R = \frac{S_{\text{raw}}}{S_{\text{artifact}}}
$$

其中 $S_{\text{raw}}$ 为原始 Chrome Trace JSON 大小，$S_{\text{artifact}}$ 为压缩 artifact 大小。

## 3.7 并行压缩

多 rank trace 天然适合分片处理。PADOC 的 parallel pipeline 将每个 rank 或 rank 子集作为工作单元并行读取和压缩，然后进行全局模板合并和 artifact 序列化。实验中压缩线程数从 1 到 64 扫描。结果显示，线程数增加并不总是单调加速：`llama_full` 在 32 workers 达到最快 357.691 s，64 workers 反而退化到 452.036 s。这说明大规模 trace 压缩最终会受到 NFS、内存带宽、merge 和序列化阶段影响。

## 3.8 Baseline 压缩器

本文实现并比较五类压缩器：PADOC、ScalaTrace、TraceZip、gzip_json 和 raw_json。raw_json 保存原始 JSON 等价表示；gzip_json 使用通用 gzip 压缩；ScalaTrace 和 TraceZip 作为结构化 trace 压缩 baseline。所有 baseline 均通过 lossless round-trip 验证。需要强调的是，ScalaTrace 和 TraceZip 的设计目标更接近字节压缩或通信 trace 结构回放，并不保留 PADOC 所需的全部 AI 分析结构。因此，本文不会声称 PADOC 总是最小，而是比较“压缩率”和“分析可用结构”之间的权衡。

---

# 第 4 章 原位分析方法

## 4.1 分析接口

PADOC 的分析模块定义统一接口。每个任务实现 `run_raw(trace)` 和可选的 `run_in_situ(compressed)`。对于 baseline，benchmark harness 先调用对应 compressor 的 decompress，得到完整 `Trace` 后运行 `run_raw`。对于 PADOC，若任务支持原位分析，则直接在 `CompressedTrace` 上运行 `run_in_situ`。

这种接口使同一任务可以比较两种执行路径：重建后扫描和压缩域直接分析。本文最终核心任务均支持 PADOC 原位分析。

## 4.2 算子热点分析

`operator_hotspot` 统计总持续时间最高的 operator 或 kernel 模板。在 raw 路径中，任务遍历所有 stream 中的所有事件，将事件名称归一化后累加 `dur`。其复杂度为 $O(|E|)$。

在 PADOC 原位路径中，任务遍历模板数组，对每个模板调用 `dur_total()`。对于常量持续时间列，`dur_total()` 为一次乘法；对于 `I32` 列，为紧凑数组求和。因此复杂度主要与模板数和数值列长度有关。当模板持续时间列为常量时，大量实例可以在 $O(1)$ 时间贡献总和。本文最大 `llama_full` trace 有 301,288,116 个事件，但模板数仅为 312，因此模板级热点分析非常快。

## 4.3 Rank 负载均衡分析

`rank_load_balance` 分析每个 rank 的 GPU compute 和 communication busy time。任务将 GPU kernel 按名称分类：NCCL 或类似集合通信 kernel 记为通信<sup>[8]</sup>，其余 kernel 记为计算。输出每类时间的 max、min、mean、stddev、cv 和 `(max-min)/mean` 等指标。

Raw 路径需要遍历全部 kernel 事件。PADOC 原位路径首先对每个 GPU 模板分类一次，然后遍历每个 rank root 下的树节点，遇到 `Gpu`、`KernelLaunch` 或 `KernelsLaunch` 时按模板和实例 id 读取持续时间并累加。该任务访问维度是 rank，而不是全局事件序列。它验证了 rank-rooted tree 对负载分析的价值。

负载不均衡指标定义为：

$$
I = \frac{\max_r x_r - \min_r x_r}{\frac{1}{N}\sum_{r=1}^{N}x_r}
$$

其中 $x_r$ 为 rank $r$ 的计算或通信忙碌时间，$N$ 为 rank 数。

## 4.4 层级 GPU kernel 热点

`layer_kernel_hotspot` 统计每个模型 layer 或重复 scope 内最耗时的 GPU kernel。任务先识别 CPU 侧 layer scope。若事件名称中显式包含 `layer.0`、`Block0`、`ViTLayer0` 等模式，系统通过 `name_nums` 解码具体 layer 编号；若没有显式 layer 名称，则对出现次数在阈值范围内的 `SameCpu` 重复 scope 使用 `scope#idx` 作为层级实例标识。

识别到 active layer 后，任务沿该 CPU 子树向下遍历，收集 `Gpu`、`KernelLaunch` 和 `KernelsLaunch` 中的 GPU kernel。对于每个 `(layer, kernel_template)`，累加次数和持续时间，最后输出 top-k 热点。若没有 CPU-GPU link，GPU kernel 无法稳定归因到 CPU layer，任务结果会退化为空或只剩无意义的低覆盖结果。

## 4.5 层级计算通信重叠

`layer_compute_comm_overlap` 分析每个 layer 或 scope 内 GPU compute kernel 与 communication kernel 的时间重叠。任务同样先从 CPU layer scope 收集 GPU kernel，然后按 rank 和 layer 分组。对每组 kernel，系统将非通信 kernel 区间放入 compute 集合，将通信 kernel 区间放入 comm 集合，并执行区间合并。

设合并后的计算区间集合为 $C$，通信区间集合为 $M$，二者覆盖的时间集合分别为 $U(C)$ 和 $U(M)$。本文使用重叠时间和总时间等指标描述 overlap：

$$
T_{\text{overlap}} = |U(C) \cap U(M)|
$$

该任务比模板热点更重，因为它不仅要归因 kernel，还要进行区间排序和合并。实验中它是四个大数据集上最慢或接近最慢的核心任务。

## 4.6 层级 Rank 负载均衡

`layer_rank_balance` 统计每个 layer 或 scope 在不同 rank 上的 GPU compute 和 communication 时间差异。它与 rank_load_balance 的区别是增加了 layer 维度：同一个全局 rank 负载可能看似均衡，但某些 layer 内部可能存在局部 straggler。任务输出每个 layer 下不同 rank 的计算、通信时间统计。

该任务验证了 PADOC 的组合访问能力：它既需要 CPU layer scope，又需要 GPU kernel link，还需要 rank 根结构。如果压缩表示只保存全局事件序列或只保存模板频次，就难以高效回答这一问题。

## 4.7 复杂度分析

设事件数为 $|E|$，模板数为 $|T|$，结构树节点数为 $|N|$，层级 GPU 可归因 kernel 引用数为 $|G_L|$。raw 路径通常需要 $O(|E|)$ 扫描；若还需要按时间或 stream 排序，则可能需要 $O(|E|\log |E|)$ 或额外内存。

PADOC 原位路径的复杂度与任务访问维度相关。`operator_hotspot` 主要为 $O(|T|)$ 加列求和。`rank_load_balance` 为 $O(|T| + |N|)$。三类 layer-aware 任务为 $O(|N| + |G_L|)$，其中 overlap 还包含每个 `(rank, layer)` 分组内的区间排序和合并。由于 $|T|$ 通常远小于 $|E|$，模板聚合任务收益最大；层级任务虽然更重，但其访问对象仍为结构化 link 和可归因 kernel，而不是无差别重建全部事件。

---

# 第 5 章 实验设计

## 5.1 实验环境

实验在 NUMA-balanced 集群节点上运行。小到中等规模数据使用 sc1 节点，配置为 32 NUMA cores 和 256 GiB 内存；大规模 LLaMA 数据使用 sc4 节点，配置为 64 NUMA cores 和 256 GiB 内存。实验命令使用 `numactl --interleave=all` 以降低 NUMA 分配偏差。trace 文件和 artifact 存储在共享 NFS 路径 `/mnt/treasure/ljx/`。Rust 项目使用 release 构建。

由于 Markdown 初稿不包含最终答辩版硬件表，转 LaTeX 前应补充 CPU 型号、内存频率、磁盘或 NFS 配置、Rust 编译器版本和操作系统版本。

## 5.2 数据集

本文使用四个真实 AI 工作负载。它们覆盖推理、稠密训练、world-model 训练和千卡大模型训练。

**表 5-1 数据集概况**

| 数据集 | 工作负载 | Ranks / GPUs | 事件数 | 原始大小 |
|---|---|---:|---:|---:|
| `leworldmodel_full` | LeWorldModel inference | 2 | 3,469,389 | 884.37 MiB |
| `qwen3_full` | Qwen3 dense training | 256 | 33,813,574 | 6.91 GiB |
| `unifolm_full` | UniFolm world-model training | 4 | 80,223,071 | 22.43 GiB |
| `llama_full` | LLaMA-70B training | 1024 | 301,288,116 | 69.95 GiB |

四个数据集均为整数微秒时间戳。数据规模差异较大，有利于观察 PADOC 在不同 rank 数、不同事件密度和不同模型结构下的表现。

## 5.3 对比方法

实验比较以下方法。

1. PADOC：本文系统，输出结构化压缩 artifact，支持原位分析。
2. ScalaTrace：结构化 trace 压缩 baseline。
3. TraceZip：重复 trace 结构压缩 baseline。
4. gzip_json：对 JSON 表示使用 gzip 压缩。
5. raw_json：保存原始 JSON 等价表示。

对于压缩实验，报告 artifact 大小、压缩比、最快压缩时间和吞吐。对于分析实验，PADOC 直接读取 `CompressedTrace` 并执行原位分析；baseline 需先 decompress 到 raw trace 后运行相同 raw 任务。对于新引入的 layer-aware GPU 任务，本文不报告跨 compressor speedup，因为 baseline 尚未针对等价 CPU-GPU attribution 逻辑重新实现和测量；本文使用 PADOC 原位时间和 kernel-link 消融验证这些任务。

## 5.4 核心分析任务

本文最终核心分析任务如下。

**表 5-2 核心分析任务**

| 任务 | 分析目标 | 依赖结构 |
|---|---|---|
| `operator_hotspot` | 按总持续时间输出热点 operator/kernel 模板 | 模板列 |
| `rank_load_balance` | 比较不同 rank 的 GPU 计算和通信时间 | rank-rooted node tree |
| `layer_kernel_hotspot` | 输出每个 layer 或重复 scope 内的 GPU kernel 热点 | CPU tree + CPU-GPU kernel link |
| `layer_compute_comm_overlap` | 分析每个 layer 内计算与通信区间重叠 | CPU tree + CPU-GPU kernel link |
| `layer_rank_balance` | 分析每个 layer 在不同 rank 上的负载差异 | CPU tree + CPU-GPU kernel link |

早期实验中的 `stream_load_balance`、`layer_operator_balance` 和 global `compute_comm_overlap` 不作为核心结果。`stream_load_balance` 更多描述 stream 并行机会而非负载均衡，`layer_operator_balance` 主要依赖名称模式而非结构树，global overlap 已由 layer-aware overlap 替代。

## 5.5 评价指标

压缩效果使用 artifact 大小和压缩比评价。压缩性能使用最快时间和吞吐评价。分析性能使用 read time、deserialize 或 decompress time、analyze time、total time 和 peak RSS 评价。结构消融使用可归因 GPU refs、总 GPU refs、coverage 和 result row count 评价。存储拆解使用各 encoded region 的 zstd 后字节数评价。扩展性实验使用 GPU 数、压缩线程数、synthetic layers 和 synthetic iterations 的扫描结果评价。

## 5.6 实验可复现性

本项目的主要实验结果保存在 `results/remaining/` 下。综合结果文件为 `results/remaining/paper_results_summary.md`，核心 layer-aware 分析结果为 `results/remaining/core_layer_analysis.tsv`，kernel-link 消融结果为 `results/remaining/core_kernel_link_coverage.tsv` 和 `results/remaining/core_kernel_link_ablation.tsv`。压缩和分析命令由 `scripts/` 目录下脚本驱动，Rust benchmark harness 负责统一输出 TSV 和 Markdown 报告。

---

# 第 6 章 实验结果与分析

## 6.1 压缩效果

PADOC 在四个数据集上的压缩结果如表 6-1 所示。最佳 workers 来自线程数扫描。

**表 6-1 PADOC 压缩结果**

| 数据集 | PADOC artifact | 压缩比 | 最佳 workers | 最快压缩时间 | 吞吐 |
|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 37.52 MiB | 23.57x | 2 | 13.687 s | 64.6 MB/s |
| `qwen3_full` | 272.23 MiB | 26.00x | 16 | 38.413 s | 184.3 MB/s |
| `unifolm_full` | 741.08 MiB | 31.00x | 16 | 199.686 s | 115.0 MB/s |
| `llama_full` | 2.40 GiB | 29.18x | 32 | 357.691 s | 200.2 MB/s |

四个数据集的压缩比位于 23.57 倍至 31.00 倍之间。`llama_full` 原始大小为 69.95 GiB，压缩后为 2.40 GiB，说明 PADOC 可以将千卡级 trace 保存为可管理的单个 artifact。相比 raw JSON，压缩后的存储和传输成本显著降低。

与 baseline 的压缩比比较见表 6-2。

**表 6-2 不同压缩器压缩比比较**

| 数据集 | PADOC | ScalaTrace | TraceZip | gzip_json | raw_json |
|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 23.57x | 60.97x | 32.76x | 21.31x | 1.21x |
| `qwen3_full` | 26.00x | 34.30x | 26.59x | 17.71x | 1.27x |
| `unifolm_full` | 31.00x | 82.39x | 47.50x | 27.70x | 1.25x |
| `llama_full` | 29.18x | 34.94x | 28.24x | 21.59x | 1.30x |

ScalaTrace 在 `leworldmodel_full` 和 `unifolm_full` 上明显更小，在 `qwen3_full` 和 `llama_full` 上也略优或接近。这说明若唯一目标是字节最小化，PADOC 并非总是最优。本文的核心观点不是“PADOC 压缩比最高”，而是“PADOC 在保持竞争性压缩率的同时保留结构，使分析任务可直接运行”。这一点将在后续分析和消融实验中体现。

## 6.2 核心原位分析性能

表 6-3 汇总五个核心任务在 PADOC 上的分析时间。Read + deserialize 是加载 zstd/msgpack artifact 并构造 `CompressedTrace` 的时间；Max analyze time 是五个任务中最慢任务的纯分析时间；Total time range 是各任务端到端时间范围。

**表 6-3 PADOC 核心分析任务性能**

| 数据集 | Read + deserialize | Max analyze time | 最慢任务 | Total time range | Peak RSS |
|---|---:|---:|---|---:|---:|
| `leworldmodel_full` | 1.460 s | 0.446 s | `layer_rank_balance` | 1.462-1.905 s | 0.55 GiB |
| `qwen3_full` | 11.659 s | 5.948 s | `layer_compute_comm_overlap` | 11.672-17.607 s | 5.04 GiB |
| `unifolm_full` | 37.138 s | 8.605 s | `layer_compute_comm_overlap` | 37.173-45.743 s | 14.24 GiB |
| `llama_full` | 109.766 s | 59.063 s | `layer_compute_comm_overlap` | 109.858-168.829 s | 34.32 GiB |

结果显示，PADOC 能够在单进程中加载最大 2.40 GiB 的 `llama_full` artifact，并完成五个核心任务。端到端时间主要由 artifact load 和 deserialize 决定；模板级 `operator_hotspot` 在 `llama_full` 上仅需 0.091 s 分析时间，`rank_load_balance` 需 2.074 s，而三个 layer-aware 任务更慢，尤其 `layer_compute_comm_overlap` 需 59.063 s。这符合第 4 章复杂度分析：layer-aware overlap 需要沿 CPU-GPU link 收集 kernel，并对区间进行合并。

这一结果支持两个结论。第一，结构化压缩并未阻止大规模 trace 的单 artifact 分析，即使最大数据集有 1024 rank 和 3.01 亿事件。第二，分析任务之间的成本差异与访问模式一致：模板聚合最快，rank tree walk 次之，层级 GPU attribution 和 overlap 最重。

## 6.3 CPU-GPU kernel link 消融

为了验证 CPU-GPU kernel link 的语义作用，本文比较默认 PADOC 与 `padoc_no_kernel_links`。该消融关闭 CPU launch 与 GPU kernel 的关联边，但保留其他压缩机制。表 6-4 展示三类 layer-aware GPU 任务的可归因 kernel 引用数。

**表 6-4 Kernel-link 语义消融**

| 数据集 | 默认可归因 GPU refs | 默认覆盖率 | `no_kernel_links` 可归因 refs | 结果 |
|---|---:|---:|---:|---|
| `leworldmodel_full` | 4,637 / 29,589 | 15.67% | 0 / 29,589 | layer-aware rows disappear |
| `qwen3_full` | 1,592,830 / 1,806,096 | 88.19% | 0 / 1,806,096 | layer-aware rows disappear |
| `unifolm_full` | 449,519 / 7,953,432 | 5.65% | 0 / 7,953,432 | layer-aware rows disappear |

`qwen3_full` 是最有代表性的例子，因为 profiler scope 暴露了较完整的重复模型结构。默认 PADOC 能将 1,592,830 个 GPU kernel 引用归因到 layer 或 repeated scope，覆盖率为 88.19%；关闭 kernel link 后，三类 layer-aware 任务的 attributed refs 和 result rows 均为 0。这说明 kernel link 并非只为加速而存在，而是定义了从 CPU 模型结构到 GPU 执行事件的语义映射。

`leworldmodel_full` 和 `unifolm_full` 的覆盖率较低，主要原因是 trace 中存在更多初始化、utility 或框架级 GPU 工作，这些工作不位于清晰的重复模型 scope 下。即便如此，关闭 link 后结果行同样消失，仍验证了机制的必要性。

## 6.4 内存表示分析

表 6-5 展示 PADOC artifact 反序列化后的主要内存统计。

**表 6-5 PADOC 内存表示**

| 数据集 | Templates | Constant cols | i32 cols | i64 cols | Accounted in-memory |
|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 4,094 | 1,748 | 6,511 | 0 | 0.28 GiB |
| `qwen3_full` | 5,498 | 4,880 | 6,194 | 0 | 2.53 GiB |
| `unifolm_full` | 16,897 | 5,147 | 28,909 | 0 | 6.72 GiB |
| `llama_full` | 312 | 4 | 716 | 0 | 22.10 GiB |

所有数据集最终都没有 `i64` 数值列。时间戳经过每 rank 起点归一化后均可放入 `i32`，常量列进一步压缩了重复字段。这解释了为什么内存中不再由原始 `i64` 数组主导。

`llama_full` 的 accounted in-memory 为 22.10 GiB，而进程 peak RSS 为 34.32 GiB。二者差异来自反序列化、zstd 缓冲、allocator 碎片、临时结构和分析中间状态。对论文而言，22.10 GiB 是更能体现核心表示的数据，因为它排除了部分工程实现和运行时开销；34.32 GiB 则代表真实进程峰值。本文在正文中同时报告二者，但强调工程优化如 lazy loading 或 mmap 不改变结构化压缩的核心思想。

## 6.5 磁盘存储拆解

表 6-6 展示各数据集 artifact 的主要 on-disk 贡献区域。

**表 6-6 On-disk storage breakdown**

| 数据集 | Artifact | 主要贡献区域 |
|---|---:|---|
| `leworldmodel_full` | 37.52 MiB | args columns 20.50 MB, node instance refs 13.57 MB, rank node tree 12.95 MB, timestamp columns 4.71 MB |
| `qwen3_full` | 272.23 MiB | timestamp columns 125.50 MB, node instance refs 100.64 MB, rank node tree 100.15 MB, args columns 44.78 MB |
| `unifolm_full` | 741.08 MiB | args columns 352.40 MB, node instance refs 264.15 MB, rank node tree 258.25 MB, timestamp columns 144.34 MB |
| `llama_full` | 2.40 GiB | timestamp columns 1.07 GB, rank node tree 946.11 MB, node instance refs 925.15 MB, args columns 295.25 MB |

对最大数据集，timestamp columns、rank node tree 和 node instance refs 是主要贡献区域。详细拆解见表 6-7。

**表 6-7 `llama_full` 主要 on-disk 区域**

| Region | zstd bytes | Contribution |
|---|---:|---:|
| `ts_columns` | 1,073,867,885 | 41.7% |
| `rank_node_tree` | 946,110,875 | 36.8% |
| `node_instance_refs` | 925,145,515 | 35.9% |
| `args_columns` | 295,247,430 | 11.5% |
| `ids_pids_phases_streams` | 153,426,779 | 6.0% |
| `dur_columns` | 101,498,433 | 3.9% |
| `name_nums` | 2,715,745 | 0.1% |

表中 contribution 不应简单相加到 100%，因为该 breakdown 是对各 region 独立编码后的贡献估计，不是最终 zstd stream 的精确分区。该结果说明，时间戳在磁盘上确实占较大比例。未来可以尝试分段线性预测加残差编码，将残差从 `i32` 降至 `i16` 或 `i8`；但经过 zstd 后是否仍能获得显著收益需要实测。若残差编码破坏了 zstd 对连续模式的识别，最终 artifact 可能未必更小。因此本文不将该优化作为当前贡献，而列为未来工作。

## 6.6 存储和分析消融

PADOC 提供多种 ablation preset，包括关闭结构压缩、关闭 anchor matching、关闭 SLP、关闭参数去重、关闭 kernel link、关闭 name pattern 和 minimal。表 6-8 汇总存储消融。

**表 6-8 存储消融摘要**

| 数据集 | 默认 PADOC | 最小 preset | 最大 preset | 范围 |
|---|---:|---:|---:|---|
| `leworldmodel_full` | 37.5 MiB / 23.57x | `padoc_minimal`, 36.0 MiB / 24.54x | `padoc_no_args_dedup`, 37.6 MiB / 23.55x | narrow |
| `qwen3_full` | 272.2 MiB / 25.99x | `padoc_no_kernel_links`, 259.9 MiB / 27.23x | `padoc_no_anchor`, 275.5 MiB / 25.69x | narrow |
| `unifolm_full` | 741.1 MiB / 30.99x | `padoc_minimal`, 672.5 MiB / 34.16x | `padoc_no_args_dedup`, 743.0 MiB / 30.91x | moderate |

部分 minimal preset 在磁盘上更小。这并不否定 PADOC 设计，反而说明“最小字节流”和“分析友好表示”不是同一个目标。表 6-9 展示 `operator_hotspot` 的分析消融例子。

**表 6-9 `operator_hotspot` 分析消融示例**

| 数据集 | Preset | Artifact | Deserialize/decompress | Analyze | Total | RSS |
|---|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | default | 37.5 MiB | 1.4287 s | 0.0024 s | 1.4853 s | 0.55 GiB |
| `leworldmodel_full` | minimal | 36.0 MiB | 2.5881 s | 0.0026 s | 2.6379 s | 1.40 GiB |
| `qwen3_full` | default | 272.2 MiB | 11.1339 s | 0.0100 s | 11.5418 s | 5.04 GiB |
| `qwen3_full` | minimal | 265.5 MiB | 19.2806 s | 0.0103 s | 19.6071 s | 10.60 GiB |
| `unifolm_full` | default | 741.1 MiB | 35.0344 s | 0.0287 s | 36.2310 s | 14.24 GiB |
| `unifolm_full` | minimal | 672.5 MiB | 72.7825 s | 0.0284 s | 73.6399 s | 41.35 GiB |

在三个数据集上，minimal 虽可能更小，但加载更慢且内存更高。例如 `unifolm_full` default 的 RSS 为 14.24 GiB，minimal 为 41.35 GiB。原因是 minimal 去掉部分结构化和紧凑化机制后，虽然 zstd 后字节数下降，但反序列化时需要更大的运行时对象或更低效的访问路径。该结果支持本文的 co-design 观点：PADOC 的目标是压缩表示与分析性能的整体优化，而不是单独最小化磁盘大小。

## 6.7 与历史 baseline 分析速度比较

早期实验比较了四个任务在 PADOC 和 baseline 上的端到端速度：`operator_hotspot`、`stream_load_balance`、`layer_operator_balance` 和 `rank_load_balance`。虽然其中三个任务不再作为最终核心任务，但结果仍可作为背景证据，说明 PADOC 的模板和树表示相较 reconstruct-then-scan baseline 有优势。

**表 6-10 历史任务端到端 speedup**

| 数据集 | operator_hotspot | stream_load_balance | layer_operator_balance | rank_load_balance |
|---|---:|---:|---:|---:|
| `leworldmodel_full` | 2.6x | 2.0x | 2.3x | 2.0x |
| `qwen3_full` | 2.2x | 1.6x | 1.9x | 1.7x |
| `unifolm_full` | 3.0x | 2.3x | 2.6x | 2.3x |
| `llama_full` | 4.0x | 3.0x | 3.5x | 3.2x |

若只看 `llama_full` 的 analyze-only 时间，`operator_hotspot` 为 0.082 s，对比最佳 baseline 106.69 s，约 1,301 倍；`rank_load_balance` 为 2.086 s，对比最佳 baseline 19.30 s，约 9.3 倍。`stream_load_balance` analyze-only 没有明显优势，因为它本身接近逐 kernel 聚合，不能充分利用模板结构。这也解释了为什么最终论文不再将 stream balance 作为核心任务。

本文不会将表 6-10 作为新 layer-aware GPU 任务的跨 compressor 结论，因为 baseline 尚未实现等价的 CPU-GPU attribution。最终核心证据是表 6-3 的 PADOC 原位可扩展性和表 6-4 的结构语义消融。

## 6.8 GPU 数扩展性

为了观察 rank 数增长时的行为，实验从 `llama_full` 中抽取不同 GPU 数的子集，并使用 32 workers 压缩。结果见表 6-11。

**表 6-11 GPU 数扩展性**

| GPUs | Events | Raw size | Artifact | Ratio | Compress time |
|---:|---:|---:|---:|---:|---:|
| 1 | 316,746 | 74.81 MiB | 2.99 MiB | 25.03x | 1.737 s |
| 8 | 2,607,995 | 622.99 MiB | 23.35 MiB | 26.68x | 4.665 s |
| 64 | 19,544,859 | 4.55 GiB | 165.23 MiB | 28.19x | 48.408 s |
| 256 | 75,749,224 | 17.59 GiB | 621.61 MiB | 28.97x | 115.421 s |

随着 GPU 数增加，原始大小和 artifact 大小近似线性增长，压缩比从 25.03x 提升到 28.97x。这说明更多 rank 暴露了更多重复结构，有利于模板和结构共享。

## 6.9 压缩线程扩展性

表 6-12 展示 qwen、unifolm 和 llama 三个数据集的压缩线程扫描。

**表 6-12 压缩线程数扩展性**

| Workers | `qwen3_full` | `unifolm_full` | `llama_full` |
|---:|---:|---:|---:|
| 1 | 290.937 s | 570.182 s | 3153.696 s |
| 2 | 159.692 s | 321.551 s | 1648.617 s |
| 4 | 88.036 s | 200.491 s | 933.970 s |
| 8 | 49.993 s | 203.894 s | 562.100 s |
| 16 | 38.413 s | 199.686 s | 397.718 s |
| 32 | 43.288 s | 206.084 s | 357.691 s |
| 64 | 60.492 s | 206.789 s | 452.036 s |

`qwen3_full` 在 16 workers 最快，`unifolm_full` 在 16 workers 附近饱和，`llama_full` 在 32 workers 最快。64 workers 退化说明压缩 pipeline 中存在不可并行阶段或共享资源瓶颈，包括 NFS 读取、内存带宽、全局模板 merge 和 zstd 序列化。

## 6.10 Synthetic 扩展性

Synthetic 实验改变模型 layers 和 iterations，观察重复结构增长时压缩率变化。

**表 6-13 Synthetic layers / iterations**

| Sweep | Values | 结果 |
|---|---|---|
| Layers | 8, 16, 32, 64, 128 | Events 从 1,216 线性增长到 19,456；压缩比保持在约 28x 至 30x |
| Iterations | 1, 2, 4, 8, 16 | Events 从 304 线性增长到 4,864；压缩比从 20.47x 提升到 30.42x |

结果符合预期：当重复 layer 或 iteration 增加时，模板和结构复用更充分，压缩比保持稳定或提升。Synthetic 实验不能替代真实 trace，但能验证系统对重复维度的扩展趋势。

## 6.11 实验结论

综合实验结果，本文得到以下结论。

1. PADOC 可以将真实 AI profiler trace 压缩到 23.57x 至 31.00x，能够处理 301M events / 1024 ranks 的大规模训练 trace。
2. PADOC 不总是字节最小，但其结构化表示支持原位分析；这是与 ScalaTrace、TraceZip 和通用压缩器的核心区别。
3. 五个核心任务覆盖模板、rank 和 layer-aware GPU 三类访问维度。最大数据集上，任务端到端时间在 109.858 s 至 168.829 s 之间，说明压缩表示可直接用于大规模分析。
4. Kernel-link 消融直接验证了 CPU-GPU provenance link 的必要性。关闭 link 后，layer-aware GPU 分析的可归因引用和结果行降为 0。
5. 存储和分析消融表明，较小 artifact 不一定带来更快加载或更低内存。分析友好结构与字节压缩率之间存在可度量权衡。

---

# 第 7 章 讨论

## 7.1 关于压缩比的解释

本文不将 PADOC 描述为所有场景下压缩比最高的方法。实验中 ScalaTrace 在多个数据集上更小，尤其在重复结构规则的 trace 上优势明显。PADOC 保留 rank tree、CPU-GPU link、参数列和名称数字列，这些信息会占用额外空间。保留这些结构的目的不是压缩字节最少，而是让热点、rank、layer-aware GPU 分析能够在压缩表示上执行。

因此，正确的论文论点应是：PADOC 在保持竞争性压缩率的同时，提供面向分析的结构化压缩表示；其价值通过原位分析时间、kernel-link 消融和 minimal preset 的加载/内存对比体现。

## 7.2 关于层级分析覆盖率

层级 GPU 分析依赖 profiler 中的 CPU scope 和 correlation 信息。如果模型代码或 profiler 没有清晰记录 layer scope，或者大量 GPU 工作发生在初始化、数据搬运、框架 utility 中，可归因覆盖率会降低。`qwen3_full` 覆盖率为 88.19%，适合作为主要展示数据；`leworldmodel_full` 和 `unifolm_full` 覆盖率较低，但仍证明 link 是必要的。

未来可以通过显式模型层注解、框架级 scope 规范或对 profiler output 的预处理提高覆盖率。PADOC 的结构表示可以承接这些更高质量的 annotation。

## 7.3 关于内存占用

磁盘上 `llama_full` artifact 为 2.40 GiB，但加载后 peak RSS 为 34.32 GiB，accounted representation 为 22.10 GiB。这一差距来自三方面。第一，zstd/msgpack 解码需要临时缓冲。第二，Rust 对象、Vec capacity、树节点和哈希结构有运行时开销。第三，分析任务可能分配中间 map、interval 和 JSON 输出。

本文不把这部分作为主要理论贡献，但它是系统工程中的重要问题。未来可以使用 mmap-backed column、lazy decode、arena allocation、按任务加载部分 region 或 streaming analysis 来降低峰值内存。即便如此，当前结果已经能在 256 GiB 节点上分析 1024-rank trace。

## 7.4 关于时间戳进一步压缩

On-disk breakdown 显示时间戳列在最大数据集上贡献约 1.07 GB，是重要优化对象。分段线性拟合加残差编码是一条可行路线：对单调或近线性的时间戳序列保存 segment 参数，并将残差降至 `i16` 或 `i8`。这种方法理论上可以降低内存和磁盘占用，访问时通过 segment index 做 $O(\log k)$ 或 $O(1)$ 定位。

但该方法未必在 zstd 后必然更优。现有 `i32` 时间戳列经过 zstd 已有较强压缩；残差编码可能引入 segment metadata，并改变字节分布。对于本文，稳妥做法是将其列为未来工作，只有在完整实现并比较 artifact size、load time 和 analysis time 后再作为结论。

## 7.5 威胁与局限

本文实验仍有局限。第一，当前可用真实数据集不包含 MoE 和 ViT trace，无法验证专家路由或视觉模型结构对压缩和分析的影响。第二，新 layer-aware GPU 任务尚未对 ScalaTrace、TraceZip、gzip_json 和 raw_json 实现完全等价的 attribution baseline，因此本文不报告其跨 compressor speedup。第三，完整 1024-rank LLaMA 上的 8-preset ablation 成本较高，当前完整 ablation 主要覆盖 leworldmodel、qwen3 和 unifolm。第四，系统目前分析任务主要是单线程，layer-aware overlap 在大 trace 上仍有 59 s 分析时间，存在并行优化空间。

这些局限不影响本文主要结论，但需要在最终论文中如实说明。

## 7.6 未来工作

未来工作包括五个方向。第一，扩展数据集，加入 MoE、ViT 和更多推理服务 trace。第二，实现 lazy loading 和列级 mmap，降低加载峰值内存。第三，对 layer-aware 任务进行 rank 并行和 layer 分组并行，降低 overlap 分析时间。第四，研究时间戳残差编码、node tree delta encoding 和更紧凑的实例引用表示。第五，建立更标准的 profiler annotation 规范，使模型 layer、module、expert 和 pipeline stage 信息能够稳定进入 trace。

---

# 第 8 章 结论

本文研究大规模 AI 性能剖析轨迹的结构化压缩与原位分析问题。针对原始 JSON trace 体积大、完整重建成本高、传统压缩表示缺乏 AI 分析结构的问题，本文设计并实现 PADOC 系统。PADOC 将事件归并为模板，将实例字段保存为类型化列，并构建 rank-rooted node tree；系统显式保留 CPU launch 与 GPU kernel 的 provenance link，使 layer-aware GPU 分析可以直接在压缩表示上执行。

实验在四个真实 AI 工作负载上进行，覆盖最高 301,288,116 个事件和 1024 ranks。PADOC 获得 23.57x 至 31.00x 压缩比，并在最大数据集上以 2.40 GiB artifact 支持五个核心任务的原位分析，端到端时间为 109.858 s 至 168.829 s。Kernel-link 消融显示，关闭 CPU-GPU link 后 layer-aware GPU 分析的可归因引用和结果行均降为 0，证明该结构对分析语义是必要的。存储和分析消融进一步说明，字节最小化与分析友好表示之间存在权衡，PADOC 的贡献在于两者的共同设计。

总体而言，本文证明了面向分析的结构化压缩是处理大规模 AI profiler trace 的有效路线。PADOC 不是单纯的文件压缩器，而是将压缩后的 artifact 作为可查询数据结构，为模型层级瓶颈定位、rank 负载分析和大规模离线性能诊断提供基础。

---

# 参考文献

[1] Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library[C]. Advances in Neural Information Processing Systems, 2019.

[2] The Chromium Projects. Trace Event Format[EB/OL]. https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md.

[3] Collet Y, Kucherawy M. Zstandard Compression and the application/zstd Media Type[S]. RFC 8878, 2021.

[4] Deutsch P. DEFLATE Compressed Data Format Specification version 1.3[S]. RFC 1951, 1996.

[5] MessagePack. MessagePack Specification[EB/OL]. https://github.com/msgpack/msgpack/blob/master/spec.md.

[6] Noeth M, Ratn P, Mueller F, Schulz M, de Supinski B R. ScalaTrace: Scalable compression and replay of communication traces for high-performance computing[C]. Proceedings of the International Parallel and Distributed Processing Symposium, 2007.

[7] Knüpfer A, Rössel C, an Mey D, et al. Score-P: A Joint Performance Measurement Run-Time Infrastructure for Periscope, Scalasca, TAU, and Vampir[C]. Tools for High Performance Computing, 2012.

[8] NVIDIA. NVIDIA Collective Communications Library Documentation[EB/OL]. https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/.

[9] 清华大学图书馆. 本科生论文写作指南[EB/OL]. https://lib.tsinghua.edu.cn/info/1073/1978.htm.

[10] PADOC 项目实验结果汇总. `results/remaining/paper_results_summary.md`, 2026.

---

# 致谢

感谢指导教师在课题选题、系统设计和实验分析方面给予的指导。感谢课题组同学在大规模 trace 收集、集群实验环境维护和性能分析讨论中提供帮助。感谢开源社区提供 Rust、PyTorch、zstd、MessagePack 等基础工具，使本文系统实现和实验验证成为可能。

本节为初稿，最终提交前应根据实际指导教师、合作同学、课题来源和资助情况补充或删改，篇幅控制在学校要求范围内。

---

# 声明

本人郑重声明：所呈交的学位论文，是本人在导师指导下，独立进行研究工作所取得的成果。尽我所知，除文中已经注明引用的内容外，本学位论文的研究成果不包含任何他人享有著作权的内容。对本论文所涉及的研究工作做出贡献的其他个人和集体，均已在文中以明确方式标明。

作者签名：`（请填写）`

日期：`（请填写）`

---

# 附录 A 外文资料的调研阅读报告

本附录为 Markdown 初稿占位。正式论文中可围绕以下外文资料撰写调研阅读报告：

1. PyTorch profiler 和 Chrome Trace Event Format 文档，说明 AI profiler trace 的字段模型和导出格式。
2. ScalaTrace 相关论文，说明 HPC 通信 trace 中重复结构压缩思想。
3. Zstandard 和 MessagePack 文档，说明本文持久化层使用的通用压缩与二进制序列化技术。

调研报告应包含资料来源、核心观点、与本文工作的关系和参考文献。

---

# 附录 B 实验复现命令与结果文件

## B.1 主要结果文件

**表 B-1 结果文件索引**

| 文件 | 内容 |
|---|---|
| `results/remaining/paper_results_summary.md` | 论文结果总表与解释 |
| `results/remaining/core_layer_analysis.tsv` | 五个核心任务在四个数据集上的 PADOC 分析时间 |
| `results/remaining/core_kernel_link_coverage.tsv` | kernel-link 语义消融覆盖率 |
| `results/remaining/core_kernel_link_ablation.tsv` | kernel-link 消融 timing |
| `results/remaining/on_disk_breakdown.txt` | artifact on-disk 区域拆解 |
| `results/remaining/ablation_storage_from_artifacts.tsv` | 存储消融 |
| `results/remaining/ablation_analyze.tsv` | 分析消融 |
| `results/remaining/gpu_scalability.md` | GPU 数扩展性 |
| `results/remaining/compress_scalability_full.md` | 压缩线程扩展性 |
| `results/remaining/synthetic_scalability.md` | synthetic layers / iterations 扩展性 |

## B.2 复现命令

```bash
cargo build --release --bin padoc --example inspect_artifact

# 压缩实验和 artifact 生成
scripts/compress_all.sh

# artifact 内存和磁盘拆解
scripts/inspect_all.sh

# 小数据集分析
scripts/analyze_small.sh

# LLaMA 大数据集分析
scripts/analyze_llama.sh
```

当前最终核心任务由 `src/analysis/mod.rs` 的 registry 暴露，包括 `operator_hotspot`、`rank_load_balance`、`layer_kernel_hotspot`、`layer_compute_comm_overlap` 和 `layer_rank_balance`。

## B.3 数据质量检查

实验汇总中记录的数据质量检查包括：

1. `padoc_5task_analysis.tsv` 为 20 行，对应历史 4 datasets x 5 tasks。
2. `ablation_analyze.tsv` 为 120 行，对应 3 datasets x 8 presets x 5 tasks。
3. `core_layer_analysis.tsv` 为 20 行，对应 4 datasets x 5 current core tasks。
4. `core_kernel_link_coverage.tsv` 为 18 行，对应 3 datasets x 2 presets x 3 layer-aware tasks。

---

# 附录 C 转 LaTeX 前待补充信息

1. 补齐封面中的院系、专业、姓名、学号、指导教师和日期。
2. 按学院要求替换授权说明和声明页模板，并补充签名。
3. 补充最终实验机器的 CPU 型号、操作系统、Rust 版本和存储配置。
4. 将本文中的 ASCII 示意图替换为正式绘图。
5. 按 GB/T 7714 或学院指定格式核对参考文献条目、引用位置和页码。
6. 根据答辩要求决定是否将历史 baseline speedup 表放入正文或附录。

---

# 在学期间参加课题的研究成果

`（请填写已发表论文、录用论文、专利、软件著作权或其他正式成果；如无，按学校说明最终版可删除本节。）`

---

# 综合论文训练记录表

`（该表通常由院系模板单独填写并装订在最后。Markdown 初稿仅保留位置提示，最终版请替换为正式记录表。）`
