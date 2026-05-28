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

本文在四个真实 AI 工作负载上评估系统：LeWorldModel 推理、Qwen3 稠密训练、UniFolm world-model 训练和 1024 rank 的 LLaMA-70B 训练，事件数从 346.9 万到 3.01 亿，原始轨迹规模从 884.37 MiB 到 69.95 GiB。实验结果表明，PADOC 将这些轨迹压缩到 37.17 MiB 至 2.44 GiB，对应 23.79 倍至 31.08 倍压缩比。虽然 ScalaTrace 在部分数据集上获得更高字节压缩比，但 PADOC 保留可查询结构，并能在最大数据集上以单个合并 artifact 完成五个核心任务，端到端总时间为 133.99 s 至 226.27 s，accounted resident representation 为 14.38 GiB。CPU-GPU 映射消融显示，在 Qwen3 数据集上，默认 PADOC 可将 1,592,830 个 GPU kernel 引用归因到层级或重复 scope，覆盖率为 88.19%；运行时通过 correlation id 动态重建映射也可恢复 1,588,915 个引用，覆盖率为 87.98%，但需要额外构建索引并依赖 correlation id 的作用域一致性。综合来看，PADOC 支持的核心结论是：面向分析共同设计的结构化压缩表示可以在保持竞争性压缩率的同时，为大规模 AI 性能轨迹提供可扩展的原位分析能力。

关键词：AI 性能剖析；轨迹压缩；原位分析；结构化压缩；GPU kernel；负载均衡

---

# ABSTRACT

Large-scale artificial intelligence training and inference workloads generate massive profiling traces. Chrome Trace and PyTorch Profiler JSON files contain CPU operators, GPU kernels, communication kernels, ranks, streams, threads and runtime arguments, and a single thousand-GPU training iteration may produce tens or hundreds of gigabytes of trace data. General-purpose compressors reduce storage overhead but usually require full decompression and reconstruction before analysis. Existing trace compressors for high-performance computing can achieve strong byte-level compression, but they often regularize or discard structures that are essential for AI performance analysis, such as model scopes, CPU-GPU launch provenance and rank-level execution hierarchy. This thesis studies how to compress AI profiling traces into an analysis-ready representation that preserves structure and supports in-situ queries without materializing the original event stream.

This thesis presents PADOC, a structural compression and in-situ analysis system for AI profiler traces. PADOC groups raw events into templates, stores per-instance timestamps, durations, arguments and numeric name components in typed columns, and builds a rank-rooted call tree. The tree explicitly preserves CPU launch to GPU kernel provenance links. PADOC further reduces redundancy with repeated CPU subtree compression, anchor matching, argument deduplication, name-pattern normalization and compact integer columns. Based on this representation, PADOC supports five core analysis tasks: operator hotspot, rank load balance, layer-aware GPU kernel hotspot, layer-aware compute-communication overlap, and layer-aware rank balance. The three layer-aware GPU tasks directly rely on the CPU tree and CPU-GPU provenance links, which makes them suitable for validating the semantic value of structural compression.

The system is evaluated on four real AI workloads: LeWorldModel inference, Qwen3 dense training, UniFolm world-model training and a 1024-rank LLaMA-70B training trace. The traces contain 3.47 million to 301.29 million events, with raw sizes from 884.37 MiB to 69.95 GiB. PADOC compresses these traces to 37.17 MiB to 2.44 GiB, corresponding to compression ratios from 23.79x to 31.08x. Although ScalaTrace achieves smaller byte streams on some datasets, PADOC preserves queryable structure and can analyze the largest trace as a single merged artifact. On the LLaMA-70B trace, the five core tasks finish in 133.99 s to 226.27 s end-to-end, with 14.38 GiB accounted resident representation. A CPU-GPU mapping ablation shows that on Qwen3, PADOC attributes 1,592,830 GPU kernel references to layer or repeated scopes with 88.19% coverage, while dynamic correlation lookup reconstructs 1,588,915 references with 87.98% coverage but requires an additional runtime index and depends on correlation-id scope consistency. These results support the main conclusion that analysis-aware structural compression can provide competitive compression ratio and scalable in-situ analysis for large AI profiling traces.

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

需要澄清的是，on-disk breakdown 中的历史字段名容易被误解为仅表示“软连接边”。实际上，`node_instance_refs` 和 `rank_node_tree` 代表节点实例引用和树结构，不只是 CPU-GPU link。对最终 `llama_full` artifact，`rank_node_tree` 和 `node_instance_refs` 分别贡献约 987.71 MB 和 925.15 MB 的 zstd 后 region 大小，但它们对应整棵树和实例引用，而不是仅有 kernel link。真正的关键问题不是“不保存显式 link 就无法分析”，因为也可以在分析时根据 correlation id 重建映射；关键问题是显式保存映射能否提供更稳定、更直接的 layer-aware 查询路径。第 6 章的映射消融围绕这一点展开。

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

对于压缩实验，报告 artifact 大小、压缩比、最快压缩时间和吞吐。对于分析实验，PADOC 直接读取 `CompressedTrace` 并执行原位分析；baseline 需先 decompress 到 raw trace 后运行相同 raw 任务。对于新引入的 layer-aware GPU 任务，本文不报告跨 compressor speedup，因为 baseline 尚未针对等价 CPU-GPU attribution 逻辑重新实现和测量；本文使用 PADOC 原位时间和 CPU-GPU 映射消融验证这些任务。

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

压缩效果使用 artifact 大小和压缩比评价。压缩性能使用最快时间和吞吐评价。分析性能使用 read time、deserialize 或 decompress time、analyze time、total time 和 accounted resident representation 评价。结构消融使用可归因 GPU refs、总 GPU refs、coverage 和 result row count 评价。存储拆解使用各 encoded region 的 zstd 后字节数评价。扩展性实验使用 GPU 数、压缩线程数、synthetic layers 和 synthetic iterations 的扫描结果评价。

## 5.6 实验可复现性

本项目的主要实验结果保存在 `results/remaining/` 下。最终论文使用的综合结果文件为 `results/remaining/final_paper/final_experiment_results.md`，核心分析结果为 `results/remaining/final_paper/core_layer_analysis_sparse_v7.tsv`，磁盘和内存拆解为 `results/remaining/final_paper/on_disk_breakdown_sparse_v7.txt`，补充消融结果为 `results/remaining/final_paper/no_structural_core_ablation.tsv` 和 `results/remaining/final_paper/dynamic_kernel_mapping_ablation.tsv`。压缩和分析命令由 `scripts/` 目录下脚本驱动，Rust benchmark harness 负责统一输出 TSV 和 Markdown 报告。

---

# 第 6 章 实验结果与分析

## 6.1 压缩效果

表 6-1 展示 PADOC 与四类 baseline 的压缩结果。PADOC 数据使用最终 sparse-slot artifact，baseline 数据使用相同 trace 上的 v6 artifact。表中给出每个方法的具体大小和压缩比，避免只报告最佳 baseline。

**表 6-1 不同压缩器压缩效果**

| 数据集 | PADOC | ScalaTrace | TraceZip | gzip_json | raw_json |
|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 37.17 MiB / 23.79x | 14.31 MiB / 60.97x | 28.27 MiB / 32.76x | 42.97 MiB / 21.31x | 732.79 MiB / 1.21x |
| `qwen3_full` | 274.41 MiB / 25.79x | 208.97 MiB / 34.30x | 279.17 MiB / 26.59x | 400.09 MiB / 17.71x | 5.43 GiB / 1.27x |
| `unifolm_full` | 739.05 MiB / 31.08x | 278.82 MiB / 82.39x | 483.62 MiB / 47.50x | 829.34 MiB / 27.70x | 18.01 GiB / 1.25x |
| `llama_full` | 2.44 GiB / 28.72x | 2.00 GiB / 34.94x | 2.48 GiB / 28.24x | 3.24 GiB / 21.59x | 53.63 GiB / 1.30x |

PADOC 的压缩比为 23.79 倍至 31.08 倍。ScalaTrace 在部分数据集上更小，说明若唯一目标是最小字节流，PADOC 并非总是最优。本文的核心观点是压缩与分析共同设计：PADOC 在保持竞争性压缩率的同时保留模板列、rank 树和 CPU-GPU provenance，使后续分析能够直接在压缩表示上运行。

压缩线程数扫描显示，PADOC 对大规模 trace 的压缩可并行化但会在 16 至 32 workers 附近饱和。`qwen3_full` 在 16 workers 最快，为 38.413 s；`unifolm_full` 在 16 workers 最快，为 199.686 s；`llama_full` 在 32 workers 最快，为 357.691 s。64 workers 退化说明 pipeline 后段受到 NFS、内存带宽、全局模板合并和序列化影响。

## 6.2 文件存储拆解

表 6-2 展示最终 PADOC artifact 的 on-disk region breakdown。每个 region 独立序列化并 zstd 压缩后统计，因此该表是贡献剖析，不要求各列加和等于 artifact 大小。

**表 6-2 PADOC on-disk region breakdown**

| 数据集 | Artifact | ts | dur | ids/pids/streams | name nums | args | tree + refs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 37.17 MiB | 4.50 MiB | 0.32 MiB | 0.04 MiB | 0.05 MiB | 19.55 MiB | 24.96 MiB |
| `qwen3_full` | 274.41 MiB | 119.69 MiB | 13.81 MiB | 0.03 MiB | 0.32 MiB | 42.71 MiB | 193.61 MiB |
| `unifolm_full` | 739.05 MiB | 137.66 MiB | 6.36 MiB | 3.66 MiB | 0.89 MiB | 336.07 MiB | 497.06 MiB |
| `llama_full` | 2.44 GiB | 1.00 GiB | 96.80 MiB | 146.32 MiB | 2.59 MiB | 281.57 MiB | 1.78 GiB |

表 6-3 给出若干关键 region 的 msgpack 到 zstd 压缩比。`llama_full` 的 timestamp zstd 后仍有 1.00 GiB，说明时间戳是主要磁盘开销之一；同时 tree + refs 在最大数据集上贡献约 1.78 GiB，说明结构化表示本身也是重要成本。

**表 6-3 关键 region 的原始序列化大小与压缩比**

| 数据集 | ts msgpack / zstd / ratio | dur msgpack / zstd / ratio | args msgpack / zstd / ratio | tree+refs msgpack / zstd / ratio |
|---|---:|---:|---:|---:|
| `leworldmodel_full` | 15.54 / 4.50 MiB / 3.46x | 3.26 / 0.32 MiB / 10.07x | 50.73 / 19.55 MiB / 2.59x | 249.10 / 24.96 MiB / 9.98x |
| `qwen3_full` | 161.15 / 119.69 MiB / 1.35x | 33.66 / 13.81 MiB / 2.44x | 272.99 / 42.71 MiB / 6.39x | 1877.09 / 193.61 MiB / 9.70x |
| `unifolm_full` | 382.20 / 137.66 MiB / 2.78x | 69.46 / 6.36 MiB / 10.91x | 1269.92 / 336.07 MiB / 3.78x | 3340.31 / 497.06 MiB / 6.72x |
| `llama_full` | 1435.98 / 1024.12 MiB / 1.40x | 247.47 / 96.80 MiB / 2.56x | 1942.74 / 281.57 MiB / 6.90x | 12973.13 / 1824.24 MiB / 7.11x |

## 6.3 常驻内存拆解

磁盘文件与分析常驻内存的差距来自两个事实。第一，artifact 在磁盘上经过 zstd 压缩，而内存中需要可随机访问的列、树节点和向量。第二，Rust 对象、`Vec` 元数据、capacity、哈希结构和 allocator 会产生运行时开销。表 6-4 只统计加载后压缩表示自身的 accounted resident size，不包含 transient load buffer。

**表 6-4 PADOC resident representation breakdown（Arena 优化后）**

| 数据集 | Accounted resident | ts | dur | id/pid/stream | arena (tree) | args | name_nums |
|---|---:|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 0.147 GiB | 0.013 GiB | 0.012 GiB | 0.000 GiB | 0.070 GiB | 0.049 GiB | 0.002 GiB |
| `qwen3_full` | 1.298 GiB | 0.126 GiB | 0.096 GiB | 0.044 GiB | 0.523 GiB | 0.488 GiB | 0.021 GiB |
| `unifolm_full` | 3.624 GiB | 0.299 GiB | 0.264 GiB | 0.275 GiB | 1.002 GiB | 1.697 GiB | 0.054 GiB |
| `llama_full` | 10.430 GiB | 1.122 GiB | 0.770 GiB | 0.842 GiB | 3.734 GiB | 2.535 GiB | 1.426 GiB |

PADOC 使用 arena 化调用树存储：将递归 `Node` 树在加载后转换为扁平 `NodeArena`（连续数组 + 索引引用），然后释放原始树。相比优化前（递归 `Vec<Node>` 存储），树结构内存降低约 57%，总 accounted resident 降低 23-38%。以 `llama_full` 为例，accounted resident 为 10.430 GiB，其中 arena (tree) 为 3.734 GiB，args 为 2.535 GiB，name_nums 为 1.426 GiB，时间戳列为 1.122 GiB。

## 6.4 核心原位分析性能

表 6-5 展示五个核心任务在最终 PADOC artifact 上的端到端时间。加载到内存时间定义为 `read_secs + decompress_secs`，分析时间是任务本身的时间。

**表 6-5 核心任务总体性能**

| 数据集 | Artifact | 加载到内存 | 最长分析时间 | 最慢任务 | 端到端范围 | Resident |
|---|---:|---:|---:|---|---:|---:|
| `leworldmodel_full` | 37.17 MiB | 3.075 s | 0.567 s | `layer_kernel_hotspot` | 3.096-3.641 s | 0.237 GiB |
| `qwen3_full` | 274.41 MiB | 14.021 s | 9.126 s | `layer_compute_comm_overlap` | 14.147-23.146 s | 1.899 GiB |
| `unifolm_full` | 739.05 MiB | 87.547 s | 12.920 s | `layer_kernel_hotspot` | 88.070-100.468 s | 4.678 GiB |
| `llama_full` | 2.44 GiB | 133.878 s | 92.393 s | `layer_compute_comm_overlap` | 133.992-226.272 s | 14.375 GiB |

表 6-6 展示两个代表数据集的单任务 breakdown。模板级热点只需要遍历模板和持续时间列；rank 负载需要遍历 rank tree；三个 layer-aware 任务需要从 CPU scope 沿 CPU-GPU link 收集 GPU kernel，其中 overlap 还需要区间排序和合并。

**表 6-6 代表数据集的单任务时间**

| 数据集 | 任务 | 加载到内存 | 分析 | 端到端 |
|---|---|---:|---:|---:|
| `qwen3_full` | `operator_hotspot` | 14.021 s | 0.126 s | 14.147 s |
| `qwen3_full` | `rank_load_balance` | 14.021 s | 0.387 s | 14.408 s |
| `qwen3_full` | `layer_kernel_hotspot` | 14.021 s | 3.089 s | 17.110 s |
| `qwen3_full` | `layer_compute_comm_overlap` | 14.021 s | 9.126 s | 23.146 s |
| `qwen3_full` | `layer_rank_balance` | 14.021 s | 5.336 s | 19.357 s |
| `llama_full` | `operator_hotspot` | 133.878 s | 0.114 s | 133.992 s |
| `llama_full` | `rank_load_balance` | 133.878 s | 3.184 s | 137.063 s |
| `llama_full` | `layer_kernel_hotspot` | 133.878 s | 23.389 s | 157.267 s |
| `llama_full` | `layer_compute_comm_overlap` | 133.878 s | 92.393 s | 226.272 s |
| `llama_full` | `layer_rank_balance` | 133.878 s | 42.084 s | 175.962 s |

## 6.5 与 baseline 的分析速度对比

本节展示 4 个核心分析任务在所有 compressor 上的完整对比。4 个任务分别对应 PADOC 的 4 个访问维度：按算子类型过滤（`operator_hotspot`）、按 rank 遍历（`rank_load_balance`）、按模型层遍历（`layer_compute_comm_overlap`）和按时间访问（`gpu_bubble_rate`）。所有基线均实现了相同的分析逻辑（通过 `run_raw` 在完整 Trace 上执行），确保比较公平。

Benchmark 采用 batch 模式：每个 compressor 加载一次（read + decompress），然后顺序执行所有 4 个分析任务，各自独立计时。表中仅列 `analyze_secs`（纯分析时间，不含加载）。

**表 6-7 跨压缩器分析时间对比（analyze_secs）**

*leworldmodel_full (3.5M events, 2 ranks, AMD GPU)*

| Compressor | operator_hotspot | rank_load_balance | layer_overlap | gpu_bubble |
|---|---:|---:|---:|---:|
| **PADoC** | **0.009 s** | **0.032 s** | **0.455 s** | **0.033 s** |
| raw_json | 1.921 s (213×) | 0.029 s (0.9×) | 2.031 s (4.5×) | 0.025 s (0.8×) |
| gzip_json | 0.888 s (99×) | 0.029 s (0.9×) | 1.990 s (4.4×) | 0.025 s (0.8×) |
| ScalaTrace | 0.887 s (99×) | 0.030 s (0.9×) | 2.019 s (4.4×) | 0.027 s (0.8×) |
| TraceZip | 0.825 s (92×) | 0.029 s (0.9×) | 2.560 s (5.6×) | 0.025 s (0.8×) |

*qwen3_full (33.8M events, 256 ranks, Ascend NPU)*

| Compressor | operator_hotspot | rank_load_balance | layer_overlap | gpu_bubble |
|---|---:|---:|---:|---:|
| **PADoC** | **0.065 s** | **0.171 s** | **5.638 s** | **0.425 s** |
| raw_json | 14.637 s (225×) | 0.762 s (4.5×) | 25.207 s (4.5×) | 0.519 s (1.2×) |
| gzip_json | 14.720 s (226×) | 0.822 s (4.8×) | 26.977 s (4.8×) | 0.515 s (1.2×) |
| ScalaTrace | 6.580 s (101×) | 0.772 s (4.5×) | 25.900 s (4.6×) | 0.531 s (1.2×) |
| TraceZip | 6.256 s (96×) | 0.695 s (4.1×) | 27.869 s (4.9×) | 0.513 s (1.2×) |

*unifolm_full (80.2M events, 4 ranks, NVIDIA CUDA)*

| Compressor | operator_hotspot | rank_load_balance | layer_overlap | gpu_bubble |
|---|---:|---:|---:|---:|
| **PADoC** | **0.182 s** | **1.046 s** | **8.346 s** | **1.436 s** |
| raw_json | 42.266 s (232×) | 1.608 s (1.5×) | 46.774 s (5.6×) | 1.330 s (0.9×) |
| gzip_json | 50.229 s (276×) | 1.716 s (1.6×) | 52.179 s (6.3×) | 1.521 s (1.1×) |
| ScalaTrace | 32.930 s (181×) | 1.569 s (1.5×) | 45.110 s (5.4×) | 1.302 s (0.9×) |
| TraceZip | 33.685 s (185×) | 1.397 s (1.3×) | 56.177 s (6.7×) | 1.194 s (0.8×) |

*llama_full (301M events, 1024 ranks, NVIDIA CUDA)*

| Compressor | operator_hotspot | rank_load_balance | layer_overlap | gpu_bubble |
|---|---:|---:|---:|---:|
| **PADoC** | **0.081 s** | **1.714 s** | **49.579 s** | **2.725 s** |
| ScalaTrace | 86.397 s (1067×) | 9.301 s (5.4×) | 192.915 s (3.9×) | 5.758 s (2.1×) |
| TraceZip | 85.482 s (1055×) | 8.516 s (5.0×) | 187.345 s (3.8×) | 5.506 s (2.0×) |
| raw_json | OOM | OOM | OOM | OOM |
| gzip_json | OOM | OOM | OOM | OOM |

注：llama_full 的 raw_json 和 gzip_json 解压后需要 819-825 GiB 内存，超出实验服务器 503 GiB RAM 限制，无法完成分析。

**分析**：4 个任务的加速比反映了 PADOC 4 种访问维度的不同优势：

1. **按算子类型过滤**（operator_hotspot）：PADoC 直接在模板表上求和（O(|T|) 而非 O(|E|)），获得 **92-1067×** 加速。这是模板化压缩的核心收益。
2. **按 rank 遍历**（rank_load_balance）：PADoC 的 rank-rooted tree 使 per-rank GPU 统计无需全局扫描，获得 **4-5×** 加速（大规模数据集）。小数据集（leworldmodel 仅 2 ranks）优势不明显。
3. **按 layer 遍历**（layer_compute_comm_overlap）：PADoC 利用 CPU-GPU link 直接从 layer scope 收集 kernel 并计算 overlap，获得 **3.8-6.7×** 加速。
4. **按时间访问**（gpu_bubble_rate）：所有 compressor 均可高效执行流式窗口遍历，PADoC 优势较小（**1-2×**），说明该维度的瓶颈在于数据加载而非分析算法。

**表 6-8 跨压缩器常驻内存对比（分析时 peak RSS）**

| 数据集 | PADoC | raw_json | gzip_json | ScalaTrace | TraceZip |
|---|---:|---:|---:|---:|---:|
| leworldmodel_full | 0.6 GiB | 9.0 GiB | 9.0 GiB | 3.1 GiB | 3.1 GiB |
| qwen3_full | 4.5 GiB | 81.9 GiB | 82.2 GiB | 22.8 GiB | 23.0 GiB |
| unifolm_full | 15.3 GiB | 232.4 GiB | 233.2 GiB | 71.4 GiB | 72.9 GiB |
| llama_full | 29.4 GiB | OOM (est. 819 GiB) | OOM (est. 825 GiB) | 221.8 GiB | 221.8 GiB |

PADoC 的分析常驻内存比 ScalaTrace/TraceZip 低 **5-7.5×**，比 raw_json/gzip_json 低 **15-28×**。这是因为 PADoC 的原位分析直接在压缩表示（模板+树）上操作，无需将每个事件展开为独立的 Event 对象。ScalaTrace/TraceZip 虽然通过跨 rank 字典共享减少了字符串冗余，但分析时仍需完全解压为事件级 Trace 结构。

## 6.6 分析消融

本节围绕四个问题做消融：结构信息、时间戳整型宽度、时间戳残差压缩，以及 CPU-GPU 映射。

第一，表 6-8 对比默认 PADOC 与关闭结构合并的 `no_structural` preset。关闭结构合并后，部分 layer-aware 遍历在这些数据集上更快，因为重复 scope 被展开为更直接的节点形态；但它会显著增加常驻峰值内存，并削弱结构化表示的压缩目的。因此该实验支持的结论不是“结构合并总让每个查询更快”，而是“结构压缩降低 resident memory，并显式保留 layer/rank 访问语义”。

**表 6-8 结构合并消融**

| 数据集 | Preset | Artifact | Accounted resident | `rank_load_balance` 分析 | `layer_compute_comm_overlap` 分析 |
|---|---|---:|---:|---:|---:|
| `qwen3_full` | default | 272.23 MiB | 1.899 GiB | 0.212 s | 6.798 s |
| `qwen3_full` | no structural | 268.48 MiB | 3.558 GiB | 0.350 s | 1.516 s |
| `unifolm_full` | default | 741.08 MiB | 4.678 GiB | 0.634 s | 9.638 s |
| `unifolm_full` | no structural | 692.71 MiB | 9.472 GiB | 1.246 s | 5.941 s |

第二，所有最终 artifact 的时间戳列均为常量或 `i32`，没有 `i64` 列。以 `llama_full` 为例，当前 timestamp resident 为 1.122 GiB；如果按 `i64` 存储相同数量的 timestamp 值，至少需要约 2.244 GiB，resident representation 至少增加约 1.122 GiB。该估算不包含额外 vector overhead，因此是保守下界。该结果说明 per-rank timestamp normalization 和 `i32` downcast 对分析常驻内存有直接作用。

第三，表 6-9 给出分段线性残差编码的列级原型结果。该原型使用整数定点斜率，保存 segment 参数和 `i8`/`i16` residual，并采用 per-column fallback。它尚未集成进主 artifact 格式，因此本文不把它作为系统主结果；但它说明时间戳和持续时间列仍有进一步降低内存的空间。

**表 6-9 分段线性时间戳/持续时间原型**

| 数据集 | Columns | Sampled values | Hybrid vs int64 memory | Hybrid vs int32 memory | Accepted cols | Encode time |
|---|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 128 | 3,906,828 | 7.06x | 3.53x | 128 | 0.153 s |
| `qwen3_full` | 128 | 38,097,496 | 4.08x | 2.04x | 118 | 1.620 s |
| `unifolm_full` | 128 | 58,928,428 | 5.87x | 2.93x | 116 | 2.333 s |
| `llama_full` | 128 | 93,249,920 | 4.04x | 2.02x | 126 | 4.087 s |

第四，表 6-10 展示 CPU-GPU 映射消融。本文不把“不保存 link 后当前实现返回空结果”作为有效证据，因为这只能说明实现路径缺失，不能说明分析本身不可行。更合理的对照是在默认 artifact 上不直接使用保存的 GPU instance 指针，而是运行时构建 `(rank, correlation) -> GPU kernel` 映射并动态查询。该路径在 `qwen3_full` 上可恢复接近默认的覆盖率，但需要额外建立映射，且对 correlation id 的重复和作用域更敏感。因此该实验验证的是显式映射与运行时重建映射之间的工程权衡，而不是证明“不维护映射就无法分析”。

**表 6-10 CPU-GPU mapping 消融**

| 数据集 | 方法 | `layer_kernel_hotspot` coverage | `layer_compute_comm_overlap` coverage | 额外建索引 | `layer_compute_comm_overlap` 分析 |
|---|---|---:|---:|---:|---:|
| `qwen3_full` | explicit mapping | 1,592,830 / 1,806,096 | 1,592,830 / 1,806,096 | 0.000 s | 9.126 s |
| `qwen3_full` | dynamic correlation lookup | 1,588,915 / 1,806,096 | 1,588,915 / 1,806,096 | 0.568 s | 7.389 s |

## 6.7 扩展性

GPU 数扩展性实验从 `llama_full` 抽取不同 rank 子集。结果显示 raw size 和 artifact size 随 GPU 数近似线性增长，压缩比从 1 GPU 的 25.03x 提升到 256 GPUs 的 28.97x，说明跨 rank 重复结构有助于模板和结构共享。

**表 6-11 GPU 数扩展性**

| GPUs | Events | Raw size | Artifact | Ratio | Compress time |
|---:|---:|---:|---:|---:|---:|
| 1 | 316,746 | 74.81 MiB | 2.99 MiB | 25.03x | 1.737 s |
| 8 | 2,607,995 | 622.99 MiB | 23.35 MiB | 26.68x | 4.665 s |
| 64 | 19,544,859 | 4.55 GiB | 165.23 MiB | 28.19x | 48.408 s |
| 256 | 75,749,224 | 17.59 GiB | 621.61 MiB | 28.97x | 115.421 s |

Synthetic layers 和 iterations 扫描进一步验证了重复结构增加时的趋势：layers 从 8 到 128 时事件数线性增长，压缩比保持在约 28x 至 30x；iterations 从 1 到 16 时，压缩比从 20.47x 提升到 30.42x。

## 6.8 实验结论

综合实验结果，本文得到以下结论。PADOC 在四个真实 AI profiler trace 上达到 23.79x 至 31.08x 压缩比；虽然不是所有数据集上的最小字节流，但保留了可查询结构。最终实现可以将 301M events / 1024 ranks 的 LLaMA trace 保存为 2.44 GiB artifact，并以 14.375 GiB accounted resident representation 完成五个核心分析任务。分析时间与访问模式一致：模板聚合最快，rank tree walk 次之，layer-aware attribution 和 overlap 最重。消融结果表明，CPU-GPU 映射可以在分析时由 correlation 动态重建，但显式保存映射提供了更直接的查询路径；时间戳 `i32` downcast 已经显著降低常驻内存，而分段线性残差编码是进一步优化方向。

---

# 第 7 章 讨论

## 7.1 关于压缩比的解释

本文不将 PADOC 描述为所有场景下压缩比最高的方法。ScalaTrace 在多个数据集上更小，尤其在重复结构规则的 trace 上优势明显。PADOC 保留 rank tree、CPU-GPU link、参数列和名称数字列，这些信息会占用额外空间。保留这些结构的目的不是压缩字节最少，而是让热点、rank、layer-aware GPU 分析能够在压缩表示上执行。

因此，本文的核心论点是：PADOC 在保持竞争性压缩率的同时，提供面向分析的结构化压缩表示。其价值通过原位分析、存储和内存 breakdown、CPU-GPU 映射消融和时间戳列消融共同体现。

## 7.2 关于层级分析覆盖率

层级 GPU 分析依赖 profiler 中的 CPU scope 和 correlation 信息。如果模型代码或 profiler 没有清晰记录 layer scope，或者大量 GPU 工作发生在初始化、数据搬运、框架 utility 中，可归因覆盖率会降低。`qwen3_full` 覆盖率为 88.19%，适合作为主要展示数据；`leworldmodel_full` 和 `unifolm_full` 覆盖率较低，说明覆盖率更多受 trace annotation 质量影响。动态 correlation lookup 能恢复接近默认的覆盖率，因此本文不把“没有显式 link 就不能分析”作为结论，而把显式映射定位为更直接、更稳定的查询路径。

未来可以通过显式模型层注解、框架级 scope 规范或对 profiler output 的预处理提高覆盖率。PADOC 的结构表示可以承接这些更高质量的 annotation。

## 7.3 关于内存占用

最终 `llama_full` artifact 磁盘大小为 2.44 GiB，accounted resident representation 为 14.375 GiB。文件和内存相差较大的根本原因不是时间戳未压缩，而是磁盘上有 zstd 压缩，内存中则需要可直接访问的对象、向量、树节点和参数列。表 6-4 显示 `llama_full` 中 node storage 为 7.679 GiB，args storage 为 2.535 GiB，timestamp 为 1.122 GiB。

这部分属于系统工程问题，不改变结构化压缩的核心思想。未来可以使用 mmap-backed column、lazy decode、arena allocation、按任务加载部分 region 或 streaming analysis 进一步降低常驻表示和运行时峰值。

## 7.4 关于时间戳进一步压缩

On-disk breakdown 显示时间戳列在最大数据集上贡献约 1.00 GiB，是重要优化对象。分段线性拟合加残差编码是一条可行路线：对单调或近线性的时间戳序列保存整数 segment 参数，并将 residual 降至 `i16` 或 `i8`。本文的列级原型已经验证该方法在 sampled columns 上可超过 2x vs `i64`，并且通常优于当前 `i32` 内存估计。

但该方法尚未进入主 artifact 格式。完整集成后还需要重新评估 artifact size、load time、analysis time 和随机访问开销，避免为了内存压缩牺牲查询性能。

## 7.5 威胁与局限

本文实验仍有局限。第一，当前真实数据集不包含 MoE 和 ViT trace，无法验证专家路由或视觉模型结构对压缩和分析的影响。第二，新 layer-aware GPU 任务尚未对 ScalaTrace、TraceZip、gzip_json 和 raw_json 实现完全等价的 attribution baseline，因此本文只在 common tasks 上报告跨 compressor speedup。第三，完整 1024-rank LLaMA 上的多 preset ablation 成本较高，当前完整 ablation 主要覆盖 leworldmodel、qwen3 和 unifolm。第四，系统目前分析任务主要是单线程，`llama_full` 上 layer-aware overlap 仍需 92.393 s 纯分析时间，存在并行优化空间。

这些局限不影响本文主要结论，但需要在最终论文中如实说明。

## 7.6 未来工作

未来工作包括五个方向。第一，扩展数据集，加入 MoE、ViT 和更多推理服务 trace。第二，实现 lazy loading 和列级 mmap，降低加载峰值内存。第三，对 layer-aware 任务进行 rank 并行和 layer 分组并行，降低 overlap 分析时间。第四，将分段线性 residual timestamp、node tree delta encoding 和更紧凑的实例引用表示集成进主格式。第五，建立更标准的 profiler annotation 规范，使模型 layer、module、expert 和 pipeline stage 信息能够稳定进入 trace。

---

# 第 8 章 结论

本文研究大规模 AI 性能剖析轨迹的结构化压缩与原位分析问题。针对原始 JSON trace 体积大、完整重建成本高、传统压缩表示缺乏 AI 分析结构的问题，本文设计并实现 PADOC 系统。PADOC 将事件归并为模板，将实例字段保存为类型化列，并构建 rank-rooted node tree；系统显式保留 CPU launch 与 GPU kernel 的 provenance link，使 layer-aware GPU 分析可以直接在压缩表示上执行。

实验在四个真实 AI 工作负载上进行，覆盖最高 301,288,116 个事件和 1024 ranks。PADOC 获得 23.79x 至 31.08x 压缩比，并在最大数据集上以 2.44 GiB artifact 支持五个核心任务的原位分析，端到端时间为 133.992 s 至 226.272 s，accounted resident representation 为 14.375 GiB。CPU-GPU 映射消融显示，运行时 correlation lookup 可以恢复接近默认的层级 GPU 覆盖率，但需要额外索引并对 correlation id 的一致性更敏感。存储和分析消融进一步说明，字节最小化与分析友好表示之间存在权衡，PADOC 的贡献在于两者的共同设计。

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

[10] PADOC 项目最终实验结果汇总. `results/remaining/final_paper/final_experiment_results.md`, 2026.

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
| `results/remaining/final_paper/final_experiment_results.md` | 最终论文结果总表与解释 |
| `results/remaining/final_paper/core_layer_analysis_sparse_v7.tsv` | 五个核心任务在四个数据集上的最终 PADOC 分析时间 |
| `results/remaining/final_paper/on_disk_breakdown_sparse_v7.txt` | 最终 artifact 的磁盘和常驻内存拆解 |
| `results/remaining/final_paper/no_structural_core_ablation.tsv` | 结构信息消融 |
| `results/remaining/final_paper/dynamic_kernel_mapping_ablation.tsv` | CPU-GPU 映射动态查找消融 |
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

1. `core_layer_analysis_sparse_v7.tsv` 为 20 行，对应 4 datasets x 5 current core tasks。
2. `on_disk_breakdown_sparse_v7.txt` 包含 4 个最终 sparse-slot artifact 的磁盘和常驻内存拆解。
3. `no_structural_core_ablation.tsv` 为 30 行，对应 3 datasets x 2 presets x 5 tasks。

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
