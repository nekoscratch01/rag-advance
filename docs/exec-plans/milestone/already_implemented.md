# Atlas 当前实现索引

更新时间：2026-05-04

这是一份短索引，不是完整实现手册。它只回答“现在做到哪里了、下一步该看哪里”。具体版本执行记录放在同目录的 milestone 文档里，实际架构细节放在 `docs/exec-plans/version-arch/`。

## 当前状态

```text
当前已完成里程碑：V1 Atlas Hybrid Kernel 的检索质量主路径
当前下一阶段：V2 Atlas Research Runtime 的脚手架设计与实现
```

V0 仍然是已实现的基础 RAG 内核：

```text
documents -> chunks -> dense vector retrieval -> cited answer -> trace -> basic eval
```

V1 在 V0 上新增：

```text
FinanceBench parent-child corpus
Dense + BM25 sparse hybrid retrieval
RRF fusion
local CrossEncoder reranker
parent evidence packer
Critic Lite
Postgres exact query cache
FinanceBench 仅检索测评
```

## 阅读入口

```text
路线图 / 设计意图：
  docs/Design-docs/00_overview.md
  docs/Design-docs/02_v1_atlas_hybrid_kernel.md
  docs/Design-docs/03_v2_atlas_research_runtime.md

V1 执行里程碑：
  docs/exec-plans/milestone/v1_hybrid_kernel_milestone.md

V1 实际架构：
  docs/exec-plans/version-arch/v1_hybrid_kernel_arch.md

V1 测评实验报告：
  benchmarks/rag_quality/financebench/reports/full_v1_retrieval_experiment.md
```

## 当前 V1 指标

主实验：

```text
benchmarks/financebench/retrieval_runs/full_v1_retrieval
```

| 模式 | doc@10 | page@10 | MRR doc | MRR page |
|---|---:|---:|---:|---:|
| dense_only | 0.467 | 0.127 | 0.233 | 0.081 |
| bm25_only | 0.787 | 0.207 | 0.448 | 0.113 |
| hybrid_rrf | 0.727 | 0.213 | 0.398 | 0.112 |
| hybrid_rrf_reranker | 0.813 | 0.267 | 0.520 | 0.146 |

## 必须记住的实现事实

```text
FinanceBench JSONL 文件是测评冻结中间产物，不是运行时存储。
运行时存储是 Postgres + Qdrant。
V1 cache 已实现，后端是 Postgres query_cache，不是 Redis。
Redis Queue 属于 V2 Research Runtime 的规划。
V1 已有检索测评证据，但还没有完整生成式答案可靠性测评。
```

## 当前边界

已实现：

```text
V0 RAG kernel
V1 hybrid retrieval
V1 local reranker
V1 parent-child evidence
V1 Postgres exact cache
V1 Critic Lite code path
V1 FinanceBench 仅检索测评
```

未实现：

```text
ResearchJob / ResearchJobEvent
Redis Queue / worker pool
GraphRAG
structured data analysis
streaming ingestion
Memory / Skills
MCP tool layer
cloud production deployment
```

仍需测评证明：

```text
generated answer citation_doc/page_hit
answer_numeric_match
unsupported_answer_rate
cache warm run
Critic Lite false-insufficient rate
```
