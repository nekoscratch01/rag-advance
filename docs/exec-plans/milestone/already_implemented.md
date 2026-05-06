# Atlas 当前实现索引

更新时间：2026-05-06

这是一份短索引，不是完整实现手册。它只回答“现在做到哪里了、下一步该看哪里”。具体版本执行记录放在同目录的 milestone 文档里，实际架构细节放在 `docs/exec-plans/version-arch/`。

## 当前状态

```text
当前已完成里程碑：V1 Atlas Advanced Hybrid Kernel 的 Design 主链路
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
Query Orchestrator / QueryPlan
RetrievalTask
TextHybridProvider
Dense / BM25 / textual Table lanes
Provider-local Weighted RRF fusion
plan-aware local CrossEncoder reranker
EvidencePack / parent evidence builder
Evidence Evaluator
Citation Verifier
Postgres exact query cache
Design trace table family
FinanceBench retrieval-only 测评
Full V1 generated-answer benchmark runner
```

## 阅读入口

```text
路线图 / 设计意图：
  docs/Design-docs/00_ATLAS_OVERVIEW.md
  docs/Design-docs/01_V1_ADVANCED_HYBRID_KERNEL.md
  docs/Design-docs/02_V2_RESEARCH_RUNTIME.md

V1 执行里程碑：
  docs/exec-plans/milestone/v1_advanced_hybrid_kernel_milestone.md

V1 实际架构：
  docs/exec-plans/version-arch/v1_advanced_hybrid_kernel_arch.md

V1 测评实验报告：
  benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506/report.md
  benchmarks/rag_quality/v1_hybrid_provider_reset/report.md
```

## 当前 V1 指标

主实验：

```text
benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506
```

| 模式 | doc@10 | page@10 | MRR doc | MRR page | MAP doc | MAP page |
|---|---:|---:|---:|---:|---:|---:|
| dense_only | 0.467 | 0.127 | 0.233 | 0.081 | 0.208 | 0.079 |
| bm25_only | 0.787 | 0.207 | 0.448 | 0.113 | 0.404 | 0.112 |
| hybrid_rrf | 0.727 | 0.213 | 0.398 | 0.112 | 0.343 | 0.105 |
| hybrid_rrf_reranker | 0.813 | 0.267 | 0.520 | 0.146 | 0.460 | 0.139 |

## 必须记住的实现事实

```text
FinanceBench JSONL 文件是测评冻结中间产物，不是运行时存储。
运行时存储是 Postgres + Qdrant。
V1 cache 已实现，后端是 Postgres query_cache，不是 Redis。
Redis Queue 属于 V2 Research Runtime 的规划。
V1 已有检索测评证据和 full generated-answer benchmark runner，但还没有归档的全量生成式答案可靠性报告。
```

## 当前边界

已实现：

```text
V0 RAG kernel
V1 hybrid retrieval
V1 local reranker
V1 parent-child evidence
V1 QueryPlan / RetrievalTask
V1 TextHybridProvider lanes
V1 Weighted RRF trace
V1 EvidencePack
V1 Evidence Evaluator / Citation Verifier
V1 trace table family
V1 Postgres exact cache
V1 Critic Lite code path
V1 FinanceBench 仅检索测评
V1 component-level benchmark runner
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
