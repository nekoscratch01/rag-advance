# V1 里程碑：Atlas Hybrid Kernel

更新时间：2026-05-04

## 版本契约

| 版本 | 名称 | 核心目标 | 主要产物 |
|---|---|---|---|
| V1 | Atlas Hybrid Kernel | 提升检索质量和答案可靠性 | hybrid retrieval、reranker、cache、Critic Lite、质量对比 |

设计来源：

```text
docs/Design-docs/02_v1_atlas_hybrid_kernel.md
```

实际架构说明：

```text
docs/exec-plans/version-arch/v1_hybrid_kernel_arch.md
```

当前实现索引：

```text
docs/exec-plans/milestone/already_implemented.md
```

## 里程碑状态

```text
状态：检索质量主路径已经实现并完成仅检索测评
边界：完整生成式答案可靠性测评尚未完成
```

V1 当前已经具备可运行的 hybrid retrieval 栈、FinanceBench corpus 准备与导入路径、parent-child evidence 模型、本地 reranker、Postgres exact cache、Critic Lite、trace metadata，以及仅检索测评证据。

## 相比 Design 的实际变化

### 1. FinanceBench 成为 V1 的核心验证集

Design 文档里 V1 是通用 hybrid retrieval 蓝图。实际执行时，我们把质量验证收敛到 FinanceBench，因为它提供：

```text
150 条 open-source QA cases
gold document name
gold evidence string
gold evidence page number
财报 PDF 场景，强依赖公司名、年份、filing、财务科目等精确词匹配
```

这个取舍让 V1 从“实现 hybrid retrieval 组件”变成了“能被实验验证的检索质量升级”。

### 2. Parent-Child RAG 取代相邻 chunk 猜合并

最初设计允许 Evidence Builder 合并相邻 chunk。实际执行中改成显式 parent-child schema：

```text
ParentBlock = page / table / section 级 readable block
ChildChunk  = retrieval chunk
```

检索发生在 child chunk 上；证据展示、citation 和 context packing 回到 parent block。

这个取舍的好处是 evidence 可追溯，不再靠“相邻页/相邻 chunk 猜测”。代价是 ingestion 必须保证 `parent_id` 和 `child_ids` 稳定。

### 3. JSONL 是冻结中间产物，不是运行时存储

FinanceBench 准备阶段会生成：

```text
corpus/financebench/manifest.jsonl
corpus/financebench/parsed/pages.jsonl
corpus/financebench/parsed/parent_blocks.jsonl
corpus/financebench/parsed/child_chunks.jsonl
```

这些文件用于可复现实验、PDF 解析审计、page 对齐和 chunk id 稳定性检查。线上查询不会读取这些 JSONL。

`scripts/ingest_financebench.py` 会把它们导入：

```text
Postgres: documents / parent_blocks / chunks
Qdrant: child dense vectors + BM25 sparse vectors
```

### 4. BM25 留在 Qdrant，没有引入 OpenSearch

实际实现选择：

```text
FastEmbed Qdrant/bm25
Qdrant sparse vector
Modifier.IDF
```

取舍：

```text
收益：本地 Mac 迭代更轻，少一个服务，dense 和 sparse 都在 Qdrant
成本：词法检索能力不如 OpenSearch 完整
后路：保留 LexicalRetriever 边界，未来可以用 OpenSearch 替换 Qdrant BM25
```

### 5. V1 实现了 cache，但不是 Redis

V1 的版本契约要求 cache 能力。实际实现为：

```text
Postgres query_cache table
cache key schema: atlas-query-cache-v2
trace fields: cache_hit / cache_key / cache_latency_ms / cache_status
```

Redis 当前没有实现。它后续可以作为 cache 性能后端，也会在 V2 的 Redis Queue / worker pool 中更重要。

### 6. Reranker 成为主质量路径的必要环节

实验显示 RRF-only 并不足够。当前最强检索路径是：

```text
dense + BM25 -> RRF -> local CrossEncoder reranker -> parent evidence
```

默认本地 reranker：

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

取舍：

```text
收益：在 FinanceBench retrieval 指标上明显提升排序质量
成本：Mac CPU 延迟显著上升
```

### 7. 新增仅检索测评

原本的测评 runner 会调用 `/v1/query`，因此依赖 LLM API。为了先隔离检索质量，实际新增：

```text
src/atlas/benchmark/financebench_retrieval.py
```

这使我们能在不调用 LLM 的情况下评估：

```text
dense_only
bm25_only
hybrid_rrf
hybrid_rrf_reranker
```

## 已实现产物

```text
src/atlas/datasets/financebench.py
src/atlas/datasets/financebench_importer.py
scripts/prepare_financebench.py
scripts/ingest_financebench.py
src/atlas/db/models.py
src/atlas/vector/collections.py
src/atlas/embeddings/bm25_sparse.py
src/atlas/retrieval/candidate.py
src/atlas/retrieval/dense_retriever.py
src/atlas/retrieval/bm25_retriever.py
src/atlas/retrieval/fusion.py
src/atlas/retrieval/reranker.py
src/atlas/retrieval/hybrid_retriever.py
src/atlas/retrieval/mode_switching.py
src/atlas/query_runtime/cache.py
src/atlas/query_runtime/critic_lite.py
src/atlas/query_runtime/evidence_builder.py
src/atlas/query_runtime/citation_builder.py
src/atlas/query_runtime/trace_logger.py
src/atlas/benchmark/financebench.py
src/atlas/benchmark/financebench_retrieval.py
```

## Benchmark 证据

主实验：

```text
benchmarks/financebench/retrieval_runs/full_v1_retrieval
```

人类可读实验报告：

```text
benchmarks/rag_quality/financebench/reports/full_v1_retrieval_experiment.md
```

主 retrieval-only 结果：

| 模式 | doc@10 | page@10 | MRR doc | MRR page | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| dense_only | 0.467 | 0.127 | 0.233 | 0.081 | 27 | 88 |
| bm25_only | 0.787 | 0.207 | 0.448 | 0.113 | 4 | 7 |
| hybrid_rrf | 0.727 | 0.213 | 0.398 | 0.112 | 45 | 59 |
| hybrid_rrf_reranker | 0.813 | 0.267 | 0.520 | 0.146 | 747 | 1261 |

核心解释：

```text
BM25 是强 baseline，不是可有可无的补充。
RRF-only 不会自动优于 BM25-only。
Hybrid + reranker 是当前实验中最强的 retrieval path。
page-level recall 仍然偏弱，是后续答案可靠性的主要风险。
```

## 尚未完成

```text
生成式答案测评尚未完成
citation_doc/page_hit 尚未在 generated answers 上测量
answer_numeric_match 尚未测量
unsupported_answer_rate 尚未测量
cache warm 测评尚未完整报告
Critic Lite 有代码路径，但仍需正式测评证据
最佳路径 page@10 仍只有 0.267
```

## 是否进入 V2

判断：

```text
可以进入 V2 脚手架。
不应直接进入完整 planner / 报告生成器质量承诺。
```

允许进入的 V2 范围：

```text
ResearchJob schema
ResearchJobEvent schema
/v1/research/jobs API skeleton
JobManager
minimal queue/worker boundary
artifact store
复用 V1 retrieval 的 subquestion executor
job-level budget controls
```

暂时不要做：

```text
完整报告生成器质量承诺
reflexive loop
GraphRAG
structured data analysis
front-end
```
