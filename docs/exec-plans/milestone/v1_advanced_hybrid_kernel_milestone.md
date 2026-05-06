# V1 里程碑：Atlas Advanced Hybrid Kernel

更新时间：2026-05-06

## 版本契约

| 版本 | 名称 | 核心目标 | 主要产物 |
|---|---|---|---|
| V1 | Atlas Advanced Hybrid Kernel | 把单次问答的 evidence retrieval 做到强、可解释、可评估、可扩展 | QueryPlan、TextHybridProvider、Weighted RRF、EvidencePack、Evaluator/Verifier、trace 表族、full V1 eval |

设计来源：

```text
docs/Design-docs/01_V1_ADVANCED_HYBRID_KERNEL.md
```

实际架构说明：

```text
docs/exec-plans/version-arch/v1_advanced_hybrid_kernel_arch.md
```

## 里程碑状态

```text
状态：V1 Design 主链路已按 Advanced Hybrid Evidence Kernel 形态落地
边界：V1 仍是 Evidence Kernel，不是 V2 Research Runtime，也不是 V4 结构化数据内核
证据：检索质量有既有 FinanceBench retrieval-only 报告；full V1 generated-answer benchmark runner 已补齐，但全量生成式可靠性报告仍需单独跑数归档
```

这轮 V1 对齐把原来的 hybrid retrieval 主路径推进为正式 evidence kernel：

```text
User Query
  -> Query Orchestrator
  -> Retrieval Plan
  -> TextHybridProvider
  -> Dense / BM25 / Table textual lanes
  -> Provider-local Weighted RRF
  -> Reranker
  -> Evidence Builder / EvidencePack
  -> Evidence Evaluator
  -> Answer Generator
  -> Citation Verifier
  -> Trace + Eval + Cache
```

## 已实现产物

### Contract / Plan

```text
src/atlas/query_orchestrator/schema.py
src/atlas/query_orchestrator/service.py
src/atlas/query_orchestrator/fallback.py
src/atlas/query_orchestrator/llm_planner.py
src/atlas/query_orchestrator/validator.py
src/atlas/retrieval/models/retrieval_task.py
src/atlas/retrieval/models/candidate.py
src/atlas/retrieval/models/evidence_contract.py
src/atlas/query_runtime/verification.py
configs/finance_metric_ontology.yaml
```

### Retrieval / Fusion / Rerank

```text
src/atlas/retrieval/providers/text_hybrid/provider.py
src/atlas/retrieval/providers/text_hybrid/lanes.py
src/atlas/retrieval/ranking/fusion.py
src/atlas/retrieval/ranking/reranker.py
```

### Evidence / Verification / Runtime

```text
src/atlas/query_runtime/evidence_builder.py
src/atlas/query_runtime/evidence_evaluator.py
src/atlas/query_runtime/citation_verifier.py
src/atlas/query_runtime/critic_lite.py
src/atlas/query_runtime/service.py
src/atlas/query_runtime/cache.py
```

### Trace / API / Eval

```text
src/atlas/api/routes/query.py
src/atlas/db/models.py
src/atlas/db/repositories.py
src/atlas/db/session.py
src/atlas/eval/v1_full.py
src/atlas/benchmark/financebench.py
src/atlas/benchmark/financebench_retrieval.py
```

## 相比旧 V1 的实际变化

### 1. QueryPlan 成为数据流入口

旧 V1 主要从原始 query 和 request options 直接进入 retriever。现在查询会先生成正式 `QueryPlan`：

```text
original_query
standalone_query
query_type
entities / periods / metrics
retrieval_units
metadata_filter
metadata.known_providers
metadata.executable_providers
risk_flags
budget
```

实现策略是 LLM structured planner + deterministic fallback + validator。
没有 `OPENAI_API_KEY` 时 fallback 仍可用；有 key 时默认 planner model 为 `gpt-5-nano`。
V1 planner prompt 使用 `known_providers=["hybrid","sql","graph"]` 表达语义意图。
V1 runtime 使用 `executable_providers=["hybrid"]` 执行当前能力。
`sql` / `graph` 在 plan 中合法，但会在 V1 execution 中生成 `skipped_non_executable` ProviderResult。

### 2. RetrievalTask 固定 provider 输入

`QueryPlan` 会编译为 `RetrievalTask`，明确：

```text
provider
query_text
metadata_filter
provider_status
unsupported_reason
internal_lanes
must_have_terms
should_terms
unit_weight
```

### 2.1 ProviderRouter contract bump

本阶段新增 runtime 收口层：

```text
ProviderRouter
ProviderResult
SourceAnchor
```

实际行为：

```text
hybrid task -> TextHybridProvider -> evidence
sql task    -> skipped_non_executable -> trace only
graph task  -> skipped_non_executable -> trace only
```

默认不做 `hybrid_backfill`，避免把 SQL/Graph 语义意图伪装成 hybrid。

### 2.2 LLM client adapter

Planner 和 Answer Generator 不再直接实例化 OpenAI SDK。

```text
llm.clients.LLMClient
llm.clients.OpenAIClient
```

任务层只描述 planner / answer generation；底层 provider 替换、mock 测试、usage/latency trace 由 client adapter 承担。

这让 provider 不再只面对一个字符串，而是面对可解释的检索任务。
V1 live runtime 只执行 `provider="hybrid"`；`sql` / `graph` 只保留为未来 provider contract，不在 V1 中执行。

### 3. TextHybridProvider 成为 V1 主检索边界

新增 provider 边界：

```text
src/atlas/retrieval/providers/text_hybrid/
```

Provider 内部 lanes：

```text
Dense Lane
BM25 Lane
Metric Alias Lane
Section Lane
Table Lane
```

这些 lane 不是 Planner 可选 provider。Planner 只能输出 provider 级别的 `hybrid` / `sql` / `graph` unit；Dense/BM25/Metric Alias/Section/Table 都由 `TextHybridProvider` 内部根据 hybrid task 字段和配置派生。

当前 sparse 输入规则：

```text
dense_text = unit.text
sparse_text = unit.text + should_terms + repeat(must_have_terms, 3)
```

`repeat(must_have_terms, 3)` 是当前实现的 sparse boost：不修改 BM25 底层公式，而是在 sparse input 中重复关键 term，提高词法匹配偏好，同时避免 hard filter 误杀 evidence。

V1 的 Table Lane 是 row/page textual lane。结构化 table store、SQL provider、cell provenance 仍推迟到 V4。

### 4. Fusion 升级为 multi-lane Weighted RRF

Weighted RRF 保留每个 candidate 的：

```text
lane
rank
raw score
weight
weighted contribution
fusion score
```

为什么不用 raw score 相加：dense cosine、BM25 sparse score、table textual score 的尺度不同，直接相加会让某一路因为分布尺度而支配排序。RRF 用 rank 做归一化，再用 lane weight 表达策略偏好。

### 5. Reranker 接入 QueryPlan / RetrievalUnit

Reranker 仍保留本地 CrossEncoder 主路径，但输入现在包含：

```text
query
standalone query
entities / periods / metrics
retrieval unit
must-have / should terms
candidate text
```

trace 会记录 input rank、output rank、score、model、latency、top-N/top-M。

### 6. EvidencePack 成为正式输出

Evidence Builder 现在正式输出 `EvidenceBlock` 与 `EvidencePack`，并记录：

```text
child -> parent
dedupe
merge
token budget
query unit coverage
prompt inclusion
drop reason
```

### 7. Evidence Evaluator 与 Citation Verifier 拆出

`critic_lite.py` 现在作为兼容层，内部转调：

```text
src/atlas/query_runtime/evidence_evaluator.py
src/atlas/query_runtime/citation_verifier.py
```

生成前判断：

```text
supported
insufficient
contradicted
partially_supported
```

生成后检查：

```text
citation 是否来自 evidence
citation 是否支持关键数字
doc/page metadata 是否一致
```

系统不会自动补 citation；unsupported 与 warning 分开。

### 8. Trace 表族落地

V1 Design trace 表族已经落库：

```text
query_plans
retrieval_tasks
retrieval_results
candidates
evidence_blocks
evidence_packs
evidence_evaluations
answers
citations
citation_verifications
```

同时继续复用：

```text
query_runs
retrieval_events
generation_events
query_cache
eval_runs
eval_results
```

### 9. Full V1 eval 输出组件级指标

`src/atlas/eval/v1_full.py` 会从 `/v1/query/{id}/trace` 抽取：

```text
retrieval metrics
evidence metrics
answer metrics
latency metrics
component presence
failure buckets
```

`src/atlas/benchmark/financebench.py` 的报告新增 V1 Component Benchmarks 区块，用来逐节点检查主链路是否真的跑通。

## Benchmark 证据

已有 retrieval-only 主实验仍然有效：

```text
benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506/report.md
```

主 retrieval-only 结果：

| 模式 | doc@10 | page@10 | MRR doc | MRR page | MAP doc | MAP page | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_only | 0.467 | 0.127 | 0.233 | 0.081 | 0.208 | 0.079 | 33 | 67 |
| bm25_only | 0.787 | 0.207 | 0.448 | 0.113 | 0.404 | 0.112 | 6 | 10 |
| hybrid_rrf | 0.727 | 0.213 | 0.398 | 0.112 | 0.343 | 0.105 | 47 | 61 |
| hybrid_rrf_reranker | 0.813 | 0.267 | 0.520 | 0.146 | 0.460 | 0.139 | 502 | 690 |

新增 full V1 benchmark runner 能额外测：

```text
citation_doc_hit
citation_page_hit
answer_gold_contains
answer_numeric_match
unsupported_answer_rate
false_insufficient_rate
cache_hit_rate
component presence
plan / retrieval / reranker / generation / cache latency
evidence selected / dropped / coverage
V1 failure buckets
```

注意：

```text
retrieval-only 指标不能证明 generated answer reliability。
全量 generated-answer 可靠性报告需要单独运行 /v1/query benchmark 后写入 benchmarks/。
```

## 尚未完成

```text
全量 FinanceBench generated-answer reliability report 尚未归档
contextual chunk enrichment 仍未作为 ingestion 默认产物
table-aware metadata 仍是 textual lane，不是结构化 table store
SQLProvider / financial_facts / cell provenance 推迟到 V4
GraphProvider 推迟到 V3
Redis Queue / worker pool 属于 V2 Research Runtime，不属于 V1
```

## 验收命令

基础检查：

```bash
python -m compileall -q src/atlas scripts
pytest -q
```

仅检索测评：

```bash
ATLAS_RETRIEVAL_MODE=hybrid \
ATLAS_BM25_ENABLED=true \
ATLAS_QDRANT_COLLECTION=atlas_financebench_v1 \
python -m atlas.benchmark.financebench_retrieval \
  --cases evals/financebench_cases.yaml \
  --modes dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker \
  --top-k 10
```

Full V1 generated-answer benchmark：

```bash
ATLAS_LLM_MODEL=gpt-5-nano \
python -m atlas.benchmark.financebench \
  --cases evals/financebench_cases.yaml \
  --modes dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker \
  --top-k 10 \
  --cache-policy off \
  --warm-cache
```
