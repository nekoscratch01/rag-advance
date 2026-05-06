# V1 实际架构：Atlas Advanced Hybrid Kernel

更新时间：2026-05-06

本文记录当前代码里的 V1 真实架构。Design 文档说明意图；本文说明已经落地的 runtime、API、DB 表、trace 字段和 eval 指标。

## 1. V1 一句话解释

Atlas V1 Advanced Hybrid Kernel 是一个单次问答的 Evidence Kernel：

```text
它把用户问题编译成 QueryPlan，
再把 QueryPlan 编译成 RetrievalTask，
交给 TextHybridProvider 的 dense / BM25 / textual table lanes，
用 provider-local Weighted RRF 合并，
用 CrossEncoder reranker 精排，
构造 EvidencePack，
生成前做 evidence sufficiency/conflict 检查，
生成后做 citation support check，
最后把 trace / eval / cache 写回可审查记录。
```

它不是 V2 Research Runtime，也不是 V4 structured data runtime。

## 2. 在线主链路

```text
User Query
  -> Query Orchestrator
  -> Retrieval Plan
  -> TextHybridProvider
  -> Dense Lane / BM25 Lane / Table Lane
  -> Provider-local Weighted RRF
  -> Reranker
  -> Evidence Builder
  -> Evidence Evaluator
  -> Answer Generator
  -> Citation Verifier
  -> Trace + Eval + Cache
```

代码入口：

```text
src/atlas/api/routes/query.py
src/atlas/query_runtime/service.py
```

主要 API：

```text
POST /v1/query/plan
POST /v1/retrieve
POST /v1/query
GET  /v1/query/{query_id}/trace
```

## 3. Query Orchestrator

实现位置：

```text
src/atlas/query_orchestrator/
```

核心职责：

```text
rewrite
entity / period / metric extraction
metric ontology expansion
query decomposition
retrieval unit generation
validator
fallback plan
```

输入：

```text
user query
```

输出：

```text
QueryPlan
```

重要文件：

```text
schema.py       QueryPlan / RetrievalUnit 等 contract
service.py      planner service
fallback.py     deterministic planner
llm.py          OpenAI structured planner
validators.py  grounding validator
```

默认 planner model：

```text
gpt-5-nano
```

没有 `OPENAI_API_KEY` 时，fallback planner 仍然能生成 plan。

## 4. Retrieval Plan / Task

实现位置：

```text
src/atlas/retrieval/retrieval_task.py
```

`QueryPlan` 会编译为 `RetrievalTask`：

```text
task_id
unit_id
query_text
filters
lanes
lane_budget
must_have_terms
should_terms
unit_weight
```

为什么需要这一层：

```text
QueryPlan 是语义计划。
RetrievalTask 是 provider 可执行计划。
```

这样 reranker、provider trace 和 eval 都能知道候选证据来自哪个 query unit。

## 5. TextHybridProvider

实现位置：

```text
src/atlas/retrieval/providers/text_hybrid/provider.py
src/atlas/retrieval/providers/text_hybrid/lanes.py
```

Provider 内部 lanes：

| Lane | 当前实现 | 边界 |
|---|---|---|
| Dense Lane | Qdrant dense vector over child chunks | 语义召回 |
| BM25 Lane | Qdrant sparse BM25 over child chunks | 精确词召回 |
| Metric Alias Lane | ontology-expanded BM25 query | 金融指标别名 |
| Section Lane | textual section hints | filing/table title 约束的轻量版本 |
| Table Lane | row/page textual BM25 lane | V1 不做 SQL / cell provenance |

V1 的 Table Lane 是 textual lane。它可以搜索 row label、年份、指标别名，但不维护结构化 table store。

## 6. Weighted RRF

实现位置：

```text
src/atlas/retrieval/fusion.py
```

公式：

```text
score(d) = sum(weight_i / (rrf_k + rank_i(d)))
```

Weighted RRF 的作用：

```text
1. 把 dense、BM25、metric_alias、table 等不同 lane 的候选合并。
2. 避免 raw score 尺度不一致的问题。
3. 用 lane weight 表达策略偏好。
4. 保留每个 candidate 的 contribution trace。
```

trace 会保留：

```text
lane
rank
raw score
weight
weighted contribution
fusion score
```

## 7. Reranker

实现位置：

```text
src/atlas/retrieval/reranker.py
```

当前主路径：

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Reranker 输入包含：

```text
original query
standalone query
entities / periods / metrics
retrieval unit text
must-have / should terms
candidate text
```

输出 trace：

```text
input_rank
output_rank
score
model
latency_ms
top_n
top_m
query_plan_id
retrieval_task_id
retrieval_unit_id
```

## 8. Evidence Builder / EvidencePack

实现位置：

```text
src/atlas/query_runtime/evidence_builder.py
src/atlas/retrieval/evidence_contract.py
```

核心职责：

```text
child -> parent
dedupe
merge
token budget
max blocks
query unit coverage
prompt inclusion / drop reason
```

输出：

```text
EvidenceBlock
EvidencePack
```

EvidencePack 再被转换成旧 runtime 兼容的 `Evidence`，所以 `/v1/query` 旧调用仍可运行。

## 9. Evidence Evaluator

实现位置：

```text
src/atlas/query_runtime/evidence_evaluator.py
```

生成前状态：

```text
supported
insufficient
contradicted
partially_supported
```

关键机制：

```text
如果 evidence 为空，返回 insufficient。
如果多个 supported evidence 对同一关键数字明显冲突，返回 contradicted。
部分覆盖实体/年份/指标时返回 partially_supported，但不一定阻断生成。
```

## 10. Answer Generator

实现位置：

```text
src/atlas/llm/openai_client.py
src/atlas/llm/prompts.py
src/atlas/query_runtime/service.py
```

默认 runtime model：

```text
gpt-5-nano
```

生成只接收 EvidencePack 选中的 evidence，不会读取 FinanceBench JSONL。

## 11. Citation Verifier

实现位置：

```text
src/atlas/query_runtime/citation_verifier.py
src/atlas/query_runtime/critic_lite.py
```

生成后检查：

```text
citation marker 是否存在于 answer
citation_id 是否来自 evidence
引用的关键数字是否出现在 cited evidence
document_id / page_start / page_end metadata 是否一致
```

策略：

```text
不自动补 citation。
invalid / missing citation 属于 unsupported。
数字或 doc/page metadata 不一致属于 warning，除非引用本身不可用。
```

## 12. Trace + Eval + Cache

### Trace 表族

新增 Design trace tables：

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

复用旧表：

```text
query_runs
retrieval_events
generation_events
query_cache
eval_runs
eval_results
```

实现位置：

```text
src/atlas/db/models.py
src/atlas/db/repositories.py
src/atlas/db/session.py
```

Trace API：

```text
GET /v1/query/{query_id}/trace
```

返回：

```text
query
result
retrieval
generation
latency
cache
stages
metadata
v1_trace
model
```

### Eval

实现位置：

```text
src/atlas/eval/v1_full.py
src/atlas/benchmark/financebench.py
src/atlas/benchmark/financebench_retrieval.py
```

Full V1 eval 从 trace 抽取：

```text
component presence
retrieval metrics
evidence metrics
answer metrics
citation metrics
latency metrics
failure buckets
```

### Cache

当前 V1 cache：

```text
Postgres query_cache
cache key schema: atlas-query-cache-v2
```

V1 没有实现 Redis。Redis Queue 属于 V2 Research Runtime。

## 13. 总体验收映射

| V1 Design 节点 | 代码模块 | API | DB 表 | Trace 字段 | Eval 指标 |
|---|---|---|---|---|---|
| User Query | `api/routes/query.py`、`query_runtime/service.py` | `POST /v1/query` | `query_runs` | `query.user_query` | `latency.total_latency_ms` |
| Query Orchestrator | `query_orchestrator/service.py` | `POST /v1/query/plan` | `query_plans` | `result.details.query_plan`、`metadata.query_plan` | `component_presence.query_orchestrator`、`plan_latency_ms` |
| Retrieval Plan | `retrieval/retrieval_task.py` | `POST /v1/query/plan`、`POST /v1/retrieve` | `retrieval_tasks` | `result.details.retrieval_tasks` | `component_presence.retrieval_plan` |
| TextHybridProvider | `retrieval/providers/text_hybrid/provider.py` | `POST /v1/retrieve`、`POST /v1/query` | `retrieval_results` | `result.details.retrieval_trace` | `component_presence.text_hybrid_provider`、`retrieval_latency_ms` |
| Dense Lane | `dense_retriever.py`、`text_hybrid/lanes.py` | same as provider | `candidates` | `lanes`、`lane_attributions` | `component_presence.dense_lane` |
| BM25 Lane | `bm25_retriever.py`、`text_hybrid/lanes.py` | same as provider | `candidates` | `lanes`、`lane_attributions` | `component_presence.bm25_lane` |
| Table Lane | `text_hybrid/lanes.py` | same as provider | `candidates` | `lanes`、`lane_attributions` | `component_presence.table_lane` |
| Provider-local Weighted RRF | `retrieval/fusion.py` | same as provider | `candidates` | `lane_contributions`、`fusion_score` | `component_presence.provider_local_weighted_rrf` |
| Reranker | `retrieval/reranker.py` | same as provider | `candidates` | `reranker`、`reranker_input` | `component_presence.reranker`、`reranker_latency_ms` |
| Evidence Builder | `query_runtime/evidence_builder.py` | `POST /v1/query` | `evidence_blocks`、`evidence_packs` | `evidence_pack`、`coverage` | `selected_block_count`、`dropped_block_count`、`coverage missing counts` |
| Evidence Evaluator | `query_runtime/evidence_evaluator.py` | `POST /v1/query` | `evidence_evaluations` | `critic.evidence_evaluation` | `evaluation_status`、`unsupported/insufficient buckets` |
| Answer Generator | `llm/openai_client.py` | `POST /v1/query` | `answers`、`generation_events` | `result.answer`、`generation` | `answer_gold_contains`、`answer_numeric_match`、`generation_latency_ms` |
| Citation Verifier | `query_runtime/citation_verifier.py` | `POST /v1/query` | `citations`、`citation_verifications` | `critic.citation_verification`、`result.citations` | `citation_doc_hit`、`citation_page_hit`、`citation warnings` |
| Trace + Eval + Cache | `db/repositories.py`、`eval/v1_full.py`、`query_runtime/cache.py` | `GET /v1/query/{id}/trace` | `query_cache`、`eval_runs`、`eval_results` | `v1_trace`、`cache`、`latency` | `component benchmarks`、`failure_buckets`、`cache_hit_rate` |

## 14. 仍然不是 V1 的内容

```text
ResearchJob / ResearchJobEvent
Redis Queue / worker pool
GraphRAG provider
SQLProvider
financial_facts table
cell-level provenance
streaming ingestion
```

这些内容分别属于 V2、V3、V4 或更后续版本。
