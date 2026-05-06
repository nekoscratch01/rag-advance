# V1 实际架构：Atlas Advanced Hybrid Kernel

更新时间：2026-05-06

本文记录当前 V1 runtime 的真实架构。Design 文档说明目标；本文说明已经落地的模块、API、DB trace payload、eval 入口和仍需补齐的 TODO。

核心事实：

```text
最终架构：hybrid / sql / graph 三个 provider。
V1 runtime：只启用 hybrid provider。
V1 provider 实现：TextHybridProvider。
SQLProvider：未实现。
GraphProvider：未实现。
```

---

## 1. Runtime 总览

V1 是单次问答的 hybrid-only Evidence Kernel：

```text
POST /v1/query
  -> QueryRuntime
  -> QueryOrchestrator
  -> QueryPlan
  -> RetrievalTask
  -> TextHybridProvider
  -> Qdrant dense + Qdrant BM25 sparse
  -> Python Weighted RRF
  -> optional CrossEncoder reranker
  -> parent-child EvidencePack
  -> Evidence Evaluator
  -> Answer Generator
  -> Citation Verifier
  -> Postgres trace tables + query_cache
```

运行时存储：

```text
Postgres:
  documents
  parent_blocks
  chunks
  query_runs
  retrieval_events
  generation_events
  query_cache
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

Qdrant:
  dense child vectors
  BM25 sparse child vectors
  chunk payload metadata
```

FinanceBench JSONL 是冻结测评产物，不是 runtime 存储。

---

## 2. Provider 架构现状

| Provider | 代码状态 | V1 可执行 | 说明 |
|---|---|---:|---|
| `hybrid` | `src/atlas/retrieval/providers/text_hybrid/` | 是 | V1 唯一 provider。内部有 dense、BM25、metric_alias、section、table textual lanes。 |
| `sql` | 无 `SQLProvider` runtime | 否 | V4 候选。V1 不做 Text-to-SQL、计算、cell provenance。 |
| `graph` | 无 `GraphProvider` runtime | 否 | V3 候选。V1 不做 GraphRAG 主路径。 |

配置入口：

```text
src/atlas/core/config.py
  Settings.query_planner_enabled_providers = "hybrid"
  enabled_query_providers(settings)
```

依赖装配：

```text
src/atlas/api/dependencies.py
  get_retriever() -> TextHybridProvider
```

V1 的 `QueryPlan` 可以保留最终 provider vocabulary：

```text
ProviderName = Literal["hybrid", "sql", "graph"]
```

但当前 executable plan 只能使用：

```text
provider = "hybrid"
```

如果本地代码仍看到 `retrievers` 字段，应按 provider 语义解释：

```text
retrievers = ["hybrid"]
```

`dense`、`bm25`、`table`、`metric_alias`、`section` 是 `TextHybridProvider` 内部 lane，不是顶层 provider。

---

## 3. API

实现位置：

```text
src/atlas/api/routes/query.py
```

主要 API：

```text
POST /v1/query/plan
POST /v1/retrieve
POST /v1/query
GET  /v1/query/{query_id}
GET  /v1/query/{query_id}/trace
```

请求形态：

```json
{
  "query": "What is the FY2018 capital expenditure amount for 3M?",
  "top_k": 8,
  "filters": {
    "document_ids": ["doc_..."]
  },
  "options": {
    "return_trace": true,
    "retrieval_mode": "hybrid",
    "reranker_enabled": true
  }
}
```

`POST /v1/query/plan` 只返回 plan 和 tasks，不执行检索。
`POST /v1/retrieve` 执行 planner + retrieval，不生成答案。
`POST /v1/query` 执行完整问答链路。

---

## 4. Query Orchestrator

实现位置：

```text
src/atlas/query_orchestrator/schema.py
src/atlas/query_orchestrator/service.py
src/atlas/query_orchestrator/llm_planner.py
src/atlas/query_orchestrator/prompts.py
src/atlas/query_orchestrator/fallback.py
src/atlas/query_orchestrator/validator.py
src/atlas/query_orchestrator/ontology.py
configs/finance_metric_ontology.yaml
```

当前职责：

```text
1. 从 user query 生成 QueryPlan。
2. 抽取或 canonicalize entity / period / metric。
3. 生成 retrieval_units。
4. 用 validator 检查 grounding。
5. LLM planner 不可用或不合格时 fallback。
```

默认模型：

```text
Settings.query_planner_model = "gpt-5-nano"
```

没有 `OPENAI_API_KEY` 时：

```text
build_fallback_plan(...)
```

仍会生成保守 hybrid plan。

### 4.1 Planner Prompt Paradox

实际风险：

```text
schema 知道 hybrid / sql / graph。
runtime 只装配 TextHybridProvider。
如果 prompt 让 LLM 自由选择 provider，LLM 会为 SQL-like 或 graph-like query 输出 sql / graph。
这些 provider 在 V1 不可执行。
```

V1 架构要求：

```text
prompt 必须动态注入 enabled_providers。
V1 enabled_providers 必须是 ["hybrid"]。
disabled provider 必须在 prompt 中标注为不可输出。
```

代码锚点：

```text
src/atlas/core/config.py
  enabled_query_providers(settings)
  Settings.query_planner_enabled_providers
  Settings.query_planner_retry_count

src/atlas/query_orchestrator/prompts.py
  QUERY_PLANNER_INSTRUCTIONS
  build_query_planner_input(...)
```

当前实现：

```text
prompts.py 会把 enabled_query_providers(settings) 注入 planner instructions。
llm_planner.py 会按 enabled_providers 收窄 schema enum。
validation 失败会把错误反馈给 LLM 重试。
```

### 4.2 Compound Unit Retry

`RetrievalUnit` 必须是 single-provider unit。

错误形态：

```yaml
retrieval_units:
  - unit_id: u1
    provider: [sql, hybrid]
```

V1 正确处理：

```text
1. validator 报 compound_unit_must_be_split 或 disabled_provider。
2. LLM planner 带错误原因重试。
3. 重试 prompt 要求只输出 enabled_providers 里的 single-provider units。
4. retry 次数耗尽后，fallback 到 deterministic hybrid-only plan。
```

代码锚点：

```text
src/atlas/query_orchestrator/schema.py
  RetrievalUnit._retrievers_single_provider(...)

src/atlas/core/config.py
  Settings.query_planner_retry_count
```

TODO：

```text
把 disabled_provider retry reason 和 retry_count 写入 query_plans / trace payload。
继续用测试锁住 fallback.py 不输出 sql / graph。
```

### 4.3 Unit-first 约束

当前实现可以返回完整 `QueryPlan`，但架构上应逐步收敛到 Unit-first：

```text
LLM 主要提出 retrieval_units。
系统从 original query + accepted units + ontology derive constraints。
validator 检查 derived constraints 是否 grounded。
compiler 生成最终 QueryPlan / RetrievalTask。
```

原因：

```text
避免 LLM 同时在 global periods 和 unit terms 中声明互相冲突的事实。
```

---

## 5. RetrievalTask 编译

实现位置：

```text
src/atlas/retrieval/retrieval_task.py
```

输入：

```text
QueryPlan.retrieval_units
QueryPlan.metadata_filter
RetrievalUnit.metadata_filter
```

输出：

```text
RetrievalTask:
  task_id
  plan_id
  unit_id
  provider
  query_text
  metadata_filter
  provider_status
  unsupported_reason
  internal_lanes
  must_have_terms
  should_terms
  top_k
  weight
  lane_weights
  metadata
```

字段边界：

```text
QueryPlan.metadata_filter / RetrievalUnit.metadata_filter:
  contract 层字段。

RetrievalTask.metadata_filter:
  provider 执行层字段。
  它应由 plan.metadata_filter + unit.metadata_filter 合并而来。
```

计划层不再保留旧 `filters` 字段；请求 API 的 `filters` 仍是调用方级别的外部过滤入口，不属于 QueryPlan contract。

---

## 6. TextHybridProvider

实现位置：

```text
src/atlas/retrieval/providers/text_hybrid/provider.py
src/atlas/retrieval/providers/text_hybrid/lanes.py
```

依赖：

```text
src/atlas/retrieval/dense_retriever.py
src/atlas/retrieval/bm25_retriever.py
src/atlas/retrieval/fusion.py
src/atlas/retrieval/reranker.py
```

Provider 内部 lanes：

| Lane | backend | V1 边界 |
|---|---|---|
| `dense` | Qdrant dense vector | semantic child chunk recall |
| `bm25` | Qdrant BM25 sparse vector | lexical child chunk recall |
| `metric_alias` | BM25 sparse text with ontology aliases | 金融指标别名召回 |
| `section` | BM25 sparse text with section hints | filing/table title 的轻量文本约束 |
| `table` | BM25 sparse text over serialized table text | stopgap，不是 SQL |

### 6.1 内部执行流

每个 `RetrievalTask` 应执行：

```text
dense_text = task.query_text
sparse_text = task.query_text + task.should_terms + repeat(task.must_have_terms, 3)
metadata_filter = task.metadata_filter
qdrant_filter = metadata_filter -> Qdrant payload filter
```

随后：

```text
1. DenseRetriever.embed_query(dense_text)
2. Qdrant dense vector query
3. BM25SparseEncoder.embed_query(sparse_text)
4. Qdrant sparse vector query
5. TextHybridLane.annotate_candidate(...)
6. weighted_rrf_fuse(...)
7. optional rerank_with_context(...)
8. build_evidence_pack_from_candidates(...)
```

代码锚点：

```text
src/atlas/retrieval/providers/text_hybrid/lanes.py
  lane_query_text(...)
  lane_filters(...)
  annotate_candidate(...)

src/atlas/retrieval/dense_retriever.py
  DenseRetriever.retrieve_candidates(...)

src/atlas/retrieval/bm25_retriever.py
  BM25Retriever.retrieve_candidates(...)
```

当前实现注意事项：

```text
lane_query_text("metric_alias") 已拼入 should_terms。
bm25 lane 已收敛到 sparse_text = task.query_text + should_terms + repeated must_have_terms。
Dense lane 保持 dense_text = task.query_text。
must_have_terms 的 sparse boost 使用字符串重复 3 次，不改 BM25 底层公式。
```

### 6.2 metadata_filter 到 Qdrant payload filter

当前 Qdrant filter 代码位置：

```text
src/atlas/retrieval/dense_retriever.py
  _build_filter(...)

src/atlas/retrieval/bm25_retriever.py
  _build_filter(...)
```

当前已支持：

```text
document_ids -> FieldCondition(key="document_id", MatchAny(...))
```

架构目标：

```text
document_ids
company
filing_type
fiscal_year
source_title
```

都应可以从 `metadata_filter` 编译为 Qdrant payload filter，并在 candidate metadata / trace 中保留。

TODO：

```text
扩展 _build_filter(...)，不要只支持 document_ids。
确保 Qdrant payload keys 与 Postgres metadata_json 命名一致。
```

### 6.3 must_have_terms 的实际边界

当前 `must_have_terms` 会进入：

```text
RetrievalTask
lane_trace
reranker context
cache stable task payload
BM25 sparse_text boost
```

它不是默认硬过滤条件。

架构结论：

```text
must_have_terms 是 experimental retrieval variable。
它可以用于 sparse boost、reranker hint、coverage trace、post-filter ablation。
它不能单独代表 entity / period / metric 的完整语义约束。
```

TODO：

```text
补 must_have_terms hard-filter / sparse-boost / off 的真实 FinanceBench eval。
如果要做 hard lexical filter，必须单独报告误杀率。
```

### 6.4 Fusion

当前默认：

```text
src/atlas/retrieval/fusion.py
  weighted_rrf_fuse(...)
```

trace payload：

```text
lane
rank
raw_score
weight
unit_weight
lane_weight
weighted_contribution
fusion_score
fusion_rank
retrieval_task_id
retrieval_unit_id
```

当前实现是 Python Weighted RRF。Qdrant RRF 是 V1 可选 ablation，不是已证明主路径。

TODO：

```text
补 Qdrant RRF path 的等价 trace。
跑 Python Weighted RRF vs Qdrant RRF 对比。
```

---

## 7. Table Lane

V1 table lane 是：

```text
serialized table text retrieval only
```

实现位置：

```text
src/atlas/retrieval/providers/text_hybrid/lanes.py
  lane_query_text(..., lane="table")
```

它只是把 table textual hints 加到 sparse query 中，例如：

```text
table row page
row label
table title
year
metric alias
```

它不提供：

```text
SQL
calculation
cell provenance
row/column ids
cell bbox
formula verification
```

因此 `table` lane 输出的 candidate 仍然是 `provider="hybrid"` 的 text evidence。

---

## 8. Graph 边界

V1 没有 GraphProvider。

未来 graph 的合理定位：

```text
GraphProvider -> graph candidates -> source text grounding -> EvidenceBlock
```

不允许：

```text
graph node / edge / community summary -> VIP EvidenceBlock -> answer
```

原因：

```text
1. Graph hub node 容易爆炸。
2. high-degree node 会把候选池拉向泛化主题。
3. graph_score 与 BM25/dense score 不是同一尺度。
4. graph 关系不是 SQL-like exact fact。
5. 没有 source anchor 的 graph claim 不能作为 citation。
```

AAPL/MSFT/Vision Pro 这类 query 在 V1 必须走 hybrid-only path，不能声称执行了 graph traversal。

---

## 9. Evidence Builder / EvidencePack

实现位置：

```text
src/atlas/query_runtime/evidence_builder.py
src/atlas/retrieval/evidence_contract.py
```

职责：

```text
child -> parent
dedupe
merge
token budget
max blocks
query unit coverage
prompt inclusion
drop reason
```

输出：

```text
EvidenceBlock
EvidencePack
```

V1 EvidenceBlock 的 provider 应为：

```text
hybrid / text_hybrid
```

当前代码有兼容命名：

```text
metadata.provider = "text_hybrid"
metadata.retrieval_provider = "text_hybrid"
metadata.provider_contract = "TextHybridProvider"
```

架构语义：

```text
text_hybrid 是 hybrid provider 的实现名。
```

---

## 10. Evidence Evaluator / Citation Verifier

实现位置：

```text
src/atlas/query_runtime/evidence_evaluator.py
src/atlas/query_runtime/citation_verifier.py
src/atlas/query_runtime/critic_lite.py
```

生成前状态：

```text
supported
insufficient
contradicted
partially_supported
```

生成后检查：

```text
citation marker 是否存在
citation_id 是否来自 evidence set
关键数字是否出现在 cited evidence 中
document_id / page metadata 是否一致
```

边界：

```text
Citation Builder 只解析 citation。
Citation Verifier 才检查支持关系。
系统不自动补 citation。
```

---

## 11. QueryRuntime 与 Cache

实现位置：

```text
src/atlas/query_runtime/service.py
src/atlas/query_runtime/cache.py
```

运行顺序：

```text
1. normalize query
2. plan query
3. build cache key
4. query_cache lookup
5. retrieve evidence
6. pre_generation_critic
7. generate answer
8. build citations
9. post_generation_critic
10. write trace
11. supported answer 写入 query_cache
```

Cache 事实：

```text
backend: Postgres query_cache
schema: atlas-query-cache-v2
```

V1 没有 Redis cache。Redis Queue 属于 V2 Research Runtime。

---

## 12. Trace Payload

Trace API：

```text
GET /v1/query/{query_id}/trace
```

返回顶层：

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

DB 写入位置：

```text
src/atlas/db/models.py
src/atlas/db/repositories.py
src/atlas/db/session.py
```

V1 trace family：

| 表 | payload 重点 |
|---|---|
| `query_plans` | `plan_id`、`planner`、`enabled_providers`、`retrieval_units`、`metadata_filter` |
| `retrieval_tasks` | `task_id`、`unit_id`、`provider`、`query_text`、`metadata_filter`、`provider_status`、`internal_lanes` |
| `retrieval_results` | `retrieval_trace`、stage status、retrieval events |
| `candidates` | `chunk_id`、rank、lane trace、fusion trace、reranker trace |
| `evidence_blocks` | selected evidence text、page、coverage、provider metadata |
| `evidence_packs` | token budget、included/dropped blocks、drop reason |
| `evidence_evaluations` | pre-generation status、warnings、coverage |
| `answers` | answer、confidence、model、prompt_version、generation event |
| `citations` | citation id、evidence id、doc/page metadata |
| `citation_verifications` | post-generation status、warnings、unsupported reasons |

重要 nested payload：

```text
result.details.query_plan
result.details.retrieval_tasks
result.details.retrieval_trace
result.details.critic.evidence_evaluation
result.details.critic.citation_verification
v1_trace.candidates[].payload.metadata.fusion
v1_trace.candidates[].payload.metadata.lane_attributions
```

TODO：

```text
把 enabled_providers 和 disabled_provider retry reason 写入 query_plans payload。
把 metadata_filter -> qdrant_filter 编译结果写入 retrieval_tasks 或 retrieval_results payload。
```

---

## 13. Eval

实现位置：

```text
src/atlas/eval/v1_full.py
src/atlas/eval/runner.py
src/atlas/benchmark/financebench.py
src/atlas/benchmark/financebench_retrieval.py
benchmarks/rag_quality/financebench/reports/full_v1_retrieval_experiment.md
```

已存在证据：

```text
FinanceBench retrieval-only eval 已有报告。
full V1 generated-answer runner 已有代码入口。
完整生成式答案可靠性报告仍需单独跑数归档。
```

V1 eval 应覆盖：

```text
component_presence.query_orchestrator
component_presence.text_hybrid_provider
component_presence.provider_local_weighted_rrf
component_presence.reranker
retrieval doc/page hit
evidence doc/page hit
answer_numeric_match
citation_doc_hit
citation_page_hit
unsupported / insufficient buckets
latency by stage
cache hit/miss
```

新增专项 TODO：

```text
1. planner_disabled_provider_rate
2. compound_unit_retry_success_rate
3. metadata_filter_hit_rate
4. must_have_terms off/on ablation
5. sparse_text = unit.text vs unit.text + should_terms vs must_terms sparse boost ablation
6. Python Weighted RRF vs Qdrant RRF ablation
7. serialized table textual lane off/on ablation
8. AAPL/MSFT/Vision Pro hybrid-only regression case
```

---

## 14. AAPL / MSFT / Vision Pro 实际路径

用户问题：

```text
Compare AAPL and MSFT exposure to Vision Pro based on annual report evidence.
```

V1 不能执行：

```text
Graph traversal over Vision Pro hub node
SQL mention count
structured exposure calculation
cell provenance lookup
```

V1 应执行：

```text
1. QueryOrchestrator 读取 enabled_providers = ["hybrid"]。
2. Planner 生成 hybrid-only retrieval_units。
3. RetrievalTask 编译 metadata_filter，例如 filing_type / document_ids。
4. TextHybridProvider 对每个 unit 执行 dense_text = unit.text。
5. TextHybridProvider 对每个 unit 执行 sparse_text = unit.text + should_terms + repeat(must_have_terms, 3)。
6. Qdrant dense / sparse 分别召回 source chunks。
7. Weighted RRF 合并候选。
8. Reranker 精排。
9. Evidence Builder 扩展到 parent/page blocks。
10. Answer Generator 基于 annual report source evidence 做保守比较。
```

示例 hybrid units：

```yaml
retrieval_units:
  - unit_id: u0
    provider: hybrid
    purpose: original
    text: "AAPL MSFT Vision Pro annual report exposure"
    should_terms:
      - Apple Vision Pro
      - Microsoft annual report
      - mixed reality
      - spatial computing
    metadata_filter:
      filing_type: ["10-K"]

  - unit_id: u1
    provider: hybrid
    purpose: apple_source_text
    text: "Apple AAPL Vision Pro annual report product discussion"
    should_terms:
      - Apple Vision Pro
      - products
      - services

  - unit_id: u2
    provider: hybrid
    purpose: microsoft_source_text
    text: "Microsoft MSFT Vision Pro annual report product partnership discussion"
    should_terms:
      - Microsoft
      - Apple Vision Pro
      - mixed reality
      - devices
```

如果证据不足，正确输出是：

```text
当前导入的文档中没有检索到足够证据回答这个问题。
```

或带明确 caveat 的 grounded answer，而不是声称 graph/sql 已得出结论。

---

## 15. 模块映射

| 架构节点 | 模块 | API | DB / payload | Eval |
|---|---|---|---|---|
| QueryRuntime | `query_runtime/service.py` | `POST /v1/query` | `query_runs`、`details_json` | total latency、confidence |
| Query Orchestrator | `query_orchestrator/service.py` | `POST /v1/query/plan` | `query_plans.payload_json` | planner latency、disabled provider rate |
| LLM Planner | `query_orchestrator/llm_planner.py` | plan API | `planner`、`model` metadata | retry/fallback rate |
| Prompt Builder | `query_orchestrator/prompts.py` | plan API | enabled providers metadata | prompt compliance |
| Validator | `query_orchestrator/validator.py` | plan API | validation warnings/reasons | validation failure buckets |
| RetrievalTask compiler | `retrieval/retrieval_task.py` | plan/retrieve/query | `retrieval_tasks.payload_json` | task count、unit coverage |
| TextHybridProvider | `retrieval/providers/text_hybrid/provider.py` | retrieve/query | `retrieval_results`、`candidates` | retrieval latency |
| Lanes | `retrieval/providers/text_hybrid/lanes.py` | retrieve/query | `lane_trace`、`lane_attributions` | lane contribution |
| Dense retriever | `retrieval/dense_retriever.py` | retrieve/query | dense rank/score | dense hit@k |
| BM25 retriever | `retrieval/bm25_retriever.py` | retrieve/query | lexical rank/score | sparse hit@k |
| Fusion | `retrieval/fusion.py` | retrieve/query | `fusion`、`lane_contributions` | RRF ablation |
| Reranker | `retrieval/reranker.py` | retrieve/query | rerank rank/score/model | reranker lift |
| Evidence Builder | `query_runtime/evidence_builder.py` | query | `evidence_blocks`、`evidence_packs` | evidence doc/page hit |
| Verifier | `evidence_evaluator.py`、`citation_verifier.py` | query | `evidence_evaluations`、`citation_verifications` | unsupported rate |
| Cache | `query_runtime/cache.py` | query | `query_cache` | cache hit rate |

---

## 16. 当前 TODO

```text
Planner:
  - Unit-first contract 收敛

RetrievalTask:
  - 持续检查 provider_status / unsupported_reason 在 trace 中的可读性
  - metadata_filter -> Qdrant payload filter 只对真实 payload 字段做 hard filter
  - company / metric / table hints 这类语义 metadata 不应硬过滤，后续可做 boost/reranker context

TextHybridProvider:
  - must_have_terms hard-filter / sparse-boost / off ablation
  - Qdrant RRF path 与 Python Weighted RRF 对比

Table Lane:
  - 明确 serialized table text quality eval
  - 不引入 SQL / calculation / cell provenance 到 V1

Graph:
  - 保持未启用
  - 未来只作为 candidate generator
  - 设计 hub node explosion 防护

Eval:
  - full generated-answer reliability report
  - AAPL/MSFT/Vision Pro regression case
  - citation_doc/page_hit 与 answer_numeric_match 全量报告
```

---

## 17. 结论

V1 的真实架构不是“三个 provider 都已经实现”。

准确说法是：

```text
Atlas 已经把 Evidence Kernel 的 provider contract 预留为 hybrid / sql / graph。
V1 runtime 当前只启用 hybrid。
TextHybridProvider 是 V1 的唯一 provider 实现。
SQL 和 graph 在 V1 中只能作为未来架构上下文出现，不能出现在 executable plan 或答案事实里。
```
