# V1 实际架构：Atlas Advanced Hybrid Kernel

更新时间：2026-05-08

本文记录当前 V1 runtime 的真实架构。Design 文档说明目标；本文说明已经落地的模块、API、DB trace payload、eval 入口和仍需补齐的 TODO。

核心事实：

```text
最终架构：hybrid / sql / graph 三个 provider。
V1 baseline：只启用 hybrid provider。
当前默认 Atlas runtime：hybrid + graph 可执行，仍由 QueryPlan 决定是否调用 graph。
V1 provider 实现：TextHybridProvider。
SQLProvider：V1 受控单表 Text-to-SQL proof 已实现；默认仍不可执行，必须同时设置 sql_provider_enabled=true 且 query_runtime_executable_providers 显式包含 sql。
GraphProvider：V3.0 walking skeleton 已默认注册为可执行 provider；不声明质量提升。
```

---

## 1. Runtime 总览

V1 baseline 是单次问答的 hybrid-executable Evidence Kernel；当前默认 runtime 已在 ProviderRouter 下同时注册 hybrid 与 graph：

```text
POST /v1/query
  -> QueryRuntime
  -> QueryOrchestrator
  -> QueryPlan
  -> RetrievalTask
  -> ProviderRouter
       hybrid -> TextHybridProvider
              -> Qdrant dense + Qdrant BM25 sparse
              -> provider-local Python Weighted RRF
       graph  -> GraphProvider local/path walking skeleton, only when QueryPlan selects graph
       sql    -> 默认 skipped_non_executable；显式 opt-in 后进入 SQLProvider V1 单表链路
  -> CandidateAdapter
  -> CandidateFusion
  -> optional global CrossEncoder reranker
  -> EvidenceBuilder / parent-child EvidencePack
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
| `hybrid` | `src/atlas/retrieval/providers/text_hybrid/` | 是 | V1 默认 provider。内部有 dense、BM25、metric_alias、section、table textual lanes。 |
| `sql` | `src/atlas/retrieval/providers/sql/` | 默认否；显式双开关后是 | V4 SQLProvider V1 proof：单表 schema routing、identifier normalization、compiler、validator、DuckDB executor、deterministic SQL result evidence。不声明生成式答案可靠性或 cell-level citation 完成。 |
| `graph` | `src/atlas/retrieval/providers/graph/` | 当前默认是；可显式关闭 | V3.0 walking skeleton：local/path、Postgres grounding、trace auditability；不声明质量提升；只在 QueryPlan 选择 graph 时运行。 |

配置入口：

```text
src/atlas/core/config.py
  Settings.query_planner_known_providers = "hybrid,sql,graph"
  Settings.query_runtime_executable_providers = "hybrid,graph"
  IMPLEMENTED_RUNTIME_PROVIDERS = ("hybrid", "graph", "sql")
  sql_provider_enabled = false
  known_query_providers(settings)
  executable_query_providers(settings) = requested ∩ registered_runtime_provider ∩ known - conditional_non_executable(sql)
  sql 只有在 sql_provider_enabled=true 且 requested 显式包含 sql 时才可执行
```

Provider registry / dependency 装配：

```text
src/atlas/core/registry.py
  ComponentRegistry(namespace="retrieval_provider")

src/atlas/retrieval/providers/registry.py
  provider_registry built-ins = hybrid / graph / sql
  build_provider(name, ProviderBuildContext)

src/atlas/api/dependencies.py
  executable_query_providers(settings)
  build_provider("hybrid" | "graph" | opt-in "sql", ProviderBuildContext)
  get_provider_router() -> ProviderRouter(..., session_factory=SessionLocal)
```

Router 注册边界：

```text
provider 必须显式继承 RetrievalProvider ABC。
sql 是 known semantic provider；默认 runtime 视为 non-executable，显式双开关 opt-in 后才允许注册/执行。
dense / bm25 / sparse / table / section / metric_alias 是 internal lanes，注册为 provider 会被拒绝。
unknown provider 会在 router 注册期失败，而不是运行时静默降级。
```

V1 的 `QueryPlan` provider 字段是 registry-facing string，但 reserved internal lanes 会被 schema 拒绝：

```text
known semantic providers = hybrid / sql / graph
reserved internal lanes = dense / bm25 / sparse / table / section / metric_alias
```

当前 semantic plan 可以使用：

```text
provider = "hybrid" | "sql" | "graph"
```

当前默认 execution 能执行：

```text
provider = "hybrid" | "graph"
```

SQLProvider opt-in execution：

```bash
ATLAS_SQL_PROVIDER_ENABLED=true
ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid,sql,graph
```

当前 SQLProvider V1 只声明受控单表最小闭环，不声明完整答案可靠性。
SQLProvider V1 的 DuckDB timeout trace 使用 `timeout_isolation=thread_only`；当前未实现 worker-process isolation，也不声明能强杀 native DuckDB query。

V1 baseline 回退口径是：

```bash
ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid
```

Phase 6 已将 graph 从 opt-in 提升为默认可执行 provider；这不等于 graph-first，实际调用仍由 QueryPlan 控制。

如果本地代码仍看到 `retrievers` 字段，应按 provider 语义解释：

```text
retrievers = ["hybrid"]  # legacy input alias
provider = "hybrid"      # canonical field
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
默认 runtime 通过 ProviderRouter 注册 TextHybridProvider 和 GraphProvider。
如果 LLM 为 SQL-like 或 graph-like query 输出 sql / graph，这是合法 semantic plan。
sql 默认不可执行，会生成 skipped_non_executable ProviderResult；显式双开关 opt-in 后才进入 SQLProvider V1 单表链路。
graph 默认可执行 local/path walking skeleton；仍不能绕过 Evidence Kernel。
```

V1 架构要求：

```text
prompt 必须声明 known_providers = ["hybrid", "sql", "graph"]。
默认 runtime 声明 executable_providers = ["hybrid", "graph"]。
V1 baseline 可显式声明 executable_providers = ["hybrid"]。
non-executable provider 必须进入 trace，不进入 evidence。SQLProvider opt-in 成功时输出 pinned / rerankable=false 的 sql_result evidence。
默认不自动 hybrid_backfill。
```

代码锚点：

```text
src/atlas/core/config.py
  known_query_providers(settings)
  executable_query_providers(settings)
  Settings.query_planner_known_providers
  Settings.query_runtime_executable_providers
  Settings.query_planner_retry_count

src/atlas/query_orchestrator/prompts.py
  QUERY_PLANNER_INSTRUCTIONS
  build_query_planner_input(...)
```

当前实现：

```text
prompts.py 会把 known_query_providers(settings) 注入 planner instructions。
llm_planner.py 会按 known_providers 生成 schema enum。
validation 失败会把错误反馈给 LLM 重试。
ProviderRouter 会把不可执行的 sql 编译为 skipped_non_executable provider result。
默认注册 GraphProvider 后，graph task 可执行 local/path，但仍不能绕过 Evidence Kernel。
```

### 4.2 Compound Unit Retry

`RetrievalUnit` 必须是 single-provider unit。

错误形态：

```yaml
retrieval_units:
  - unit_id: u3
    provider: [sql, hybrid]
```

V1 正确处理：

```text
1. validator 报 compound_unit_must_be_split。
2. LLM planner 带错误原因重试。
3. 重试 prompt 要求输出 single-provider units。
4. sql/graph 是 known provider；sql 是 valid plan、non-executable execution，graph 在当前默认 runtime 中是 planner-selected executable provider。
```

代码锚点：

```text
src/atlas/query_orchestrator/schema.py
  RetrievalUnit.provider
  RetrievalUnit.retrievers legacy alias

src/atlas/core/config.py
  Settings.query_planner_retry_count
```

TODO：

```text
把 unknown_provider retry reason 和 retry_count 写入 query_plans / trace payload。
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
src/atlas/retrieval/models/retrieval_task.py
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

主路径结果对象：

```text
TextHybridRun
  candidates
  evidence
  evidence_pack
  trace
  latency_ms
```

`retrieve_provider_result()`、`retrieve_with_plan()`、`retrieve_candidates_with_plan()` 都从一次局部 `TextHybridRun` 派生结果；Provider singleton 不再依赖 `last_retrieval_trace` / `last_evidence_pack` 这类实例状态回传主链路数据。

依赖：

```text
src/atlas/retrieval/providers/text_hybrid/adapters/dense.py
src/atlas/retrieval/providers/text_hybrid/adapters/bm25.py
src/atlas/retrieval/providers/text_hybrid/adapters/hybrid.py
src/atlas/retrieval/providers/text_hybrid/adapters/mode_switching.py
src/atlas/retrieval/ranking/fusion.py
src/atlas/retrieval/ranking/reranker.py
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

src/atlas/retrieval/providers/text_hybrid/adapters/dense.py
  DenseRetriever.retrieve_candidates(...)

src/atlas/retrieval/providers/text_hybrid/adapters/bm25.py
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
src/atlas/retrieval/providers/text_hybrid/adapters/dense.py
  _build_filter(...)

src/atlas/retrieval/providers/text_hybrid/adapters/bm25.py
  _build_filter(...)
```

当前已支持：

```text
document_ids -> FieldCondition(key="document_id", MatchAny(...))
section_name -> section_title
document_type / filing_type -> file_type
parent_id / title / source_uri / page_start / page_end / language / embedding_model
```

边界：

```text
只对 Qdrant payload 中真实存在且白名单允许的字段做 hard filter。
不支持的 metadata_filter key 会被忽略，避免构造无效 payload filter。
```

`metadata_filter` 原始值仍会保留在 task/lane trace 中，便于审计 planner 约束和实际 payload filter 能力之间的差异。

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
src/atlas/retrieval/ranking/fusion.py
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

当前默认 runtime 已启用 GraphProvider。V3.0 已在以下路径实现 walking skeleton：

```text
src/atlas/retrieval/providers/graph/
```

V3.0 graph 的实际定位：

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

AAPL/MSFT/Vision Pro 这类 query 的 semantic plan 可以包含 graph/sql 意图。当前默认 runtime 会执行 graph task 的 local/path walking skeleton；SQL 仍 skipped。没有质量测评前，不能声称 graph 改善检索或答案可靠性。

---

## 9. Evidence Builder / EvidencePack

实现位置：

```text
src/atlas/query_runtime/evidence_builder.py
src/atlas/retrieval/models/evidence_contract.py
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

EvidenceBlock 的 contract provider 应为：

```text
V1 baseline: hybrid
current default: hybrid | graph
```

当前代码有兼容命名：

```text
metadata.provider = "text_hybrid"              # 实现元数据，不是 QueryPlan provider
metadata.retrieval_provider = "text_hybrid"    # 兼容字段，不是 canonical provider
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
schema: atlas-query-cache-v3
```

V1 没有 Redis cache。Redis Queue 属于 V2 Research Runtime。

---

## 12. Trace Payload

Trace API：

```text
GET /v1/query/{query_id}/trace
GET /v1/query/{query_id}/trace?include_raw_llm_io=true
```

默认 `include_raw_llm_io=false`。API 返回的 `v1_trace.llm_calls[]` 会 redacted
`request` / `response` / `instructions_text` / `input_text` /
`raw_output_text` / `parsed_answer_text`，`v1_trace.llm_call_evidence[]`
会 redacted `text_snapshot`。只有显式 query parameter、`X-Atlas-Include-Raw-Llm-Io`
header 或
`ATLAS_TRACE_INCLUDE_RAW_LLM_IO_DEFAULT=true` 才返回完整 raw LLM I/O。

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
| `query_runs` | `details_json.llm_io` / `planner_llm` 轻量状态/指针、query-level stage/details、latency、confidence |
| `query_plans` | `plan_id`、`planner_call_id`、`planner`、`known_providers`、`executable_providers`、`retrieval_units`、`metadata_filter` |
| `retrieval_tasks` | `task_id`、`unit_id`、`provider`、`query_text`、`metadata_filter`、`provider_status`、`internal_lanes` |
| `retrieval_results` | `provider_router_trace`、`provider_results`、`retrieval_trace`、stage status、retrieval events |
| `candidates` | `chunk_id`、rank、lane trace、fusion trace、reranker trace |
| `evidence_blocks` | selected evidence text、page、coverage、provider metadata |
| `evidence_packs` | token budget、included/dropped blocks、drop reason |
| `evidence_evaluations` | pre-generation status、warnings、coverage |
| `answers` | answer、confidence、model、prompt_version、generation event、`answer_call_id` |
| `llm_calls` | planner / answer LLM exact request/response/raw text、usage、latency、hash、retention、governance status |
| `llm_call_evidence` | answer prompt evidence snapshot、rank、provider、chunk/page、hash、retention |
| `citations` | citation id、evidence id、doc/page metadata |
| `citation_verifications` | post-generation status、warnings、unsupported reasons |

### 12.1 Generation LLM I/O observability

V1 `/v1/query` 和 V3.0 graph-enabled runtime 共用 QueryRuntime 生成链路。生成答案时，Answer Generator 的 LLM I/O 不再写入 `query_runs.details_json` 或 `answers.payload_json` 的 raw JSON 旁路。

```text
query_runs.details_json.llm_io -> status + answer_llm_call_id/error/reason
answers.answer_call_id
answers.payload_json.answer_llm_call_id
llm_calls
llm_call_evidence
```

`llm_io` 现在只是轻量指针：

```text
completed: {"status": "completed", "answer_llm_call_id": "..."}
failed:    {"status": "failed", "answer_llm_call_id": "...", "error_message": "..."}
skipped:   {"status": "skipped", "reason": "..."}
```

`llm_calls` 保存 Answer Generator 的 exact sent request（`model`、`instructions`、`input`、`max_output_tokens`、`reasoning`、`store`）、raw output text、parsed answer/confidence、usage token、latency、model、prompt_version、reasoning_effort、store、status/error_message、hash、retention_expires_at、redaction/encryption status。

`llm_call_evidence` 保存进入 answer LLM prompt 的 evidence snapshot：rank、provider、chunk_id、document_id、page_start/end、retrieval_score、token_count、text_snapshot、text_hash、retention，并尽量通过 `evidence_block_record_id` 链接 `evidence_blocks`。

AnswerRecord 的 `payload_json` 只保存：

```text
answer
confidence
model
prompt_version
generation_event
answer_llm_call_id
```

生成失败时仍会创建 answer-stage `llm_calls` 记录，保存 request、error_message 和 prompt evidence；没有调用 LLM 的路径使用 `llm_io.status="skipped"` 和 `reason`，例如 cache hit、retrieval failure、no evidence 或 pre-generation critic 拦截，不创建 answer call。

安全边界：

```text
不保存 API key。
不保存 Authorization header。
不保存 transport-level secret。
```

`llm_calls.input_text` / `request_json.input` 和 `llm_call_evidence.text_snapshot` 会持久化用户 query 与进入 prompt 的 evidence text；这属于可观察性取证数据，不应当按普通非敏日志处理。当前 trace API 默认 redacted raw LLM I/O，只在显式 opt-in 时返回完整 raw。生产环境仍需要在 trace API 权限、retention、导出和 redaction 策略里显式覆盖这一类 payload。

旧版本或迁移前写入的 legacy rows 可能仍在 `query_runs.details_json` 或 `answers.payload_json` 中包含旧 raw `llm_io` / planner prompt / answer payload。当前版本没有做 full legacy scrub migration；生产或共享环境使用前需要单独 scrub/backfill。

### 12.2 Planner LLM I/O observability

V1 planner 真实调用 LLM 时，planner-stage I/O 写入同一张结构化表：

```text
llm_calls(stage="planner")
query_plans.planner_call_id
query_runs.details_json.planner_llm -> status + planner_llm_call_id / validation_status / error
query_plans.payload_json.metadata.planner_llm_call_id
```

`llm_calls` 保存 planner 的 exact sent request、raw output text、parsed JSON、parsed plan id、usage、latency、attempt index、model、planner_version、validation_status/error、hash、retention、redaction/encryption status。

`query_runs.details_json` 和 `query_plans.payload_json` 不保存 raw planner prompt/response，只保留 `planner_llm_call_id`、planner status、validation status、fallback reason 等轻量字段。LLM validation fallback 或 exception fallback 仍会记录 `status="invalid"` / `status="failed"` 的 planner call；fallback plan 会标记 `quality_eligible=false` 和 `not_quality_reason="planner_fallback_not_quality_run"`。

重要 nested payload：

```text
query_runs.details_json.llm_io
query_runs.details_json.planner_llm
answers.payload_json.answer_llm_call_id
v1_trace.llm_calls[]
v1_trace.llm_call_evidence[]
result.details.query_plan
result.details.retrieval_tasks
result.details.provider_router_trace
result.details.provider_results
result.details.retrieval_trace
result.details.critic.evidence_evaluation
result.details.critic.citation_verification
v1_trace.candidates[].payload.metadata.fusion
v1_trace.candidates[].payload.metadata.lane_attributions
```

Multi-provider runtime 下，`ProviderRouter` 先收集 provider candidates，再由
`CandidateFusion` 做跨 provider 去重、provenance/source_anchor 合并和全局 candidate
window。V4 preflight 后，candidate contract 带 `rerankable`、`fusion_policy`
和 `structured_payload`；global reranker 只看 rerankable text candidates，`pinned` /
`supporting` structured candidates 不走 CrossEncoder，但仍可进入 EvidencePack。
`EvidenceBuilder` 再把全局排序后的 candidates 编成 prompt evidence，并统一重编号为全局 `c1..cN`。provider 内部的局部 evidence id /
rank 不改写，而是投影到 `retrieval_trace.top_k[]` 的
`original_evidence_id`、`provider_local_evidence_id`、`provider_local_rank` 和
`provider_local_provider`，用于把 LLM citation 映射回 provider-local trace。

面向 API / `details_json` 的 provider failure trace 会做轻量 scrub：不暴露
`error_message`、`planned_text` 等可能包含原始 query / planner text 的字段；内部
`ProviderResult.trace` 仍可保留 debugging 信息供 router 组装和测试断言使用。

TODO：

```text
把 metadata_filter -> qdrant_filter 编译结果写入 retrieval_tasks 或 retrieval_results payload。
补 legacy details_json / answers payload raw LLM I/O scrub/backfill migration。
```

---

## 13. Eval

实现位置：

```text
src/atlas/eval/v1_full.py
src/atlas/eval/runner.py
src/atlas/benchmark/financebench.py
src/atlas/benchmark/financebench_retrieval.py
benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506/report.md
benchmarks/rag_quality/v1_hybrid_provider_reset/report.md
```

已存在证据：

```text
FinanceBench retrieval-only eval 已有报告。
full V1 generated-answer runner 已有代码入口。
完整生成式答案可靠性报告仍需单独跑数归档。
```

包含 `/v1/query/{id}/trace` 输出的 FinanceBench artifact 必须按敏感产物处理。
如果 artifact 需要共享或归档到低信任位置，需要先确认 raw LLM I/O 已关闭或完成
scrub。

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
1. planner_unknown_provider_rate
2. compound_unit_retry_success_rate
3. metadata_filter_hit_rate
4. must_have_terms off/on ablation
5. sparse_text = unit.text vs unit.text + should_terms vs must_terms sparse boost ablation
6. Python Weighted RRF vs Qdrant RRF ablation
7. serialized table textual lane off/on ablation
8. AAPL/MSFT/Vision Pro semantic-provider / hybrid-execution regression case
```

---

## 14. AAPL / MSFT / Vision Pro 实际路径

用户问题：

```text
Compare AAPL and MSFT exposure to Vision Pro based on annual report evidence.
```

当前 runtime 仍不声明：

```text
full GraphRAG / hub-quality graph traversal claim
SQL mention count
structured exposure calculation
cell provenance lookup
```

V1 应执行：

```text
1. QueryOrchestrator 读取 known_providers = ["hybrid", "sql", "graph"]。
2. Planner 生成 semantic retrieval_units，可包含 hybrid/sql/graph。
3. RetrievalTask 编译 metadata_filter，例如 filing_type / document_ids。
4. ProviderRouter 按 retrieval_tasks 执行默认 executable provider = ["hybrid", "graph"]。
5. sql tasks 写入 skipped_non_executable ProviderResult；graph tasks 只有被 QueryPlan 选中时才执行 local/path。
6. TextHybridProvider 对 hybrid unit 执行 dense_text = unit.text。
7. TextHybridProvider 对 hybrid unit 执行 sparse_text = unit.text + should_terms + repeat(must_have_terms, 3)。
8. Qdrant dense / sparse 分别召回 source chunks。
9. Weighted RRF 合并候选。
10. Reranker 精排。
11. Evidence Builder 扩展到 parent/page blocks。
12. Answer Generator 基于 successful ProviderResult 的 source evidence 做保守比较。
```

示例 semantic units：

```yaml
retrieval_units:
  - unit_id: u0
    provider: graph
    purpose: product_relationship_discovery
    text: "AAPL MSFT Vision Pro product ecosystem relationships"

  - unit_id: u1
    provider: sql
    purpose: structured_exposure_count
    text: "count AAPL MSFT annual report mentions of Vision Pro or mixed reality"

  - unit_id: u2
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

  - unit_id: u3
    provider: hybrid
    purpose: apple_source_text
    text: "Apple AAPL Vision Pro annual report product discussion"
    should_terms:
      - Apple Vision Pro
      - products
      - services

  - unit_id: u4
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
| QueryRuntime | `query_runtime/service.py` | `POST /v1/query` | `query_runs.details_json`、轻量 `details_json.llm_io`、`llm_calls` | total latency、confidence |
| Backend Registry | `backends/` | runtime construction | backend names / config errors | backend replaceability |
| Ingestion Registry | `ingestion/contracts.py`、`ingestion/registry.py` | ingest API | loader/parser/chunker/indexer trace boundary | ingestion replaceability |
| Query Orchestrator | `query_orchestrator/service.py` | `POST /v1/query/plan` | `query_plans.payload_json` | planner latency、unknown provider rate |
| LLM Planner | `query_orchestrator/llm_planner.py` | plan API | `planner`、`model` metadata | retry/fallback rate |
| LLM Client Adapter | `llm/clients/` | planner / answer generator | model usage、raw provider metadata | provider error rate、latency |
| Prompt Builder | `query_orchestrator/prompts.py` | plan API | known/executable providers metadata | prompt compliance |
| Validator | `query_orchestrator/validator.py` | plan API | validation warnings/reasons | validation failure buckets |
| RetrievalTask compiler | `retrieval/models/retrieval_task.py` | plan/retrieve/query | `retrieval_tasks.payload_json` | task count、unit coverage |
| Provider ABC / Registry | `retrieval/providers/base.py`、`retrieval/providers/registry.py`、`core/registry.py` | retrieve/query | provider names、registration errors | provider registration coverage |
| ProviderRouter | `retrieval/router.py` | retrieve/query | `provider_router_trace`、`provider_results` | skipped_non_executable rate、provider latency |
| TextHybridProvider | `retrieval/providers/text_hybrid/provider.py` | retrieve/query | `retrieval_results`、`candidates` | retrieval latency |
| GraphProvider | `retrieval/providers/graph/provider.py` | retrieve/query | graph candidates、source anchors、graph trace | graph grounding coverage |
| Lanes | `retrieval/providers/text_hybrid/lanes.py` | retrieve/query | `lane_trace`、`lane_attributions` | lane contribution |
| Dense adapter | `retrieval/providers/text_hybrid/adapters/dense.py` | retrieve/query | dense rank/score | dense hit@k |
| BM25 adapter | `retrieval/providers/text_hybrid/adapters/bm25.py` | retrieve/query | lexical rank/score | sparse hit@k |
| Provider-local Fusion | `retrieval/ranking/fusion.py` | retrieve/query | `fusion`、`lane_contributions` | RRF ablation |
| CandidateAdapter / CandidateFusion | `retrieval/candidate_adapter.py`、`retrieval/candidate_fusion.py` | retrieve/query | cross-provider fusion、provider_provenance、source_anchors | cross-provider dedupe / rerank order |
| Reranker | `retrieval/ranking/reranker.py` | retrieve/query | rerank rank/score/model | reranker lift |
| Evidence Builder | `query_runtime/evidence_builder.py` | query | `evidence_blocks`、`evidence_packs` | evidence doc/page hit |
| Answer Generator | `query_runtime/service.py` | query | `answers.answer_call_id`、`llm_calls`、`llm_call_evidence`、generation event | answer metrics、usage |
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
  - 默认 Atlas runtime 已注册 GraphProvider
  - V3.0 GraphProvider 已作为 local/path candidate generator walking skeleton 落地
  - 继续补 hub node explosion 防护测评和 graph retrieval eval

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
V1 baseline 只启用 hybrid；当前默认 Atlas runtime 已启用 hybrid + graph。
TextHybridProvider 是 V1 默认 provider 实现。
V3.0 GraphProvider 已作为默认可执行 walking skeleton 存在，但不提供质量提升声明。
SQL 在 V1 中仍只能作为未来架构上下文出现，不能出现在 executable plan 或答案事实里。
```
