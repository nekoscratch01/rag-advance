# V1 — Atlas Advanced Hybrid Kernel

> 核心目标：把单次问答的 evidence retrieval 做到强、可解释、可评估、可扩展。
> 关键边界：V1 是 Evidence Kernel，不做完整 Research Runtime，不做 GraphRAG 主路径，不做 SQL Text-to-SQL，不做 Streaming Ingestion。

---

## 1. V1 的定位

V1 不应该只是：

```text
Dense + BM25 + RRF
```

那只是底座。

成熟 V1 应该是：

```text
Query Orchestrator
 -> Retrieval Plan
 -> TextHybridProvider
 -> Provider-local Fusion
 -> Reranker
 -> Evidence Builder
 -> Evidence Evaluator
 -> Citation Verifier
 -> Trace + Eval + Cache
```

V1 的真实目标是：

```text
给后续 V2 Research、V3 Graph、V4 SQL 提供统一的 Evidence Kernel。
```

V1 做强以后，V2/V3/V4 都可以复用它，而不是重新写一套检索和证据逻辑。

---

## 2. V1 总体架构

```text
                           ┌──────────────────────┐
                           │      User Query       │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Query Orchestrator  │
                           │ rewrite / extraction  │
                           │ decomposition / plan  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Retrieval Plan     │
                           │ query units / filters │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  TextHybridProvider   │
                           └──────────┬───────────┘
                                      │
      ┌───────────────────────────────┼───────────────────────────────┐
      │                               │                               │
      ▼                               ▼                               ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ Dense Lane   │              │ BM25 Lane    │              │ Table Lane   │
│ embeddings   │              │ sparse terms │              │ row/page     │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       │                             │                             │
       └─────────────────────────────┼─────────────────────────────┘
                                     │
                                     ▼
                           ┌──────────────────────┐
                           │ Provider-local Fusion │
                           │ Weighted RRF / merge  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │      Reranker         │
                           │ top-N -> top-M        │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Evidence Builder    │
                           │ dedupe / merge / pack │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Evidence Evaluator    │
                           │ sufficiency/conflict  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  Answer Generator     │
                           │ grounded answer       │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Citation Verifier     │
                           │ support check         │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Trace + Eval + Cache  │
                           └──────────────────────┘
```

---

## 3. V1 模块清单

```text
src/atlas/query_orchestrator/
  schema.py
  service.py
  prompts.py
  validators.py
  router.py

src/atlas/retrieval/providers/text_hybrid/
  provider.py
  dense.py
  bm25.py
  table_row.py
  parent_child.py
  fusion.py
  reranker.py

src/atlas/retrieval/
  candidate.py
  retrieval_task.py
  evidence.py

src/atlas/query_runtime/
  service.py
  evidence_builder.py
  evidence_evaluator.py
  citation_verifier.py
  trace_logger.py

src/atlas/cache/
  exact_cache.py
  keys.py
  stores.py

src/atlas/eval/
  financebench.py
  metrics.py
  report.py
```

---

## 4. Query Orchestrator

### 4.1 为什么 V1 需要 Query Orchestrator

如果只用原始 query 检索，长 query、比较 query、指标别名 query、表格 query 都很容易失败。

例如：

```text
What was 3M's FY2018 capital expenditure amount?
```

用户说的是：

```text
capital expenditure
```

财报里可能写的是：

```text
Purchases of property, plant and equipment
```

所以 V1 必须把 query 编译成更适合检索的任务。

### 4.2 Orchestrator 的职责

```text
- standalone rewrite
- query type classification
- entity extraction
- period extraction
- metric extraction
- metric canonicalization
- query decomposition
- query expansion
- metadata filter generation
- retrieval unit generation
- provider / lane budget assignment
```

### 4.3 Orchestrator 输出

```python
class QueryPlan:
    original_query: str
    standalone_query: str | None
    query_type: str
    entities: list[Entity]
    periods: list[Period]
    metrics: list[Metric]
    filters: dict
    retrieval_units: list[RetrievalUnit]
    risk_flags: list[str]
    budget: RetrievalBudget
```

### 4.4 RetrievalUnit

```python
class RetrievalUnit:
    unit_id: str
    purpose: str
    text: str
    retrievers: list[str]
    filters: dict
    must_have_terms: list[str]
    should_terms: list[str]
    top_k: int
    weight: float
```

### 4.5 Query 类型

```text
fact_lookup              单一事实查询
financial_numeric_fact   财务数值查询
comparison               比较问题
calculation              需要计算的问题
summarization            总结问题
explanation              解释问题
multi_hop                多跳问题
ambiguous                需要澄清或保守处理
```

### 4.6 示例

输入：

```text
Compare 3M's FY2018 capital expenditure with FY2017 and explain whether it increased.
```

输出：

```yaml
query_type: comparison
entities:
  - 3M
periods:
  - FY2018
  - FY2017
metrics:
  - capital_expenditure
filters:
  company: 3M
  filing_type: 10-K
retrieval_units:
  - unit_id: u0
    purpose: original
    text: "Compare 3M's FY2018 capital expenditure with FY2017 and explain whether it increased."
    retrievers: [dense, bm25]
    weight: 1.0
  - unit_id: u1
    purpose: comparison_operand
    text: "3M FY2018 capital expenditure"
    retrievers: [dense, bm25]
    must_have_terms: ["2018"]
    weight: 1.3
  - unit_id: u2
    purpose: comparison_operand
    text: "3M FY2017 capital expenditure"
    retrievers: [dense, bm25]
    must_have_terms: ["2017"]
    weight: 1.3
  - unit_id: u3
    purpose: metric_alias
    text: "3M 2018 2017 purchases of property plant and equipment"
    retrievers: [bm25]
    must_have_terms: ["property", "equipment"]
    weight: 1.6
  - unit_id: u4
    purpose: table_row
    text: "Purchases of property, plant and equipment 2018 2017"
    retrievers: [bm25]
    weight: 1.7
```

### 4.7 Orchestrator 的实现策略

V1 可以同时支持：

```text
rule-based extractor
LLM structured output
ontology-based metric expansion
validator
fallback plan
```

推荐策略：

```text
1. 先用规则抽 company/year/metric。
2. 再用 LLM 生成 structured QueryPlan。
3. 用 validator 检查 LLM 输出。
4. 如果 LLM 输出不合法，则 fallback 到 rule-based plan。
```

### 4.8 Validator 规则

```text
- 不允许生成原 query 中没有依据的公司、年份、指标。
- 不允许 retrieval_units 超过最大数量。
- 每个 retrieval_unit 必须保留核心实体或核心指标。
- HyDE / Query2Doc 不能用于 BM25 exact lane。
- metric_alias 必须来自 ontology。
- filters 必须来自 corpus metadata schema。
```

---

## 5. Finance Metric Ontology

V1 需要一个轻量金融指标词典。

位置：

```text
configs/finance_metric_ontology.yaml
```

示例：

```yaml
capital_expenditure:
  canonical_name: capital_expenditure
  aliases:
    - capital expenditure
    - capital expenditures
    - capex
    - capital spending
    - purchases of property, plant and equipment
    - additions to property, plant and equipment
  statement_hints:
    - cash flow
    - investing activities
  value_type: currency

revenue:
  canonical_name: revenue
  aliases:
    - revenue
    - revenues
    - net sales
    - sales
    - total revenue
    - net revenue
  statement_hints:
    - income statement
    - consolidated statements of operations
  value_type: currency

operating_cash_flow:
  canonical_name: operating_cash_flow
  aliases:
    - operating cash flow
    - cash flow from operations
    - net cash provided by operating activities
    - net cash from operating activities
  statement_hints:
    - cash flow
    - operating activities
  value_type: currency
```

第一批建议覆盖：

```text
revenue
net_income
operating_income
gross_profit
total_assets
total_liabilities
shareholders_equity
cash_and_cash_equivalents
operating_cash_flow
capital_expenditure
free_cash_flow
r_and_d_expense
sga_expense
long_term_debt
inventory
dividends
eps
```

---

## 6. Contextual Hybrid Retrieval

### 6.1 为什么需要 Contextual Retrieval

很多 chunk 单独看没有上下文。

原始 chunk：

```text
Purchases of property, plant and equipment 1,577 1,373 1,420
```

这个 chunk 缺少：

```text
公司是谁？
是哪份 filing？
表格标题是什么？
三列分别是哪几年？
单位是什么？
```

Contextual chunk 应该变成：

```text
This chunk is from 3M 2018 Form 10-K, Consolidated Statement of Cash Flows,
Investing Activities table. Amounts are in millions of USD. Columns are 2018,
2017, and 2016.

Purchases of property, plant and equipment 1,577 1,373 1,420
```

这样 dense 和 BM25 都更容易命中。

### 6.2 Parent-child chunking

V1 的索引单位和展示单位应该区分：

```text
child chunk:
  用于精确检索。

parent chunk / page block:
  用于 evidence pack。
```

结构：

```text
Document
  Page
    ParentBlock
      ChildChunk
      ChildChunk
      ChildChunk
```

检索：

```text
child chunks participate in dense/BM25 search
```

证据：

```text
parent block or page-neighborhood enters Evidence Builder
```

### 6.3 Table-aware chunking

金融文档里表格很重要。V1 至少要保留：

```text
table_id
table_title
row_label
column_headers
unit
page_number
cell_text
source_bbox optional
```

对于表格行，建议生成额外 searchable text：

```text
3M 2018 Form 10-K Consolidated Statement of Cash Flows Investing Activities
Amounts in millions USD
Row: Purchases of property, plant and equipment
2018: 1,577
2017: 1,373
2016: 1,420
```

---

## 7. TextHybridProvider

### 7.1 Provider 内部 lanes

```text
Dense lane:
  semantic retrieval over contextual child chunks

BM25 lane:
  sparse keyword retrieval over contextual text

Metric alias lane:
  ontology-expanded financial terms

Table row lane:
  row-label + year + metric search

Section-aware lane:
  search constrained by filing section / table title

Parent-child expansion:
  child hit -> parent/page evidence candidate
```

### 7.2 Candidate

```python
class Candidate:
    candidate_id: str
    provider: str = "text_hybrid"
    source_type: str  # text_chunk / table_row / page_block
    document_id: str
    chunk_id: str | None
    parent_id: str | None
    page_start: int | None
    page_end: int | None
    text: str
    lane: str
    rank: int | None
    score: float | None
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    table_rank: int | None
    rrf_score: float | None
    metadata: dict
```

不要只放一个 `score`。必须保留原始分数和 fusion 分数。

---

## 8. Weighted RRF

Provider 内部可以用 Weighted RRF。

```text
score(d) = Σ weight_i / (rrf_k + rank_i(d))
```

建议初始权重：

```yaml
original_dense: 1.0
original_bm25: 1.1
standalone_dense: 1.1
standalone_bm25: 1.2
metric_alias_bm25: 1.5
table_row_bm25: 1.7
section_bm25: 1.3
hyde_dense: 0.7
```

注意：

```text
HyDE / Query2Doc 适合语义解释类问题，不适合精确数值事实题默认高权重。
```

### 8.1 为什么不是 raw score 相加

```text
dense cosine score
BM25 score
table row score
reranker score
```

尺度不同，不能直接相加。

RRF 使用排名，更稳。

### 8.2 Provider 内与 Provider 间区别

V1 当前只有 TextHybridProvider，所以 Fusion 主要是 Provider 内部。

后续 V3/V4 加入 GraphProvider / SQLProvider 后：

```text
Provider 内可以 RRF。
Provider 间不建议裸 RRF。
跨 Provider 用 reranker + evidence quality + verifier 决策。
```

---

## 9. Reranker

V1 成熟版应该有 reranker。

典型流程：

```text
retrieval candidate pool: top 100/150
 -> reranker
 -> top 20/30
 -> Evidence Builder
```

Reranker 输入：

```text
query / retrieval_unit / candidate text
```

输出：

```text
rerank_score
rerank_rank
```

### 9.1 Reranker 策略

```text
- 默认可配置开启。
- 本地 Mac 可以用小模型 benchmark。
- 云端生产可以使用更强 cross-encoder 或 API reranker。
- Reranker 不应该替代 high-recall retrieval。
```

### 9.2 何时需要 reranker

如果 eval 显示：

```text
hybrid_hit@20 高，但 hit@5 低
```

说明答案在候选池里，但排序不够好，reranker 很有价值。

如果：

```text
hybrid_hit@20 也低
```

说明召回不足，优先修 query transform、chunking、BM25、ontology。

---

## 10. Evidence Builder

Evidence Builder 不是简单 top-k 拼接。

职责：

```text
- chunk 去重
- child hit 扩展到 parent block
- 同文档相邻 chunk 合并
- 同页/相邻页合并
- 表格行与表头/单位一起保留
- context token budget 控制
- 覆盖多个 query units
- 保留 retrieval provenance
```

输出：

```python
class EvidenceBlock:
    evidence_id: str
    source_type: str
    provider: str
    text: str
    document_id: str
    doc_name: str
    page_start: int | None
    page_end: int | None
    chunk_ids: list[str]
    candidate_ids: list[str]
    retrieval_sources: list[str]
    best_dense_rank: int | None
    best_bm25_rank: int | None
    best_rrf_score: float | None
    rerank_score: float | None
    token_count: int
    coverage: dict
    metadata: dict
```

### 10.1 Evidence coverage

Coverage 至少记录：

```text
entity_hit
period_hit
metric_hit
value_hit
query_unit_ids_covered
table_context_present
citation_ready
```

---

## 11. Evidence Evaluator

Evidence Evaluator 是 V1 的轻量 verifier。

标签：

```text
supported
insufficient
contradicted
partially_supported
```

### 11.1 Pre-generation Evaluator

在生成答案前判断：

```text
- 是否有 evidence
- evidence 是否覆盖核心 entity / period / metric
- evidence 是否来自正确文档类型
- 多个 evidence 是否明显冲突
- 是否需要扩大检索或拒答
```

### 11.2 Post-generation Citation Verifier

在生成答案后判断：

```text
- 每个 citation 是否来自 evidence set
- citation 是否真的支持对应句子
- 答案中的关键数字是否能在 cited evidence 中找到
- 引用页码/文档是否正确
```

### 11.3 不要过度依赖绝对分数阈值

Dense score、BM25 score、RRF score 跨 query 不稳定。第一版不要用一个全局分数阈值直接拒答。

更稳的 sufficiency 规则：

```text
strong insufficient:
  no evidence
  entity 不匹配
  period 不匹配
  metric 不匹配
  answer citation 不在 evidence set

warning only:
  top score low
  candidate pool low diversity
  retrieved docs too broad
```

---

## 12. Cache

V1 应有 cache，但 benchmark 默认关闭或单独报告。

### 12.1 Cache 分层

```text
QueryPlanCache:
  normalized_query + schema_version + orchestrator_version

RetrievalCache:
  query_plan_hash + corpus_version + index_version + retrieval_version

AnswerCache:
  query + evidence_hash + prompt_version + model_version
```

### 12.2 Cache key 必须包含

```text
normalized_query
corpus_version
chunk_version
index_version
retrieval_mode
dense_model_id
sparse_model_id
bm25_config_hash
rrf_k
lane_weights
reranker_id
evidence_builder_version
prompt_version
answer_model
```

---

## 13. Trace

V1 的 trace 必须能回答：

```text
为什么这条 evidence 被选中？
哪些 lane 找到了它？
reranker 为什么把它排在前面？
证据是否进入了 prompt？
答案中的引用是否支持 claim？
```

Trace 结构：

```text
QueryRun
  QueryPlan
  RetrievalTasks
  LaneResults
  FusedCandidates
  RerankedCandidates
  EvidencePack
  EvidenceEvaluation
  Prompt
  Answer
  Citations
  CitationVerification
  Metrics
```

---

## 14. Eval

V1 必须有 retrieval-only eval 和 answer eval。

### 14.1 Retrieval metrics

```text
dense_doc_hit@k
dense_page_hit@k
bm25_doc_hit@k
bm25_page_hit@k
hybrid_doc_hit@k
hybrid_page_hit@k
mrr_doc
mrr_page
query_unit_gold_hit
provider/lane contribution
```

### 14.2 Evidence metrics

```text
evidence_doc_hit@context
evidence_page_hit@context
evidence_token_count
evidence_block_count
evidence_coverage
citation_ready_rate
```

### 14.3 Answer metrics

```text
answer_normalized_hit
numeric_tolerance_hit
citation_doc_hit
citation_page_hit
unsupported_claim_rate
false_refusal_rate
```

### 14.4 Latency metrics

```text
orchestrator_ms
dense_embed_ms
dense_query_ms
bm25_embed_ms
bm25_query_ms
fusion_ms
rerank_ms
evidence_builder_ms
evaluator_ms
generator_ms
citation_verifier_ms
total_ms
```

### 14.5 Ablation

至少跑：

```text
A. dense-only original query
B. BM25-only original query
C. hybrid original query
D. hybrid + query orchestrator
E. hybrid + orchestrator + reranker
F. full V1 with evidence evaluator / citation verifier
```

---

## 15. API

### 15.1 Query

```http
POST /v1/query
```

```json
{
  "query": "What is the FY2018 capital expenditure amount for 3M?",
  "mode": "answer",
  "retrieval_mode": "advanced_hybrid",
  "use_reranker": true,
  "use_cache": false,
  "return_trace": true
}
```

### 15.2 Inspect Query Plan

```http
POST /v1/query/plan
```

返回 QueryPlan，不执行检索。

### 15.3 Retrieval-only

```http
POST /v1/retrieve
```

用于 eval 和 debug。

### 15.4 Trace

```http
GET /v1/query-runs/{query_run_id}/trace
```

---

## 16. 数据库表

```text
query_runs
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
cache_entries
eval_runs
eval_case_results
```

---

## 17. 实施计划

### V1.0 Foundation

```text
- Candidate / EvidenceBlock contract
- QueryRun trace schema
- FinanceBench adapter
- dense-only baseline
- retrieval-only eval
```

### V1.1 Text Hybrid

```text
- BM25 sparse indexing
- TextHybridProvider
- dense + BM25 lanes
- Provider-local Weighted RRF
- hybrid eval report
```

### V1.2 Query Orchestrator

```text
- QueryPlan schema
- rule-based extractor
- LLM structured output
- finance metric ontology
- query unit generation
- validator / fallback
```

### V1.3 Contextual Retrieval

```text
- contextual chunk enrichment
- parent-child chunking
- table-aware text
- section metadata
```

### V1.4 Rerank + Evidence

```text
- reranker integration
- Evidence Builder
- Evidence Evaluator
- Citation Verifier
```

### V1.5 Cache + Eval Hardening

```text
- exact cache
- ablation reports
- latency reports
- failure attribution
```

---

## 18. Definition of Done

V1 完成时必须满足：

```text
- Query Orchestrator 能输出结构化 QueryPlan。
- TextHybridProvider 支持 dense/BM25/table/alias lanes。
- Provider 内 Weighted RRF 可 trace。
- Reranker 可开关、可评测。
- Evidence Builder 输出结构化 EvidenceBlock。
- Evidence Evaluator 能判断 supported / insufficient / contradicted。
- Citation Verifier 能检查 citation 是否支持答案。
- Eval 能对 dense-only、hybrid、full V1 做对比。
- Trace 能解释 retrieval / packing / generation / citation 每一步。
```

---

## 19. 关键风险

```text
Query Orchestrator 过度生成:
  用 validator 和 max retrieval_units 控制。

BM25 / dense / table lane 噪声增加:
  用 lane weights、anchor coverage、reranker 处理。

Reranker 延迟高:
  通过 top-N 控制、缓存、可配置开关。

Evidence Builder 丢掉关键证据:
  记录候选进入/未进入 prompt 的原因。

Citation Verifier 误拒答:
  把 hard fail 和 warning 分开。
```

---

## 20. V1 结论

V1 的目标不是“加 BM25”。

V1 的目标是建立：

```text
Advanced Hybrid Evidence Kernel
```

它必须足够强，才能支撑：

```text
V2 Research Runtime
V3 Graph Context
V4 Structured Data Context
V5 Streaming Ingestion
```
