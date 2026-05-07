# Atlas Architecture Overview — Evidence-Centric Provider Architecture

> 范围：本文档重做 Atlas 总览，并把 V0–V5 的关系重新收敛到一条主线：**Evidence-Centric Provider Architecture**。
> 结论：Atlas 不应该让每个版本各自发明一套 RAG，而应该让所有能力都通过统一的 `QueryPlan -> RetrievalTask -> Candidate -> EvidenceBlock -> EvidencePack -> VerificationResult` 合约进入同一个 Evidence Kernel。

---

## 0. 一句话定义

Atlas 不是普通文档问答机器人，也不是一开始就大而全的 Agent 平台。

Atlas 应该被定义为：

```text
一个证据优先、可追踪、可评估、可扩展的 RAG / Research 后端运行时。
```

它的核心原则是：

```text
先找证据，再生成结论；
结论必须能回到证据；
证据不足时必须拒答或标记不足；
证据冲突时必须展示冲突；
每一步必须能 trace、eval、replay。
```

这意味着 Atlas 的真正核心不是 LLM，不是 Graph，不是 SQL，不是 Kafka，而是：

```text
Evidence Kernel
```

后续所有能力都只是给 Evidence Kernel 提供上下文、证据或执行能力。

---

## 1. 最重要的架构裁决

旧设计容易变成：

```text
V1 做一套 Hybrid RAG
V2 做一套 Research RAG
V3 做一套 GraphRAG
V4 做一套 SQL QA
V6 再做一套 Memory QA
```

这会导致系统越来越乱，因为每一版都有自己的检索、证据、引用、trace、eval。

新设计必须改成：

```text
V1 先做强 Evidence Kernel；
V2 只是在 Evidence Kernel 上做 Research 编排；
V3 Graph 作为 GraphProvider 接入 Evidence Kernel；
V4 SQL 作为 SQLProvider 接入 Evidence Kernel；
V5 Streaming Ingestion 负责持续更新所有 Provider 的索引；
V6 Memory 作为 MemoryProvider；
V7 MCP 把 Provider / Tool 标准化；
V8/V9 做生产化和事件触发。
```

最关键的一句话：

```text
GraphRAG、SQL、Memory、Tools 都不能绕过 Evidence Kernel。
```

---

## 2. 总体架构图

```text
                           ┌──────────────────────┐
                           │      User Query       │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Query Orchestrator  │  V1
                           │ rewrite / extraction  │
                           │ decomposition / plan  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Retrieval Plan     │
                           │ tasks / budget / use  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Provider Router    │
                           │ choose providers      │
                           └──────────┬───────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ TextHybridProvider  │     │ GraphProvider       │     │ SQLProvider         │
│ V1                  │     │ V3                  │     │ V4                  │
│ dense/sparse/table  │     │ entity/path/global  │     │ table/query/calc    │
└─────────┬───────────┘     └─────────┬───────────┘     └─────────┬───────────┘
          │                           │                           │
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ Text Candidates     │     │ Graph Candidates    │     │ SQL Candidates      │
│ chunks/pages/tables │     │ nodes/edges/paths   │     │ rows/cells/results  │
└─────────┬───────────┘     └─────────┬───────────┘     └─────────┬───────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Candidate Adapter     │
                           │ normalize to contract │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Candidate Fusion      │  V1+
                           │ provider-local RRF    │
                           │ cross-provider rerank │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Evidence Builder      │  V1
                           │ dedupe/merge/budget   │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Evidence Evaluator    │  V1/V2
                           │ sufficient/conflicted │
                           └──────────┬───────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ Single Answer       │     │ Research Runtime    │     │ Claim Audit         │
│ V1                  │     │ V2                  │     │ V2                  │
│ grounded answer     │     │ async job/report    │     │ support verification│
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

---

## 3. 三个 Plane：不要把职责混在一起

Atlas 应该拆成三个长期稳定的平面。

### 3.1 Query Plane

负责理解用户意图、生成计划、调度 Provider。

```text
Query Plane:
  - Query Orchestrator
  - Query rewrite
  - Query decomposition
  - Structured extraction
  - Provider Router
  - Research Planner
  - Budget Manager
```

Query Plane 决定：

```text
这个问题是单一事实查询，还是多步研究任务？
语义计划应该表达 `hybrid` / `sql` / `graph` 中的哪类 provider 意图？
当前 runtime 能执行哪些 provider、哪些只能进入 skipped trace？
每个 retrieval unit 分配多少 top_k、延迟、token、成本预算？
```

### 3.2 Evidence Plane

负责真正产出证据。

```text
Evidence Plane:
  - TextHybridProvider
  - GraphProvider
  - SQLProvider
  - MemoryProvider
  - ToolProvider
  - Candidate Fusion
  - Reranker
  - Evidence Builder
  - Evidence Evaluator
  - Citation Verifier
```

Evidence Plane 的关键目标是：

```text
所有来源都必须被规范化成 Candidate 和 EvidenceBlock。
```

### 3.3 Ingestion Plane

负责写入、解析、索引、持续更新。

```text
Ingestion Plane:
  - collectors
  - parser
  - chunker
  - contextual chunk enrichment
  - dense embedding indexing
  - sparse/BM25 indexing
  - graph indexing
  - table extraction
  - SQL/table loading
  - Kafka/Flink pipeline
  - DLQ
  - index versioning
```

Kafka、Flink、DLQ 属于 Ingestion Plane，**不应该进入查询链路**。

---

## 4. 版本路线总览

| 版本 | 名称 | 核心目标 | 关键边界 |
|---|---|---|---|
| V0 | Atlas Kernel | 最小可用 RAG 内核 | 单次 dense RAG 闭环 |
| V1 | Atlas Advanced Hybrid Kernel | 做强 evidence retrieval | query transform、hybrid、多路检索、rerank、evidence、verifier |
| V2 | Atlas Research Runtime | 异步复杂研究任务 | planner 调用 V1，不重写 retrieval |
| V3 | Atlas Graph Context | 图谱上下文补充 | graph provider 输出 evidence，不替代 V1 |
| V4 | Structured Data Context | 表格/SQL/数值分析 | SQL result 也变 EvidenceBlock |
| V5 | Streaming Ingestion | 持续索引 | 更新 text/graph/sql/memory indexes |
| V6 | Memory & Skills | 长期记忆和技能系统 | memory 是低/中/高可信 context，不等于事实证据 |
| V7 | MCP Tool Layer | 工具标准化 | provider/tool protocol 标准化 |
| V8 | Cloud Production | 生产部署 | scaling/observability/security |
| V9 | Event Path | 主动事件响应 | event 触发 V1/V2/V3/V4 runtime |

本文档包先覆盖 Overview 和 V1–V5。

---

## 5. V1–V5 的真实关系

```text
V1 = Evidence Kernel 的核心能力
V2 = Research 编排层，调用 V1
V3 = Graph Provider，接入 V1/V2
V4 = SQL Provider，接入 V1/V2
V5 = Streaming Ingestion，持续更新 V1/V3/V4 的索引
```

不要理解成：

```text
V1 做完后 V2 替代 V1；
V3 做完后 Graph 替代 Hybrid；
V4 做完后 SQL 替代 Text；
V5 做完后 Kafka 进入 Query Path。
```

正确理解是：

```text
V1 是底座；
V2 编排它；
V3/V4 扩展它；
V5 持续喂给它。
```

---

## 6. 核心合约

后续所有版本都围绕这条链路扩展：

```text
QueryPlan
 -> RetrievalTask
 -> Candidate
 -> EvidenceBlock
 -> EvidencePack
 -> VerificationResult
 -> Answer / Report / Audit
 -> Trace
```

### 6.1 QueryPlan

`QueryPlan` 是 Query Orchestrator 的输出。它不是自然语言，而是结构化计划。

```python
class QueryPlan:
    original_query: str
    standalone_query: str | None
    query_type: str
    entities: list[Entity]
    periods: list[Period]
    metrics: list[Metric]
    metadata_filter: dict
    retrieval_units: list[RetrievalUnit]
    budget: RetrievalBudget
    risk_flags: list[str]
```

### 6.2 RetrievalTask

`RetrievalTask` 是发给 Provider 的任务。

```python
class RetrievalTask:
    task_id: str
    provider: str
    purpose: str
    query_text: str
    metadata_filter: dict
    must_have_terms: list[str]
    should_terms: list[str]
    top_k: int
    weight: float
    budget_ms: int
```

### 6.3 Candidate

`Candidate` 是 Provider 返回的候选。

```python
class Candidate:
    candidate_id: str
    provider: str
    source_type: str
    text: str
    rank: int | None
    score: float | None
    document_id: str | None
    chunk_id: str | None
    page_start: int | None
    page_end: int | None
    graph_node_ids: list[str] | None
    graph_edge_ids: list[str] | None
    sql_query: str | None
    table_name: str | None
    row_ids: list[str] | None
    metadata: dict
```

### 6.4 EvidenceBlock

`EvidenceBlock` 是最终可能进入 prompt / report 的证据块。

```python
class EvidenceBlock:
    evidence_id: str
    source_type: str
    provider: str
    text: str
    citations: list[Citation]
    supporting_candidate_ids: list[str]
    document_id: str | None
    page_start: int | None
    page_end: int | None
    confidence: float | None
    coverage: dict
    provenance: dict
```

### 6.5 VerificationResult

```python
class VerificationResult:
    label: str  # supported / insufficient / contradicted / partially_supported
    reason: str
    evidence_ids: list[str]
    missing_requirements: list[str]
    conflict_set: list[str]
    confidence: float
```

---

## 7. Provider 不是 Lane

一个容易犯的错误是把所有检索方法都扔进同一个 RRF 池：

```text
dense lane
BM25 lane
graph lane
SQL lane
memory lane
tool lane
全部 RRF
```

这不合理，因为这些东西输出的语义不同。

正确分层是：

```text
TextHybridProvider 内部可以有 dense / BM25 / table-row / alias lane；
GraphProvider 内部可以有 local / global / DRIFT / path lane；
SQLProvider 内部可以有 SQL / formula / cell provenance lane；
跨 Provider 统一适配成 Candidate / EvidenceBlock 后，再做 rerank、verify、select。
```

Provider 与 Lane 的关系：

```text
Provider = 一个上下文来源系统
Lane     = Provider 内部的一条检索策略
```

例子：

```text
TextHybridProvider:
  lane_1 = dense semantic search
  lane_2 = BM25 sparse search
  lane_3 = metric alias BM25
  lane_4 = table row search
  lane_5 = parent-child expansion

GraphProvider:
  lane_1 = entity local search
  lane_2 = path search
  lane_3 = community/global search
  lane_4 = DRIFT-style search

SQLProvider:
  lane_1 = direct SQL lookup
  lane_2 = aggregation query
  lane_3 = formula calculation
  lane_4 = cell provenance lookup
```

---

## 8. Query 类型到主路径的路由

| Query 类型 | 主路径 | 不建议默认用 |
|---|---|---|
| 单一财务数值事实 | V1 TextHybrid + metric alias + table row + rerank | Graph global / HyDE |
| 多年份比较 | V1 decomposition + per-period retrieval + calculator | Graph global |
| 多公司比较 | V1 decomposition + metadata filter + rerank | Graph global |
| 报告型问题 | V2 Research Runtime + V1 Kernel | 单次 top-k answer |
| 实体关系问题 | V3 graph local + V1 text grounding | 纯 dense retrieval |
| 全局主题/趋势 | V3 graph global / DRIFT + V2 synthesis | 只靠 top-k chunks |
| 精确表格/聚合 | V4 SQL/table context + calculator | LLM 自己算 |
| 用户偏好/项目上下文 | V6 MemoryProvider | 当成事实证据 |
| 外部动作/工具 | V7 MCP tools | 硬编码工具调用 |

这个表是 Router 的第一版策略基础。

---

## 9. Fusion 策略：Provider 内 RRF，Provider 间 Evidence 质量排序

V1 的 TextHybridProvider 内部可以使用 Weighted RRF：

```text
score(d) = Σ weight_i / (rrf_k + rank_i(d))
```

适合放入 RRF 的是同一类输出：

```text
chunk candidates
page candidates
table-row candidates
```

不适合直接和 BM25 / Dense 裸 RRF 的是：

```text
graph path
community summary
SQL result
memory item
tool result
```

跨 Provider 选择证据时应该看：

```text
- query intent match
- source grounding strength
- citation quality
- entity/period/metric coverage
- numeric verifiability
- provider reliability
- reranker score
- evidence evaluator label
```

---

## 10. V1–V5 的阶段目标

### V1: Atlas Advanced Hybrid Kernel

目标：把单次问答的证据检索做到强。

```text
Query Orchestrator
 -> TextHybridProvider
 -> Provider-local Weighted RRF
 -> Reranker
 -> Evidence Builder
 -> Evidence Evaluator
 -> Citation Verifier
 -> Trace + Eval
```

### V2: Atlas Research Runtime

目标：复杂任务异步化、可追踪、可产出报告。

```text
ResearchJob
 -> Planner
 -> Subquestion DAG
 -> each subquestion calls V1 Kernel
 -> Evidence Packs
 -> Synthesis
 -> Report Artifact
 -> Claim Audit
```

### V3: Atlas Graph Context

目标：图谱上下文作为 Provider 接入，不替代 TextHybrid。

```text
GraphProvider
 -> entity / relation / path / community retrieval
 -> source grounding
 -> EvidenceBlock
```

### V4: Structured Data Context

目标：结构化表格、SQL、数值计算接入。

```text
SQLProvider
 -> Text-to-SQL
 -> SQL verifier
 -> calculator
 -> cell/table provenance
 -> EvidenceBlock
```

### V5: Streaming Ingestion

目标：持续写入和持续索引。

```text
Kafka/Flink/Queue
 -> parse
 -> chunk
 -> contextualize
 -> dense/BM25 index
 -> graph index
 -> table index
 -> DLQ / SLA / index versioning
```

---

## 11. 推荐目录结构

```text
src/atlas/
  query_orchestrator/
    service.py
    schema.py
    prompts.py
    validators.py
    router.py

  retrieval/
    providers/
      base.py
      text_hybrid/
        provider.py
        lanes.py
        adapters/
          dense.py
          bm25.py
          hybrid.py
          mode_switching.py
    candidate.py
    evidence.py
    fusion.py

  query_runtime/
    service.py
    evidence_builder.py
    evidence_evaluator.py
    citation_verifier.py
    trace_logger.py

  research/
    job_service.py          # V2
    planner.py
    task_runner.py
    report_writer.py
    claim_audit.py
    artifacts.py

  graph_context/
    indexer.py              # V3
    entity_resolver.py
    relationship_extractor.py
    graph_store.py
    graph_retriever.py
    evidence_adapter.py

  structured_context/
    schema_registry.py      # V4
    table_extractor.py
    sql_generator.py
    sql_verifier.py
    calculator.py
    evidence_adapter.py

  ingestion/
    parser.py
    chunker.py
    contextualizer.py
    indexer.py
    streaming/              # V5
      topics.py
      consumers.py
      dlq.py
      versioning.py
      sla.py
```

---

## 12. 长期数据流

```text
                         READ PATH

User Query
  -> Query Orchestrator
  -> Provider Router
  -> Providers
  -> Candidate Fusion
  -> Evidence Builder
  -> Evaluator
  -> Answer / Research / Audit


                         WRITE PATH

Documents / Tables / Events
  -> Collectors
  -> Parser
  -> Chunker
  -> Contextualizer
  -> Text Indexes
  -> Graph Indexes
  -> SQL/Table Store
  -> Index Version Registry
```

读路径和写路径要分离：

```text
读路径追求低延迟和证据质量；
写路径追求吞吐、稳定、重试、版本一致性。
```

---

## 13. 质量评估总线

所有版本都要共享 Eval 思路。

```text
Retrieval Eval:
  - doc_hit@k
  - page_hit@k
  - chunk_hit@k
  - MRR
  - recall by provider

Evidence Eval:
  - evidence_coverage
  - citation_doc_hit
  - citation_page_hit
  - evidence_token_efficiency
  - conflicting_evidence_detection

Answer Eval:
  - answer_normalized_hit
  - numeric_tolerance_hit
  - unsupported_claim_rate
  - refusal_correctness

Runtime Eval:
  - latency_p50/p95
  - cost_per_query
  - cache_hit_rate
  - provider_budget_usage

Research Eval:
  - subquestion_coverage
  - report_claim_support_rate
  - artifact_completeness
  - failure_attribution_accuracy

Ingestion Eval:
  - indexing_lag
  - DLQ_rate
  - reindex_success_rate
  - freshness_SLA_hit
```

---

## 14. 设计边界

### V1 不做

```text
- 异步 deep research job
- GraphRAG 主路径
- SQL Text-to-SQL
- Kafka/Flink streaming
- memory / skills
- MCP tool layer
```

### V2 不做

```text
- 重新实现 retrieval
- 绕过 V1 Evidence Kernel
- Graph index construction
- SQL warehouse
```

### V3 不做

```text
- 替代 TextHybridProvider
- 用 graph summary 直接替代 source citation
- 把所有问题都路由到 graph
```

### V4 不做

```text
- 让 LLM 自己算复杂数字
- SQL result 无 provenance
- Text-to-SQL 未验证就执行高风险查询
```

### V5 不做

```text
- 进入 query runtime 主链路
- 让 Kafka 控制业务语义
- 不做 index version 就直接更新线上索引
```

---

## 15. 参考资料

这些资料不是 Atlas 的照搬对象，而是本设计的技术参考：

- Qdrant Hybrid Queries / RRF / Weighted RRF: https://qdrant.tech/documentation/search/hybrid-queries/
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Microsoft GraphRAG Query Overview: https://microsoft.github.io/graphrag/query/overview/
- Microsoft GraphRAG DRIFT Search: https://microsoft.github.io/graphrag/query/drift_search/
- Apache Kafka Documentation: https://kafka.apache.org/documentation/
- Apache Flink Checkpointing: https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/fault-tolerance/checkpointing/
- Confluent Kafka DLQ Guide: https://www.confluent.io/learn/kafka-dead-letter-queue/

---

## 16. 最终裁决

Atlas 的主线应该是：

```text
Evidence-Centric Provider Architecture
```

当前主攻：

```text
V1 Advanced Hybrid Evidence Kernel
```

然后：

```text
V2 = Research Runtime 调用 V1
V3 = GraphProvider 接入 V1/V2
V4 = SQLProvider 接入 V1/V2
V5 = Streaming Ingestion 持续更新所有 Provider 索引
```

最重要的是：

```text
不要让每个版本变成一套新 RAG。
所有版本都必须通过统一 Evidence Contract 协同。
```
