# V3 — Atlas Graph Context

> 核心目标：把轻量 GraphRAG 作为 GraphProvider 接入 Atlas Evidence Kernel。
> 关键边界：GraphProvider 不替代 TextHybridProvider；Graph 找结构和线索，最终答案仍尽量回到 source chunk/page 做 grounding。

---

## 1. V3 的定位

V3 不是：

```text
用 GraphRAG 替代 Hybrid RAG。
```

V3 是：

```text
新增一个 GraphProvider，让系统能处理实体关系、跨文档连接、全局主题、社区级概览等问题。
```

V3 应该接入统一链路：

```text
QueryPlan
 -> Provider Router
 -> GraphProvider
 -> GraphCandidate
 -> Evidence Adapter
 -> EvidenceBlock
 -> Fusion / Rerank / Verifier
```

Graph 输出如果不能回到 source evidence，就不能作为强引用证据。

---

## 2. Graph 与 TextHybrid 的分工

| 问题类型 | TextHybridProvider | GraphProvider |
|---|---|---|
| 单一数值事实 | 强 | 弱，通常不用 |
| 表格行定位 | 强 | 弱 |
| 多实体关系 | 中 | 强 |
| 跨文档主题 | 中 | 强 |
| 全局趋势总结 | 弱/中 | 强 |
| 社区/主题概览 | 弱 | 强 |
| 引用原文证据 | 强 | 需要回源 grounding |

一句话：

```text
Graph 找结构；Text 找证据；Evidence Kernel 负责最终可信输出。
```

---

## 3. V3 总体架构

```text
                           ┌──────────────────────┐
                           │      QueryPlan        │
                           │ from V1 Orchestrator  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Provider Router    │
                           │ decide graph usage    │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    GraphProvider      │
                           └──────────┬───────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
       ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│ Local Search  │              │ Global Search │              │ DRIFT-style   │
│ entity/paths  │              │ communities   │              │ global+local  │
└──────┬────────┘              └──────┬────────┘              └──────┬────────┘
       │                              │                              │
       └──────────────────────────────┼──────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  Graph Candidates     │
                           │ nodes/edges/paths     │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Source Grounding      │
                           │ graph -> chunks/pages │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Evidence Adapter      │
                           │ Graph -> EvidenceBlock│
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Evidence Kernel       │
                           │ rerank/build/verify   │
                           └──────────────────────┘
```

---

## 4. Graph Indexing 架构

```text
Documents / Chunks
       │
       ▼
┌──────────────────────┐
│ Entity Extraction     │
│ org/person/product... │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Entity Resolution     │
│ aliases / merge / ids │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Relation Extraction   │
│ typed edges + evidence│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Graph Store           │
│ nodes / edges / facts │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Community Detection   │
│ clusters / themes     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Community Summaries   │
│ source-grounded       │
└──────────────────────┘
```

---

## 5. Graph 数据模型

### 5.1 Entity

```python
class Entity:
    entity_id: str
    name: str
    canonical_name: str
    entity_type: str
    aliases: list[str]
    source_chunk_ids: list[str]
    metadata: dict
```

### 5.2 Relationship

```python
class Relationship:
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str
    confidence: float
    source_chunk_ids: list[str]
    source_document_ids: list[str]
    metadata: dict
```

### 5.3 GraphFact

```python
class GraphFact:
    fact_id: str
    subject_entity_id: str
    predicate: str
    object_value: str
    qualifiers: dict
    source_chunk_ids: list[str]
    confidence: float
```

### 5.4 Community

```python
class Community:
    community_id: str
    level: int
    entity_ids: list[str]
    relationship_ids: list[str]
    summary: str
    source_chunk_ids: list[str]
    metadata: dict
```

### 5.5 SourceAnchor

所有 graph 信息必须有 source anchor。

```python
class SourceAnchor:
    document_id: str
    chunk_id: str
    page_start: int | None
    page_end: int | None
    text_span: str | None
```

---

## 6. GraphProvider 检索模式

### 6.1 Entity Local Search

适用：

```text
具体实体相关问题
```

流程：

```text
query entities
 -> entity lookup
 -> neighbor expansion
 -> relation ranking
 -> source chunk grounding
```

示例：

```text
How are 3M's restructuring actions related to segment margin changes?
```

### 6.2 Path Search

适用：

```text
A 和 B 有什么关系？
事件 X 如何影响指标 Y？
```

流程：

```text
entity A + entity B
 -> find paths
 -> rank paths by relation confidence / source support
 -> retrieve source chunks for path edges
```

### 6.3 Global Search

适用：

```text
全局主题 / 趋势 / corpus-level overview
```

流程：

```text
query
 -> community reports
 -> rank relevant communities
 -> map/reduce synthesis
 -> source grounding
```

### 6.4 DRIFT-style Search

适用：

```text
既需要全局主题，又需要具体实体证据的问题。
```

流程：

```text
global community hints
 -> choose promising entities/subtopics
 -> local graph search
 -> source text grounding
```

### 6.5 Graph-assisted Text Retrieval

Graph 不直接回答，只生成检索线索：

```text
query -> graph entities / related terms / neighbor concepts
 -> expand V1 TextHybrid retrieval units
```

这对 V1/V2 非常实用。

---

## 7. Provider Router 策略

GraphProvider 不应该默认参与所有 query。

### 7.1 高优先级使用 Graph

```text
- 问题包含实体关系
- 问题需要跨文档连接
- 问题要求全局主题/趋势
- 问题涉及多个实体和关系链
- V1 TextHybrid 证据不足且 query 不是简单数值题
```

### 7.2 低优先级或不使用 Graph

```text
- 单一财务数值事实
- 精确表格行查找
- 简单年份/指标查询
- 明确要求某页或某文档字段
```

### 7.3 Router 示例

```yaml
query_type: entity_relationship
providers:
  graph:
    mode: local
    top_k: 50
    budget_ms: 1200
  text_hybrid:
    top_k: 50
    budget_ms: 800

query_type: financial_numeric_fact
providers:
  text_hybrid:
    top_k: 100
  graph:
    enabled: false
```

---

## 8. Graph Candidate

```python
class GraphCandidate:
    candidate_id: str
    provider: str = "graph"
    source_type: str  # graph_node / graph_edge / graph_path / community_report
    text: str
    entity_ids: list[str]
    relationship_ids: list[str]
    community_id: str | None
    source_chunk_ids: list[str]
    source_document_ids: list[str]
    graph_score: float
    rank: int
    grounding_strength: float
    metadata: dict
```

---

## 9. Graph Evidence Adapter

GraphCandidate 不能直接等于 EvidenceBlock。

必须做 grounding：

```text
graph_node / graph_edge / graph_path / community_report
 -> source_chunk_ids
 -> fetch source text
 -> build EvidenceBlock
```

EvidenceBlock 示例：

```python
EvidenceBlock(
    source_type="graph_path_grounded_text",
    provider="graph",
    text="...source text from chunks...",
    citations=[...],
    supporting_candidate_ids=["graph_path_001"],
    provenance={
        "graph_path": ["entity_a", "edge_1", "entity_b"],
        "source_chunks": ["chunk_1", "chunk_2"]
    }
)
```

---

## 10. Fusion 策略

不要把 graph_score 和 BM25 score 直接相加。

GraphProvider 内部自己排序。

跨 Provider 时看：

```text
- 是否有 source grounding
- 是否覆盖 query entities
- 是否解释关系
- source chunks 是否可引用
- 是否与 TextHybrid evidence 一致
- reranker 对 query/evidence 的相关性评分
- Evidence Evaluator 的支持标签
```

### 10.1 Graph priority signal

```text
graph_priority = f(
  query_type,
  entity_match,
  path_quality,
  community_relevance,
  grounding_strength,
  source_diversity
)
```

但最终进入答案的必须是 EvidenceBlock。

---

## 11. Graph Store 选择

### 11.1 初期

```text
Postgres tables + adjacency lists
or
NetworkX-style local graph for prototype
```

适合：

```text
小规模、可控、低基础设施复杂度
```

### 11.2 成熟期

```text
Neo4j
or
Postgres + Apache AGE
or
graph-native store
```

选择标准：

```text
- graph size
- query complexity
- traversal latency
- ops complexity
- integration with existing metadata store
```

---

## 12. Graph Indexing 细节

### 12.1 Entity extraction

可以来源于：

```text
- rule-based NER
- domain dictionary
- LLM extraction
- metadata fields
- table headers
```

### 12.2 Entity resolution

必须处理：

```text
3M
3M Company
MMM
The Company
```

Resolution 结果要可审计：

```text
alias -> canonical entity
source evidence
confidence
```

### 12.3 Relationship extraction

关系必须有类型：

```text
owns
reports
mentions
causes
affects
belongs_to_segment
supplier_of
competitor_of
increases
reduces
```

第一版不要关系类型过多。推荐先做：

```text
mentions
related_to
reports_metric
affects
part_of
```

### 12.4 Community summary

Community summary 不能没有来源。

每个 summary 需要：

```text
community_id
summary_text
source_entity_ids
source_chunk_ids
created_by
created_at
version
```

---

## 13. Eval

### 13.1 Graph construction eval

```text
entity_precision
entity_recall
entity_resolution_accuracy
relationship_precision
relationship_source_grounding_rate
community_summary_grounding_rate
```

### 13.2 Graph retrieval eval

```text
graph_entity_hit@k
graph_path_hit@k
graph_source_chunk_hit@k
graph_grounded_evidence_rate
graph_vs_text_ablation
```

### 13.3 Answer eval

```text
relationship_answer_accuracy
source_citation_hit
unsupported_graph_claim_rate
conflict_detection_rate
```

---

## 14. API

### 14.1 Graph query debug

```http
POST /v3/graph/retrieve
```

```json
{
  "query": "How are 3M's restructuring actions related to margin changes?",
  "mode": "local",
  "return_source_grounding": true
}
```

### 14.2 Graph index inspect

```http
GET /v3/graph/entities/{entity_id}
GET /v3/graph/relationships/{relationship_id}
GET /v3/graph/communities/{community_id}
```

### 14.3 Provider integration

GraphProvider 也通过通用接口被 V1/V2 调用：

```python
provider.retrieve(task: RetrievalTask) -> list[Candidate]
```

---

## 15. Implementation Plan

### V3.0 Graph Contract

```text
- GraphCandidate schema
- Graph Evidence Adapter
- Provider Router integration
- source grounding contract
```

### V3.1 Entity Layer

```text
- entity extraction
- entity resolution
- entity table
- entity lookup
```

### V3.2 Relationship Layer

```text
- relationship extraction
- source anchors
- relation ranking
- local search
```

### V3.3 Graph Retrieval

```text
- local search
- path search
- graph-assisted text retrieval
```

### V3.4 Community / Global

```text
- community detection
- community summaries
- global search
```

### V3.5 DRIFT-style Search

```text
- global hints -> local expansion
- source grounding
- V2 research integration
```

---

## 16. Definition of Done

V3 完成时必须满足：

```text
- GraphProvider 能接收 RetrievalTask。
- Graph 能索引 entity、relationship、source anchors。
- Local search 能返回 grounded source chunks。
- Global/community search 能返回 source-grounded EvidenceBlock。
- Graph 输出不会绕过 Evidence Builder。
- Router 能区分什么时候用 Graph、什么时候不用。
- Eval 能证明 graph 对关系/全局问题有增益。
```

---

## 17. V3 结论

V3 的核心不是“上 Neo4j”。

V3 的核心是：

```text
把图谱上下文变成一个可路由、可 grounding、可审计的 Context Provider。
```

GraphRAG 只有在能回到 EvidenceBlock 时，才真正适合 Atlas。
