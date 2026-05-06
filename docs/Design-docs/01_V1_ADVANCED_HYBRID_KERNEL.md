# V1 - Atlas Advanced Hybrid Kernel

更新时间：2026-05-06

> 核心目标：把单次问答的 evidence retrieval 做到强、可解释、可评估、可扩展。
> V1 边界：V1 只启用 `hybrid` provider。`sql` 和 `graph` 是最终架构里的 provider 候选，不是 V1 runtime 能力。

---

## 1. V1 的定位

Atlas 的最终形态是三引擎 Evidence Kernel：

| Provider | 目标能力 | V1 状态 |
|---|---|---|
| `hybrid` | 文本证据检索。内部包含 dense、BM25、metric alias、section、table textual lanes。 | 已启用，是 V1 唯一 provider。 |
| `sql` | 结构化表格、SQL、计算、cell provenance。 | 未启用，属于 V4 候选能力。 |
| `graph` | 实体关系、跨文档线索、community/global context。 | 未启用，属于 V3 候选能力。 |

V1 要证明的不是“把所有 provider 都接上”，而是：

```text
QueryPlan
  -> RetrievalTask
  -> TextHybridProvider
  -> Candidate
  -> EvidenceBlock
  -> EvidencePack
  -> VerificationResult
```

这条证据链足够稳之后，后续 `sql` 和 `graph` 才能作为 provider 接入，而不是绕过 Evidence Kernel 自己生成答案。

V1 不做：

```text
ResearchJob / worker / report generator
Text-to-SQL
SQL calculation
cell-level table provenance
GraphRAG 主路径
graph summary 直接当引用证据
streaming ingestion
```

---

## 2. 总体架构

最终三引擎架构：

```text
User Query
  -> Query Orchestrator
  -> Provider Router
       enabled_providers = ["hybrid"] in V1
       future = ["hybrid", "sql", "graph"]
  -> Provider-local retrieval
       hybrid: dense + sparse + textual table lanes
       sql: structured table / calculation, future only
       graph: graph candidate generation, future only
  -> Cross-provider evidence normalization
  -> Evidence Builder
  -> Evidence Evaluator
  -> Answer Generator
  -> Citation Verifier
  -> Trace + Eval + Cache
```

V1 实际启用路径：

```text
User Query
  -> Query Orchestrator
  -> enabled_providers prompt: ["hybrid"]
  -> QueryPlan / RetrievalUnit(provider="hybrid")
  -> RetrievalTask
  -> TextHybridProvider
  -> Dense Lane + BM25 Sparse Lane + textual helper lanes
  -> Python Weighted RRF, or Qdrant RRF as an ablation path
  -> Reranker
  -> parent-child EvidencePack
  -> Evidence Evaluator
  -> Answer Generator
  -> Citation Verifier
  -> Trace + Eval + Postgres query_cache
```

关键原则：

```text
Provider 不是 lane。
V1 provider 只有 hybrid。
Dense / BM25 / table textual 是 TextHybridProvider 内部 lanes。
SQLProvider 和 GraphProvider 不应出现在 V1 executable QueryPlan 中。
```

---

## 3. QueryPlan contract

V1 的 planner 可以知道最终 provider vocabulary，但必须受 `enabled_providers` 限制。

```python
ProviderName = Literal["hybrid", "sql", "graph"]

class QueryPlan:
    original_query: str
    standalone_query: str | None
    query_type: str
    entities: list[Entity]
    periods: list[Period]
    metrics: list[Metric]
    metadata_filter: dict
    retrieval_units: list[RetrievalUnit]
    risk_flags: list[str]
    budget: RetrievalBudget
    metadata: dict

class RetrievalUnit:
    unit_id: str
    purpose: str
    text: str
    provider: ProviderName
    metadata_filter: dict
    must_have_terms: list[str]
    should_terms: list[str]
    top_k: int
    weight: float
    metadata: dict
```

如果代码仍使用 `retrievers` 字段，V1 语义应按下面理解：

```text
retrievers = ["hybrid"]
```

它不是让 LLM 输出 `["dense", "bm25"]`。`dense` 和 `bm25` 是 `hybrid` provider 内部的执行 lane。

---

## 4. Planner Prompt Paradox

### 4.1 悖论是什么

Design 文档需要告诉 planner 最终架构里有三个 provider：

```text
hybrid / sql / graph
```

但 V1 runtime 只有 `hybrid` provider。如果 prompt 只说“选择最合适的 provider”，模型会自然输出：

```yaml
provider: sql
```

或：

```yaml
provider: graph
```

这在架构上看起来合理，在 V1 runtime 中却不可执行。这就是 Planner Prompt Paradox：

```text
架构 vocabulary 比 runtime enabled providers 更大。
如果 prompt 不动态收窄，LLM 会计划出不存在的能力。
```

### 4.2 V1 的解决方式

planner prompt 必须动态注入：

```yaml
enabled_providers:
  - hybrid

disabled_providers:
  sql:
    reason: "not implemented in V1; do not output SQL units"
  graph:
    reason: "not implemented in V1; do not output Graph units"
```

prompt 还必须明确：

```text
Return only providers listed in enabled_providers.
For V1, all executable retrieval_units must use provider="hybrid".
If a query looks like SQL or graph work, express the best available textual evidence search as hybrid units.
Do not output disabled providers as placeholders.
```

### 4.3 Unit-first plan

V1 planner 应该优先生成 retrieval units，而不是让 LLM 同时自由声明两套事实。

不推荐：

```yaml
periods:
  - "2018"
retrieval_units:
  - text: "3M 2017 capex"
```

推荐：

```text
LLM proposes retrieval_units.
System derives entities / periods / metrics from original query + accepted units + ontology.
Validator checks grounding.
Compiler writes QueryPlan / RetrievalTask.
```

这样可以减少全局 slots 与局部 units 冲突。

---

## 5. Compound Unit Retry

一个 `RetrievalUnit` 只能服务一个 provider。

错误：

```yaml
unit_id: u1
text: "Compare Apple and Microsoft Vision Pro exposure and cite annual reports"
provider: [graph, hybrid]
```

正确：

```yaml
unit_id: u1
text: "Apple Microsoft Vision Pro annual report discussion"
provider: hybrid
```

未来如果同时启用 `graph`，应拆成两个 unit：

```yaml
- unit_id: u_graph_vision_pro
  provider: graph
  purpose: related_entity_candidate_generation

- unit_id: u_hybrid_source_text
  provider: hybrid
  purpose: source_text_grounding
```

V1 retry 规则：

```text
1. 如果 LLM 输出 disabled provider，带错误原因重试。
2. 如果 LLM 输出 compound provider unit，要求拆成 single-provider units 后重试。
3. 如果重试后仍不可执行，fallback 到 deterministic hybrid-only plan。
4. fallback 不生成 sql / graph unit。
```

这条规则防止 planner 把“最终架构的理想路径”误当成“当前可执行路径”。

---

## 6. TextHybridProvider 内部执行

`TextHybridProvider` 是 V1 唯一启用 provider。它内部可以有多条 lane，但对外只产出 `provider="hybrid"` 的 candidates。

每个 accepted `RetrievalUnit` 会编译成 `RetrievalTask`：

```text
unit.text
unit.metadata_filter
unit.must_have_terms
unit.should_terms
unit.top_k
unit.weight
```

执行流：

```text
1. dense_text = unit.text
2. sparse_text = unit.text + should_terms + sparse-boosted must_have_terms
3. metadata_filter = plan.metadata_filter + unit.metadata_filter
4. metadata_filter 编译为 Qdrant payload filter
5. Dense lane 用 dense_text 查询 Qdrant dense vector
6. Sparse lane 用 sparse_text 查询 Qdrant BM25 sparse vector
7. Table / metric_alias / section lanes 只是在 sparse_text 上增加受控 textual hints
8. 合并 dense + sparse candidates
9. 用 Python Weighted RRF，或 Qdrant RRF 做 ablation
10. Reranker 重排 top-N
11. Evidence Builder 做 child -> parent expansion
```

### 6.1 dense_text

```text
dense_text = unit.text
```

Dense lane 不应该强行拼入过多精确词约束。它的职责是语义召回。

### 6.2 sparse_text

```text
sparse_text = unit.text + should_terms + repeat(must_have_terms, 3)
```

BM25 sparse lane 需要指标别名、报表行名、section hint 这类词法线索。例如：

```text
unit.text:
  3M FY2018 capex

should_terms:
  capital expenditure
  purchases of property, plant and equipment
  investing activities

must_have_terms:
  3M
  FY2018
```

Sparse query 变成：

```text
3M FY2018 capex
capital expenditure
purchases of property, plant and equipment
investing activities
3M 3M 3M FY2018 FY2018 FY2018
```

这里的 `repeat(must_have_terms, 3)` 是一个故意朴素的工程实现：不去改 BM25 / sparse index 底层计分公式，而是在 sparse input 里把关键 term 重复 3 次，让倒排/稀疏匹配自然提高这些词的权重。它只进入 sparse/textual lanes，不进入 dense_text。

### 6.3 metadata_filter

`metadata_filter` 是可执行约束，不是 prompt 描述。

V1 支持的方向：

```yaml
metadata_filter:
  document_ids:
    - doc_aapl_10k_2024
  company:
    - AAPL
  filing_type:
    - 10-K
```

Provider compiler 必须把它转换成 Qdrant payload filter。已经存在于 Postgres metadata 里的字段也要保持同名，方便 trace 和 eval 对齐。

### 6.4 must_have_terms 是实验变量

`must_have_terms` 不能当作完整语义模型。

它可以用于：

```text
reranker 输入提示
candidate coverage trace
eval ablation
默认 sparse boost
可选 lexical post-filter 实验
```

它不应该默认变成所有 lanes 的硬过滤条件，原因是：

```text
1. PDF/OCR/table serialization 可能让关键词形态变化。
2. 公司名可能只在 parent/page metadata 中，不在 child text 中。
3. 财务指标常以别名出现，must_have_terms 容易误杀。
4. 一个词命中不等于 evidence 支持答案。
```

所以 V1 把 `must_have_terms` 标记为 experimental retrieval variable，需要用 FinanceBench 或专项 eval 证明收益后再提升为默认硬约束。

V1 设计默认取舍是：

```text
must_have_terms 不做 hard filter。
must_have_terms 会以 repeat=3 的方式进入 sparse_text boost。
trace 记录 sparse_boost_terms 和 sparse_boost_repeat，方便后续消融。
```

### 6.5 Fusion

默认融合策略：

```text
Python Weighted RRF
```

公式：

```text
score(d) = sum(weight_i / (rrf_k + rank_i(d)))
```

保留 trace：

```text
lane
rank
raw_score
lane_weight
unit_weight
weighted_contribution
fusion_score
```

可选实验路径：

```text
Qdrant RRF
```

Qdrant RRF 可以减少应用层合并逻辑，但 V1 仍需要保留同等 trace，否则无法解释 evidence 为什么被选中。

---

## 7. Table Lane 的 V1 边界

V1 的 table lane 是 stopgap：

```text
serialized table text retrieval only
```

它可以检索这样的文本：

```text
Consolidated Statements of Cash Flows
Purchases of property, plant and equipment
2018: 1,577
2017: 1,373
```

它不能提供：

```text
SQL 查询
精确计算
cell provenance
row_id / column_id / bbox 级引用
公式验证
跨表 join
```

因此 table lane 返回的是 text evidence，而不是 structured fact。答案里如果需要数值比较，可以基于 evidence 解释；如果需要严格计算链路，应推迟到 V4 `sql` provider。

---

## 8. Graph 的边界与 hub node 风险

`graph` 是未来 provider，但 V1 不启用。

Graph 适合做：

```text
实体关系线索
跨文档候选生成
主题/社区导航
source text retrieval expansion
```

Graph 不应该做：

```text
替代 TextHybridProvider
把 graph summary 直接当引用证据
把 graph node/edge 当 SQL-like exact fact
让高频 hub node 支配检索
```

hub node explosion 风险：

```text
Vision Pro
AI
cloud
revenue
Microsoft
Apple
```

这类节点可能连接大量 chunk 和公司。如果 GraphProvider 把 hub neighbors 全部展开，候选池会迅速爆炸，且相关性下降。

未来 GraphProvider 的正确定位是：

```text
graph -> candidate generator -> source chunk/page grounding -> EvidenceBlock
```

不是：

```text
graph node -> VIP EvidenceBlock -> answer
```

Graph 输出必须回到 source text，才能进入 EvidencePack。

---

## 9. AAPL / MSFT / Vision Pro 示例

用户问：

```text
Compare AAPL and MSFT exposure to Vision Pro based on annual report evidence.
```

最终架构里，planner 可能想用：

```text
graph:
  Vision Pro -> Apple -> Microsoft -> product/ecosystem relationships

hybrid:
  annual report source text grounding

sql:
  count mentions / segment revenue, if structured facts exist
```

但 V1 enabled providers 只有：

```yaml
enabled_providers:
  - hybrid
```

因此 V1 的 actual plan 应该是 hybrid-only：

```yaml
retrieval_units:
  - unit_id: u0
    provider: hybrid
    purpose: original
    text: "AAPL MSFT Vision Pro annual report exposure"
    should_terms:
      - Apple Vision Pro
      - Microsoft annual report
      - product ecosystem
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
      - wearables
      - services
      - products

  - unit_id: u2
    provider: hybrid
    purpose: microsoft_source_text
    text: "Microsoft MSFT Vision Pro annual report partnership product discussion"
    should_terms:
      - Microsoft
      - Apple Vision Pro
      - mixed reality
      - devices
```

V1 实际执行路径：

```text
1. planner prompt 只允许 hybrid provider。
2. TextHybridProvider 生成 dense_text = unit.text。
3. TextHybridProvider 生成 sparse_text = unit.text + should_terms + repeat(must_have_terms, 3)。
4. metadata_filter 限制 filing_type / document_ids，如果调用方提供了可执行 metadata。
5. Qdrant dense 和 Qdrant BM25 sparse 分别召回 child chunks。
6. Python Weighted RRF 或 Qdrant RRF 合并。
7. Reranker 选择最能支持 AAPL/MSFT/Vision Pro 比较的 source chunks。
8. Evidence Builder 扩展到 parent/page blocks。
9. Answer Generator 只能基于 evidence 做比较。
```

V1 不能声称：

```text
已经用 graph 分析 Vision Pro hub。
已经用 SQL 计算 exposure。
已经提供 mention count 的 cell/source provenance。
```

如果 evidence 不足，V1 应返回保守答案，并在 trace 中显示是 hybrid source evidence 不足，而不是假装 graph/sql 已经执行。

---

## 10. Evidence 与验证

V1 的 answer 只能来自 EvidencePack。

```python
class EvidenceBlock:
    evidence_id: str
    provider: str
    source_type: str
    text: str
    document_id: str
    page_start: int | None
    page_end: int | None
    chunk_ids: list[str]
    candidate_ids: list[str]
    coverage: dict
    metadata: dict
```

`provider` 在 V1 中应为：

```text
hybrid
```

`source_type` 可以是：

```text
text_chunk
page_block
serialized_table_text
```

生成前：

```text
Evidence Evaluator 判断 supported / insufficient / contradicted / partially_supported。
```

生成后：

```text
Citation Verifier 检查 citation 是否来自 evidence，以及关键数字是否被 cited evidence 支持。
```

---

## 11. Eval 要证明什么

V1 eval 不应该只证明 dense + BM25 能跑。它要证明：

```text
planner 没有输出 disabled providers
compound units 被 retry 或 fallback
hybrid provider 的 dense/sparse/table textual lanes 有可解释 contribution
metadata_filter 实际影响 Qdrant payload filter
must_have_terms 作为实验变量的收益/风险
Python Weighted RRF 与 Qdrant RRF 的差异
EvidencePack 是否包含 gold doc/page
generated answer 是否有 citation_doc/page_hit 和 numeric match
```

最低 ablation：

```text
dense_text only
sparse_text only
hybrid Python Weighted RRF
hybrid Qdrant RRF
hybrid + reranker
hybrid + reranker + Evidence Evaluator / Citation Verifier
must_have_terms off/on
table textual lane off/on
```

---

## 12. Definition of Done

V1 完成标准：

```text
1. Query Orchestrator 使用 dynamic enabled_providers prompt。
2. V1 executable QueryPlan 只包含 provider="hybrid"。
3. disabled sql/graph provider 会触发 retry 或 fallback，而不是进入 runtime。
4. compound unit 会被拆分重试。
5. TextHybridProvider 明确执行 dense_text = unit.text。
6. TextHybridProvider 明确执行 sparse_text = unit.text + should_terms + repeat(must_have_terms, 3)。
7. metadata_filter 能进入 Qdrant payload filter，并写入 trace。
8. must_have_terms 的默认行为和实验行为可通过 eval 区分。
9. Table lane 被标注为 serialized table text stopgap。
10. Graph 被标注为 future candidate generator，不能直接当 VIP EvidenceBlock。
11. AAPL/MSFT/Vision Pro 这类 graph-looking query 在 V1 走 hybrid-only source evidence path。
12. trace / eval / cache 都能复盘每一步。
```

---

## 13. V1 结论

V1 的名字是 Advanced Hybrid Kernel，但它不是“把所有高级检索都塞进 V1”。

V1 的准确结论是：

```text
Atlas 建立了 hybrid-only Evidence Kernel。
最终架构保留 hybrid / sql / graph 三个 provider 槽位。
当前 runtime 只启用 hybrid。
所有未来 provider 都必须回到 Candidate / EvidenceBlock / EvidencePack / VerificationResult 合约。
```
