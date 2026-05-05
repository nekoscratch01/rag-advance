# V1 Design Gap Audit：Advanced Hybrid Evidence Kernel

更新时间：2026-05-05

本文是 V1 Design 对齐工作的起点审查。它只记录当前实现相对
`docs/Design-docs/01_V1_ADVANCED_HYBRID_KERNEL.md` 的差距，不把 Design
当作已经实现的事实。

## 审查结论

当前 V1 已经实现并验证了 FinanceBench hybrid retrieval 主路径：

```text
Dense + BM25 -> RRF -> local reranker -> parent evidence
```

Critic Lite、citation builder、Postgres query_cache 和 trace metadata 已有代码路径，但尚未被
generated answer benchmark 正式验证。

但它还不是 Design 文档里的完整 Advanced Hybrid Evidence Kernel。核心差距是：

```text
1. Query Orchestrator / Retrieval Plan 还没有成为正式 contract。
2. TextHybridProvider 还没有 provider/lane 边界。
3. RRF 仍是 dense + lexical 双路非加权融合，不是 multi-lane Weighted RRF。
4. EvidenceBlock 已有雏形，但 EvidencePack / coverage / drop reason 不完整。
5. Critic Lite 已存在，但 Evidence Evaluator 和 Citation Verifier 还没有正式拆分。
6. Trace 仍以 query_runs + retrieval_events + generation_events + details_json 为主，
   还没有 Design 要求的完整 trace 表族。
7. Contextual retrieval 和 table-aware chunk metadata 还没有落地。
8. Retrieval-only eval 已完成，generated answer reliability eval 尚未完成。
```

## 架构节点对照

| V1 Design 节点 | 当前状态 | 当前实现位置 | 差距与后续 PR |
|---|---|---|---|
| User Query | 已实现 | `src/atlas/api/routes/query.py`、`src/atlas/query_runtime/service.py` | API 只接收 query/options，尚未显式进入 QueryPlan。PR-03 接入 plan 消费，PR-09 补 plan/retrieve API。 |
| Query Orchestrator | 未实现 | 无正式目录 | 缺 rewrite、extraction、decomposition、plan、validator。PR-02 新增 `query_orchestrator`。 |
| Retrieval Plan | 未实现 | 无正式 contract | 现在 retrieval 直接吃原 query/options。PR-01/03 新增 QueryPlan/RetrievalTask。 |
| TextHybridProvider | 部分实现 | `src/atlas/retrieval/hybrid_retriever.py`、`mode_switching.py` | 已有 hybrid 行为，但不是 provider/lane 边界。PR-04 重组到 provider。 |
| Dense Lane | 已实现 | `src/atlas/retrieval/dense_retriever.py`、`src/atlas/embeddings/bge_local.py` | 需要迁入 provider lane，并记录 retrieval_unit/lane trace。PR-04。 |
| BM25 Lane | 已实现 | `src/atlas/retrieval/bm25_retriever.py`、`src/atlas/embeddings/bm25_sparse.py` | 需要迁入 provider lane，并支持 ontology alias units。PR-02/04。 |
| Metric Alias Lane | 未实现 | 无正式 lane | Design 要求 ontology-expanded financial terms。PR-02 提供 ontology，PR-04 接入 lane。 |
| Section-aware Lane | 未实现 | 无正式 lane | Design 要求 filing section / table title 约束；当前只有 chunk/parent metadata 雏形。PR-04 做 textual section-aware lane。 |
| Table Lane | 未实现 | 无正式 lane | V1 做 row/page textual lane；结构化 table store、SQL provider、cell provenance 推迟到 V4。PR-04。 |
| Parent-child Expansion | 部分实现 | `src/atlas/query_runtime/evidence_builder.py`、`parent_blocks` / `chunks` | child->parent 已实现；需要纳入 provider/evidence contract 和 trace。PR-04/07。 |
| Provider-local Fusion | 部分实现 | `src/atlas/retrieval/fusion.py` | 当前是两路 RRF，缺 lane weight、unit weight、multi-lane contribution trace。PR-05。 |
| Reranker | 已实现 | `src/atlas/retrieval/reranker.py` | 已有 CrossEncoder；缺 QueryPlan/RetrievalUnit-aware input trace。PR-06。 |
| Evidence Builder | 部分实现 | `src/atlas/query_runtime/evidence_builder.py` | 已有 child->parent、dedupe、token budget；缺正式 EvidencePack、coverage、drop reason。PR-07。 |
| Evidence Evaluator | 部分实现 | `src/atlas/query_runtime/critic_lite.py` | Critic Lite 兼任 pre/post 检查；需要拆成 evaluator/verifier contract。PR-08。 |
| Answer Generator | 已实现 | `src/atlas/llm/openai_client.py`、`src/atlas/llm/prompts.py` | 需要在 trace 中挂接 EvidencePack 与 verification result。PR-09。 |
| Citation Verifier | 部分实现 | `src/atlas/query_runtime/citation_builder.py`、`critic_lite.py` | citation builder 只解析 marker；正式 support check 需要独立 verifier。PR-08。 |
| Trace | 部分实现 | `query_runs`、`retrieval_events`、`generation_events`、`details_json` | 缺 query_plans、retrieval_tasks、candidates、evidence_packs、verification 表族。PR-09。 |
| Eval | 部分实现 | `src/atlas/benchmark/financebench_retrieval.py`、`financebench.py` | retrieval-only 已完成；answer/evidence/latency component benchmark 需要补齐。PR-10。 |
| Cache | 部分实现 | `src/atlas/query_runtime/cache.py`、`query_cache` | exact answer cache 已有；Design 的分层 cache 暂不全做，先复用 query_cache 并纳入 trace。PR-09/10。 |

## 当前实现事实

已实现：

```text
FinanceBench parent-child corpus
Postgres documents / parent_blocks / chunks
Qdrant dense vectors + BM25 sparse vectors
DenseRetriever
BM25Retriever
RRF fusion
local CrossEncoder reranker
parent evidence packer
Critic Lite
Postgres query_cache
retrieval-only FinanceBench benchmark
```

未实现或不完整：

```text
QueryPlan / RetrievalUnit / RetrievalTask contract
LLM structured QueryPlan
finance metric ontology
TextHybridProvider provider/lane structure
Table Lane textual row/page lane
Weighted RRF
EvidencePack
Evidence coverage
Contextual chunk enrichment
table-aware chunk metadata / searchable text
Evidence Evaluator
Citation Verifier
Design trace table family
component-level latency benchmark
generated answer reliability benchmark
```

推迟到 V4：

```text
结构化 table store
SQLProvider
cell-level provenance
financial_facts table
calculator / Text-to-SQL 主路径
```

## 后续 PR 对应关系

```text
PR-01 contract 层：
  QueryPlan / RetrievalUnit / RetrievalTask / Candidate / EvidenceBlock /
  EvidencePack / VerificationResult。

PR-02 Query Orchestrator：
  LLM structured planner、fallback planner、validator、finance metric ontology。

PR-03 Retrieval Plan / Task：
  QueryPlan 编译为 RetrievalTask，并接入 query/retrieve 路径。

PR-04 TextHybridProvider：
  provider 边界、Dense Lane、BM25 Lane、Metric Alias Lane、
  Section-aware Lane、V1 textual Table Lane、provider-level parent-child expansion。

PR-05 Weighted RRF：
  multi-lane Weighted RRF 与 fusion trace。

PR-06 Reranker：
  QueryPlan/RetrievalUnit-aware reranker trace。

PR-07 EvidencePack：
  parent-child evidence contract、coverage、prompt inclusion、drop reason。

PR-08 Evaluator / Verifier：
  Evidence Evaluator、Citation Verifier，Critic Lite 兼容层。

PR-09 Trace / API / DB：
  Design trace 表族、/v1/query/plan、/v1/retrieve。

PR-10 Eval / Docs：
  component benchmark、full V1 answer eval、milestone/version-arch 回写。
```

## 风险记录

```text
LLM QueryPlan 可能过度生成：
  必须有 deterministic fallback 和 validator。

Table Lane 容易滑向 V4：
  V1 只做文本化 row/page lane，不做结构化表格数据库。

Weighted RRF 可能引入噪声：
  必须保留 lane contribution trace，并用 benchmark 对比。

Trace 表族会扩大写路径：
  必须保持 /v1/query 兼容，落库失败不能吞掉主要错误上下文。

答案可靠性不能靠 retrieval-only 指标证明：
  PR-10 必须补 generated answer citation/numeric/unsupported 指标。
```
