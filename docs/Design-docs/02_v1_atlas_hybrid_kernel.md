# V1：Atlas Hybrid Kernel

> V1 的目标是把 V0 的“能回答”升级成“更容易答对”。它重点解决单纯向量检索不稳定、引用不够精确、证据不足判断不可靠、重复查询成本高等问题。

---

## 1. V1 的目标

V0 已经可以把文档导入系统，做向量检索，并生成带引用的回答。但 V0 的检索方式比较单一。它主要依赖向量相似度，而单纯向量检索在很多场景下并不稳定。比如用户问某个具体条款、某个公司 ticker、某个数字、某个年份、某个文件名，向量检索可能会召回语义相近但并不精确的 chunk。

V1 的核心目标是增强检索质量。它会在 V0 的基础上加入关键词检索、metadata filter、结果融合、reranker、cache 和 Critic Lite。这样系统不再只靠“语义相似”，而是能同时利用精确关键词、来源信息、时间信息和重排序模型来找到更可靠的证据。

V1 做完后，系统应该能回答一个更关键的问题：

```text
为什么这个答案比 V0 更可靠？
```

答案应该来自 eval report，而不是主观感觉。V1 必须能对比 dense-only 和 hybrid + rerank 的效果，证明检索命中率、citation correctness、faithfulness 至少有部分提升。

---

## 2. V1 要解决的具体问题

V0 最常见的问题有六类。

第一类是精确词漏检。用户问“文档里有没有提到 Basel III”，向量检索可能找到很多“银行监管风险”的内容，但不一定找到真正包含 Basel III 的 chunk。关键词检索在这种情况下更可靠。

第二类是数字和日期不稳定。用户问“2024 Q3 margin 是多少”，向量检索可能找到了 margin 的讨论，但没有找到 Q3 或 2024 的具体表述。关键词检索、metadata filter 和 reranker 可以一起改善这类问题。

第三类是 top-k 召回里有很多泛泛而谈的 chunk。向量检索召回的是“相似”，但相似不等于能支持答案。V1 的 reranker 会把候选 chunk 和用户问题重新配对打分，把真正能回答问题的证据排到前面。

第四类是 evidence 太多太乱。V0 可能直接把 top 12 chunks 交给模型，里面既有相关信息，也有重复段落和弱相关内容。V1 的 Evidence Builder 会去重、合并相邻 chunk、控制上下文长度，减少模型被噪音干扰。

第五类是系统不会很好地拒答。V0 可以通过 prompt 要求模型证据不足时拒答，但这不够稳定。V1 加入 Critic Lite，专门判断当前 evidence 是否足以回答问题。

第六类是重复查询成本高。相同或极其相似的问题每次都重新检索、rerank、调用模型，会浪费时间和成本。V1 加入 exact cache 和高阈值 semantic cache，用来节省重复工作。

---

## 3. V1 系统结构

V1 的核心变化是 Query Runtime 变成一个更完整的检索控制层。它不再只调用 Qdrant，而是根据问题类型决定检索策略。

```text
┌──────────────────────────────────────────────┐
│                   用户问题                    │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                 Query Runtime                 │
│                                              │
│  标准化 query                                │
│  cache lookup                                │
│  判断问题类型                                 │
│  生成检索配置                                 │
│  生成 semantic query / keyword query          │
└───────────────────────┬──────────────────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Dense Search  │ │ Keyword Search│ │ Metadata     │
│ Qdrant        │ │ Postgres FTS  │ │ Filters      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌──────────────────┐
              │ Hybrid Fusion     │
              │ 去重 / 合并 / 加权 │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Reranker          │
              │ 精排候选 evidence │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Evidence Builder  │
              │ 合并 / 去重 / 截断 │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Critic Lite       │
              │ 判断证据是否足够   │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Answer Generator  │
              │ 生成答案和引用     │
              └──────────────────┘
```

V1 看起来组件变多了，但每个组件职责都很清楚。Query Runtime 负责决定怎么检索；Dense Search 负责语义召回；Keyword Search 负责精确召回；Fusion 负责合并；Reranker 负责精排；Evidence Builder 负责整理证据；Critic Lite 负责判断证据是否足够；Answer Generator 负责生成最终答案。

---

## 4. Query Runtime 设计

Query Runtime 是 V1 的控制中心。它接收用户问题后，不应该立刻把问题丢给 Qdrant，而是先判断这个问题应该怎么检索。不同类型的问题适合不同策略。

比如，用户问“GraphRAG 和普通 RAG 有什么区别”，这是语义型问题，向量检索通常就有不错效果。用户问“哪里提到了 Section 230”，这是关键词敏感问题，需要提高 keyword search 权重。用户问“2024 年 Q3 的 revenue 是多少”，这类问题需要识别日期和数字，后续可能还要 metadata filter。用户问“总结这些文档中关于供应链风险的内容”，这类问题需要更宽的召回和更强的 evidence consolidation。

V1 的 Query Runtime 不需要复杂 Agent。它可以先用规则做轻量分类，再逐步加小模型分类。它输出的不是自由文本 plan，而是一个检索配置。

示例输出：

```json
{
  "query_type": "keyword_sensitive",
  "use_dense": true,
  "use_keyword": true,
  "use_reranker": true,
  "dense_top_k": 30,
  "keyword_top_k": 30,
  "rerank_top_k": 8,
  "filters": {
    "source_type": null,
    "date_range": null,
    "document_ids": []
  }
}
```

这个配置会决定后续 pipeline 怎么走。这样做的好处是可解释：当某个 query 出错时，工程师可以看到系统为什么选择了 dense+keyword，为什么 top_k 是 30，为什么用了或者没用 filter。

---

## 5. Query Rewriter 设计

V1 需要一个轻量 Query Rewriter。它的作用不是改写用户真实意图，而是为不同检索器生成适合的查询形式。一个用户问题可以同时产生 semantic query、keyword query 和 metadata hints。

例如用户问：

```text
“苹果最近 10-K 里关于供应链风险怎么说？”
```

Query Rewriter 可以输出：

```json
{
  "semantic_query": "Apple annual report supply chain risk discussion",
  "keyword_query": "\"supply chain\" risk Apple 10-K",
  "metadata_hints": {
    "company": "Apple",
    "source_type": "filing",
    "form_type": "10-K"
  }
}
```

这个模块要非常克制。它不应该把用户问题改得面目全非，也不应该丢掉关键实体。很多 RAG 系统会因为 query rewrite 过度发挥导致检索错方向。V1 的 Query Rewriter 应该遵守一个原则：它只帮助检索，不改变最终回答的问题。

在 trace 中应该记录 original_query、semantic_query、keyword_query 和 metadata_hints。这样 eval 发现错误时，可以判断是不是 rewrite 阶段丢掉了重要信息。

---

## 6. Keyword Retriever 设计

Keyword Retriever 负责精确匹配。V1 可以先用 Postgres full-text search，不必一开始引入 OpenSearch。Postgres FTS 对中小规模文档已经够用，而且部署复杂度低。等 chunk 数达到数百万以上、关键词检索延迟变高或需要更复杂搜索语法时，再接 OpenSearch。

Keyword Retriever 适合处理以下场景：

```text
精确术语：Basel III、Section 230、SOX、GDPR
股票代码：AAPL、TSLA、NVDA
日期：2024 Q3、March 2026
文件名或标题：annual_report_2024.pdf
短语查询：“supply chain concentration”
```

Keyword Retriever 的输出结构应该和 Dense Retriever 一致。它也返回 Candidate Chunk，只是 score 的含义不同。统一输出结构很重要，因为后面的 Fusion 和 Reranker 不应该关心候选是从 dense 还是 keyword 来的。

统一 Candidate 结构可以是：

```text
candidate_id
chunk_id
document_id
retriever_type
raw_score
rank
text
metadata
```

V1 的 trace 里应该能看到 dense search 找到了什么、keyword search 找到了什么。如果 hybrid 结果提升了质量，eval 应该能证明是 keyword 补回了 dense 漏掉的证据。

---

## 7. Metadata Filter 设计

Metadata Filter 用来缩小检索范围。它可以按文档、来源、时间、语言、文件类型、标签等字段过滤。V0 里 metadata 主要用于展示 citation，V1 开始 metadata 会成为检索质量的一部分。

典型 filter 包括：

```text
document_ids：只在指定文档里查
source_type：只查 filing / report / news
date_range：只查某个时间范围
language：只查中文或英文
file_type：只查 PDF 或 Markdown
section_title：只查某类章节
```

Filter 要谨慎使用。过强的 filter 会导致正确证据被排除。比如用户问“苹果最近报告里怎么说”，如果系统错误识别 source_type，只查 filing，就可能漏掉用户上传的内部 research report。因此 V1 的 filter 应该分成 hard filter 和 soft boost。

Hard filter 是用户明确要求的限制，比如“只看这份文档”。Soft boost 是系统推断出来的偏好，比如 query 里出现“10-K”，那 filing 可以加权，但不一定完全排除其他来源。V1 初期可以先实现 hard filter，后面再实现 soft boost。

---

## 8. Hybrid Fusion 设计

Hybrid Fusion 负责把 dense 和 keyword 的结果合并。它要解决三个问题：去重、合并分数、排序。

同一个 chunk 可能同时被 dense 和 keyword 检索到。Fusion 应该把它们合并成一个候选，而不是重复传给 reranker。一般来说，同时被 dense 和 keyword 找到的 chunk 更可能是好证据，因此可以给它更高权重。

V1 推荐使用简单、可解释的融合方式，例如 Reciprocal Rank Fusion 的思想。不用一开始训练复杂融合模型。Fusion 可以这样理解：

```text
dense 排名靠前 → 加分
keyword 排名靠前 → 加分
两个检索器都找到 → 再加分
metadata 匹配 → 再加分
```

Fusion 输出的候选数量可以控制在 40-80 个之间，再交给 reranker 精排。不要直接把 dense top 30 + keyword top 30 全部塞给 LLM，因为候选里会有大量重复和弱相关内容。

Fusion 结果也要写 trace。每个候选应该记录：

```text
chunk_id
dense_rank
dense_score
keyword_rank
keyword_score
fused_score
metadata_boost
```

这样后面可以分析：正确证据到底是 dense 找到的，还是 keyword 找到的，还是融合后才排上来的。

---

## 9. Reranker 设计

Reranker 是 V1 的关键组件。Retriever 的任务是“多找一点”，Reranker 的任务是“排得更准”。它把用户问题和候选 chunk 放在一起判断相关性，比单纯向量相似度更精确。

典型流程是：

```text
Dense Retriever 返回 30 个候选
Keyword Retriever 返回 30 个候选
Fusion 合并成 40-50 个候选
Reranker 精排
保留 top 5-10 个 Evidence
```

Reranker 可以使用 cross-encoder 模型，也可以先用可调用的 rerank API。它不应该由大语言模型完成，因为 rerank 是高频任务，用大模型成本太高。Reranker 的输入是 query 和 candidate text，输出是 rerank_score。

Reranker 的失败模式也需要考虑。如果 reranker 服务不可用，系统应该降级为使用 fused_score 排序，而不是整个 query 失败。V1 的降级策略可以是：

```text
Reranker 正常：使用 rerank_score 排序。
Reranker 失败：使用 fused_score 排序，并在 trace 中标记 reranker_unavailable。
```

Reranker 的效果必须通过 eval 证明。V1 的 eval report 应该对比 rerank 前后的 hit@5。如果 reranker 没有提升，说明候选召回、模型选择或 chunk 质量有问题。

---

## 10. Evidence Builder 设计

Evidence Builder 负责把 reranker 之后的候选整理成适合 LLM 使用的 evidence pack。它不是简单地取 top 8，而是要做去重、合并、截断和 citation metadata 保留。

它主要做五件事：

```text
1. 去掉重复 chunk。
2. 合并同一文档中相邻的 chunks。
3. 去掉过短或噪音明显的片段。
4. 控制总 token 数。
5. 保留每段 evidence 的 source metadata。
```

例如，reranker top 10 里可能有 4 个 chunk 都来自同一页相邻段落。如果全部传给模型，会浪费 token，还可能让答案重复。Evidence Builder 可以把相邻 chunks 合并成一个 evidence block，并保留 chunk_ids 列表。

Evidence Builder 不建议用 LLM 大量改写证据。证据进入生成模型之前最好保持原文，因为一旦在 evidence 阶段被改写，就很难判断后续答案到底是基于原文还是基于改写后的二手文本。V1 的 Evidence Builder 应该尽量 deterministic。

Evidence Pack 可以这样设计：

```json
{
  "evidence_blocks": [
    {
      "evidence_id": "ev_001",
      "document_id": "doc_001",
      "chunk_ids": ["chunk_12", "chunk_13"],
      "source_title": "annual_report.pdf",
      "page_range": [12, 13],
      "section_title": "Risk Factors",
      "text": "...",
      "score": 0.91
    }
  ]
}
```

这个结构会被 Answer Generator、Critic Lite、Citation Builder 和 Eval Harness 共用。

---

## 11. Critic Lite 设计

Critic Lite 是 V1 用来判断证据是否足够的模块。它不负责写答案，只负责判断当前 evidence pack 是否能支撑回答。这个模块可以用小模型，也可以先用一组规则 + LLM judge 混合实现。

Critic Lite 的输入是：

```text
user_query
evidence_pack
optional draft_answer
```

输出是：

```text
supported：证据足够，可以回答。
insufficient：证据不足，应该拒答或部分回答。
conflicted：证据之间有冲突，答案需要展示冲突。
```

例如，用户问“这家公司 2025 年收入是多少”，但 evidence 里只有 2023 年收入和 2024 年收入。Critic Lite 应该输出 insufficient，而不是让 Answer Generator 推测 2025 年数据。

Critic Lite 的价值在于把“要不要回答”从生成器里拆出来。生成器通常倾向于回答，因为它被训练成帮助用户完成任务。Critic Lite 则应该更保守，专门判断证据边界。

Critic Lite 的判断也必须写 trace。每次 query 应该记录：

```text
critic_judgment
critic_reason
missing_evidence
conflict_notes
```

V1 不做 reflexive loop。也就是说，Critic Lite 判断 insufficient 后，系统不自动补检索多轮。补检索放到 V2。V1 的行为是：证据不足就明确返回 insufficient，或者生成一个带限制说明的部分答案。

---

## 12. Cache Layer 设计

V1 必须有 cache layer。Cache 的目标有两个：降低重复查询延迟，降低重复模型调用成本。具体 backend 可以先用 Postgres exact cache，也可以换成 Redis；这里的关键不是 Redis 这个组件本身，而是 cache key、失效策略、trace 可见性和 benchmark 默认关闭 cache。

第一层是 Exact Cache。它基于 normalized query 的 hash。如果用户问完全相同的问题，并且数据版本和 prompt 版本没有变化，系统可以直接返回缓存结果。

Exact Cache 的 key 可以是：

```text
atlas:cache:exact:{query_hash}:{retrieval_version}:{prompt_version}
```

value 包含 answer、citations、confidence、created_at、trace_id 等字段。

第二层是 Semantic Cache。它基于 query embedding 查找非常相似的历史问题。Semantic Cache 要非常谨慎，因为两个问题语义相似不代表答案可以复用。建议只在相似度非常高时使用，例如 0.95 以上，而且 time-sensitive query 不走 semantic cache。

Cache 的失效策略可以先用 TTL。比如普通知识查询缓存 1 小时，文档内查询缓存 24 小时，时效性查询缓存 5 分钟。V1 不需要复杂事件驱动 cache invalidation，因为系统还没有持续数据流入。

如果 cache 命中，trace 里也要记录。不要让 cache 成为不可见的黑盒。Eval 时可以选择关闭 cache，确保评估的是检索和生成能力，而不是历史答案复用。

---

## 13. V1 查询流程

V1 的完整查询流程如下：

```text
1. Gateway 接收 query。
2. Query Runtime 标准化 query。
3. 查询 exact cache。
4. 查询 semantic cache。
5. Cache miss 后判断 query type。
6. Query Rewriter 生成 semantic query、keyword query、metadata hints。
7. Dense Retriever 查询 Qdrant。
8. Keyword Retriever 查询 Postgres FTS。
9. Fusion 合并 dense 和 keyword 结果。
10. Reranker 对候选证据精排。
11. Evidence Builder 整理 evidence pack。
12. Critic Lite 判断 evidence 是否足够。
13. Answer Generator 生成答案。
14. Citation Builder 输出引用。
15. Trace Logger 记录完整链路。
16. Response 写入 cache 并返回。
```

和 V0 相比，V1 的链路更长，但每一步都有明确目的。V1 的挑战不是“能不能跑”，而是“每增加一个模块，是否真的提升质量”。因此 V1 必须围绕 eval 做开发，而不是围绕组件数量做开发。

---

## 14. V1 API 变化

V1 的 API 可以沿用 V0，但 query options 需要更丰富。

`POST /v1/query` 的 body 可以支持：

```json
{
  "query": "Where does the document mention Basel III?",
  "filters": {
    "document_ids": [],
    "source_type": null,
    "date_range": null
  },
  "options": {
    "use_cache": true,
    "use_keyword": true,
    "use_reranker": true,
    "top_k": 8,
    "return_trace": true
  }
}
```

返回值除了 V0 的 answer、confidence、citations，还可以增加 retrieval_summary：

```json
{
  "answer": "...",
  "confidence": "supported",
  "citations": [...],
  "retrieval_summary": {
    "cache_hit": false,
    "query_type": "keyword_sensitive",
    "dense_candidates": 30,
    "keyword_candidates": 30,
    "reranked_candidates": 8
  },
  "trace_id": "tr_001"
}
```

这类字段对前端不是必须，但对开发和展示非常有价值。别人看到响应就能理解系统不是简单向量检索，而是一个可解释的检索流水线。

---

## 15. V1 Trace 设计

V1 的 trace 要比 V0 更细。因为 V1 有多个 retriever、fusion、reranker、critic，如果 trace 不细，系统出错后很难判断是哪一步的问题。

每次 query 至少记录：

```text
query_type
semantic_query
keyword_query
filters_applied
cache_hit
dense_results
keyword_results
fusion_results
rerank_results
evidence_pack
critic_judgment
answer
citations
latency_breakdown
cost_estimate
```

Latency breakdown 很重要。它可以告诉你系统慢在哪里：是 Qdrant 慢，Postgres FTS 慢，reranker 慢，还是 LLM 慢。

例如：

```json
{
  "latency_breakdown_ms": {
    "cache_lookup": 12,
    "query_rewrite": 180,
    "dense_search": 45,
    "keyword_search": 38,
    "fusion": 5,
    "rerank": 320,
    "critic": 480,
    "generation": 2400
  }
}
```

这种数据会直接指导后续优化。如果 generation 占 70% 时间，优化 Qdrant 不会显著改善整体延迟。如果 reranker 太慢，就需要 batch、换模型或降低候选数量。

---

## 16. V1 Evaluation Harness 设计

V1 的 eval 要从“最终答案评估”扩展为“检索链路评估”。因为 V1 的主要目标是提升 retrieval quality，所以 eval 必须能证明 dense-only、keyword-only、hybrid、hybrid+rerank 的差异。

V1 推荐四层评估：

```text
1. Dense Retrieval Eval
   看向量检索是否召回 expected source。

2. Keyword Retrieval Eval
   看关键词检索是否召回 expected source。

3. Hybrid + Rerank Eval
   看融合和精排后正确 chunk 是否进入 top 5。

4. Answer + Citation Eval
   看最终答案是否忠实于 evidence，citation 是否正确。
```

Eval report 示例：

```text
Eval Run: 2026-xx-xx
Total cases: 120

Dense hit@10: 0.70
Keyword hit@10: 0.63
Hybrid hit@10: 0.83
Rerank hit@5: 0.78
Faithfulness: 0.86
Citation correctness: 0.81
Insufficient detection accuracy: 0.74
Average latency: 6.1s
Average cost: $0.006/query

Top improvements vs V0:
1. Keyword-sensitive queries improved from 0.52 to 0.79 hit@10.
2. Citation correctness improved from 0.68 to 0.81.
3. Insufficient detection reduced unsupported answers by 24%.

Top failures:
1. Query rewrite removed key entity in 5 cases.
2. Reranker preferred generic background chunks in 7 cases.
3. Critic Lite missed weak evidence in 4 cases.
```

这个 report 是 V1 的核心交付之一。如果 V1 没有 eval report，就很难说明 hybrid retrieval 和 reranker 真的带来了价值。

---

## 17. V1 容量规划

V1 相比 V0 增加了 keyword search、reranker 和 cache，因此延迟和资源会增加。合理目标如下：

| 指标 | V1 目标 |
|---|---:|
| 文档数 | 10万 - 50万 |
| Chunk 数 | 100万 - 500万 |
| 普通查询 QPS | 5 - 20 |
| 并发查询 | 20 - 100 |
| 查询延迟 | p95 5 - 10 秒 |
| Cache hit 延迟 | p95 < 1 秒 |
| Eval cases | 100 - 300 |
| Research job | 暂不正式支持 |

V1 的实际瓶颈一般不在 Gateway，而在 reranker 和 LLM。Reranker 如果是本地模型，需要考虑 batch 推理和并发限制。如果是外部 API，需要考虑 rate limit 和成本。LLM 生成阶段仍然是最大延迟来源，因此 V1 应该记录 latency breakdown，并用 cache 降低重复查询成本。

---

## 18. V1 失败模式与降级

V1 的组件更多，因此必须设计降级策略。

如果 Keyword Retriever 失败，系统可以只走 Dense Retriever，并在 trace 中标记 keyword_unavailable。答案仍然可以返回，但检索质量可能下降。

如果 Reranker 失败，系统可以使用 fusion score 排序。不要让 reranker 成为单点失败。

如果 Critic Lite 失败，系统可以降级为 Answer Generator 自带拒答规则，但要在 trace 中标记 critic_unavailable。此时 confidence 可以更保守，例如标记为 `unknown` 或要求生成器更谨慎。

如果 cache backend 失败，系统不使用 cache，直接走完整查询链路。Cache 是性能优化，不应该影响功能正确性。

如果 Query Rewriter 失败，可以退回 original query 同时用于 dense 和 keyword search。rewrite 是增强项，不应该阻断主路径。

这些降级策略体现一个原则：V1 的主路径仍然应该能在组件部分失效时返回基于证据的答案，只是质量或速度下降。

---

## 19. V1 完成标准

V1 完成应达到：

```text
1. 支持 dense search 和 keyword search。
2. 支持 metadata filter。
3. 支持 hybrid fusion。
4. 支持 reranker。
5. 支持 Evidence Builder 去重、合并、截断。
6. 支持 Critic Lite 的 supported / insufficient / conflicted 判断。
7. 支持 exact cache。
8. 支持高阈值 semantic cache。
9. 支持完整 latency breakdown。
10. 支持 dense-only vs hybrid+rerank 的 eval 对比。
11. Eval report 能证明至少部分 query 类型质量提升。
```

V1 的核心不是“我用了很多组件”，而是“我能证明这些组件让系统更可靠”。如果 eval 没有提升，说明设计需要回头调整，而不是继续往 V2 堆 Agent。

---

## 20. V1 不做什么

V1 不做复杂 Planner。Query Runtime 可以判断 query type，但不应该拆成多步研究计划。

V1 不做 reflexive loop。Critic Lite 判断 insufficient 后，系统可以拒答或部分回答，但不自动补检索多轮。多轮补检索是 V2 的能力。

V1 不做 GraphRAG。Graph retrieval 放到 V3。

V1 不做结构化数据 Text-to-SQL。数值和表格分析放到 V4。

V1 不做 Kafka/Flink。持续写入路径仍然后移。

V1 的一句话目标是：

```text
把 V0 的单向量检索升级成可解释、可评估、可降级的 Hybrid RAG 内核。
```
