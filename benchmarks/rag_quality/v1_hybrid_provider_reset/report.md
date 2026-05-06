# V1 Hybrid Provider Reset 基准报告

更新时间：2026-05-06

## 0. 结论先行

这轮报告的对象不是重新证明 V1 已经解决 FinanceBench，而是给
`TextHybridProvider` reset 后的策略空间建立一份可复查地图：query rewrite、filter、
fusion、candidate shape 和 reranker input 分别会带来什么收益、失败形态和取舍。

本轮新增的是一个离线 synthetic smoke harness：

```bash
python -m atlas.benchmark.v1_hybrid_provider_reset \
  --out benchmarks/rag_quality/v1_hybrid_provider_reset/smoke_runs \
  --run-id smoke_20260506
```

产物：

```text
benchmarks/rag_quality/v1_hybrid_provider_reset/smoke_runs/smoke_20260506/
  summary.json
  cases.jsonl
  report.md
```

重要边界：

```text
这不是 FinanceBench 全量 retrieval-only 重跑。
这不是 generated-answer 可靠性测评。
它不调用 Postgres、Qdrant、OpenAI 或本地 CrossEncoder。
它只用于验证 reset 后的消融维度、trace 形状和失败案例是否可解释。
```

历史 V1 retrieval-only 结论仍沿用 milestone 中记录的主实验：

| 模式 | doc@10 | page@10 | MRR doc | MRR page | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| dense_only | 0.467 | 0.127 | 0.233 | 0.081 | 27 | 88 |
| bm25_only | 0.787 | 0.207 | 0.448 | 0.113 | 4 | 7 |
| hybrid_rrf | 0.727 | 0.213 | 0.398 | 0.112 | 45 | 59 |
| hybrid_rrf_reranker | 0.813 | 0.267 | 0.520 | 0.146 | 747 | 1261 |

这组历史数字给 reset 的主判断是：

```text
BM25 是强 baseline。
RRF-only 不自动优于 BM25-only。
Hybrid + reranker 是当前 retrieval-only 最强路径，但 page@10 仍偏弱。
retrieval-only 不能证明最终答案可靠。
```

## 1. 本轮 Smoke 设计

synthetic corpus 覆盖 5 个小案例：

| case | 目标 |
|---|---|
| smoke_3m_capex_2018 | capex 问法和年报行文不完全一致 |
| smoke_3m_capex_compare | 2018/2017 多年份比较 |
| smoke_apple_net_sales_2019 | net sales / revenue alias |
| smoke_msft_capex_2020 | capex 对应 additions to property and equipment |
| smoke_ibm_dividends_2021 | IBM alias / full company name 风险 |

核心指标只看 doc/page/parent rank，不测答案生成：

```text
doc_hit@1/@3
page_hit@1/@3
parent_hit@1/@3
MRR_doc / MRR_page
failure_reasons
```

## 2. Query Rewrite 消融

| 变体 | page@1 | page@3 | answer_terms@3 | MRR page | 观察 |
|---|---:|---:|---:|---:|---|
| unit.text | 0.800 | 1.000 | 1.000 | 0.900 | 一个 3M capex case 的正确页被排到第 2 |
| +should_terms | 1.000 | 1.000 | 1.000 | 1.000 | 局部词法提示能救回年报真实措辞 |
| +ontology aliases | 1.000 | 1.000 | 1.000 | 1.000 | 对 capex / net sales / dividends 这类同义指标更稳 |

发现：

```text
unit.text 是必要的 dense 安全锚点，但不够。
FinanceBench 的财务科目常以年报原文出现，不一定使用用户问题中的词。
should_terms 能把 planner 的局部意图交给 sparse/alias lane；本轮 smoke 已保持 dense 只看 unit.text。
ontology aliases 能把 capex -> purchases/additions 这类映射显式化。
```

失败案例：

```text
smoke_3m_capex_2018
用户问 capital expenditure，正确页写的是 purchases of property, plant and equipment。
只用 unit.text 时，含有 capital expenditure 字样但不给金额的 MD&A 页更容易靠前。
```

取舍：

```text
+should_terms / +ontology aliases 会增加 lexical recall，也会增加噪声入口。
因此 aliases 不应直接变成全局 hard filter，而应作为 lane query / boost / reranker context。
```

## 3. Filter Strategy 消融

| 变体 | page@1 | page@3 | answer_terms@3 | MRR page | 失败 |
|---|---:|---:|---:|---:|---|
| no hard filter | 1.000 | 1.000 | 1.000 | 1.000 | 无 |
| metadata_filter only | 1.000 | 1.000 | 1.000 | 1.000 | 无 |
| must_have hard filter | 0.800 | 0.800 | 0.800 | 0.800 | page_miss@3: 1, answer_terms_miss@3: 1 |
| must terms sparse boost | 1.000 | 1.000 | 1.000 | 1.000 | 无 |

发现：

```text
metadata_filter 是低风险过滤，但只应对真实存在的 Qdrant payload 字段做 hard filter。
`document_id`、`section_title`、`file_type`、`page_start/page_end` 这类字段适合进入 Qdrant must condition。
`company`、`metric`、table hints 这类 planner 语义字段如果 payload 中不存在，不应硬过滤；它们应保留在 trace、sparse boost 或 reranker context 中。
must_have_terms 做硬过滤风险更高，因为证据文本可能使用 alias 或正式公司全称。
must_terms_sparse_boost 更适合当前 V1：它表达偏好，但不直接杀掉候选。
```

失败案例：

```text
smoke_ibm_dividends_2021
query 使用 IBM，正确 evidence 使用 International Business Machines / cash dividends paid。
must_have hard filter 要求 IBM 字面出现在候选文本，导致正确页被过滤掉。
```

取舍：

```text
硬过滤适合 document_id、filing、明确 page range 这类可靠 metadata。
实体简称、年份、指标 alias 更适合 boost / reranker context。
如果必须 hard filter，应先做 alias normalization，并记录 dropped_expected 风险。
```

## 4. Fusion 消融

| 变体 | page@1 | page@3 | MRR page | 状态 |
|---|---:|---:|---:|---|
| dense-only | 1.000 | 1.000 | 1.000 | smoke baseline |
| sparse-only | 1.000 | 1.000 | 1.000 | smoke baseline |
| Python Weighted RRF | 1.000 | 1.000 | 1.000 | 当前可执行 |
| Qdrant RRF | - | - | - | planned_not_run |

这组 synthetic smoke 太小，dense/sparse/RRF 都能命中，所以不能用它证明 fusion 优劣。
真正的方向仍要看历史 FinanceBench retrieval-only 数字：

```text
dense_only 明显弱于 BM25-only。
hybrid_rrf 的 page@10 稍好于 BM25-only，但 doc@10 / MRR doc 不一定更好。
hybrid_rrf_reranker 才是历史主实验中的最强路径。
```

Python Weighted RRF 的当前价值：

```text
可在 provider 内合并 dense、bm25、metric_alias、section、table textual lanes。
保留 lane_contributions、lane_weight、unit_weight、fusion_score。
方便解释“为什么这个 parent block 被选入 evidence pack”。
```

Qdrant RRF 的计划价值：

```text
如果 Qdrant hybrid query / server-side RRF 在本地版本可用，可以减少 Python 多路请求和合并成本。
但必须确认返回结果能保留 lane provenance，否则会损失 V1 reset 最需要的可解释 trace。
```

风险：

```text
不要把 raw dense score、BM25 score、table textual score 直接相加。
这些分数尺度不同，容易让某一路支配排序。
RRF 解决的是 rank normalization，不解决“候选是否回答问题”；后者仍需要 reranker / verifier。
```

## 5. Candidate Shape 消融

| 变体 | page@1 | page@3 | MRR page | 失败 |
|---|---:|---:|---:|---|
| child chunk | 1.000 | 1.000 | 1.000 | 无 |
| parent block | 1.000 | 1.000 | 1.000 | 无 |
| page neighborhood | 1.000 | 1.000 | 1.000 | 无 |
| token budget 18 | 0.600 | 0.600 | 0.600 | token_budget drop: 2 |

发现：

```text
child chunk 适合检索排序，但经常缺单位、年份列、表头或相邻行。
parent block 是 V1 当前更合理的 evidence 形态：检索小，展示大。
page neighborhood 能救跨页/表头分离问题，但会快速吃掉 context budget。
token budget 是真实风险：正确 parent 被找到了，也可能在 packing 阶段被挤掉。
```

失败案例：

```text
shape_token_budget_18
3M capex 单年和 3M capex 比较两个 case 中，正确 evidence 被 token budget drop。
这类失败不属于 retriever miss，而是 evidence packing / budget allocation 问题。
```

取舍：

```text
候选 shape 不应只按 retrieval rank 决定，还要看 answerability。
表格类问题更需要 parent/page context。
短事实问题可以用较小 parent block。
多年份比较需要保留同一 row 的多个年份，不要只截取命中的年份 token。
```

## 6. Reranker Input 消融

| 变体 | page@1 | page@3 | MRR page | 观察 |
|---|---:|---:|---:|---|
| original query + candidate | 0.800 | 1.000 | 0.900 | 一个 capex case 正确页只排第 2 |
| current unit + candidate | 1.000 | 1.000 | 1.000 | 当前 smoke 最稳之一 |
| local terms + candidate | 1.000 | 1.000 | 1.000 | 当前 smoke 最稳之一 |
| full plan summary + candidate | 0.800 | 1.000 | 0.900 | summary 不一定比局部 unit 更好 |
| full plan/all units + candidate | 1.000 | 1.000 | 1.000 | smoke 命中，但有噪声扩散风险 |

发现：

```text
original query 太短，可能无法告诉 reranker 年报里的真实指标措辞。
current unit 比 original query 更接近检索任务。
local terms 能把 must/should/alias 显式给 reranker，是当前推荐默认。
full plan summary 对单一候选不总是更好，可能把局部排序问题变成全局摘要匹配问题。
full plan/all units 信息最多，但最容易把其他 unit 的词带进当前候选判断。
```

失败/风险案例：

```text
smoke_3m_capex_2018
original query 和 full plan summary 都让正确页排第 2。
原因是 “capital expenditure” 字面页和 “purchases of property...” 真实答案页竞争。
local terms 把 annual report wording 带入后，正确页回到第 1。
```

取舍：

```text
reranker 输入越长，召回语义越丰富，但也越容易引入 cross-unit 噪声和延迟。
V1 的合理默认是 original query + current unit + local must/should terms。
full plan/all units 更适合二次诊断或 multi-hop 汇总，不适合作为所有候选的默认输入。
```

## 7. 建议的下一轮真实测评

下一轮应从 synthetic smoke 进入真实 FinanceBench retrieval-only ablation，但仍不调用生成：

```text
1. 在真实 Qdrant/Postgres 索引上跑 query rewrite 三档。
2. 比较 metadata_filter、must_have hard filter、must_terms sparse boost。
3. 保留 dense-only / sparse-only / Python Weighted RRF baseline。
4. 如果本地 Qdrant 版本支持 server-side RRF，再加 Qdrant RRF；否则保持 planned。
5. 对 parent block / page neighborhood / token budget 输出 dropped_expected 桶。
6. reranker input 至少比较 original、current unit、local terms、full plan summary。
```

生成式可靠性测评要单独做，不要用 retrieval-only 结果替代：

```text
citation_doc/page_hit
answer_numeric_match
unsupported_answer_rate
false_insufficient_rate
cache warm/cold
Critic Lite pre/post status
```

## 8. 当前推荐配置

在没有全量真实 ablation 之前，V1 provider reset 的保守推荐是：

```text
query rewrite:
  unit.text + local should_terms + ontology aliases

filter:
  metadata_filter only for known Qdrant payload keys
  semantic metadata stays in trace / sparse boost / reranker context
  must_have_terms as sparse boost, not hard filter

fusion:
  Python provider-local Weighted RRF
  keep dense-only and sparse-only as regression baselines
  Qdrant RRF remains planned until provenance is verified

candidate shape:
  retrieve child chunk
  pack parent block
  allow page neighborhood only for table/header split cases
  trace token_budget drops

reranker input:
  original query + current unit + local must/should terms + candidate
  avoid full plan/all units as default
```

## 9. 未完成

```text
尚未跑真实 FinanceBench reset ablation。
尚未测 generated-answer reliability。
尚未验证 Qdrant server-side RRF 是否能保留 lane provenance。
尚未把 provider reset 指标接入长期 dashboard。
```
