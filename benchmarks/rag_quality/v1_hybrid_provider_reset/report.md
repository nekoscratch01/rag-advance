# V1 Hybrid Provider Reset 基准报告

更新时间：2026-05-06

## 0. 先把边界说清楚

这份报告不是“完整 FinanceBench 结论”的替代品。

它有两层数据：

```text
1. Provider reset synthetic regression
   位置：benchmarks/rag_quality/v1_hybrid_provider_reset/smoke_runs/smoke_20260506/
   数量：100 条 synthetic case
   目的：检查 query rewrite、filter、fusion、candidate shape、reranker input 的策略形状和失败案例

2. FinanceBench retrieval-only full run
   位置：benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506/
   数量：150 条 FinanceBench case，4 个 retrieval mode，共 600 个 mode-case
   目的：给 V1 retrieval 主路径提供真实 doc/page hit、MRR、MAP、latency 数据
```

重要限制：

```text
synthetic regression 不调用 Postgres / Qdrant / OpenAI / CrossEncoder。
FinanceBench retrieval-only full run 调用 Postgres + Qdrant + local reranker，但不调用生成模型。
两者都不能证明 generated-answer reliability。
逐 case 的 cases.jsonl 是本地调试产物，体积大，不作为当前提交归档文件。
```

---

## 1. 完整 FinanceBench Retrieval-only 数据

运行命令：

```bash
ATLAS_RETRIEVAL_MODE=hybrid \
ATLAS_BM25_ENABLED=true \
ATLAS_QDRANT_COLLECTION=atlas_financebench_v1 \
python -m atlas.benchmark.financebench_retrieval \
  --cases evals/financebench_cases.yaml \
  --modes dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker \
  --top-k 10 \
  --out benchmarks/rag_quality/financebench/retrieval_runs \
  --run-id full_v1_retrieval_20260506
```

归档产物：

```text
benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506/
  summary.json
  report.md
```

结果：

| Mode | n | doc@10 | page@10 | MRR doc | MRR page | MAP doc | MAP page | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_only | 150 | 0.467 | 0.127 | 0.233 | 0.081 | 0.208 | 0.079 | 33 | 67 |
| bm25_only | 150 | 0.787 | 0.207 | 0.448 | 0.113 | 0.404 | 0.112 | 6 | 10 |
| hybrid_rrf | 150 | 0.727 | 0.213 | 0.398 | 0.112 | 0.343 | 0.105 | 47 | 61 |
| hybrid_rrf_reranker | 150 | 0.813 | 0.267 | 0.520 | 0.146 | 0.460 | 0.139 | 502 | 690 |

失败桶：

| Bucket | Count |
|---|---:|
| dense_missed_bm25_hit | 20 |
| hybrid_found_reranker_lost | 3 |
| reranker_improved_page_rank | 24 |
| both_dense_and_hybrid_missed | 115 |
| errors | 0 |

结论：

```text
BM25-only 是非常强的 FinanceBench baseline。
RRF-only 不自动优于 BM25-only；它更像候选池扩展机制。
Hybrid RRF + reranker 是当前 retrieval-only 最强路径。
page@10 最高只有 0.267，说明 retrieval 仍是瓶颈，不能直接证明答案可靠。
MAP_doc / MAP_page 已补齐；真正的答案正确性仍要看 generated-answer eval。
```

---

## 2. Provider Reset Synthetic Regression

运行命令：

```bash
python -m atlas.benchmark.v1_hybrid_provider_reset \
  --out benchmarks/rag_quality/v1_hybrid_provider_reset/smoke_runs \
  --run-id smoke_20260506
```

归档产物：

```text
benchmarks/rag_quality/v1_hybrid_provider_reset/smoke_runs/smoke_20260506/
  summary.json
  report.md
```

本轮 synthetic regression 使用 100 条 query variants，由 5 个基础案例派生：

| base case | 目标 |
|---|---|
| smoke_3m_capex_2018 | capex 问法和年报行文不完全一致 |
| smoke_3m_capex_compare | 2018/2017 多年份比较 |
| smoke_apple_net_sales_2019 | net sales / revenue alias |
| smoke_msft_capex_2020 | capex 对应 additions to property and equipment |
| smoke_ibm_dividends_2021 | IBM alias / full company name 风险 |

注意：

```text
这 100 条不是 FinanceBench 真实 100 条。
它们是 synthetic regression，用于放大策略差异和失败桶，不用于宣称真实质量。
```

### 2.1 Query Rewrite

| 变体 | n | page@1 | page@3 | answer_terms@3 | MRR page | MAP page |
|---|---:|---:|---:|---:|---:|---:|
| unit.text | 100 | 0.850 | 1.000 | 1.000 | 0.925 | 0.925 |
| +should_terms | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| +ontology aliases | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

发现：

```text
unit.text 是 dense 的安全锚点，但 FinanceBench 指标常用年报原文而不是用户措辞。
should_terms 能把 planner 的局部词法意图交给 sparse lane。
ontology aliases 能把 capex -> purchases/additions 这类映射显式化。
```

### 2.2 Filter Strategy

| 变体 | n | page@1 | page@3 | answer_terms@3 | MRR page | MAP page | 失败 |
|---|---:|---:|---:|---:|---:|---:|---|
| no hard filter | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 无 |
| metadata_filter only | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 无 |
| must_have hard filter | 100 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | page_miss@3: 20, answer_terms_miss@3: 20 |
| must terms sparse boost | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 无 |

发现：

```text
metadata_filter 只适合真实存在的 Qdrant payload 字段，例如 document_id、section_title、file_type、page range。
company / metric / table hints 这类 planner 语义字段不应硬塞进 payload filter。
must_have_terms hard filter 容易误杀，例如 IBM vs International Business Machines。
must_terms_sparse_boost 更适合当前 V1：表达偏好，但不直接杀掉候选。
```

Sparse boost 实现细节：

```text
不改 BM25 / sparse index 底层公式。
在 sparse input 中把每个 must_have term 重复 3 次。
例：Apple 2023 revenue -> Apple 2023 revenue Apple Apple Apple 2023 2023 2023
只进入 sparse/textual lanes，不进入 dense_text。
```

### 2.3 Fusion

| 变体 | n | page@1 | page@3 | answer_terms@3 | MRR page | MAP page | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| dense-only | 100 | 0.820 | 1.000 | 1.000 | 0.910 | 0.910 | synthetic baseline |
| sparse-only | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | synthetic baseline |
| Python Weighted RRF | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 当前可执行 |
| Qdrant RRF | 0 | - | - | - | - | - | planned_not_run |

解释：

```text
synthetic corpus 太小，不能证明 fusion 真实优劣。
它只能说明 provider-local Weighted RRF 的 trace、lane contribution 和排序形状可复查。
真实优劣看 FinanceBench 150 条 full run：RRF-only 不自动赢 BM25-only，reranker 才带来主要增益。
```

### 2.4 Candidate Shape

| 变体 | n | page@1 | page@3 | answer_terms@3 | MRR page | MAP page | 失败 |
|---|---:|---:|---:|---:|---:|---:|---|
| child chunk | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 无 |
| parent block | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 无 |
| page neighborhood | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 无 |
| token budget 18 | 100 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | token_budget drop: 40 |

发现：

```text
child chunk 适合检索排序，但经常缺单位、年份列、表头或相邻行。
parent block 是 V1 当前更合理的 evidence 形态：检索小，展示大。
page neighborhood 能救跨页/表头分离问题，但会快速吃掉 context budget。
token budget 会把已经找对的 evidence 挤掉，所以必须进入 failure bucket。
```

### 2.5 Reranker Input

| 变体 | n | page@1 | page@3 | answer_terms@3 | MRR page | MAP page |
|---|---:|---:|---:|---:|---:|---:|
| original query + candidate | 100 | 0.850 | 1.000 | 1.000 | 0.925 | 0.925 |
| current unit + candidate | 100 | 0.850 | 1.000 | 1.000 | 0.925 | 0.925 |
| local terms + candidate | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full plan summary + candidate | 100 | 0.850 | 1.000 | 1.000 | 0.925 | 0.925 |
| full plan/all units + candidate | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

发现：

```text
original query 太短，可能无法告诉 reranker 年报里的真实指标措辞。
local terms 能把 must/should/alias 显式给 reranker，是当前推荐默认。
full plan/all units 信息最多，但容易把其他 unit 的词带进当前候选判断，不适合作为默认输入。
```

---

## 3. 当前推荐配置

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

## 4. 未完成

```text
尚未跑真实 FinanceBench provider-reset ablation。
尚未测 generated-answer reliability。
尚未验证 Qdrant server-side RRF 是否能保留 lane provenance。
尚未把 provider reset 指标接入长期 dashboard。
```
