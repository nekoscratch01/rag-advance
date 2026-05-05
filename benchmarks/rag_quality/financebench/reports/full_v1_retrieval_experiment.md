# FinanceBench V1 Retrieval Benchmark Report

## 摘要

本次实验的目标不是评估 Atlas V1 最终回答是否“看起来正确”，而是先回答一个更基础的问题：**新的检索架构是否真的比 V0 的 Dense-only 更可靠？**

因此，本实验刻意绕开 LLM 生成阶段，只评估 retrieval。这样可以把问题收窄到一个可复现、可解释的范围内：

```text
同一 FinanceBench corpus
同一 parent-child chunking
同一 Qdrant collection
同一 gold doc/page
比较 dense、BM25、RRF、RRF+reranker 的召回质量
```

最终结果显示：V1 主路径 `Hybrid RRF + Reranker` 在本轮 retrieval-only 实验中是整体最强方案。它相对 Dense-only 明显提升：

```text
doc@10:  0.467 -> 0.813
page@10: 0.127 -> 0.267
MRR_doc: 0.233 -> 0.520
MAP_doc: 0.208 -> 0.460
```

但实验也揭示了一个重要事实：**BM25-only 本身就是非常强的 baseline，而 RRF-only 并不自动优于 BM25-only。** 换句话说，V1 真正成立的原因不是“把 dense 和 BM25 加起来就更好”，而是：

```text
BM25 提供强 lexical recall
Dense 提供候选多样性
RRF 扩展候选池
Reranker 重新判断相关性
Parent evidence 负责把 child hit 还原成可读证据
```

这个实验也说明，后续优化重点不应该急着上 LLM，而应该优先改进 parser、chunking、英文 embedding baseline 和 reranker candidate depth。

## 1. 实验背景

Atlas V0 的检索方式是 Dense-only：用户问题经过 BGE-small embedding 后，在 Qdrant 中做向量检索，再把 top-k chunks 交给生成模型。

这个设计在普通文档 RAG 上是一个合理起点，但 FinanceBench 的问题类型更苛刻。它经常要求定位：

```text
特定公司
特定年份
特定 filing
特定财务科目
表格里的某一行或某一页
```

例如：

```text
What is the FY2018 capital expenditure amount for 3M?
Which debt securities are registered to trade under 3M's name as of Q2 2023?
What was the key agenda of AMCOR's 8-K filing dated July 1st 2022?
```

这些问题不只是语义相似度问题。公司名、年份、filing 类型、财务指标名称这些硬关键词非常关键。因此，V1 引入了 BM25 sparse retrieval、RRF fusion 和本地 reranker。

本实验要验证的是：这条路线是否真的提升了检索质量。

## 2. 实验设置

### 2.1 数据集

本次使用 FinanceBench open-source subset：

```text
150 QA cases
84 PDF manifest records
12013 parsed pages
18999 child chunks
0 prepare failures
```

生成的 corpus artifacts：

```text
corpus/financebench/manifest.jsonl
corpus/financebench/parsed/pages.jsonl
corpus/financebench/parsed/parent_blocks.jsonl
corpus/financebench/parsed/child_chunks.jsonl
evals/financebench_cases.yaml
```

### 2.2 Parent-Child RAG 结构

V1 不再用“相邻 chunk 猜合并”的方式构造 evidence，而是显式采用 parent-child schema：

```text
ParentBlock = page-level readable block
ChildChunk  = retrieval chunk
```

检索发生在 child chunk 上，最终展示和评估时再回到 parent page：

```text
child hit
  -> parent_id
  -> parent block
  -> evidence c1 / c2 / c3
```

这对于 FinanceBench 很重要，因为答案经常来自表格页。检索小 chunk 有利于召回，生成和引用时则需要完整 page 级上下文。

### 2.3 Page 对齐

准备 corpus 时发现，FinanceBench 的 raw page number 不能直接当作 PDF parser 的 page number 使用。第一条 3M case 就体现了这个问题：

```text
evidence_page_num_raw: 59
evidence_page_num_normalized: 60
page_num_normalized: 60
```

对应 parent block：

```text
parent_id: fbpar_faf10c53fc6f1a8a_p0060
page_start: 60
page_end: 60
```

因此，本实验统一用 normalized page 做 page hit 评估，否则会产生系统性误判。

### 2.4 检索模式

主实验比较四种模式：

```text
dense_only
bm25_only
hybrid_rrf
hybrid_rrf_reranker
```

主要配置：

```text
embedding_model: BAAI/bge-small-zh-v1.5
bm25_model: Qdrant/bm25
bm25_language: english
bm25_k: 1.2
bm25_b: 0.75
bm25_avg_len: 256.0
dense_top_k: 50
bm25_top_k: 50
rrf_k: 60
rrf_top_k: 40
reranker_model: cross-encoder/ms-marco-MiniLM-L6-v2
reranker_top_k: 30
reranker_output_k: 10
```

### 2.5 评估指标

本实验只看 retrieval：

```text
doc_hit@1/@3/@5/@10
page_hit@1/@3/@5/@10
MRR_doc
MRR_page
MAP_doc
MAP_page
latency_p50/p95
failure buckets
```

本实验不评估：

```text
answer_numeric_match
citation_hit
unsupported_answer_rate
```

这些属于生成侧指标，需要后续调用 `/v1/query` 和 LLM。

## 3. 实验执行过程

### 3.1 先补 retrieval-only benchmark runner

原有 benchmark runner 是通过 HTTP 调 `/v1/query`，它会进入 AnswerGenerator，因此会依赖 OpenAI API。为了只测召回，我新增了：

```text
src/atlas/benchmark/financebench_retrieval.py
```

它直接构造本地 retriever，并在 Python 内部跑：

```text
EvalCase
  -> Retriever.retrieve()
  -> Evidence[]
  -> compare with gold doc/page
  -> write summary/cases/report
```

这一步的意义是把 retrieval benchmark 从 generation benchmark 中解耦出来。

### 3.2 先用 1-document corpus 做 smoke test

全量跑之前，先用 3M 单文档 corpus 检查基础链路。第一次导入失败，因为当前 Python 环境没有 `fastembed`：

```text
ModuleNotFoundError: No module named 'fastembed'
```

BM25 sparse retrieval 依赖 `fastembed` 的 `Qdrant/bm25`，因此安装依赖后重跑：

```bash
python -m pip install 'fastembed>=0.8.0'
```

随后 1-document import 成功：

```text
Imported FinanceBench: 1 documents, 160 parents, 280 children, 280 vectors.
Collection: atlas_financebench_import_smoke
```

在 smoke test 中，BM25 能返回带 `lexical_rank`、`lexical_score` 和 `parent_id` 的 candidates；Hybrid RRF 能把 child hits 解析回 parent evidence；MiniLM reranker 能加载并保留 dense/BM25/RRF 的 raw ranks。

这个阶段证明链路是通的。

### 3.3 1-case smoke 暴露 reranker 价值

用第一条 3M case 跑 retrieval-only smoke：

```text
What is the FY2018 capital expenditure amount for 3M?
```

结果显示，Dense、BM25 和 RRF 都能找到正确文档，但正确页没进入 top10；加入 reranker 后，正确页进入 top5/top10。

这只是单 case 现象，不能作为最终结论，但它提示：reranker 可能对 page-level ranking 有真实价值。

### 3.4 全量导入时遇到 PDF 文本问题

全量导入 `corpus/financebench` 时，第一次失败在 Postgres 写入阶段：

```text
psycopg.DataError: PostgreSQL text fields cannot contain NUL (0x00) bytes
```

问题来自部分 PDF 页面经 `pypdf` 提取后包含 `\x00`。Postgres `text` 字段不接受 NUL byte。

修复方式是在 frozen corpus importer 写入 DB 前做存储层清洗：

```text
_clean_text(value).replace("\x00", "")
```

这里没有修改 raw JSONL artifacts，因为 raw artifacts 应该保持可追溯；清洗只发生在导入数据库时。

### 3.5 全量导入成功

修复后重跑全量导入，最终成功：

```text
Imported FinanceBench: 84 documents, 12013 parents, 18999 children, 18999 vectors.
Collection: atlas_financebench_v1
```

Qdrant points 逐步增长：

```text
6272
11776
17280
18999
```

这个阶段主要耗时在本地 BGE embedding 和 BM25 sparse encoding。

## 4. 主实验结果

主实验 run：

```text
benchmarks/financebench/retrieval_runs/full_v1_retrieval
```

规模：

```text
150 cases
4 modes
600 records
duration_seconds: 141.098
errors: 0
```

结果：

| Mode | doc@1 | doc@3 | doc@5 | doc@10 | page@1 | page@3 | page@5 | page@10 | MRR doc | MRR page | MAP doc | MAP page | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_only | 0.133 | 0.293 | 0.353 | 0.467 | 0.060 | 0.093 | 0.107 | 0.127 | 0.233 | 0.081 | 0.208 | 0.079 | 27ms | 88ms |
| bm25_only | 0.320 | 0.507 | 0.620 | 0.787 | 0.073 | 0.140 | 0.153 | 0.207 | 0.448 | 0.113 | 0.404 | 0.112 | 4ms | 7ms |
| hybrid_rrf | 0.253 | 0.487 | 0.593 | 0.727 | 0.080 | 0.120 | 0.167 | 0.213 | 0.398 | 0.112 | 0.343 | 0.105 | 45ms | 59ms |
| hybrid_rrf_reranker | 0.373 | 0.633 | 0.740 | 0.813 | 0.100 | 0.173 | 0.213 | 0.267 | 0.520 | 0.146 | 0.460 | 0.139 | 747ms | 1261ms |

## 5. 结果分析

### 5.1 Dense-only 是弱 baseline

Dense-only 的 doc@10 只有 0.467，page@10 只有 0.127。对于 FinanceBench 这种财报问答，这说明单纯依赖当前 dense embedding 不够。

主要原因可能有三个：

```text
1. 当前 embedding 是 BAAI/bge-small-zh-v1.5，本身不是英文财报专用模型。
2. FinanceBench 问题高度依赖硬关键词，例如公司名、年份、filing、财务科目。
3. PDF 表格经 pypdf 拉平成文本后，语义结构被破坏，dense matching 容易受噪声影响。
```

### 5.2 BM25-only 比预期强

BM25-only 的表现非常突出：

```text
doc@10: 0.787
page@10: 0.207
MRR_doc: 0.448
MAP_doc: 0.404
p50: 4ms
```

这说明 FinanceBench 当前任务中 lexical signal 非常强。很多问题不是“语义像不像”，而是“有没有精确命中正确的财务术语和年份”。

典型样例：

```text
financebench_id_00941
Q: Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?
dense_only: doc rank 6, page miss
bm25_only: doc rank 2, page rank 2
```

```text
financebench_id_01319
Q: What is the quantity of restructuring costs directly outlined in AES Corporation's income statements for FY2022?
dense_only: doc rank 9, page miss
bm25_only: doc rank 2, page rank 2
```

这也是后续架构判断中最重要的发现之一：BM25 不是辅助模块，而是强基础召回器。

### 5.3 RRF-only 并没有自然超过 BM25-only

Hybrid RRF-only 相比 Dense-only 有明显提升，但 doc 级指标不如 BM25-only：

```text
bm25_only doc@10: 0.787
hybrid_rrf doc@10: 0.727
```

这说明当前 dense 分支质量偏弱时，RRF 不一定产生纯收益。它可能把 BM25 的强结果往后稀释。

这个发现修正了一个常见误解：

```text
Dense + BM25 + RRF 不等于自动最好。
```

在这个 corpus 和模型设置下，RRF-only 只能算中间候选融合层，不能作为最终排序层。

### 5.4 RRF + Reranker 才是当前最强路径

`hybrid_rrf_reranker` 在所有核心 retrieval 指标上最高：

```text
doc@10: 0.813
page@10: 0.267
MRR_doc: 0.520
MRR_page: 0.146
MAP_doc: 0.460
MAP_page: 0.139
```

这说明 V1 主路径的关键不是 RRF 本身，而是：

```text
RRF 负责扩大候选池
Reranker 负责重新排序
```

但是代价也很明显：

```text
p50: 747ms
p95: 1261ms
```

在 Mac 本地场景下，这个延迟是可以接受但不能忽视的。

### 5.5 Page-level recall 仍然偏弱

即便最强模式 `hybrid_rrf_reranker`，page@10 也只有 0.267。

这说明系统已经比较擅长找到正确文档，但还不够稳定地定位正确页。后续提升 citation 和 answer grounding，瓶颈大概率在：

```text
PDF table parsing
chunking strategy
page-level parent granularity
English embedding baseline
reranker candidate depth
```

## 6. Failure Buckets

主实验 failure buckets：

```text
dense_missed_bm25_hit: 20
hybrid_found_reranker_lost: 3
reranker_improved_page_rank: 24
both_dense_and_hybrid_missed: 115
errors: 0
```

### 6.1 dense_missed_bm25_hit

这个 bucket 表示 BM25 找到了 Dense 没找到的结果，说明 lexical retrieval 是必要组件。

样例：

```text
financebench_id_01935
Q: What was the key agenda of the AMCOR's 8k filing dated 1st July 2022?
dense_only: doc miss, page miss
bm25_only: doc rank 7, page rank 7
```

### 6.2 reranker_improved_page_rank

这个 bucket 表示 reranker 把正确页排得更靠前，或从 miss 变成 hit。

数量：

```text
24
```

样例：

```text
financebench_id_01148
Q: What industry does AMCOR primarily operate in?
hybrid_rrf: doc rank 9, page miss
hybrid_rrf_reranker: doc rank 2, page rank 6
```

这证明 reranker 确实提供了排序收益。

### 6.3 hybrid_found_reranker_lost

reranker 也会伤害个别 case：

```text
hybrid_found_reranker_lost: 3
```

样例：

```text
financebench_id_00822
Q: Were there any board member nominees who had substantially more votes against joining than the other nominees?
hybrid_rrf: doc rank 2, page rank 2
hybrid_rrf_reranker: doc miss, page miss
```

这个 bucket 很重要。它说明 reranker 虽然总体正收益，但仍需要持续监控，不能只看平均值。

## 7. 参数消融

### 7.1 RRF k

主实验用的是 `rrf_k=60`，但 sweep 发现它不是最优点。

| rrf_k | doc@10 | page@10 | MRR doc | MRR page | MAP doc | MAP page | p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.720 | 0.220 | 0.372 | 0.116 | 0.332 | 0.111 | 75ms |
| 30 | 0.727 | 0.220 | 0.400 | 0.114 | 0.344 | 0.107 | 86ms |
| 60 | 0.727 | 0.213 | 0.398 | 0.112 | 0.343 | 0.105 | 74ms |
| 100 | 0.727 | 0.213 | 0.397 | 0.112 | 0.342 | 0.105 | 62ms |

观察：

```text
rrf_k=30 更均衡。
rrf_k=10/30 的 page@10 高于 60/100。
```

因此，如果保留 RRF-only 路径，默认 `rrf_k` 更适合从 60 调到 30。

### 7.2 Reranker candidate depth

固定：

```text
rrf_k=30
rrf_top_k=40
reranker_output_k=10
```

比较：

```text
reranker_top_k = 10 / 30 / 40
```

结果：

| reranker_top_k | doc@10 | page@10 | MRR doc | MRR page | MAP doc | MAP page | p50 | p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.727 | 0.213 | 0.485 | 0.143 | 0.426 | 0.137 | 291ms | 556ms |
| 30 | 0.813 | 0.267 | 0.520 | 0.146 | 0.460 | 0.139 | 773ms | 895ms |
| 40 | 0.807 | 0.287 | 0.521 | 0.159 | 0.457 | 0.152 | 977ms | 1769ms |

结论：

```text
top_k=10 太小。
top_k=30 是 doc 指标和延迟的较好折中。
top_k=40 的 page 指标更好，但 p95 延迟明显变差。
```

如果 V1 优先 citation/page hit，`reranker_top_k=40` 值得考虑；如果优先 Mac 本地响应，`reranker_top_k=30` 更合理。

## 8. 实验后的架构判断

本实验修正了几个判断。

第一，BM25 必须保留。它不是一个补充模块，而是 FinanceBench 当前场景下的强基础召回器。

第二，RRF-only 不能作为最终排序方案。它相对 Dense-only 有提升，但不如 BM25-only 稳。

第三，V1 主路径成立的关键是 reranker。只有 `Hybrid RRF + Reranker` 在本轮实验中全面超过 Dense-only、BM25-only 和 RRF-only。

第四，page-level recall 仍然不足。即使最强模式 page@10 也只有 0.267，说明后续主要矛盾不在 LLM，而在 retrieval corpus 本身。

## 9. 推荐配置

当前最稳的 V1 retrieval 配置：

```text
mode: hybrid_rrf_reranker
dense_top_k: 50
bm25_top_k: 50
rrf_k: 30
rrf_top_k: 40
reranker_top_k: 30
reranker_output_k: 10
```

如果优先 page/citation hit：

```text
reranker_top_k: 40
```

如果需要低延迟 baseline：

```text
mode: bm25_only
```

## 10. 下一步实验

本轮还没有证明当前架构是全局最优。下一步应该继续做：

```text
1. 英文 embedding baseline
   bge-small-en / e5-small-v2 / jina-embeddings-v2-small-en

2. BM25 avg_len sweep
   当前 256 是默认值，不一定匹配 FinanceBench chunk 长度。

3. table-aware parser
   page-level recall 弱，很可能和 PDF table flattening 有关。

4. reranker_top_k 与 RRF k 联合 sweep
   当前只做了局部消融。

5. 生成式 benchmark
   在 retrieval 稳定后再测 answer_numeric_match、citation_hit、unsupported_answer_rate。
```

## 结论

这次 benchmark 证明了 V1 的主检索方向是有实验依据的：

```text
Dense-only 弱。
BM25 强。
RRF-only 不够稳。
Hybrid RRF + Reranker 最强。
```

但它也说明，V1 还没有到“检索质量已经解决”的阶段。真正影响下一阶段答案可靠性的，不是先换 LLM，而是继续提高 page-level evidence recall。
