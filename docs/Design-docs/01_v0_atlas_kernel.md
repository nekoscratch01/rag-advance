# V0：Atlas Kernel

> V0 是 Atlas 的最小可用版本。它不追求复杂 Agent，不追求 GraphRAG，不追求流式写入。它只做一件事：让系统能够基于一批文档，返回带引用、可追踪、可评估的答案。

---

## 1. V0 的目标

V0 的目标是建立一个可靠的 RAG 后端内核。这里的“可靠”不是说模型永远回答正确，而是说系统的每一次回答都有清楚的来源、清楚的执行记录、清楚的质量评估。如果答案错了，工程师能够知道错误发生在哪一步：是文档解析错了、chunk 切得不好、向量检索没找回来、引用构造错了，还是模型在生成阶段加入了证据外的信息。

V0 做完后，系统应该可以完成以下任务：用户导入一批 PDF、Markdown 或 TXT 文档；系统解析文档、切分 chunk、生成 embedding、写入向量库；用户提问后，系统检索最相关的 chunks，基于这些 chunks 生成答案，并返回 citations 和 trace_id。系统还应该能跑一组基础 eval cases，用来评估检索命中率、引用命中率、答案忠实度、延迟和成本。

V0 不应该做太多事情。它不做复杂 Planner，不做多 Agent，不做异步 research job，不做 Kafka/Flink，不做 Neo4j，不做 ClickHouse，不做完整权限系统，也不做前端。V0 的价值在于把最底层的证据链和问答链路做稳。只要这个内核足够扎实，后面的 V1/V2 才有意义。

---

## 2. V0 的系统结构

V0 的系统结构应该尽量简单，但不能玩具化。它至少包含 FastAPI、Postgres、Qdrant、Ingestion Pipeline、Query Runtime 和 Evaluation Harness。

```text
┌──────────────────────────────────────────────┐
│                 CLI / API Client              │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                 FastAPI Gateway               │
│                                              │
│  POST /v1/documents/ingest   导入文档          │
│  POST /v1/query              普通查询          │
│  GET  /v1/query/{id}         查看查询记录       │
│  POST /v1/eval/run           运行评估          │
│  GET  /v1/health             健康检查          │
└───────────────────────┬──────────────────────┘
                        │
        ┌───────────────┴────────────────┐
        │                                │
        ▼                                ▼
┌──────────────────────┐        ┌──────────────────────┐
│   Ingestion Pipeline  │        │     Query Runtime     │
│                      │        │                      │
│  解析文档             │        │  接收问题             │
│  切分 chunks          │        │  向量检索             │
│  生成 embedding       │        │  证据整理             │
│  写入 Postgres/Qdrant │        │  生成答案和引用        │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│       Postgres        │        │        Qdrant         │
│                      │        │                      │
│  documents            │        │  chunk vectors        │
│  chunks               │        │  metadata payload     │
│  query_runs           │        │  nearest search       │
│  retrieval_events     │        │                      │
│  eval_runs            │        │                      │
└──────────────────────┘        └──────────────────────┘
```

这个结构的好处是实现成本低、部署简单、扩展方向清楚。Postgres 是系统的权威元数据存储，Qdrant 是向量检索层。FastAPI 负责对外暴露 API，Ingestion Pipeline 负责把文档变成可检索的 chunks，Query Runtime 负责把用户问题变成带引用的回答。Evaluation Harness 负责衡量系统是否真的变好了。

---

## 3. FastAPI Gateway 设计

Gateway 是 V0 对外的统一入口。所有外部请求都应该经过 Gateway，包括文档导入、普通查询、查看查询记录、运行评估和健康检查。V0 的 Gateway 不需要完整的企业级认证授权系统，但它应该从第一天就有清晰的 API 结构、标准化错误响应、请求日志和 trace_id 注入。

Gateway 用 FastAPI 实现。FastAPI 的 async 特性适合处理大量 I/O 型请求，例如等待数据库、向量库和模型 API 返回。不过需要明确一点：Gateway 本身通常不是系统瓶颈，真正的瓶颈在下游的 embedding、LLM、Qdrant 查询和 Postgres 写入。因此 V0 不要把 Gateway 设计得过度复杂，重点应该放在清晰的请求契约和可靠的错误处理上。

V0 推荐暴露以下 endpoint：

```text
POST /v1/documents/ingest
  导入一批文档。body 包含 file_path、source_uri 或上传文件信息。

POST /v1/query
  发起普通查询。body 包含 query、top_k、filters、options。

GET /v1/query/{query_id}
  查看一次查询的结果和 trace 摘要。

POST /v1/eval/run
  运行一组 eval cases，返回 eval_run_id。

GET /v1/eval/{eval_run_id}
  查看评估结果。

GET /v1/health
  返回服务健康状态，包括 Postgres 和 Qdrant 是否可用。
```

请求处理流程应该保持固定。请求进入后，先经过 Request ID / Trace ID 注入，再做日志记录，然后做请求体校验，再进入业务 handler。handler 调用下游模块后，Gateway 负责把结果格式化成统一响应。任何错误都不应该直接把 stack trace 暴露给调用方，而应该返回标准错误结构。

标准错误响应可以设计为：

```json
{
  "error_code": "INVALID_REQUEST",
  "error_message": "Missing required field: query",
  "trace_id": "tr_abc123",
  "details": {}
}
```

错误码可以先分为几类：`INVALID_*` 表示请求不合法，`UPSTREAM_*` 表示下游组件错误，`INTERNAL_*` 表示系统内部错误，`EVAL_*` 表示评估相关错误。V0 不需要非常复杂的错误体系，但一定要从第一版开始标准化，否则后面 debug 会很痛苦。

Gateway 的容量规划可以保守一点。V0 如果部署为 1-2 个 FastAPI worker，每个 worker 2-4 vCPU、4-8GB 内存，处理 1-5 QPS 的查询完全足够。真正需要注意的是 query handler 不要在请求线程里做长时间阻塞。如果文档导入任务很大，V0 可以先同步执行，但应该在结构上为后续异步 ingestion worker 留接口。

---

## 4. Ingestion Pipeline 设计

Ingestion Pipeline 负责把原始文档转化为可检索的 chunks。它是 V0 中最基础也最容易被低估的模块。很多 RAG 系统回答质量差，并不是因为模型不够强，而是文档解析不干净、chunk 切得不合理、metadata 丢失、引用无法回溯。

V0 的 Ingestion Pipeline 应该支持三类文件：PDF、Markdown、TXT。HTML 可以作为可选项，但不要在第一版强行处理太多复杂格式。每个文档导入时，系统先计算 content_hash。如果 hash 已存在，可以跳过重复导入，或者记录为重复来源。这样可以避免同一份文档被多次写入，导致检索结果重复。

完整导入流程如下：

```text
接收文档
  ↓
计算 content_hash
  ↓
解析文本和基础 metadata
  ↓
按章节/段落/长度切分 chunks
  ↓
为每个 chunk 生成 embedding
  ↓
chunks 文本和 metadata 写入 Postgres
  ↓
chunk vectors 和 payload 写入 Qdrant
  ↓
记录 ingestion_run 状态
```

文档解析阶段要尽量保留结构信息。对于 PDF，至少应该保留页码；对于 Markdown，应该保留 heading；对于 TXT，也可以通过空行和段落识别构造 section。V0 不要求完美解析所有版式，但要求每个 chunk 能追溯回原始文档位置。否则后面 citation 就只能停留在“来自某文件”，无法做到“来自某页某段”。

Chunking 规则建议如下：

```text
1. 优先按标题和段落切分。
2. 如果段落过长，再按句子或 token 长度切分。
3. 如果段落过短，和相邻段落合并。
4. 每个 chunk 目标长度 400-700 tokens。
5. 相邻 chunk 保留 50-100 tokens overlap。
6. 每个 chunk 必须保留 document_id、page_range、section_title。
```

V0 不建议一开始使用过度复杂的语义 chunking。复杂 chunking 会引入更多不可控变量，不利于初期评估。先用稳定的规则式 chunking，配合 eval 找出失败案例，再逐步优化，会更工程化。

---

## 5. Embedding 模块设计

Embedding 模块负责把 chunk 文本转换成向量。V0 应该把 embedding 做成独立接口，而不是把某个模型硬编码在 ingestion 里。这样后面更换 OpenAI embedding、本地 embedding、bge-m3 或其他模型时，不需要重写整个 pipeline。

推荐抽象如下：

```text
Embedder
  embed_texts(texts: list[str]) -> list[vector]
```

这个接口看起来简单，但实际需要处理 batch、retry、timeout、模型名称记录和缓存。Embedding 服务最容易因为批量太大、网络超时、模型 API 限流而失败。V0 不需要实现特别复杂的弹性机制，但至少应该支持 batch embedding 和失败重试。

每个 embedding 写入时都要记录使用的 embedding_model 和 embedding_dim。原因是后面如果换模型，旧向量和新向量不能混在同一个 collection 里，否则相似度会失真。可以在 Postgres 的 chunks 表里记录 embedding_model，也可以在 Qdrant collection 名里体现模型版本。

V0 可以先不做完整 embedding cache，但建议保留 text_hash。后续如果同一个 chunk 被重复处理，可以通过 text_hash 复用已有 embedding。对于大量文档导入场景，embedding cache 能明显降低成本和时间。

---

## 6. Postgres 设计

Postgres 是 V0 的权威数据存储。它不负责向量相似度检索，但负责保存文档、chunk 文本、查询记录、检索事件和评估结果。可以把 Postgres 理解成 Atlas 的“记账系统”：系统做过什么、用过什么证据、生成过什么答案，都应该能在 Postgres 里查到。

V0 推荐的核心表包括：

```text
documents
chunks
ingestion_runs
query_runs
retrieval_events
generation_events
eval_cases
eval_runs
eval_results
```

`documents` 表保存文档级 metadata。`chunks` 表保存 chunk 文本和来源信息。`query_runs` 记录每次查询。`retrieval_events` 记录每次检索召回了哪些 chunk、分数是多少、排名是多少。`generation_events` 记录模型调用、token 使用、延迟和输出。`eval_cases` 和 `eval_results` 用于评估。

V0 不建议把所有大文本和所有 trace 都无限期存在 Postgres 中而不做管理。虽然 V0 数据量不大，但数据模型应该为后续扩展留空间。比如 query_runs 可以保留完整答案，retrieval_events 保留 chunk_id 和分数，而非常长的 prompt 可以后续写到对象存储或专门的 trace 系统。

Postgres 的关键不是复杂，而是清楚。每张表都应该回答一个问题：这条数据是系统事实的一部分，还是一次执行过程的一部分？Document 和 Chunk 是系统事实；QueryRun、RetrievalEvent、GenerationEvent 是执行过程。把这两类数据分清楚，后面 debug 和 eval 会容易很多。

---

## 7. Qdrant 设计

Qdrant 是 V0 的向量检索层。它负责存储 chunk 向量，并根据用户 query 的 embedding 返回最相似的 chunks。V0 只需要一个核心 collection，例如 `atlas_chunks`。

Collection 的 point 可以这样设计：

```text
point_id = chunk_id
vector = chunk embedding
payload = {
  document_id,
  chunk_id,
  title,
  source_uri,
  file_type,
  page_start,
  page_end,
  section_title,
  language,
  created_at
}
```

Qdrant payload 不建议承载所有权威信息。它应该包含检索时需要过滤和展示的必要字段，但 chunk 的完整文本最好仍然保存在 Postgres 中。这样做的原因是：Qdrant 是检索系统，不应该成为所有文档内容的唯一来源；如果未来重建 collection、换向量库或做蓝绿迁移，Postgres 仍然保留权威文本。

V0 的检索模式很简单：

```text
query text
  ↓
query embedding
  ↓
Qdrant search top_k
  ↓
根据 chunk_id 从 Postgres 取 chunk text
  ↓
构造 Evidence
```

V0 可以先支持基础 metadata filter，例如 file_type、document_id、date_range、language。即使 filter 早期用得不多，也应该在接口上保留。V1 的 hybrid retrieval 和 source-specific query 会大量依赖 filter。

容量上，V0 如果目标是 10万 - 100万 chunks，单节点 Qdrant 就可以满足。为了工程上接近生产，可以用 Docker Compose 起一个 Qdrant 服务，后续再迁移到 cluster 或 managed Qdrant。

---

## 8. Query Runtime 设计

Query Runtime 是 V0 的核心读路径。它负责接收用户问题，检索证据，生成答案，输出引用和 trace。V0 的 Query Runtime 不需要 Planner，也不需要复杂 Agent。它应该是一条稳定、可复现、可评估的流水线。

流程如下：

```text
用户问题
  ↓
标准化 query
  ↓
生成 query embedding
  ↓
Qdrant 向量检索
  ↓
从 Postgres 取回 chunk 文本
  ↓
构造 Evidence Pack
  ↓
LLM 基于 Evidence 生成答案
  ↓
Citation Builder 生成引用
  ↓
Trace Logger 记录执行过程
  ↓
返回结果
```

V0 的 Evidence Pack 不需要复杂压缩。可以选择 top_k=8 或 top_k=12 的 chunks，把它们按相似度排序后交给模型。需要注意的是，Prompt 要明确要求模型只基于给定证据回答。如果证据不足，必须输出 insufficient。这个规则比“回答得丰富”更重要。

V0 的回答应该返回结构化 JSON，而不是只返回字符串：

```json
{
  "query_id": "q_001",
  "answer": "...",
  "confidence": "supported",
  "citations": [
    {
      "citation_id": "c1",
      "document_id": "doc_001",
      "chunk_id": "chunk_123",
      "source_title": "report.pdf",
      "page_start": 12,
      "page_end": 13,
      "section_title": "Risk Factors"
    }
  ],
  "trace_id": "tr_001"
}
```

这个结构可以让后续所有能力自然接上。前端可以显示 citations，eval 可以检查 citation 是否命中，trace 系统可以用 trace_id 找到完整执行过程。

---

## 9. Citation Builder 设计

Citation Builder 是 V0 必须认真做的模块。很多 RAG demo 的引用只是把检索到的文档列在答案后面，这不是真正的 citation。真正有用的 citation 应该能告诉用户：答案中的结论来自哪份文档、哪一页、哪个 chunk，最好还能显示一段 supporting text。

V0 的 Citation Builder 可以先做 chunk-level citation。它不要求每句话都有独立引用，但最终答案至少要列出支撑答案的 evidence。每条 citation 至少包含：

```text
citation_id
document_id
chunk_id
source_title
page_start
page_end
section_title
supporting_text
retrieval_score
```

其中 supporting_text 可以直接使用 chunk text 的前几百字，或者从 chunk 中截取与 query 更相关的句子。V0 不需要复杂的 claim-to-citation alignment，但建议在 prompt 中要求模型在回答中使用 `[c1]`、`[c2]` 这样的引用标记。这样后续 V1/V2 可以逐步升级成 claim-level citation。

Citation Builder 还需要处理一个现实问题：检索到的 chunk 可能很相关，但并不直接支持最终答案中的某个具体 claim。因此 V0 的答案生成 prompt 应该让模型引用 evidence id，而不是让后处理盲目把 top chunks 都挂上去。引用应该是生成过程的一部分，而不只是展示层附加物。

---

## 10. Trace Logger 设计

Trace Logger 记录系统执行过程。V0 不需要上 Langfuse 或 OpenTelemetry 也能做 trace，但必须有最小 trace 数据。没有 trace 的系统，后面无法 debug，也无法说服别人这是一个工程项目而不是 demo。

每次 query 至少记录：

```text
query_id
trace_id
user_query
normalized_query
retrieved_chunk_ids
retrieval_scores
selected_evidence_ids
prompt_version
model_name
input_tokens
output_tokens
latency_ms
answer
confidence
citations
error_message
```

Trace 的价值在于定位失败。例如 eval 发现某个问题回答错了，工程师应该可以打开 query_run，看到 top_k 里有没有正确 chunk。如果正确 chunk 没被召回，问题在 retriever 或 chunking；如果召回了但没进入 answer，问题在 evidence selection；如果进入了但答案仍然错，问题在 prompt 或模型生成。

V0 的 Trace Logger 可以直接写 Postgres。后续 V1/V2 再接 Langfuse 或 OpenTelemetry。不要为了追求完整 observability 一开始就接太多系统，但也不要没有 trace。

---

## 11. Evaluation Harness 设计

Evaluation Harness 是 V0 的关键模块。它不是额外功能，而是系统能持续迭代的基础。V0 至少要能跑一组固定问题，并输出检索和答案的基本指标。

Eval case 可以这样定义：

```yaml
- id: q001
  question: "What does the document say about supply chain risk?"
  expected_sources:
    - "annual_report_2024.pdf"
  expected_keywords:
    - "supply chain"
    - "supplier concentration"
  tags:
    - retrieval
    - citation
```

V0 的 eval 指标不需要特别复杂，但要覆盖四件事：

```text
Retrieval hit@k：正确文档或 chunk 是否出现在 top-k 结果里。
Citation hit：最终引用是否包含 expected source。
Faithfulness：答案是否被 evidence 支持。
Latency / cost：平均延迟和成本是多少。
```

Faithfulness 可以先用 LLM-as-judge，也可以先用简单规则辅助判断。关键不是一开始做到完美评估，而是从 V0 开始建立“每次改动都能比较”的习惯。比如 chunk size 从 500 tokens 改成 800 tokens 后，retrieval hit@5 是上升还是下降？reranker 后面加入后，citation correctness 有没有提升？这些都需要 eval harness 来回答。

V0 eval report 可以输出：

```text
Eval Run: 2026-xx-xx
Total cases: 50

Retrieval hit@5: 0.72
Citation hit: 0.68
Faithfulness: 0.80
Average latency: 3.9s
Average cost: $0.004/query

Top failures:
1. q012 - expected source not retrieved
2. q019 - answer used retrieved evidence but citation was missing
3. q031 - evidence was insufficient but model still answered
```

这个 report 是后续 V1/V2 最重要的比较基线。

---

## 12. V0 数据流示例

下面用一个简单问题说明 V0 的完整数据流。

用户问：

```text
“What are the main risks mentioned in the uploaded annual report?”
```

系统执行：

```text
1. Gateway 接收请求，生成 query_id 和 trace_id。
2. Query Runtime 标准化 query。
3. Embedder 生成 query embedding。
4. Qdrant 返回 top 12 chunks。
5. Postgres 根据 chunk_id 取回 chunk 文本和 metadata。
6. Evidence Builder 构造 evidence pack。
7. LLM 基于 evidence pack 生成答案。
8. Citation Builder 生成引用列表。
9. Trace Logger 写入 query_runs、retrieval_events、generation_events。
10. Gateway 返回 answer、confidence、citations、trace_id。
```

如果 evidence 中没有足够信息，系统应该返回：

```text
confidence = insufficient
answer = 当前导入的文档中没有足够证据回答这个问题。
```

这比模型编造一个看起来合理的风险列表更重要。

---

## 13. V0 容量规划

V0 的容量目标应该服务于“验证内核”，而不是服务于“生产规模”。合理目标如下：

| 指标 | V0 目标 |
|---|---:|
| 文档数 | 1万 - 10万 |
| Chunk 数 | 10万 - 100万 |
| 普通查询 QPS | 1 - 5 |
| 并发查询 | 5 - 20 |
| 查询延迟 | p95 3 - 8 秒 |
| Eval cases | 50 - 100 |
| Research job | 不支持 |
| GraphRAG | 不支持 |
| Streaming ingestion | 不支持 |

如果使用本地 Docker Compose，V0 可以跑在一台开发机或一台小型云服务器上。Postgres、Qdrant 和 FastAPI 都可以单实例。V0 不追求高可用，但要保证可复现、可评估、可部署。

---

## 14. V0 失败模式与降级

V0 虽然简单，但仍然需要定义失败模式。否则系统一出错就会表现得不可预测。

如果文档解析失败，Ingestion Pipeline 应该把该文档标记为 failed，并记录错误原因，而不是让整个批次失败。对于批量导入，单个文档失败不应该影响其他文档。

如果 embedding 调用失败，可以做有限重试。重试仍失败时，把 chunk 标记为 embedding_failed，后续可以重新处理。不要把没有 embedding 的 chunk 写入 Qdrant，否则检索会出现不完整状态。

如果 Qdrant 不可用，query 应该返回 `UPSTREAM_VECTOR_STORE_UNAVAILABLE`，而不是让 LLM 没有证据地回答。V0 的原则是没有证据就不答。

如果 LLM 生成失败，系统应该返回标准错误，并把失败写入 generation_events。后续可以根据 trace_id 复查失败原因。

如果 eval 运行失败，应该保留已经完成的 case 结果，并标记 eval_run 为 partial_failed。这样不会因为一个 case 出错导致整次评估不可用。

---

## 15. V0 完成标准

V0 不是“能问答”就完成。它应该达到以下标准：

```text
1. 支持 PDF / Markdown / TXT 导入。
2. 每个文档会被切成带 metadata 的 chunks。
3. Chunks 文本写入 Postgres，vectors 写入 Qdrant。
4. 用户可以发起普通 query。
5. 系统返回 answer、confidence、citations、trace_id。
6. 每次 query 都记录 retrieval_events 和 generation_events。
7. 至少有 50 条 eval cases。
8. 可以输出 eval report。
9. Docker Compose 可以一键启动。
10. README 能清楚说明架构和数据流。
```

V0 的最终效果应该是：别人不用看代码，只看 API 响应、trace 和 eval report，就能理解这个系统是如何基于证据回答问题的。

---

## 16. V0 不做什么

最后明确 V0 的非目标，避免范围失控。

V0 不做 GraphRAG。关系型问题后面 V3 再做。

V0 不做 deep research。复杂异步研究任务后面 V2 再做。

V0 不做 Kafka/Flink。持续写入路径后面 V5 再做。

V0 不做 ClickHouse。结构化数值查询后面 V4 再做。

V0 不做 Memory 和 Skills。长期记忆和技能系统后面 V6 再做。

V0 不做复杂多租户和权限系统。可以保留 user_id 字段，但不要让权限系统拖慢内核建设。

V0 的核心只有一句话：

```text
让系统能够可靠地基于文档证据回答问题，并且每个回答都可追踪、可引用、可评估。
```
