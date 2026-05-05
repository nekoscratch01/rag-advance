# V0.0 API-first 实现框架（蓝图归档）

> 本文档是 Atlas V0.0 的实现协议。它把 V0 蓝图收窄成一个小而真实、可以本地跑通的 API-first RAG 内核。

---

## 1. 目标

V0.0 只验证一条最短的证据优先 RAG 闭环：

```text
PDF / Markdown / TXT 文档
  -> 本地 BGE embedding
  -> Postgres 保存元数据和 trace
  -> Qdrant 做向量检索
  -> FastAPI 查询接口
  -> gpt-5-nano 基于 evidence 生成答案
  -> 返回 citations 和 trace_id
```

V0.0 的目标不是做完整产品，而是证明：系统的每个回答都能回到被检索出来的文档证据。

---

## 2. 产品边界

### V0.0 要做

```text
FastAPI 服务
Postgres
Qdrant
PDF / Markdown / TXT 文档导入
规则式 chunking
本地 BGE small embedding
Dense vector retrieval
gpt-5-nano 答案生成
Chunk-level citations
Query trace 记录
Smoke eval cases
Docker Compose
```

### V0.0 不做

```text
前端
认证 / 多租户权限
Hybrid retrieval
Keyword search
Reranker
Critic Lite
Research jobs
GraphRAG
Kafka / Flink
Memory / Skills
MCP tool layer
```

这个边界很重要。V0.0 的任务是跑通内核，不是提前把 V1/V2/V3 的复杂度拉进来。

---

## 3. 默认模型选择

### LLM

```env
ATLAS_LLM_PROVIDER=openai
ATLAS_LLM_MODEL=gpt-5-nano
OPENAI_API_KEY=...
```

`OPENAI_API_KEY` 只能从本机环境变量读取。它不能写进源码、日志、README 示例、Docker image、eval report 或 trace。

使用 `gpt-5-nano` 意味着：系统会把用户 query 和被选中的 top-k evidence 文本发送给 OpenAI 做答案生成。系统不会发送 API key，不会发送整个仓库，也不会自动发送所有文档。

### Embedding

```env
ATLAS_EMBEDDING_PROVIDER=local
ATLAS_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
ATLAS_EMBEDDING_DIM=512
```

Embedding 在本地运行。后续可能切换 embedding 模型，所以 chunks 和 Qdrant collection 都必须记录 `embedding_model` 和 `embedding_dim`。

---

## 4. 建议文件结构

```text
rag-advance/
  pyproject.toml
  docker-compose.yml
  .env.example
  README.md

  samples/
    demo_knowledge.md
    atlas_notes.txt

  evals/
    smoke_cases.yaml

  src/
    atlas/
      main.py

      core/
        config.py
        errors.py
        ids.py
        logging.py

      api/
        routes/
          health.py
          documents.py
          query.py

      db/
        session.py
        models.py
        repositories.py

      vector/
        qdrant_client.py
        collections.py

      ingestion/
        loaders.py
        chunker.py
        service.py

      embeddings/
        base.py
        bge_local.py

      llm/
        base.py
        openai_client.py
        prompts.py

      retrieval/
        dense_retriever.py
        evidence.py

      query_runtime/
        service.py
        citation_builder.py
        trace_logger.py

      eval/
        runner.py
        metrics.py
```

---

## 5. API 设计

### 健康检查

```text
GET /v1/health
```

返回服务和依赖状态。

响应示例：

```json
{
  "status": "ok",
  "postgres": "ok",
  "qdrant": "ok"
}
```

### 文档导入

```text
POST /v1/documents/ingest
```

V0.0 接收本地 PDF / Markdown / TXT 文件路径。文件上传可以后续再做。

请求示例：

```json
{
  "paths": ["samples/demo_knowledge.md"],
  "source_uri": "local:samples/demo_knowledge.md",
  "metadata": {
    "source_type": "sample"
  }
}
```

响应示例：

```json
{
  "ingestion_run_id": "ing_...",
  "documents": [
    {
      "document_id": "doc_...",
      "title": "demo_knowledge.md",
      "status": "ingested",
      "chunk_count": 8
    }
  ]
}
```

### 普通查询

```text
POST /v1/query
```

请求示例：

```json
{
  "query": "Atlas V0 的目标是什么？",
  "top_k": 8,
  "filters": {},
  "options": {
    "return_trace": true
  }
}
```

响应示例：

```json
{
  "query_id": "q_...",
  "trace_id": "tr_...",
  "answer": "... [c1]",
  "confidence": "supported",
  "citations": [
    {
      "citation_id": "c1",
      "document_id": "doc_...",
      "chunk_id": "chk_...",
      "source_title": "demo_knowledge.md",
      "section_title": "V0 Goal",
      "supporting_text": "..."
    }
  ]
}
```

### 查询记录

```text
GET /v1/query/{query_id}
```

返回一次 query 的结果和 trace 摘要。

---

## 6. 数据库表

### documents

```text
document_id
title
source_uri
file_type
content_hash
language
metadata_json
created_at
```

### chunks

```text
chunk_id
document_id
chunk_index
text
text_hash
section_title
token_count
embedding_model
embedding_dim
metadata_json
created_at
```

### query_runs

```text
query_id
trace_id
user_query
normalized_query
answer
confidence
citations_json
model_name
prompt_version
latency_ms
created_at
error_message
```

### retrieval_events

```text
event_id
query_id
chunk_id
rank
retrieval_score
retriever_type
created_at
```

### generation_events

```text
event_id
query_id
model_name
prompt_version
input_tokens
output_tokens
latency_ms
status
error_message
created_at
```

---

## 7. Qdrant Collection

Collection 名称：

```text
atlas_chunks_bge_small_zh_v1_5
```

Point 设计：

```text
point_id = chunk_id
vector = local BGE embedding
payload = {
  document_id,
  chunk_id,
  title,
  source_uri,
  file_type,
  section_title,
  language,
  embedding_model
}
```

Postgres 是权威数据源。Qdrant 只是向量检索索引，不保存系统事实的唯一副本。

---

## 8. Ingestion 流程

```text
1. 接收本地 PDF / Markdown / TXT 路径。
2. 校验路径位于允许的知识库目录。
3. 校验文件类型。
4. PDF 按页读取文本，Markdown / TXT 直接读取文本。
5. 计算 content_hash。
6. 如果 content_hash 已存在，则跳过重复导入。
7. 提取 title 和基础 metadata。
8. 按页、heading、段落和长度切分 chunks。
8. 批量调用本地 BGE 生成 embedding。
9. 写入 document 和 chunks 到 Postgres。
10. 写入 chunk vectors 到 Qdrant。
11. 返回导入摘要。
```

Chunking 策略：

```text
目标长度：400-700 tokens
Overlap：50-100 tokens
优先保留 Markdown heading 和段落边界
每个 chunk 必须保留 document_id、chunk_index、section_title
```

---

## 9. Query Runtime 流程

```text
1. 创建 query_id 和 trace_id。
2. 标准化 query。
3. 本地生成 query embedding。
4. 查询 Qdrant top_k。
5. 从 Postgres 取回 chunk text 和 metadata。
6. 构造 evidence pack。
7. 调用 gpt-5-nano，只传 query 和 evidence。
8. 要求回答使用 [c1]、[c2] 等引用标记。
9. 构造 citation objects。
10. 写入 query_runs、retrieval_events、generation_events。
11. 返回 answer、confidence、citations、trace_id。
```

证据不足时的行为：

```text
如果检索不到 evidence，或者 evidence 明显不足，系统应该返回证据不足，而不是猜测答案。
```

---

## 10. Prompt 契约

答案生成 prompt 必须强制以下规则：

```text
只能使用提供的 evidence。
如果 evidence 不足，必须明确说明证据不足。
不要使用 evidence 外的信息。
事实性结论必须使用 [c1]、[c2] 这样的 citation marker。
不要引用无法支撑该结论的 evidence。
回答要简洁。
```

V0.0 只承诺 chunk-level citation，不承诺 claim-level citation。Claim-level citation 放到后续版本。

---

## 11. Smoke Eval

V0.0 的 eval 先保持很小：

```text
5-10 条 eval cases
检查 expected source 是否被召回
检查 citation 是否包含 expected source
记录 latency
包含至少一个 insufficient-evidence case
```

Case 示例：

```yaml
- id: smoke_001
  question: "Atlas V0 的目标是什么？"
  expected_sources:
    - "demo_knowledge.md"
  expected_keywords:
    - "证据"
    - "引用"
    - "trace"
```

V0.0 不需要完美的 faithfulness judge。它需要的是一个可重复运行的 smoke report，用来证明链路是可测量的。

---

## 12. 完成标准

V0.0 完成时必须满足：

```text
1. docker compose up 可以启动 Postgres 和 Qdrant。
2. FastAPI 服务可以本地启动。
3. /v1/health 能返回依赖状态。
4. /v1/documents/ingest 可以导入 PDF / Markdown / TXT 样本文档。
5. Chunks 写入 Postgres。
6. Vectors 写入 Qdrant。
7. /v1/query 返回 answer、citations、confidence、trace_id。
8. /v1/query/{query_id} 可以查看已保存的 query run。
9. 对证据不足的问题，系统会拒答或明确限制答案。
10. Smoke eval 能输出基础 report。
```

---

## 13. 实现顺序

```text
1. 项目 scaffold 和配置系统。
2. Docker Compose：Postgres + Qdrant。
3. SQLAlchemy models 和 database session。
4. Qdrant collection bootstrap。
5. 本地 BGE embedder。
6. PDF / Markdown / TXT loader 和 chunker。
7. Ingestion service 和 endpoint。
8. Dense retriever。
9. OpenAI gpt-5-nano client。
10. Query runtime 和 citation builder。
11. Trace persistence。
12. Smoke eval。
13. README 运行说明。
```

---

## 14. 设计风险

### PDF 只做最小可用解析

PDF 在 V0.0 中只做文本抽取和页码 metadata，不承诺表格结构、版式还原、页眉页脚清洗或 OCR。

### BGE Small 只是起步选择

`BAAI/bge-small-zh-v1.5` 适合中文优先的 V0.0。如果后续语料变成英文或多语言，需要重建索引并切换到更合适的 embedding 模型。

### gpt-5-nano 仍然会接收 evidence

本地 embedding 可以避免把全文发给云端 embedding 服务。但答案生成时，选中的 evidence 仍然会发送给 OpenAI。V0.0 只应使用允许发送到 OpenAI 的样本文档。

### Trace 不能泄露密钥

Trace 可以保存 model name、prompt version、latency、citations、retrieved chunk IDs，但不能保存 API key 或完整环境变量。

---

## 15. 核心原则

V0.0 的验收标准不是“答案看起来聪明”。

真正的测试是：

```text
我们能不能检查 answer、citation、retrieved evidence 和 trace，
然后准确理解系统为什么这样回答？
```
