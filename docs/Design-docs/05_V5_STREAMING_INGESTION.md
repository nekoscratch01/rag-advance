# V5 — Streaming Ingestion

> 核心目标：让 Atlas 从批量导入升级为持续写入、持续解析、持续索引的系统。
> 关键边界：V5 是 Ingestion Plane，不进入 Query Runtime 主链路；Kafka/Flink 更新索引，不参与回答逻辑。

---

## 1. V5 的定位

V0–V4 主要解决读路径：

```text
用户问问题 -> 找证据 -> 生成答案/报告
```

V5 解决写路径：

```text
新数据持续进入 -> 解析 -> 切块 -> 索引 -> 更新 provider -> 保证 freshness
```

V5 不改变 Evidence Kernel 的基本逻辑。

它负责让这些索引持续更新：

```text
Text index
BM25 / sparse index
Graph index
SQL/table store
Memory index
```

---

## 2. V5 总体架构

```text
                           ┌──────────────────────┐
                           │   External Sources    │
                           │ pdf/api/rss/db/files  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │      Collectors       │
                           │ pull/push/connectors  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Ingestion Gateway   │
                           │ validate/dedupe/hash  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Kafka Topics       │
                           │ raw/parsed/chunk/...  │
                           └──────────┬───────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌────────────────┐            ┌────────────────┐            ┌────────────────┐
│ Parser Worker  │            │ Chunk Worker   │            │ Table Worker   │
│ pdf/html/txt    │            │ contextualize  │            │ cells/facts    │
└───────┬────────┘            └───────┬────────┘            └───────┬────────┘
        │                             │                             │
        ▼                             ▼                             ▼
┌────────────────┐            ┌────────────────┐            ┌────────────────┐
│ Parsed Pages   │            │ Chunk Events   │            │ Table Events   │
└───────┬────────┘            └───────┬────────┘            └───────┬────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Indexing Workers   │
                           └──────────┬───────────┘
                                      │
      ┌───────────────────────────────┼───────────────────────────────┐
      │                               │                               │
      ▼                               ▼                               ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ Text Index   │              │ Graph Index  │              │ SQL Store    │
│ dense/BM25   │              │ entities/rel │              │ tables/facts │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       │                             │                             │
       └─────────────────────────────┼─────────────────────────────┘
                                     │
                                     ▼
                           ┌──────────────────────┐
                           │ Index Version Registry│
                           │ readiness/promotion  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Monitoring + DLQ      │
                           │ lag/errors/retries    │
                           └──────────────────────┘
```

---

## 3. V5 不进入 Query Path

错误理解：

```text
User Query -> Kafka -> Flink -> Answer
```

正确理解：

```text
Kafka/Flink 是写入和索引系统。
Query Runtime 读取已经准备好的 index。
```

原因：

```text
- Query Path 要低延迟。
- Streaming Path 要高吞吐和可重试。
- 两者的 SLA 不同。
- 两者的错误处理不同。
```

---

## 4. Ingestion Plane 与 Evidence Plane 的关系

```text
V5 writes indexes.
V1/V3/V4 read indexes.
```

具体：

```text
V1 TextHybridProvider:
  读取 dense index、BM25 index、chunk metadata。

V3 GraphProvider:
  读取 graph nodes、edges、community summaries。

V4 SQLProvider:
  读取 structured tables、financial_facts、cell provenance。
```

V5 的输出是：

```text
index_version
corpus_version
ingestion_status
freshness_status
```

---

## 5. Topic 设计

### 5.1 核心 topics

```text
atlas.raw_document
atlas.document_validated
atlas.document_parsed
atlas.page_extracted
atlas.chunk_created
atlas.chunk_contextualized
atlas.embedding_requested
atlas.embedding_indexed
atlas.bm25_indexed
atlas.table_extracted
atlas.financial_fact_created
atlas.graph_entity_extracted
atlas.graph_relationship_extracted
atlas.graph_indexed
atlas.index_ready
atlas.index_promoted
atlas.dlq
```

### 5.2 Topic 命名原则

```text
atlas.{domain}.{event}
```

或者简化：

```text
atlas.raw_document
atlas.parsed_page
atlas.chunk
atlas.embedding
atlas.graph
atlas.table
atlas.dlq
```

---

## 6. Event Schema

所有事件都必须可追踪。

```python
class IngestionEvent:
    event_id: str
    event_type: str
    source: str
    document_id: str | None
    corpus_version: str
    index_version: str | None
    payload: dict
    idempotency_key: str
    attempt: int
    created_at: datetime
    trace_id: str
```

### 6.1 Idempotency key

必须防重复处理。

```text
idempotency_key = hash(event_type + document_id + content_hash + stage_version)
```

---

## 7. Corpus Version 与 Index Version

V5 必须明确两个版本。

### 7.1 Corpus Version

表示原始资料集合版本。

```text
哪些 documents 属于当前 corpus？
它们的 content_hash 是什么？
```

### 7.2 Index Version

表示索引版本。

```text
dense index version
BM25 index version
graph index version
SQL table version
```

### 7.3 为什么要分开

同一份 corpus 可以有多个 index：

```text
chunk_size=800 的 index
chunk_size=1200 的 index
BGE-small 的 index
BGE-M3 的 index
BM25 avg_len 不同的 index
```

Eval 和 Cache 必须依赖 index_version。

---

## 8. Index Promotion

不要边写边直接污染线上索引。

推荐：

```text
build new index version
 -> validate index completeness
 -> run smoke eval
 -> mark index ready
 -> promote active index version
```

ASCII：

```text
new documents
   │
   ▼
build index_v42_shadow
   │
   ▼
validate + smoke eval
   │
   ├── fail -> DLQ / retry / keep old index
   │
   ▼
promote index_v42_active
```

---

## 9. DLQ

DLQ 是 V5 必须有的能力。

DLQ 处理：

```text
- parser failure
- invalid schema
- embedding failure
- table extraction failure
- graph extraction failure
- index write failure
- unknown exception
```

DLQ event 必须包含：

```text
original_event
error_type
error_message
stack_trace optional
attempt_count
stage
created_at
```

### 9.1 DLQ 策略

```text
retryable:
  network error
  rate limit
  temporary model/API failure

non_retryable:
  corrupted file
  unsupported file type
  invalid schema
  extraction impossible
```

---

## 10. Processing Guarantees

V5 不应该一开始承诺绝对 exactly-once 到所有外部系统。

更务实的目标：

```text
at-least-once processing
+ idempotent writes
+ deterministic IDs
+ versioned indexes
+ replay support
```

成熟后再用：

```text
Kafka transactions
Flink checkpointing
transactional sinks
```

---

## 11. Worker 设计

### 11.1 Parser Worker

```text
input: raw_document
output: document_parsed / page_extracted
```

职责：

```text
- PDF / HTML / TXT parsing
- page extraction
- document metadata
- parser version recording
```

### 11.2 Chunk Worker

```text
input: page_extracted
output: chunk_created
```

职责：

```text
- page-aware chunking
- parent-child chunks
- table row chunking
- deterministic chunk_id
```

### 11.3 Contextualizer Worker

```text
input: chunk_created
output: chunk_contextualized
```

职责：

```text
- add document/page/table context
- preserve source text
- generate index text
```

### 11.4 Embedding Worker

```text
input: chunk_contextualized
output: embedding_indexed
```

职责：

```text
- dense embedding
- sparse/BM25 vector generation if needed
- Qdrant upsert
```

### 11.5 Graph Worker

```text
input: chunk_contextualized
output: graph_entity_extracted / graph_relationship_extracted
```

职责：

```text
- entity extraction
- relationship extraction
- source anchors
- graph upsert
```

### 11.6 Table Worker

```text
input: document_parsed / page_extracted
output: table_extracted / financial_fact_created
```

职责：

```text
- table detection
- cell extraction
- numeric normalization
- financial_facts upsert
```

---

## 12. Backfill vs Streaming

V5 要同时支持：

```text
backfill:
  对历史 corpus 全量重建索引。

streaming:
  对新文档增量处理。
```

Backfill 特点：

```text
- 高吞吐
- 可并行
- 可重跑
- 关注 completeness
```

Streaming 特点：

```text
- 低延迟
- 关注 freshness
- 需要 DLQ 和 retry
```

---

## 13. Consistency Model

查询时系统要知道自己用的是哪个版本。

```text
QueryRun.index_version = active_index_version
```

如果文档刚进入系统但未索引完成：

```text
status = ingesting / indexed / active
```

可以支持：

```text
read_active_only:
  只查询 active index

read_latest_ready:
  查询 ready 但未 promoted 的 index

read_workspace_latest:
  对 workspace 内新上传文档临时查询
```

---

## 14. Freshness SLA

V5 要定义 freshness。

示例：

```text
small document:
  uploaded -> searchable within 2 minutes

large PDF:
  uploaded -> searchable within 10 minutes

graph index:
  uploaded -> graph searchable within 30 minutes

SQL facts:
  uploaded -> structured query ready within 10 minutes
```

指标：

```text
ingestion_lag_seconds
parse_lag_seconds
embedding_lag_seconds
index_promotion_lag_seconds
DLQ_rate
retry_rate
freshness_SLA_hit_rate
```

---

## 15. Observability

必须能看到：

```text
- 每个 topic lag
- 每个 worker throughput
- 每个 stage error rate
- DLQ 原因分布
- index completeness
- active index version
- freshness SLA
- cost per indexed document
```

事件 trace：

```text
document_id
 -> raw_document event
 -> parsed events
 -> chunk events
 -> embedding indexed
 -> BM25 indexed
 -> graph indexed
 -> table indexed
 -> index ready
```

---

## 16. API

### 16.1 Submit document

```http
POST /v5/ingestion/documents
```

```json
{
  "source_uri": "s3://bucket/report.pdf",
  "workspace_id": "ws_123",
  "ingestion_mode": "streaming"
}
```

### 16.2 Ingestion status

```http
GET /v5/ingestion/documents/{document_id}/status
```

### 16.3 Index versions

```http
GET /v5/indexes/versions
POST /v5/indexes/{index_version}/promote
```

### 16.4 DLQ

```http
GET /v5/dlq
POST /v5/dlq/{event_id}/retry
POST /v5/dlq/{event_id}/ignore
```

---

## 17. Security

V5 写路径要处理：

```text
- source authentication
- document ownership
- workspace isolation
- PII detection optional
- malware scanning optional
- allowed file types
- audit log
```

索引必须带 tenant/workspace metadata：

```text
workspace_id
tenant_id
access_policy
```

否则后续 query path 会有数据泄露风险。

---

## 18. Implementation Plan

### V5.0 Versioned Batch Ingestion

```text
- corpus_version
- index_version
- deterministic ids
- batch reindex
- promotion model
```

### V5.1 Queue-based Ingestion

```text
- simple queue / Redis / Postgres queue
- parser/chunker/indexer workers
- DLQ table
- retry policy
```

### V5.2 Kafka Topics

```text
- Kafka topic schema
- raw_document -> parsed -> chunk -> indexed
- idempotent consumers
```

### V5.3 Streaming Index Updates

```text
- incremental dense/BM25 upsert
- graph incremental update
- SQL facts incremental load
```

### V5.4 Flink / Stream Processing

```text
- optional Flink for large-scale transforms
- checkpointing
- backpressure handling
- exactly-once where needed
```

### V5.5 SLA + Observability

```text
- lag dashboards
- freshness metrics
- DLQ dashboard
- index completeness report
```

---

## 19. Definition of Done

V5 完成时必须满足：

```text
- 新文档能通过 streaming path 进入系统。
- 每个 ingestion stage 有事件和 trace。
- 失败消息进入 DLQ，可重试或忽略。
- Text index / Graph index / SQL store 能增量更新。
- index_version 可构建、验证、promote。
- QueryRun 能记录使用的 active index version。
- 有 freshness SLA 和 lag 监控。
```

---

## 20. V5 结论

V5 的核心不是“上 Kafka”。

V5 的核心是：

```text
让 Atlas 的所有 Context Provider 都能被持续、可靠、可观测地更新。
```

Kafka/Flink 是手段，不是目标。

目标是：

```text
持续写入 + 可重试 + 可版本化 + 可回放 + 可观测。
```
