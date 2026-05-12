# V4 前置里程碑：Atlas Industrialization Preflight

更新时间：2026-05-09

## 版本契约

| 版本 | 名称 | 核心目标 | 主要产物 |
|---|---|---|---|
| V4 Preflight | Atlas Industrialization Preflight | 在 SQLProvider 前补齐 backend、candidate、ingestion、async facade 边界 | backend registry、typed candidate policy、ingestion registry、QueryRuntime async facade |

设计来源：

```text
docs/Design-docs/04_V4_STRUCTURED_DATA_CONTEXT.md
```

Phase 1 实际架构：

```text
docs/exec-plans/version-arch/v4_structured_ingestion_phase1_arch.md
```

## 当前状态

```text
状态：V4 前置工业化接口已落地；SQLProvider V1 受控单表 Text-to-SQL proof 已接入
边界：SQLProvider 默认不执行；只有 sql_provider_enabled=true 且 query_runtime_executable_providers 显式包含 sql 时才可执行；不声明结构化检索/答案质量提升或生成式答案可靠性
目标：让后续 SQLProvider 不污染 ProviderRouter / reranker / EvidenceBuilder 主链路
Phase 1：显式 opt-in V4 ingestion profile 与 structured artifact contract/proof 已接入；默认 PDF/MD/TXT ingestion 不变
```

## 已实现内容

### Backend Registry

已新增按类型拆分的 backend registry：

```text
src/atlas/backends/
  EmbeddingBackend
  SparseBackend
  RerankerBackend
  LLMClientBackend
  AnswerGeneratorBackend
  VectorStoreBackend
  GraphStoreBackend
```

默认注册：

```text
local_bge -> LocalBGEEmbedder
fastembed_bm25 -> BM25SparseEncoder
cross_encoder -> CrossEncoderReranker
openai -> OpenAIClient / OpenAIAnswerGenerator
qdrant -> QdrantClient
postgres_graph -> PostgresGraphStore
```

新增配置：

```text
ATLAS_EMBEDDING_BACKEND=local_bge
ATLAS_SPARSE_BACKEND=fastembed_bm25
ATLAS_RERANKER_BACKEND=cross_encoder
ATLAS_LLM_CLIENT_BACKEND=openai
ATLAS_ANSWER_GENERATOR_BACKEND=openai
ATLAS_VECTOR_STORE_BACKEND=qdrant
ATLAS_GRAPH_STORE_BACKEND=postgres_graph
```

Provider registry 现在通过 backend build context 获取 embedder、sparse encoder、reranker、
vector store 和 graph store。非法 backend name 会抛出明确 `CONFIGURATION_ERROR`。

### Typed Candidate / Fusion Policy

`Candidate` 新增：

```text
rerankable: bool
fusion_policy: ranked | pinned | supporting
structured_payload: dict
```

当前策略：

```text
hybrid / graph grounded text -> ranked + rerankable
future sql_result / table_cell / calculation -> 可表达为 pinned/supporting + rerankable=false
```

`CandidateFusion` 仍做跨 provider 去重和 provenance/source_anchor 合并；global reranker
只接收 `rerankable=true` 且 text source type 可 rerank 的候选。`pinned` / `supporting`
候选不走 CrossEncoder，但可以进入 EvidencePack。

provider 身份以 Router / ProviderResult 的可执行 provider 为准（当前 `hybrid` / `graph`）。
候选自身的 implementation/source 标签只进入 `candidate_provider` / `source_provider`
等 metadata；`sql` 这类 non-executable provider 名称只保留为 `reported_provider`，
不能伪装成 provider / candidate_provider。

SQLProvider V1 opt-in 后，`sql` 是实际 ProviderResult provider；其 candidate 使用
`source_type=sql_result`、`rerankable=false`、`fusion_policy=pinned`，并把 SQL、
结果行、safe/raw identifier map、used column ids 放入 structured_payload。

### Ingestion Registry

已新增 ingestion contracts / registry：

```text
DocumentLoader
DocumentParser
Chunker
ParentBlockBuilder
VectorIndexer
StructuredExtractor
```

默认 built-ins：

```text
local document loader
PDF / Markdown / TXT parser
default text chunker
page/document parent block builder
qdrant vector indexer
noop structured extractor
```

`IngestionService` 保持原 API 行为，但主流程改为编排 registry 组件：

```text
load -> parse -> chunk -> parent blocks -> vector index
```

Phase 1 profile gate：

```text
默认 profile:
  PDF / Markdown / TXT 维持既有 ingestion 行为。
  CSV / XLSX / HTML / HTM 仍为 unsupported file type。

V4 profile:
  调用方必须显式传入 ingestion_profile=v4。
  当前 /documents/ingest 请求体没有新增字段；可通过 metadata.atlas_ingestion_profile=v4 opt in。
  CSV / XLSX / HTML / HTM 会通过 profile gate 进入 ingestion。
  CSV / XLSX / HTML / HTM table intake 不作为普通 TextChunk 发往主文本索引。
  tabular intake 不写 legacy raw-row ParentBlock。
  table-only / artifact-only intake 没有 indexable chunks 时不 prepare 或写入 vector index。
  HTML / HTM 已纳入 tabular helper；table-like elements 不落成 ChildChunk。
  CSV 优先使用 structured table/schema-card contract 生成 StructuredArtifact。
  V4 profile 的 indexable chunks 写入独立 v4_qdrant_collection，不写默认 V1/V3 collection。
  V4 profile 的 indexable chunks 仍带 metadata namespace。
  duplicate skip 是 profile-aware；默认 profile 与 V4 profile 不跨 profile 互相吞。
```

### V4 Structured Contract / Writer

已落地 Phase 1 contract/proof：

```text
SourceLocator:
  已包含 precision / confidence / method / exact。
  已包含 storage locator。

Stable identity:
  已提供 stable id / content hash。

Canonical contracts:
  ParentChunk / ChildChunk
  SchemaRoutingCard / TableCard / ColumnCard / ProfileCard

StructuredArtifactWriter:
  service 已接入 writer。
  raw artifact 写完整 StructuredArtifact envelope。
  manifest 可审计。
  schema_routing_card 不物化为 table_asset。
  schema version 冲突或 unsupported 会 fail-fast。
  partial failure 不 silent pass。
  service 在 structured artifact 写入后若后续失败，会把 manifest 标记为 orphaned。
```

### SQLProvider V1 Opt-in Proof

已新增受控单表 SQLProvider 包：

```text
src/atlas/retrieval/providers/sql/
  intent.py
  schema_router.py
  identifiers.py
  compiler.py
  validator.py
  duckdb_index.py
  executor.py
  evidence.py
  provider.py
```

主链路：

```text
question -> SQLIntentGate -> AtlasSchemaRouter -> IdentifierNormalizer/SQLSchemaContext
-> SQLCompiler -> SQLValidator -> DuckDBExecutor -> SQLResultEvidence
```

关键取舍：

```text
只支持单表 SELECT。
默认 no LLM 时只生成保守启发式 SQL，否则返回 compiler_failed。
sqlglot / duckdb 是 structured-sql optional extra，不在默认 import path 强依赖。
缺少 duckdb 时执行阶段给清晰 diagnostic，不影响默认 hybrid/graph runtime。
DuckDB sandbox 的 external access / extension autoload-autoinstall / community extensions / lock_configuration 是关键设置；任一失败都会 fail closed 为 execution_failed。
SQLProvider V1 timeout isolation 是 timeout_isolation=thread_only，只取消等待线程结果，不声明 worker-process isolation 或能强杀 native DuckDB query。
SQLResultEvidence 是确定性文本，不做 LLM summary。
```

### Async Facade

已新增：

```text
async_engine
AsyncSessionLocal
get_async_db
QueryRuntime.arun(...)
```

当前是 async facade + sync runtime thread offload，不是 full async repository migration。
`POST /v1/query` 优先调用 `QueryRuntime.arun()`；`QueryRuntime.run()` 保留给 CLI / legacy tests。

## 验证

```text
python -m compileall -q src/atlas scripts
pytest -q: 281 passed, 1 skipped, 2 warnings
targeted ingestion/structured tests: 60 passed, 2 warnings
targeted backend/provider identity tests: 79 passed, 2 warnings
targeted SQLProvider V1 tests: 20 passed, 1 skipped, 2 warnings
git diff --check
```

重点测试：

```text
tests/test_backend_registry.py
tests/test_ingestion_registry.py
tests/test_v4_ingestion_phase1.py
tests/test_async_runtime_facade.py
tests/test_provider_router_contract.py
tests/test_sql_provider_v1.py
```

## 明确非声明

```text
SQLProvider V1 不是默认执行路径。
SQLProvider V1 只声明受控单表最小闭环，不声明完整 Text-to-SQL 质量。
calculator / formula runtime 尚未实现。
多表 SQL / JOIN / CTE / 子查询 / window 等能力尚未实现。
SQL timeout 仍是 thread-only，不是 worker-process sandbox。
cell-level citation / source cell provenance 尚未完成。
full AsyncSession repository path 尚未完成。
外部 plugin discovery / Python entry points 尚未实现。
CSV / XLSX / HTML / HTM ingestion opt-in 不代表结构化数值问答质量已达成。
tabular profile intake 不产生可用于答案的 raw row TextChunk。
Excel advanced adapter 不是本阶段完成能力。
PDF advanced table adapter 不是本阶段完成能力。
```

## 下一步

```text
1. V4.1 structured storage / table-cell schema。
2. V4.2 SQLProvider V1 从 proof 推进到测评样本集。
3. V4.3 SQL verifier / calculator。
4. V4.4 SQLResultEvidenceBlock + citation/cell provenance。
5. 继续把 repository / graph store 主路径迁到 native AsyncSession。
```
