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

## 当前状态

```text
状态：V4 前置工业化接口已落地
边界：不实现 SQLProvider；不声明结构化检索/答案质量提升
目标：让后续 SQLProvider 不污染 ProviderRouter / reranker / EvidenceBuilder 主链路
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
pytest -q: 222 passed, 2 warnings
targeted backend/provider identity tests: 79 passed, 2 warnings
git diff --check
```

重点测试：

```text
tests/test_backend_registry.py
tests/test_ingestion_registry.py
tests/test_async_runtime_facade.py
tests/test_provider_router_contract.py
```

## 明确非声明

```text
SQLProvider 尚未实现。
Text-to-SQL 尚未实现。
structured table/cell storage 尚未实现。
full AsyncSession repository path 尚未完成。
外部 plugin discovery / Python entry points 尚未实现。
```

## 下一步

```text
1. V4.1 structured storage / table-cell schema。
2. V4.2 SQLProvider candidate contract。
3. V4.3 SQL verifier / calculator。
4. V4.4 SQLResultEvidenceBlock + citation/cell provenance。
5. 继续把 repository / graph store 主路径迁到 native AsyncSession。
```
