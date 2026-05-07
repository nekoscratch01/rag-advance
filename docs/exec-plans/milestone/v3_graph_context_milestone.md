# V3.0 里程碑：Atlas GraphProvider Walking Skeleton

更新时间：2026-05-07

## 版本契约

| 版本 | 名称 | 核心目标 | 主要产物 |
|---|---|---|---|
| V3.0 | Atlas GraphProvider Walking Skeleton | 把 graph context 作为可路由、可 grounding、可审计的 provider 接入 Evidence Kernel | graph schema、fixture loader、PostgresGraphStore、GraphProvider local/path、graph evidence grounding、ProviderRouter contract |

设计来源：

```text
docs/Design-docs/03_V3_GRAPH_CONTEXT.md
```

实际架构：

```text
docs/exec-plans/version-arch/v3_graph_provider_arch.md
```

## 当前状态

```text
状态：V3.0 walking skeleton 已实现
默认：V1 runtime 仍是 hybrid-only
启用：graph 是 opt-in provider，配置口径为 ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid,graph
证据：证明 provider contract、Postgres grounding pivot、trace auditability
非目标：不证明 retrieval / answer quality lift
```

## 实现快照

每个实现阶段均做过 multi-agent review，并在阶段结束后形成 committed / pushed snapshot。当前可见提交包括：

```text
1c8c2fd Add graph provider schema contracts
c9235e3 Add graph fixture loader and store
de650e9 Add graph provider skeleton
1b40e67 Add graph evidence grounding
```

Phase 5 已完成 V3.0 runtime opt-in 装配，最终实现事实：

```text
IMPLEMENTED_RUNTIME_PROVIDERS = ("hybrid", "graph")
get_graph_provider() -> GraphProvider(PostgresGraphStore)
get_provider_router() 仅在 executable providers 包含 graph 时注册 GraphProvider
QueryRuntime 无显式 provider_router 时可按配置 auto-wire GraphProvider
trace_logger 能把 graph evidence 记录为 retriever_type="graph"
```

## 已实现产物

### Contracts / Models

```text
src/atlas/retrieval/contracts.py
src/atlas/retrieval/providers/graph/models.py
src/atlas/retrieval/providers/graph/store.py
src/atlas/retrieval/providers/graph/cache.py
```

已落地：

```text
GraphEntity
GraphRelationship
GraphPath
GraphNeighborhood
GraphCandidate
GraphFilters
GraphStore Protocol
GraphCache Protocol
NoOpGraphCache
SourceAnchor.graph_ids
ProviderResult.evidence / evidence_pack
```

### Storage / Loader

```text
src/atlas/db/models.py
src/atlas/db/session.py
src/atlas/retrieval/providers/graph/fixture.py
src/atlas/retrieval/providers/graph/postgres_store.py
tests/fixtures/graph/hub_fixture.json
```

Postgres graph tables：

```text
graph_indexes
graph_entities
graph_relationships
graph_entity_anchors
graph_relationship_anchors
graph_communities
```

Fixture loader 能验证 JSON shape、canonical hash、graph_version 冲突、anchor chunk 是否存在、hub-like entity 信号，并支持 idempotent load 与 replace。

### Provider / Grounding

```text
src/atlas/retrieval/providers/graph/provider.py
src/atlas/retrieval/providers/graph/evidence.py
src/atlas/retrieval/router.py
src/atlas/query_runtime/service.py
```

V3.0 实际路径：

```text
JSON fixture
  -> Postgres graph tables
  -> PostgresGraphStore
  -> GraphProvider
  -> graph local/path items
  -> SourceAnchor(chunk_id)
  -> chunks.text hydration
  -> Candidate(provider="graph", source_type="text_chunk", text=chunks.text)
  -> Evidence(text=chunks.text)
  -> ProviderRouter / QueryRuntime
```

Graph-only `summary`、`description`、`text_span`、`path_text` 只可作为审计/metadata 输入的一部分，不能成为 prompt-visible evidence text。

## API / Contract 行为

V3.0 不新增独立 `/v3/graph/*` HTTP API。Graph 通过统一 provider contract 接入：

```text
GraphProvider.retrieve_provider_result(...) -> ProviderResult
ProviderResult.candidates -> tuple[Candidate, ...]
ProviderResult.evidence -> tuple[Evidence, ...]
ProviderResult.evidence_pack -> EvidencePack | None
ProviderResult.trace -> graph audit payload
```

支持模式：

```text
local：解析 1 个实体，取 capped neighborhood，回源 chunk
path：解析 2 个实体，找 1-2 hop path，回源 chunk
```

显式返回 unsupported / empty 的模式：

```text
global
community
drift
```

默认 hub 防护：

```text
DEFAULT_DEGREE_CAP = 25
DEFAULT_MAX_HOPS = 2
DEFAULT_MAX_PATHS = 20
DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT = 3
```

## 测试与验证

最新可引用的 V3.0 Phase 5 实现验证：

```text
pytest -q: 126 passed, 2 warnings
Phase 5 targeted tests: 39 passed, 2 warnings
```

这代表 V3.0 Phase 5 implementation verification；仍只证明 contract、grounding 和 opt-in runtime 装配，不证明检索质量或答案质量提升。

覆盖重点：

```text
tests/test_graph_fixture_loader.py
tests/test_graph_store.py
tests/test_graph_provider.py
tests/test_graph_grounding.py
tests/test_provider_router_contract.py
tests/test_retrieval_plan_task.py
```

测试证明的内容：

```text
fixture schema / hash / replace / missing chunk validation
Postgres graph_version scoping
alias entity lookup
document_ids / chunk_ids / relation_types filters
hub degree cap and truncation trace
local neighborhood retrieval
one-hop / two-hop path retrieval
grounded Candidate uses chunks.text
Evidence uses hydrated chunk text, not graph-only text
unsafe graph summary/description/path_text/text_span not prompt-visible
ProviderRouter can execute registered graph provider and serialize trace without candidate text
executable_query_providers 默认只返回 hybrid，opt-in 时返回 hybrid/graph，并过滤 sql
dependencies 只在 opt-in 时注册 GraphProvider
QueryRuntime 可按配置 auto-wire GraphProvider
ready graph task 缺少注册 provider 时返回 provider_not_registered:graph
graph evidence trace event retriever_type = graph
```

## 明确非声明

V3.0 不声明：

```text
检索质量提升
生成式答案可靠性提升
FinanceBench 指标提升
global/community/DRIFT search 已可用
Graph summary 可以作为 citation
Graph relation 是 SQL-like exact fact
默认 V1 query path 被 graph 替换
```

SQL 仍是未来 structured EvidenceBlock / structured fact provider，不绕过 Evidence Kernel，也不作为 graph provider 的替代路径。

## 已知缺口

```text
没有 graph retrieval eval 或 answer eval。
没有 entity extraction / relation extraction pipeline。
没有 entity resolution 质量评估。
没有 community/global/DRIFT 可执行路径。
没有 graph-assisted text retrieval。
没有独立 graph debug HTTP API。
GraphCache 只有 Protocol 和 NoOpGraphCache，没有真实后端。
Graph fixture 是测试/开发入口，不是持续 ingestion pipeline。
```

## 下一步

```text
1. 为 graph local/path 建 retrieval-only eval，不把结果包装成答案质量声明。
2. 补 source-grounded entity / relationship extraction pipeline。
3. 再设计 V3.1+ community/global/DRIFT，并先补 grounding/eval。
```
