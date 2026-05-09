# V3.0 里程碑：Atlas GraphProvider Walking Skeleton

更新时间：2026-05-08

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
默认：runtime 注册 hybrid + graph 两个可执行 provider
启用：graph 由 QueryPlan 选择调用；不对所有 query 强制运行
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

Phase 5 完成 V3.0 runtime opt-in 装配。Phase 6 将 graph 提升为默认可执行 provider，最终实现事实：

```text
IMPLEMENTED_RUNTIME_PROVIDERS = ("hybrid", "graph")
Settings.query_runtime_executable_providers = "hybrid,graph"
executable_query_providers 校验 requested 是否 known/registered；unknown/reserved 直接配置错误，sql 保持 non-executable placeholder
get_graph_provider() -> provider_registry.build("graph", ProviderBuildContext(...))
graph store 只能经 graph_store_backend / build_graph_store() 注入
get_provider_router() 按 executable providers 从 provider registry 注册 TextHybridProvider + GraphProvider
QueryRuntime 无显式 provider_router 时可按配置 auto-wire GraphProvider
trace_logger 能把 graph evidence 记录为 retriever_type="graph"
```

2026-05-08 补做 provider industrialization refactor：V3 不再只是“能注册一个
GraphProvider”，而是把 graph 固定为 provider 架构里的一个一等路由分支。这个 refactor
是 V4 structured provider 前置债务清理，不代表 SQL / structured data 已经实现。

落地边界：

```text
RetrievalProvider ABC 强制 provider 继承
ComponentRegistry / provider_registry 负责 provider factory 注册
TextHybridProvider / GraphProvider 显式继承 RetrievalProvider
SQL 保持 known semantic provider，但 runtime registration 被拒绝
reserved internal lanes 不能冒充 provider
ProviderRouter.aretrieve() 在 session_factory 存在时并发调度 hybrid/graph
CandidateAdapter / CandidateFusion 把 provider output 收口到统一 candidate window
global reranker 在 hybrid + graph 统一候选上执行
EvidenceBuilder 只消费全局排序后的 candidates
```

2026-05-09 补做 V4 preflight industrialization：backend registry、typed candidate
policy、ingestion registry 和 QueryRuntime async facade 已落地。该阶段仍不实现
SQLProvider；它只是让 SQLProvider 后续能通过 typed candidate / Evidence Kernel 接入。

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
ProviderResult.candidates
ProviderResult.evidence / evidence_pack 仅作为兼容 fallback，不再是主输出边界
```

Provider industrialization 新增/纳入：

```text
src/atlas/core/registry.py
src/atlas/retrieval/providers/base.py
src/atlas/retrieval/providers/registry.py
src/atlas/retrieval/candidate_adapter.py
src/atlas/retrieval/candidate_fusion.py
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
GraphProvider.aretrieve_candidates(ctx: RetrievalContext) -> ProviderResult
GraphProvider.retrieve_provider_result(...) -> legacy sync wrapper
ProviderResult.candidates -> tuple[Candidate, ...]
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

最新可引用的 V3.0 runtime refactor 验证：

```text
python -m compileall -q src/atlas scripts
pytest -q: 181 passed, 2 warnings
targeted provider/router/planner tests: 112 passed, 2 warnings
git diff --check
```

这代表 V3.0 runtime + provider industrialization verification。当前仍只证明
contract、grounding、runtime 装配、parallel router、candidate fusion/global rerank
链路能跑通，不证明检索质量或答案质量提升。

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
executable_query_providers 默认返回 hybrid/graph，并过滤 sql
ProviderRouter 拒绝把 sql 或 dense/bm25/table 等 internal lane 注册为 provider
ProviderRouter 并发执行 hybrid/graph，provider 失败时保留 failed ProviderResult 和 partial trace
CandidateFusion 将 hybrid/graph candidates 合并后统一送入 global reranker
同 chunk / 同 parent 去重时保留跨 provider provenance 和 graph source_anchor
API/details 中 provider failure trace 会 scrub error_message/planned_text
cache schema 升级到 atlas-query-cache-v3，并纳入 executable providers / global rerank policy
dependencies 默认注册 GraphProvider；显式 hybrid-only 配置可回到 V1 baseline
QueryRuntime 默认 auto-wire GraphProvider
默认注册 graph 时，hybrid-only QueryPlan 不会调用 GraphProvider
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
默认 runtime 变成 graph-first 或所有 query 强制跑 graph
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
没有 full native AsyncSession repository 主路径迁移；当前 `QueryRuntime.arun()`
是 async facade，DB repository / graph store 主体仍通过 sync runtime thread offload 过渡。
backend registry / ingestion registry 已有 V4 preflight 基础；外部 plugin discovery、
FinanceBench importer 复用 ingestion contract、structured extraction/indexing 仍未完成。
没有 SQLProvider / Text-to-SQL / structured table-cell storage。
structured candidate policy 已有 pinned/supporting 基础，但真实 cell provenance 仍待 V4 SQLProvider 落地。
```

## 下一步

```text
1. 为 graph local/path 建 retrieval-only eval，不把结果包装成答案质量声明。
2. 补 source-grounded entity / relationship extraction pipeline。
3. 再设计 V3.1+ community/global/DRIFT，并先补 grounding/eval。
```
