# V3.0 实际架构：Atlas GraphProvider Walking Skeleton

更新时间：2026-05-07

本文记录 V3.0 GraphProvider 的真实实现。Design 文档说明目标；本文说明已经落地的模块、数据流、DB 表、trace payload 和边界。

核心事实：

```text
V3.0 已实现 opt-in GraphProvider walking skeleton。
默认 V1 runtime 仍是 hybrid-only。
GraphProvider 只支持 local / path。
GraphProvider 不直接把 graph text 变成 evidence。
Evidence text 必须来自 Postgres chunks.text。
V3.0 不声明 retrieval / answer quality lift。
SQL 仍是未来 structured EvidenceBlock / structured fact provider，不绕过 Evidence Kernel。
```

---

## 1. 总体路径

V3.0 的真实数据流：

```text
JSON graph fixture
  -> load_graph_fixture(...)
  -> Postgres graph tables
  -> PostgresGraphStore
  -> GraphProvider
  -> graph item / graph candidate metadata
  -> SourceAnchor(chunk_id)
  -> Postgres chunks
  -> Candidate(provider="graph", source_type="text_chunk", text=chunks.text)
  -> Graph evidence builder
  -> EvidenceBlock / Evidence(text=chunks.text)
  -> ProviderRouter / QueryRuntime
  -> trace
```

关键 pivot：

```text
graph object 本身不是证据。
graph object 找到 source anchor。
source anchor 找到 chunk。
chunk.text 才是 prompt-visible evidence。
```

因此 V3.0 的 GraphProvider 是 source-grounded candidate generator，不是独立回答器。

---

## 2. 模块映射

| 架构节点 | 模块 | 说明 |
|---|---|---|
| Graph contracts | `src/atlas/retrieval/providers/graph/models.py` | GraphEntity / Relationship / Path / Candidate / Filters |
| GraphStore Protocol | `src/atlas/retrieval/providers/graph/store.py` | store boundary |
| Graph cache contract | `src/atlas/retrieval/providers/graph/cache.py` | GraphCache Protocol + NoOpGraphCache |
| Fixture loader | `src/atlas/retrieval/providers/graph/fixture.py` | JSON fixture validation/load/replace |
| Postgres store | `src/atlas/retrieval/providers/graph/postgres_store.py` | entity lookup、neighbors、paths、anchors |
| Provider | `src/atlas/retrieval/providers/graph/provider.py` | local/path retrieval and trace |
| Evidence adapter | `src/atlas/retrieval/providers/graph/evidence.py` | candidates -> EvidencePack / Evidence |
| Runtime contract | `src/atlas/retrieval/contracts.py` | SourceAnchor / ProviderResult |
| Router | `src/atlas/retrieval/router.py` | provider grouping, execution, skipped trace |
| DB models | `src/atlas/db/models.py` | graph tables |
| DB bootstrap | `src/atlas/db/session.py` | lightweight graph table migrations |

---

## 3. Storage

V3.0 使用现有 Postgres runtime storage，不引入 Neo4j、Redis Queue 或 graph-native store。

### 3.1 Tables

```text
graph_indexes
  graph_version
  corpus_version
  fixture_schema_version
  fixture_hash
  loader_version
  row_counts_json
  status
  loaded_at
  metadata_json

graph_entities
  graph_version
  entity_id
  canonical_name
  canonical_name_norm
  entity_type
  aliases_json
  metadata_json

graph_relationships
  graph_version
  relationship_id
  source_entity_id
  target_entity_id
  relation_type
  confidence
  metadata_json

graph_entity_anchors
  graph_version
  anchor_id
  entity_id
  chunk_id
  text_span
  text_span_hash
  metadata_json

graph_relationship_anchors
  graph_version
  anchor_id
  relationship_id
  chunk_id
  text_span
  text_span_hash
  metadata_json

graph_communities
  graph_version
  community_id
  level
  summary
  metadata_json
```

重要约束：

```text
graph_version 是 graph object identity 的命名空间。
entity / relationship / anchor 均按 graph_version scoped。
anchors 通过 chunk_id 指向 runtime chunks 表。
community 表已存在，但 V3.0 不执行 community/global search。
```

### 3.2 Fixture Loader

入口：

```text
load_graph_fixture_file(db, path, replace=False)
load_graph_fixture(db, fixture, replace=False)
validate_graph_fixture(fixture)
```

行为：

```text
要求 fixture_schema_version = graph_fixture_v1。
计算 canonical JSON sha256 fixture_hash。
graph_version 已存在且 hash 相同 -> noop。
graph_version 已存在且 hash 不同 -> 默认报 conflict，replace=True 时清旧行再写入。
entity / relationship / community / anchor id 不允许重复。
relationship 必须引用已知 entity。
anchor.chunk_id 必须已存在于 chunks。
fixture 必须包含 hub-like entity 信号。
```

fixture 可以保存 graph-only summary / description；这些只用于审计和未来开发，不能成为 Evidence text。

---

## 4. GraphStore Protocol

`GraphStore` 是 provider 与存储之间的边界：

```python
get_entity(db, entity_id, *, graph_version) -> GraphEntity | None

find_entities(
    db,
    *,
    query_text,
    filters,
    aliases=(),
    limit=10,
) -> tuple[GraphEntityMatch, ...]

get_neighbors(
    db,
    *,
    entity_id,
    degree_cap=DEFAULT_DEGREE_CAP,
    relation_types=None,
    filters=None,
) -> GraphNeighborhood

find_paths(
    db,
    *,
    source_entity_id,
    target_entity_id,
    max_hops=DEFAULT_MAX_HOPS,
    degree_cap=DEFAULT_DEGREE_CAP,
    relation_types=None,
    filters=None,
    max_paths=DEFAULT_MAX_PATHS,
) -> tuple[GraphPath, ...]

get_relationships(db, ids, *, graph_version) -> tuple[GraphRelationship, ...]

get_chunks_for_entities(
    db,
    entity_ids,
    *,
    graph_version,
    max_source_chunks_per_result=DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
) -> dict[str, tuple[SourceAnchor, ...]]

get_chunks_for_relationships(
    db,
    relationship_ids,
    *,
    graph_version,
    max_source_chunks_per_result=DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
) -> dict[str, tuple[SourceAnchor, ...]]
```

`PostgresGraphStore` 的实际能力：

```text
entity_id lookup
canonical / alias exact match
canonical / alias partial match
entity_types filter
relation_types filter
document_ids / chunk_ids anchor filter
degree-capped neighborhood
one-hop and two-hop path search
source anchor hydration to SourceAnchor
```

---

## 5. GraphProvider

入口：

```text
GraphProvider.retrieve_provider_result(...)
```

返回：

```text
ProviderResult(
  provider="graph",
  status="executed" | "empty",
  candidates=tuple[Candidate, ...],
  evidence=tuple[Evidence, ...],
  evidence_pack=EvidencePack | None,
  trace={...},
)
```

### 5.1 Entity Resolution

解析顺序：

```text
1. query_plan.entities
2. task.metadata.entities / graph_entities / aliases
3. task.query_text fallback
```

trace 会记录：

```text
entity_resolution.source
entity_resolution.attempts
resolved_entities[]
```

### 5.2 Mode Selection

支持：

```text
local
path
```

明确不支持：

```text
global
community
drift
```

选择规则：

```text
如果 metadata 明确 graph_mode / graph_search_mode / graph_retrieval_mode，则尝试使用该 graph mode。
如果 resolved entity >= 2，则推断 path。
如果 resolved entity == 1，则推断 local。
否则 empty: no_resolved_entity。
```

普通 `metadata.mode` 不控制 graph state machine；这是为了避免 text-hybrid 的 dense/bm25 mode 误伤 graph provider。

### 5.3 Local

流程：

```text
resolved entity
  -> store.get_neighbors(...)
  -> graph_neighborhood item
  -> relationship/entity source anchors
  -> chunks hydration
  -> grounded Candidate
```

输出 trace：

```text
degree_seen
neighbors_examined
neighbors_returned
truncated
hub_cap_applied
truncated_reason
cap_config
```

### 5.4 Path

流程：

```text
resolved source entity + target entity
  -> store.find_paths(max_hops=2)
  -> graph_path item
  -> relationship source anchors
  -> chunks hydration
  -> grounded Candidate
```

输出 metadata：

```text
graph_path.path_id
graph_path.graph_version
graph_path.source_entity_id
graph_path.target_entity_id
graph_path.entity_ids
graph_path.relationship_ids
graph_path.hops
```

### 5.5 Defaults / Hub Guard

```text
DEFAULT_DEGREE_CAP = 25
DEFAULT_MAX_HOPS = 2
DEFAULT_MAX_PATHS = 20
DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT = 3
```

这些值可以通过 task metadata 或 options 覆盖：

```text
degree_cap / graph_degree_cap
max_hops / graph_max_hops
max_paths / graph_max_paths
max_source_chunks_per_result / graph_max_source_chunks_per_result
```

hub node 被 capped，不允许无限展开。

---

## 6. Candidate / Evidence Grounding

V3.0 的安全规则：

```text
GraphCandidate.graph_text 不是 Evidence text。
graph summary / description / narrative / path_text 不是 Evidence text。
SourceAnchor.text_span 不是 Evidence text。
Candidate.text 必须是 chunks.text。
Evidence.text 必须来自 Candidate.text，也就是 chunks.text。
```

实际 Candidate 形态：

```python
Candidate(
    provider="graph",
    source_type="text_chunk",
    text=chunks.text,
    chunk_id=anchor.chunk_id,
    document_id=chunk.document_id,
    retrieved_by=("graph",),
    lane="local" | "path",
    fusion_score=graph_score,
)
```

关键 metadata：

```text
provider
query_plan_id
graph_candidate_id
entity_ids
relationship_ids
graph_score
grounding_strength
grounded_source_chunk_ids
source_anchor
retrieval_task_id
retrieval_unit_id
graph.provider
graph.provider_version
graph.source_type
graph.mode
graph.graph_version
graph.graph_path
```

`source_anchor` 会保留 document/chunk/page/graph_ids 等 provenance，但会清理 prompt-visible 风险字段。

Evidence pack：

```text
evidence_builder = graph_grounded_chunk_pack_v1
provider = graph
provider_version = 3.0.0
source_type = text_chunk
parent_expansion = False
```

V3.0 不做 parent expansion；它证明 chunk grounding pivot，而不是完整 V1 parent-child evidence strategy。

---

## 7. ProviderRouter / QueryRuntime

`ProviderRouter` 已支持注册多个 provider：

```python
ProviderRouter({"hybrid": text_provider, "graph": graph_provider})
```

router 行为：

```text
按 RetrievalTask.provider group。
task.provider_status = skipped_non_executable 时写 skipped ProviderResult。
已注册 provider 执行 retrieve_provider_result。
ProviderResult.evidence 汇总进入 QueryRuntime。
provider_results 和 provider_router_trace 写入 trace。
```

默认 V1 仍是：

```text
ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid
IMPLEMENTED_RUNTIME_PROVIDERS = ("hybrid", "graph")
```

V3.0 graph opt-in 口径：

```bash
ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid,graph
```

Phase 5 已接入 opt-in 装配：

```text
src/atlas/api/dependencies.py
  get_graph_provider() -> GraphProvider(PostgresGraphStore, max_context_tokens=settings.max_context_tokens)
  get_provider_router() 只在 executable_query_providers(settings) 包含 "graph" 时注册 graph

src/atlas/query_runtime/service.py
  QueryRuntime 无显式 provider_router 时，按 settings auto-wire GraphProvider

src/atlas/retrieval/router.py
  ready task 但 provider 未注册时返回 provider_not_registered:{provider}

src/atlas/query_runtime/trace_logger.py
  graph evidence 记录为 retriever_type="graph"
```

无论是否 opt-in，默认 runtime 不变成 graph-first；graph 只在显式 `hybrid,graph` 时进入 executable provider set。

---

## 8. Trace Auditability

GraphProvider trace 包含：

```text
provider
provider_version
query_plan_id
planner
status
reason
tasks[]
graph_candidates[]
candidate_count
evidence_count
evidence_pack_id
dropped_evidence_count
retrieval_latency_ms
truncated
hub_cap_applied
degree_seen
neighbors_examined
neighbors_returned
paths_seen
paths_returned
truncated_reason
truncated_reasons
cap_config
relation_types
```

每个 task trace 包含：

```text
task_id
unit_id
query_text
provider_status
unsupported_reason
graph_filters
entity_resolution
resolved_entities
mode
status
reason
cap_config
grounded_candidate_count
latency_ms
```

ProviderRouter candidate trace 不序列化 candidate text；这避免 trace 中泄露完整 evidence body，同时保留 chunk/source anchor 审计信息。

---

## 9. GraphCache

V3.0 只有 contract：

```text
GraphCache.get(key)
GraphCache.set(key, value, ttl_seconds=None)
GraphCache.delete_prefix(prefix)
NoOpGraphCache
```

当前没有 Redis 或真实 graph cache backend。V1 cache 仍是 Postgres `query_cache`；Redis Queue 属于 V2 Research Runtime，不属于 V3.0。

---

## 10. Tests

最新可引用的 V3.0 Phase 5 实现验证：

```text
pytest -q: 126 passed, 2 warnings
Phase 5 targeted tests: 39 passed, 2 warnings
```

重点测试文件：

```text
tests/test_graph_fixture_loader.py
tests/test_graph_store.py
tests/test_graph_provider.py
tests/test_graph_grounding.py
tests/test_provider_router_contract.py
tests/test_retrieval_plan_task.py
```

验证范围：

```text
fixture validation / idempotency / replace
missing chunk rejection
hub-like entity requirement
Postgres graph_version scoping
entity alias lookup
document/chunk/relation filters
degree cap truncation
local neighborhood retrieval
path retrieval and ranking
source anchor hydration limits
grounded Candidate text = chunks.text
Evidence text = chunks.text
graph-only toxic text does not enter prompt / trace / evidence metadata
zero token budget drops graph evidence through EvidencePack
ProviderRouter can execute a registered graph provider
default executable providers remain hybrid-only
opt-in executable providers include graph and still filter sql
dependencies register GraphProvider only when opted in
QueryRuntime auto-wires GraphProvider when explicitly executable
ready graph task without registered provider returns provider_not_registered:graph
graph evidence persists retrieval events as retriever_type="graph"
```

这些测试证明 contract 和 grounding 安全，不证明 retrieval 或 answer quality。

---

## 11. Non-Claims

V3.0 不能被描述为：

```text
GraphRAG 质量主路径已完成。
Graph 提升了 FinanceBench retrieval 指标。
Graph 提升了生成式答案可靠性。
global/community/DRIFT 已实现。
Graph summary 可以直接引用。
Graph relation 可以作为 SQL-like exact fact。
默认 V1 runtime 改成 graph 路径。
SQL 已经实现。
```

准确说法是：

```text
V3.0 证明 graph provider contract 可以被路由、可以从 Postgres graph object pivot 到 chunks.text、可以留下可审计 trace。
默认 Atlas runtime 仍走 V1 hybrid provider。
GraphProvider 是 opt-in walking skeleton。
```

---

## 12. 后续缺口

```text
V3.1：entity extraction / resolution pipeline。
V3.2：relationship extraction / source anchor pipeline。
V3.3：graph retrieval eval。
V3.4：community/global source-grounded search。
V3.5：DRIFT-style search。
V4：SQLProvider / structured EvidenceBlock / structured fact provider。
```
