# V4 Phase 1 实际架构：Structured Ingestion Write-Path Proof

更新时间：2026-05-09

本文记录 V4 Phase 1 已落地的真实架构变化，以及后续补入的 SQLProvider V1 受控单表 proof。V4 ingestion 仍是显式 opt-in 的 offline write-path proof；SQLProvider V1 也是显式双开关 opt-in，不改变当前默认 query runtime 主路径。

核心事实：

```text
V4 Phase 1 已接入 ingestion profile gate。
V4 Phase 1 已接入 StructuredArtifactWriter、raw envelope 和 manifest。
V4 Phase 1 已有 SourceLocator / stable id / canonical chunks/cards contract。
V4 Phase 1 已有 CSV proof 和 Document proof。
V4 Phase 1 使用独立 v4_qdrant_collection。
SQLProvider V1 已有受控单表 Text-to-SQL 最小闭环。
默认 runtime 仍不执行 sql；必须 sql_provider_enabled=true 且 query_runtime_executable_providers 显式包含 sql。
当前不声明多表 SQL、复杂公式、cell-level citation 或生成式答案可靠性。
```

---

## 1. 边界

V4 Phase 1 的边界是 ingestion 写入路径：

```text
local document
  -> loader / parser
  -> V4 profile gate
  -> structured extractor or structured table contract
  -> StructuredArtifactWriter
  -> raw StructuredArtifact envelope + manifest
  -> optional V4 namespaced chunks
  -> v4_qdrant_collection
```

默认 runtime 不注册或执行 `SQLProvider`。显式双开关 opt-in 后，SQLProvider V1 只做单表 table/numeric/aggregation/filtering/ranking/top-k/grouped statistics/simple lookup 闭环，不把 retrieval-only、ingestion-only 或 SQL result proof 解释为生成式答案可靠性证据。

---

## 2. Profile Gate

入口仍是 ingestion service。默认 profile 保持 V1/V3 行为：

```text
PDF / Markdown / TXT -> 既有文本 ingestion。
CSV / XLSX / HTML / HTM -> 仍是 unsupported file type。
```

V4 profile 必须显式 opt in：

```text
ingestion_profile = v4
或当前 API 形态下 metadata.atlas_ingestion_profile = v4
```

V4 profile 下：

```text
CSV / XLSX / HTML / HTM 可以通过 profile gate 进入 ingestion。
tabular intake 不作为普通 TextChunk 写入主文本索引。
tabular intake 不写 legacy raw-row ParentBlock。
table-only / artifact-only intake 没有 indexable chunks 时不 prepare 或写入 vector index。
HTML / HTM table-like elements 不落成 ChildChunk。
duplicate skip 是 profile-aware；默认 profile 与 V4 profile 不跨 profile 去重吞掉对方。
Markdown / TXT 等 document proof 的 indexable chunks 带 V4 namespace。
```

---

## 3. Contracts

已落地的 contract 层：

| contract | 作用 |
|---|---|
| `SourceLocator` | source / storage / page / table / row / column / cell / char / bbox locator；包含 precision、confidence、method、exact |
| `stable_id(...)` / `content_hash(...)` | 为 document、table、row、cell、parent chunk、child chunk、card 提供稳定 identity |
| `ParsedDocumentIR` / `DocumentElementIR` | document proof 的 canonical document IR，可投影回 legacy `LoadedDocument` |
| `ParentChunk` / `ChildChunk` | canonical parent-child chunk contract；tabular DocumentIR 会拒绝走普通 text chunking |
| `SchemaRoutingCard` / `TableCard` / `ColumnCard` / `ProfileCard` | routing/card contract；routing card 不是答案 evidence，也不物化为 `table_asset` |
| `ArtifactManifest` / `ProvenancePolicy` | artifact 批次、source locator 和 materialization policy 的审计 envelope |

CSV proof 使用 structured table/schema-card helper 生成 `table` 与 `schema_routing_card` 等 StructuredArtifact。Document proof 证明 structured document IR、SourceLocator、stable id、canonical parent/child chunks 和 V4 index namespace 可以走通；它不是高级 PDF/Excel table adapter。

---

## 4. Writer And Storage

`StructuredArtifactWriter` 是 Phase 1 的写入边界：

```text
raw artifact 写完整 StructuredArtifact envelope。
manifest 可审计，记录 document_id / ingestion_run_id / artifact ids / status / errors。
schema version 冲突或 unsupported artifact type 会 fail-fast。
service 写入 structured artifact 后若后续 ingestion 失败，会把 manifest 标记为 orphaned。
```

运行时存储边界：

```text
Postgres:
  documents / ingestion_runs 继续记录 ingestion 状态。
  StructuredArtifactRecord 记录 structured artifact batch。

Qdrant:
  V4 profile 的 indexable chunks 写入 settings.v4_qdrant_collection。
  vector index prepare 延迟到存在 indexable chunks 时执行。
  不写默认 V1/V3 collection。
```

---

## 5. 模块映射

| 架构节点 | 模块 |
|---|---|
| profile gate / orchestration | `src/atlas/ingestion/service.py` |
| ingestion contracts | `src/atlas/ingestion/contracts.py` |
| structured IR/contracts | `src/atlas/ingestion/structured/contracts.py` |
| structured document chunking proof | `src/atlas/ingestion/structured/chunking.py` |
| structured table proof | `src/atlas/ingestion/structured/tables.py` |
| structured artifact writer | `src/atlas/ingestion/structured/writer.py` |
| built-in extractor/indexer routing | `src/atlas/ingestion/builtins.py` |
| V4 collection config | `src/atlas/core/config.py` |
| SQLProvider V1 opt-in runtime | `src/atlas/retrieval/providers/sql/` |

---

## 6. SQLProvider V1 Opt-in Proof

实现边界：

```text
question
  -> SQLIntentGate
  -> AtlasSchemaRouter
  -> IdentifierNormalizer / SQLSchemaContext
  -> SQLCompiler
  -> SQLValidator
  -> DuckDBExecutor
  -> deterministic SQLResultEvidence / pinned Candidate
```

配置边界：

```bash
ATLAS_SQL_PROVIDER_ENABLED=true
ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid,sql,graph
```

默认值仍保持：

```text
sql_provider_enabled=false
query_runtime_executable_providers=hybrid,graph
```

当前只支持单表受控 SELECT。SQLValidator 默认拒绝多语句、DDL/DML、JOIN、CTE、子查询、window、UDF、COPY/EXPORT、ATTACH/DETACH、PRAGMA/CALL、INSTALL/LOAD、read_csv/read_parquet/read_json、外部 URL/file FROM、集合运算、SELECT *、DISTINCT/HAVING/CAST/date functions。

DuckDB / sqlglot 是 optional extra：

```bash
pip install "atlas-rag-kernel[structured-sql]"
```

缺少 duckdb 时默认 import 不崩；只有执行 SQLProvider 时返回清晰 runtime diagnostic。

---

## 7. 明确非声明

```text
SQLProvider V1 不是默认 runtime 主路径。
SQLProvider V1 不是多表 SQL 引擎。
复杂公式 calculator 尚未实现。
cell-level citation / source cell provenance 尚未完成。
生成式答案可靠性尚未由 V4 Phase 1 证明。
CSV / XLSX / HTML / HTM opt-in ingestion 不等于结构化数值问答质量达成。
tabular profile intake 不产生可用于答案的 raw row TextChunk。
```
