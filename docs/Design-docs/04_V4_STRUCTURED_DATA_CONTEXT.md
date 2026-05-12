# V4 — Structured Data Context

> 核心目标：让 Atlas 能处理结构化表格、SQL 查询、数值计算和可追溯的表格证据。
> 关键边界：V4 是 SQLProvider / TableProvider，不让 LLM 自己猜数字或裸算复杂公式；SQL 结果也必须转成 EvidenceBlock。
> 当前实现：Phase 1 是 contract / ingestion proof；SQLProvider V1 受控单表 Text-to-SQL 最小闭环已接入，但默认关闭，不声明多表/复杂公式/cell citation 或生成式答案可靠性已完成。

---

## 1. V4 的定位

Finance / analytics / enterprise RAG 里，很多问题不是“找一段文本”，而是：

```text
- 查某个表格数值
- 做同比/环比
- 算 margin
- 汇总多个时期
- 排序多个公司
- 根据条件筛选
```

这些问题如果只交给 LLM 读 chunk，很容易：

```text
- 数字读错
- 年份列错位
- 单位搞错
- 公式算错
- 引用无法追溯到 cell
```

V4 的目标是把结构化数据变成 Atlas 的一个 Provider：

```text
SQLProvider / TableProvider
```

并且让 SQL / table result 进入同一个 Evidence Kernel。

---

## 2. V4 总体架构

```text
                         WRITE PATH

PDF / CSV / Excel / HTML Tables
        │
        ▼
┌──────────────────────┐
│ Table Extraction      │
│ rows/cells/headers    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Table Normalization   │
│ units/years/metrics   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Schema Registry       │
│ table/columns/types   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Structured Store      │
│ DuckDB/ClickHouse     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Cell Provenance       │
│ doc/page/table/cell   │
└──────────────────────┘


                         READ PATH

User Query
   │
   ▼
Query Orchestrator
   │
   ▼
Provider Router
   │
   ▼
SQLProvider / TableProvider
   │
   ▼
Text-to-SQL / Direct Lookup / Calculator
   │
   ▼
SQL Verifier + Result Formatter
   │
   ▼
Table EvidenceBlock
   │
   ▼
Evidence Kernel
```

---

## 3. V4 与 V1/V2/V3 的关系

```text
V1:
  负责 Evidence Kernel、QueryPlan、EvidenceBlock、Verifier。

V2:
  可以在 research job 中调用 SQLProvider 做数值子任务。

V3:
  Graph 可以辅助发现实体/指标关系，但 SQLProvider 负责精确数值。

V4:
  新增结构化数据 Provider，不替代 V1。
```

---

## 4. SQLProvider 适合的问题

```text
- 精确数值查询
- 多行/多列聚合
- 年份比较
- 多公司比较
- 排序/筛选
- ratio/margin/YoY/CAGR
- 已结构化 financial facts 查询
```

不适合单独处理：

```text
- 文本解释
- 管理层讨论
- 风险因素描述
- 没有结构化表的数据
```

这些需要 TextHybridProvider 或 GraphProvider 补充。

---

## 5. 数据模型

当前 Phase 1 已落地的 contract/proof 包含：

```text
- SourceLocator 支持 precision / confidence / method / exact 标记，以及 storage locator。
- stable_id / content_hash 用于稳定身份和内容校验。
- canonical ParentChunk / ChildChunk / SchemaRoutingCard / TableCard / ColumnCard / ProfileCard。
- StructuredArtifact envelope 保留 source locator、provenance policy、schema routing card、artifact manifest。
```

### 5.1 TableAsset

```python
class TableAsset:
    table_id: str
    document_id: str
    page_start: int
    page_end: int
    table_title: str | None
    source_type: str  # pdf_table / csv / excel / html
    extraction_method: str
    extraction_confidence: float
    metadata: dict
```

### 5.2 TableColumn

```python
class TableColumn:
    column_id: str
    table_id: str
    name: str
    canonical_name: str | None
    data_type: str
    unit: str | None
    period: str | None
    metadata: dict
```

### 5.3 TableRow

```python
class TableRow:
    row_id: str
    table_id: str
    row_index: int
    row_label: str | None
    canonical_metric: str | None
    metadata: dict
```

### 5.4 TableCell

```python
class TableCell:
    cell_id: str
    table_id: str
    row_id: str
    column_id: str
    raw_value: str
    normalized_value: float | str | None
    unit: str | None
    bbox: dict | None
    page_number: int | None
    provenance: dict
```

### 5.5 FinancialFact

为了 FinanceBench，可以抽象出财务事实表。

```python
class FinancialFact:
    fact_id: str
    company: str
    fiscal_year: int | None
    fiscal_period: str | None
    metric: str
    value: float
    unit: str
    scale: str | None
    currency: str | None
    source_document_id: str
    source_page: int | None
    source_table_id: str | None
    source_cell_ids: list[str]
    confidence: float
```

---

## 6. Structured Store 选择

### 6.1 DuckDB

适合：

```text
- 本地开发
- 文件级 analytics
- FinanceBench benchmark
- 中小规模表格查询
```

优点：

```text
- 简单
- 快
- 无服务端
- 适合 parquet/csv
```

### 6.2 Postgres

适合：

```text
- 与 metadata store 统一
- OLTP + 少量结构化查询
- provenance 强绑定
```

### 6.3 ClickHouse

适合：

```text
- 大规模 OLAP
- 高并发聚合
- 时间序列/事件分析
- V5+ 持续写入后的分析场景
```

V4 初期建议：

```text
DuckDB + Postgres metadata
```

成熟后再引入 ClickHouse。

---

## 7. Table Extraction

V4 写路径要处理：

```text
PDF tables
HTML tables
CSV / Excel
structured financial datasets
```

### 7.1 PDF 表格的难点

```text
- 表头跨行
- 年份列错位
- 单位写在表格上方
- 括号表示负数
- 空白表示 0 或 not applicable
- 多页表格
- row label 换行
```

### 7.2 Table normalization

必须处理：

```text
$1.577 billion
1.577 billion dollars
$1,577 million
1,577
(1,577)
—
N/A
```

归一化字段：

```text
raw_value
numeric_value
sign
scale
currency
unit
period
```

---

## 8. Query Orchestrator 对 V4 的支持

V1 QueryPlan 要能路由到 SQLProvider。

示例输入：

```text
What was 3M's free cash flow in FY2018?
```

如果 `free_cash_flow` 没有直接值，则 QueryPlan 可以生成：

```yaml
query_type: calculation
metrics:
  - operating_cash_flow
  - capital_expenditure
formula:
  name: free_cash_flow
  expression: operating_cash_flow - capital_expenditure
retrieval_units:
  - unit_id: u_sql
    provider: sql
    purpose: structured_calculation
    text: "3M operating cash flow capital expenditure FY2018"
  - unit_id: u_text
    provider: hybrid
    purpose: source_text_support
    text: "3M FY2018 operating cash flow capital expenditure annual report wording"
```

V4 SQLProvider 负责查两个输入值并计算。

---

## 9. Text-to-SQL

### 9.1 输入

```text
- user query
- QueryPlan
- schema registry
- allowed tables
- metric ontology
- security policy
```

### 9.2 输出

```python
class SQLPlan:
    sql: str
    tables: list[str]
    columns: list[str]
    filters: dict
    expected_result_shape: str
    calculation: dict | None
    safety_notes: list[str]
```

### 9.3 SQL Verifier

执行前必须验证：

```text
- SQL 只能读，不允许写。
- 表名/列名必须存在。
- WHERE 条件必须来自 QueryPlan 或 schema。
- 不允许无限制全表扫描，除非 query 类型允许。
- LIMIT 必须存在或由聚合保证结果大小。
- 计算公式必须在 formula registry 中存在。
```

---

## 10. Formula Registry

位置：

```text
configs/finance_formula_registry.yaml
```

示例：

```yaml
free_cash_flow:
  expression: operating_cash_flow - capital_expenditure
  inputs:
    - operating_cash_flow
    - capital_expenditure
  output_unit: currency

operating_margin:
  expression: operating_income / revenue
  inputs:
    - operating_income
    - revenue
  output_unit: percentage

yoy_growth:
  expression: (current_value - prior_value) / prior_value
  inputs:
    - current_value
    - prior_value
  output_unit: percentage
```

### 10.1 Calculator

不要让 LLM 算。

```python
def calculate_free_cash_flow(operating_cash_flow, capital_expenditure):
    return operating_cash_flow - capital_expenditure
```

LLM 只解释，不计算核心数值。

---

## 11. SQL Result EvidenceBlock

SQL 结果必须转 EvidenceBlock。

```python
class SQLResultEvidenceBlock(EvidenceBlock):
    source_type: str = "sql_result"
    sql_query: str
    result_rows: list[dict]
    source_cell_ids: list[str]
    source_document_ids: list[str]
    source_pages: list[int]
    calculation: dict | None
```

示例：

```json
{
  "source_type": "sql_result",
  "provider": "sql",
  "text": "3M FY2018 capital expenditure = 1,577 USD million, from 3M 2018 10-K cash flow table.",
  "sql_query": "SELECT value FROM financial_facts WHERE company='3M' AND fiscal_year=2018 AND metric='capital_expenditure'",
  "citations": [
    {
      "document_id": "3M_2018_10K",
      "page": 60,
      "table_id": "tbl_cash_flow_001",
      "cell_ids": ["cell_123"]
    }
  ]
}
```

---

## 12. SQL + Text Evidence Fusion

很多时候 SQL 给数字，TextHybrid 给解释。

例如：

```text
Why did free cash flow decrease?
```

需要：

```text
SQLProvider:
  operating cash flow, capex, FCF values

TextHybridProvider:
  management discussion explaining cash flow changes
```

Evidence Builder 要能组合：

```text
Numeric EvidenceBlock
+ Text EvidenceBlock
```

---

## 13. API

### 13.1 Structured query debug

```http
POST /v4/sql/query
```

```json
{
  "query": "What was 3M's free cash flow in FY2018?",
  "return_sql": true,
  "return_provenance": true
}
```

### 13.2 Schema inspect

```http
GET /v4/sql/schemas
GET /v4/sql/tables/{table_id}
GET /v4/sql/cells/{cell_id}
```

### 13.3 Provider integration

```python
SQLProvider.retrieve(task: RetrievalTask) -> list[Candidate]
```

---

## 14. Eval

### 14.1 SQL accuracy

```text
text_to_sql_exact_match
execution_accuracy
result_shape_accuracy
filter_accuracy
```

### 14.2 Numeric accuracy

```text
numeric_exact_hit
numeric_tolerance_hit
unit_correctness
scale_correctness
formula_correctness
```

### 14.3 Provenance accuracy

```text
cell_hit
row_hit
page_hit
document_hit
table_hit
```

### 14.4 Answer quality

```text
answer_normalized_hit
calculation_explanation_correctness
citation_support_rate
```

---

## 15. Implementation Plan

### V4 Phase 1 Ingestion Profile

当前 Phase 1 是 contract / proof，已接入显式 opt-in 的 V4 ingestion profile：

```text
默认 ingestion profile:
  PDF / Markdown / TXT 行为保持不变。
  CSV / XLSX / HTML / HTM 仍返回 unsupported file type。

V4 ingestion profile:
  只在调用方显式传入 ingestion_profile=v4，
  或在当前 /documents/ingest 形态下显式传入 metadata.atlas_ingestion_profile=v4 时启用。
  允许 CSV / XLSX / HTML / HTM 通过 profile gate 进入 ingestion。
  tabular source 不作为普通 TextChunk 发往主文本索引，也不写 legacy raw-row ParentBlock。
  V4 profile 的 indexable chunks 使用独立 v4_qdrant_collection，并保留 metadata namespace。
  duplicate skip 按 ingestion profile 隔离，默认 profile 与 V4 profile 不互相吞。
  service 已接入 StructuredArtifactWriter，raw_artifacts 写完整 StructuredArtifact envelope。
  manifest 可审计；schema_routing_card 不物化为 table_asset。
  schema version 冲突或 unsupported 会 fail-fast；partial failure 不 silent pass。
  service 在 structured artifact 写入后若后续失败，会把 manifest 标记为 orphaned。
```

本阶段边界：

```text
SQLProvider V1 只是受控单表 Text-to-SQL proof，默认关闭。
不声明完整 Text-to-SQL 质量。
不实现 calculator / formula runtime。
不声明多表 DuckDB query capability；DuckDB 仍是受控 derived index / structured query 层。
CSV / XLSX / HTML / HTM 只作为 V4 profile 下的 ingestion 输入，不等于结构化数值问答已可用。
tabular profile intake 不产生可用于答案的 raw row TextChunk。
Excel/PDF advanced adapter 不是 Phase 1 完成能力。
```

### V4.0 Structured Contract

```text
- SQLProvider interface
- SQLResultEvidenceBlock
- TableAsset / Cell / FinancialFact models
```

### V4.1 Table Extraction

```text
- PDF table extraction pipeline
- table metadata
- cell provenance
- table searchable text
```

### V4.2 Structured Store

```text
- DuckDB integration
- schema registry
- financial_facts table
- provenance tables
```

### V4.3 Text-to-SQL

```text
- SQLPlan schema
- structured prompt
- SQL verifier
- read-only execution
```

### V4.4 Calculator

```text
- formula registry
- finance calculator
- numeric normalization
- tolerance rules
```

### V4.5 Evidence Integration

```text
- SQL result -> EvidenceBlock
- SQL + Text evidence fusion
- V2 research integration
```

---

## 16. Definition of Done

V4 完成时必须满足：

```text
- 能抽取表格并保留 cell provenance。
- 能把财务指标写入 structured store。
- SQLProvider 能接收 RetrievalTask。
- Text-to-SQL 输出可验证 SQLPlan。
- SQL Verifier 能阻止不安全或不合法 SQL。
- Calculator 能处理核心财务公式。
- SQL result 能转成 EvidenceBlock。
- 答案中的数字能追溯到 table/cell/page/document。
```

---

## 17. V4 结论

V4 的核心不是“加一个数据库”。

V4 的核心是：

```text
让结构化数值事实成为可计算、可验证、可引用的证据来源。
```

SQLProvider 必须和 TextHybridProvider、GraphProvider 一样，通过统一 Evidence Kernel 输出。
