# Atlas V0.0 / V0.1

Atlas V0 是一个 API-first 的证据优先 RAG 内核。它做一条可追踪、可引用、可评估的 RAG 闭环：

```text
PDF / Markdown / TXT
  -> 本地 BGE embedding
  -> Postgres + Qdrant
  -> FastAPI 查询
  -> gpt-5-nano 基于 evidence 生成答案
  -> citations + trace_id
```

## 运行方式

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

启动 Postgres 和 Qdrant：

```bash
docker compose up -d
```

本项目的 Docker Compose 会把 Postgres 映射到本机 `15432`，避免和本机已有
Postgres 的 `5432` 冲突。

准备本地环境变量：

```bash
cp .env.example .env
```

然后在 `.env` 里填入：

```env
OPENAI_API_KEY=...
```

启动 API：

```bash
uvicorn atlas.main:app --reload
```

## 导入文档

```bash
curl -s http://localhost:8000/v1/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "paths": ["samples/demo_knowledge.md", "samples/atlas_notes.txt"],
    "metadata": {"source_type": "sample"}
  }' | python -m json.tool
```

第一次导入会下载并加载 `BAAI/bge-small-zh-v1.5`，可能需要一些时间。

`/documents/ingest` 只允许读取 `ATLAS_DOCUMENT_ROOTS` 里的文件，默认是
`samples,corpus`。真实知识库文档建议放进 `corpus/`，不要把任意本机路径暴露给
API。

V0 支持：

```text
PDF
Markdown
TXT
```

PDF 会按页提取文本，并把 `page_start` / `page_end` 写入 chunks、Qdrant payload、
citations 和 trace。

## 查询

```bash
curl -s http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Atlas V0.0 的目标是什么？",
    "top_k": 8
  }' | python -m json.tool
```

返回会包含：

```text
answer
confidence
citations
trace_id
```

Citation 只来自模型答案里实际写出的 `[c1]`、`[c2]` marker。系统不会在模型漏写
marker 时自动把 top-k evidence 补成 citations。

V0.0 默认对 GPT-5 nano 使用 `reasoning.effort=low`。这是为了避免默认 reasoning
消耗过多 `max_output_tokens`，同时比 `minimal` 更稳一点。

查看 query trace：

```bash
curl -s http://localhost:8000/v1/query/{query_id} | python -m json.tool
```

查看结构化 trace：

```bash
curl -s http://localhost:8000/v1/query/{query_id}/trace | python -m json.tool
```

结构化 trace 会返回：

```text
query       query_id、trace_id、用户问题
result      answer、confidence、citations
retrieval   top_k chunks、rank、score、source、section、preview
generation  model、prompt_version、input_tokens、output_tokens、latency
latency     total_latency_ms、generation_latency_ms
```

如果 Qdrant retrieval 失败，系统会返回标准错误，并尽量落一条带 `trace_id` 的
`query_runs` 失败记录，方便复查。

查看系统观测摘要：

```bash
curl -s http://localhost:8000/v1/observability/summary | python -m json.tool
```

它会汇总：

```text
storage      documents、chunks、ingestion_runs
queries      query_runs、retrieval_events、confidence 分布、最近 10 次查询
generation   generation_events、completed/failed、token 总量、model 分布
latency      query 和 generation 的 avg/min/max
```

## Smoke Eval

导入样本文档后运行：

```bash
python -m atlas.eval.runner --base-url http://localhost:8000 --cases evals/smoke_cases.yaml
```

正式 V0 eval case 文件：

```bash
python -m atlas.eval.runner --base-url http://localhost:8000 --cases evals/v0_cases.yaml
```

Eval report 会输出：

```text
Citation source hit
Confidence hit
Average keyword score
Average latency
Total input/output tokens
Top failures
```

V0.1 的 eval 仍然是 smoke eval，不是完整 faithfulness judge。它的目的主要是确认
每次改动后，检索、引用、confidence 和生成成本没有明显退化。

## Eval API

运行 eval 并把结果写入 Postgres：

```bash
curl -s http://localhost:8000/v1/eval/run \
  -H 'Content-Type: application/json' \
  -d '{
    "cases_path": "evals/smoke_cases.yaml",
    "top_k": 8
  }' | python -m json.tool
```

查看 eval run：

```bash
curl -s http://localhost:8000/v1/eval/{eval_run_id} | python -m json.tool
```

查看最近 eval runs：

```bash
curl -s http://localhost:8000/v1/eval | python -m json.tool
```

## DBeaver 查看 Postgres

Postgres 运行在 Docker 容器里，通过本机端口 `15432` 暴露给 DBeaver。

连接参数：

```text
Host: localhost
Port: 15432
Database: atlas
Username: atlas
Password: atlas
```

重点表：

```text
documents          原始文档记录
chunks             检索用文本块
ingestion_runs     导入任务记录
query_runs         一次 query 的总记录
retrieval_events   query 召回了哪些 chunk
generation_events  LLM 调用、token、latency、status
eval_runs          eval 任务汇总
eval_results       eval 单 case 结果
```

常用 SQL：

```sql
select document_id, title, file_type, source_uri, created_at
from documents;

select chunk_id, document_id, chunk_index, section_title, token_count, left(text, 120) as preview
from chunks
order by document_id, chunk_index;

select query_id, trace_id, confidence, latency_ms, created_at
from query_runs
order by created_at desc;

select
  q.query_id,
  q.trace_id,
  q.confidence,
  q.latency_ms as total_latency_ms,
  g.model_name,
  g.input_tokens,
  g.output_tokens,
  g.latency_ms as generation_latency_ms,
  g.status
from query_runs q
left join generation_events g on g.query_id = q.query_id
order by q.created_at desc;
```

## Qdrant 查看向量库

Qdrant dashboard：

```text
http://localhost:6333/dashboard
```

查看 collection：

```bash
curl -s http://localhost:6333/collections | python -m json.tool
```

查看向量点数量和配置：

```bash
curl -s http://localhost:6333/collections/atlas_chunks_bge_small_zh_v1_5 | python -m json.tool
```

查看 payload 示例：

```bash
curl -s -X POST \
  http://localhost:6333/collections/atlas_chunks_bge_small_zh_v1_5/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{"limit": 2, "with_payload": true, "with_vector": false}' \
  | python -m json.tool
```

## V1 FinanceBench Hybrid 闭环

V1 的执行记录和实际架构在：

```text
docs/exec-plans/milestone/v1_advanced_hybrid_kernel_milestone.md
docs/exec-plans/version-arch/v1_advanced_hybrid_kernel_arch.md
```

准备 FinanceBench frozen corpus：

```bash
python scripts/prepare_financebench.py \
  --out corpus/financebench \
  --evals evals/financebench_cases.yaml \
  --strict
```

导入 parent/child artifacts 到 V1 hybrid collection：

```bash
ATLAS_RETRIEVAL_MODE=hybrid \
ATLAS_BM25_ENABLED=true \
ATLAS_QDRANT_COLLECTION=atlas_financebench_v1 \
python scripts/ingest_financebench.py \
  --corpus corpus/financebench \
  --batch-size 64
```

启动 V1 查询服务：

```bash
ATLAS_RETRIEVAL_MODE=hybrid \
ATLAS_BM25_ENABLED=true \
ATLAS_RERANKER_ENABLED=true \
ATLAS_QDRANT_COLLECTION=atlas_financebench_v1 \
uvicorn atlas.main:app --reload
```

运行质量对比 benchmark：

```bash
python -m atlas.benchmark.financebench \
  --cases evals/financebench_cases.yaml \
  --modes dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker \
  --cache-policy off \
  --warm-cache
```

只评估召回质量，不调用 LLM：

```bash
ATLAS_RETRIEVAL_MODE=hybrid \
ATLAS_BM25_ENABLED=true \
ATLAS_QDRANT_COLLECTION=atlas_financebench_v1 \
python -m atlas.benchmark.financebench_retrieval \
  --cases evals/financebench_cases.yaml \
  --modes dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker
```

生成式 benchmark 会输出到：

```text
benchmarks/rag_quality/financebench/runs/<run_id>/
```

retrieval-only benchmark 会输出到：

```text
benchmarks/rag_quality/financebench/retrieval_runs/<run_id>/
```

## V0.1 验收流程

```text
1. docker compose up -d
2. uvicorn atlas.main:app --reload
3. GET /v1/health 返回 ok
4. POST /v1/documents/ingest 导入 samples
5. POST /v1/query 返回 answer、confidence、citations、trace_id
6. GET /v1/query/{query_id}/trace 能看到 retrieval 和 generation
7. GET /v1/observability/summary 能看到系统摘要
8. DBeaver 能看到 query_runs、retrieval_events、generation_events
9. python -m atlas.eval.runner 输出 smoke eval report
```

## 开发数据重置

清理 query/eval trace，但保留 documents/chunks：

```bash
curl -s http://localhost:8000/v1/admin/reset-dev-data \
  -H 'Content-Type: application/json' \
  -d '{"scope": "traces", "confirm": "RESET_DEV_DATA"}' \
  | python -m json.tool
```

清理全部开发数据，并重建 Qdrant collection：

```bash
curl -s http://localhost:8000/v1/admin/reset-dev-data \
  -H 'Content-Type: application/json' \
  -d '{"scope": "all", "confirm": "RESET_DEV_DATA"}' \
  | python -m json.tool
```

这个接口只用于本地开发，不是生产管理接口。

## 安全边界

`OPENAI_API_KEY` 只从本机环境变量读取，不写入源码、日志、README 示例、Docker image 或 trace。

本地 BGE embedding 不会把文档发给云端 embedding 服务。但查询时，被检索出来并进入
`ATLAS_MAX_CONTEXT_TOKENS` 预算内的 evidence 会发送给 OpenAI，用于 `gpt-5-nano`
生成答案。V0.0 只应该导入允许发送给 OpenAI 的样本文档或非敏感文档。

`ATLAS_DOCUMENT_ROOTS` 是本地文件读取边界。默认只允许导入 `samples/` 和
`corpus/` 下的 PDF、Markdown、TXT。

## V0 baseline 不做

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
```
