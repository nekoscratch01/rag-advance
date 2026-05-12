# Atlas Evidence Kernel

Atlas is an evidence-first, traceable, provider-based RAG / Evidence Kernel: it turns questions into auditable evidence flows, then asks the answer model to cite only what it actually used.

```text
User Query
  -> Query Orchestrator -> QueryPlan -> RetrievalTask
  -> ProviderRouter
       hybrid -> dense + BM25 + Weighted RRF
       graph  -> V3 local/path graph grounding -> chunks.text
       sql    -> V4 opt-in single-table SQL result evidence
  -> Candidate Fusion / Rerank
  -> EvidencePack -> EvidenceEvaluator
  -> Answer LLM -> Citation / Trace
```

## Current Status

| Area | Status |
|---|---|
| V1 hybrid baseline | Implemented. FinanceBench retrieval-only eval is archived at `benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506/`; best archived path is `hybrid_rrf_reranker` with doc@10 `0.813`, page@10 `0.267`, and 0 errors. |
| V3 GraphProvider | Walking skeleton implemented. `graph` is executable by default, but only runs when `QueryPlan` selects graph. This proves provider contract, Postgres grounding, and trace auditability, not GraphRAG quality lift. |
| V4 SQLProvider V1 | Opt-in proof implemented. It is a controlled single-table Text-to-SQL / SQL result evidence path, disabled by default. |
| Tri-provider full stack | Synthetic acceptance passed. Final artifact: `benchmarks/system_acceptance/tri_provider_full_stack/tri_provider_20260511T075338Z_unknown/`. |
| V2 Research Runtime | Not implemented. Redis Queue, research jobs, planner/report generator quality, and long-running research runtime remain design-stage work. |

Runtime storage is Postgres + Qdrant. FinanceBench JSONL files are frozen benchmark artifacts, not runtime storage. V1 cache uses Postgres `query_cache`, not Redis.

## Full-Stack Acceptance

Latest acceptance proves this chain closes end to end:

```text
Query Orchestrator -> QueryPlan -> RetrievalTask -> ProviderRouter
  -> hybrid / graph / sql -> Candidate -> Fusion / Rerank
  -> EvidencePack -> EvidenceEvaluator -> Answer LLM -> Citation / Trace
```

Artifact:

```text
benchmarks/system_acceptance/tri_provider_full_stack/tri_provider_20260511T075338Z_unknown/
```

What it proves:

- Real OpenAI calls were used for planner, SQL compiler, and answer generation.
- Planner, SQL compiler, and answer LLM used `gpt-5-nano` with `reasoning.effort=low`.
- Seven provider ablations behaved as expected: six failed on the corresponding missing evidence branch; `hybrid+graph+sql` was supported.
- SQL computed the gold revenue value `123456`.
- The final answer included citations for SQL result evidence and supplier dependency evidence.
- Secret scan passed.

What it does not claim:

- Not a FinanceBench answer benchmark.
- Not a GraphRAG retrieval eval.
- Not a broad Text-to-SQL benchmark.
- Not proof of multi-table SQL.
- Not proof of generated-answer reliability outside the synthetic fixture.
- Not Postgres/Qdrant live structured ingestion proof; the structured table mode is synthetic.

## Quickstart

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional SQL proof dependencies:

```bash
python -m pip install -e ".[structured-sql]"
```

Start Postgres and Qdrant:

```bash
docker compose up -d
```

Create local env:

```bash
cp .env.example .env
```

Set only your local `.env`:

```env
OPENAI_API_KEY=...
```

Run the API:

```bash
uvicorn atlas.main:app --reload
curl -s http://localhost:8000/v1/health | python -m json.tool
```

Ingest sample documents:

```bash
curl -s http://localhost:8000/v1/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "paths": [
      "samples/v1_evidence_kernel.md",
      "samples/demo_knowledge.md",
      "samples/atlas_notes.txt"
    ],
    "metadata": {"source_type": "sample"}
  }' | python -m json.tool
```

Query:

```bash
curl -s http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Atlas V1 Evidence Kernel 的 runtime 主路径是什么？",
    "top_k": 8
  }' | python -m json.tool
```

Trace:

```bash
curl -s http://localhost:8000/v1/query/{query_id}/trace | python -m json.tool
```

## Provider Config

Default runtime:

```bash
ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid,graph
```

Hybrid-only V1 baseline:

```bash
ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid
```

SQLProvider V1 opt-in:

```bash
ATLAS_SQL_PROVIDER_ENABLED=true
ATLAS_QUERY_RUNTIME_EXECUTABLE_PROVIDERS=hybrid,sql,graph
```

SQLProvider V1 is intentionally narrow: single-table controlled `SELECT`, deterministic SQL result evidence, no multi-table SQL claim, and no generated-answer reliability claim by itself.

## Verification

Syntax check:

```bash
python -m compileall -q src/atlas scripts
```

Smoke eval after sample ingestion:

```bash
python -m atlas.eval.runner \
  --base-url http://localhost:8000 \
  --cases evals/smoke_cases.yaml
```

FinanceBench retrieval-only eval:

```bash
python scripts/prepare_financebench.py \
  --out corpus/financebench \
  --evals evals/financebench_cases.yaml \
  --strict

ATLAS_RETRIEVAL_MODE=hybrid \
ATLAS_BM25_ENABLED=true \
ATLAS_QDRANT_COLLECTION=atlas_financebench_v1 \
python scripts/ingest_financebench.py \
  --corpus corpus/financebench \
  --batch-size 64

ATLAS_RETRIEVAL_MODE=hybrid \
ATLAS_BM25_ENABLED=true \
ATLAS_QDRANT_COLLECTION=atlas_financebench_v1 \
python -m atlas.benchmark.financebench_retrieval \
  --cases evals/financebench_cases.yaml \
  --modes dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker \
  --top-k 10
```

Tri-provider live synthetic acceptance:

```bash
ATLAS_RUN_LIVE_ACCEPTANCE=1 \
OPENAI_API_KEY=... \
python scripts/tri_provider_acceptance.py --combo hybrid+graph+sql
```

## Repo Layout

```text
src/atlas/       API, query orchestration/runtime, retrieval providers, ingestion, DB, vector helpers
docs/            Documentation map, design intent, milestones, actual architecture records
benchmarks/      Retrieval quality reports and full-stack acceptance artifacts
samples/         Safe sample documents
evals/           Smoke and FinanceBench case definitions
scripts/         Ingestion and acceptance utilities
```

## Safety

- Do not write secrets into source, docs, traces, benchmark artifacts, or Docker images.
- README examples use only `OPENAI_API_KEY=...`.
- Local BGE embedding keeps embedding local, but answer generation sends selected evidence to the configured LLM provider.
- `ATLAS_DOCUMENT_ROOTS` is the file-read boundary; default roots are `samples,corpus`.
- Retrieval-only metrics do not prove answer reliability.

Start with `docs/README.md` for the documentation map. Use `docs/exec-plans/version-arch/` for implementation facts and `benchmarks/` for quality claims.
