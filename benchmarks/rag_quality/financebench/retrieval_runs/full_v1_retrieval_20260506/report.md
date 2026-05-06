# FinanceBench Retrieval-only Ablation

- Run ID: `full_v1_retrieval_20260506`
- Generated: `2026-05-06T07:22:55.103863+00:00`
- Cases: `evals/financebench_cases.yaml`
- Collection: `atlas_financebench_v1`
- Embedding: `BAAI/bge-small-zh-v1.5`
- BM25: `Qdrant/bm25`
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`

## Metrics

| Mode | n | doc@1 | doc@3 | doc@5 | doc@10 | page@1 | page@3 | page@5 | page@10 | MRR doc | MRR page | MAP doc | MAP page | p50 ms | p95 ms | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_only` | 150 | 20/150 (0.133) | 44/150 (0.293) | 53/150 (0.353) | 70/150 (0.467) | 9/150 (0.060) | 14/150 (0.093) | 16/150 (0.107) | 19/150 (0.127) | 0.233 | 0.081 | 0.208 | 0.079 | 33 | 67 | 0 |
| `bm25_only` | 150 | 48/150 (0.320) | 76/150 (0.507) | 93/150 (0.620) | 118/150 (0.787) | 11/150 (0.073) | 21/150 (0.140) | 23/150 (0.153) | 31/150 (0.207) | 0.448 | 0.113 | 0.404 | 0.112 | 6 | 10 | 0 |
| `hybrid_rrf` | 150 | 38/150 (0.253) | 73/150 (0.487) | 89/150 (0.593) | 109/150 (0.727) | 12/150 (0.080) | 18/150 (0.120) | 25/150 (0.167) | 32/150 (0.213) | 0.398 | 0.112 | 0.343 | 0.105 | 47 | 61 | 0 |
| `hybrid_rrf_reranker` | 150 | 56/150 (0.373) | 95/150 (0.633) | 111/150 (0.740) | 122/150 (0.813) | 15/150 (0.100) | 26/150 (0.173) | 32/150 (0.213) | 40/150 (0.267) | 0.520 | 0.146 | 0.460 | 0.139 | 502 | 690 | 0 |

## Failure Buckets

| Bucket | Count |
| --- | ---: |
| `dense_missed_bm25_hit` | 20 |
| `hybrid_found_reranker_lost` | 3 |
| `reranker_improved_page_rank` | 24 |
| `both_dense_and_hybrid_missed` | 115 |
| `errors` | 0 |
