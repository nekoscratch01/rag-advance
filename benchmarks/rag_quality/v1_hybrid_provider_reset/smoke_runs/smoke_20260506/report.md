# V1 Hybrid Provider Reset 离线 Smoke

- 运行 ID：`smoke_20260506`
- 生成时间：`2026-05-06T05:42:44.227744+00:00`
- 案例数：`5`
- 变体数：`20`

## 分组结果

### query_rewrite

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rewrite_unit_text` | 5 | 0 | 0.800 | 1.000 | 1.000 | 0.900 | - |
| `rewrite_should_terms` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `rewrite_ontology_aliases` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |

### filter_strategy

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `filter_no_hard_filter` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `filter_metadata_only` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `filter_must_have_hard` | 5 | 0 | 0.800 | 0.800 | 0.800 | 0.800 | answer_terms_miss@3:1, page_miss@3:1 |
| `filter_must_terms_sparse_boost` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |

### fusion

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fusion_dense_only` | 5 | 0 | 0.800 | 1.000 | 1.000 | 0.900 | - |
| `fusion_sparse_only` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `fusion_python_weighted_rrf` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `fusion_qdrant_rrf_planned` | 0 | 5 | - | - | - | - | planned_not_run:5 |

### candidate_shape

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `shape_child_chunk` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `shape_parent_block` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `shape_page_neighborhood` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `shape_token_budget_18` | 5 | 0 | 0.600 | 0.600 | 0.600 | 0.600 | answer_terms_miss@3:2, expected_evidence_dropped_by_token_budget:2, page_miss@3:2 |

### reranker_input

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rerank_original_query_candidate` | 5 | 0 | 0.800 | 1.000 | 1.000 | 0.900 | - |
| `rerank_current_unit_candidate` | 5 | 0 | 0.800 | 1.000 | 1.000 | 0.900 | - |
| `rerank_local_terms_candidate` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `rerank_full_plan_summary_candidate` | 5 | 0 | 0.800 | 1.000 | 1.000 | 0.900 | - |
| `rerank_full_plan_all_units_candidate` | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | - |

## 说明

filter_must_terms_sparse_boost 使用 repeat(must_have_terms, 3) 的 sparse input 重复词策略，不改 BM25 底层公式。

这是离线 synthetic smoke，用来检查消融维度、trace 字段和排序/过滤取舍是否可解释；它不是 FinanceBench 全量质量结论，也不是 generated-answer reliability 结论。
