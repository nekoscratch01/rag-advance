# V1 Hybrid Provider Reset 离线 Smoke

- 运行 ID：`smoke_20260506`
- 生成时间：`2026-05-06T07:21:19.013943+00:00`
- 案例数：`100`
- 变体数：`20`

## 分组结果

### query_rewrite

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | MAP page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rewrite_unit_text` | 100 | 0 | 0.850 | 1.000 | 1.000 | 0.925 | 0.925 | - |
| `rewrite_should_terms` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `rewrite_ontology_aliases` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |

### filter_strategy

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | MAP page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `filter_no_hard_filter` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `filter_metadata_only` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `filter_must_have_hard` | 100 | 0 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | answer_terms_miss@3:20, page_miss@3:20 |
| `filter_must_terms_sparse_boost` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |

### fusion

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | MAP page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fusion_dense_only` | 100 | 0 | 0.820 | 1.000 | 1.000 | 0.910 | 0.910 | - |
| `fusion_sparse_only` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `fusion_python_weighted_rrf` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `fusion_qdrant_rrf_planned` | 0 | 100 | - | - | - | - | - | planned_not_run:100 |

### candidate_shape

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | MAP page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `shape_child_chunk` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `shape_parent_block` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `shape_page_neighborhood` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `shape_token_budget_18` | 100 | 0 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | answer_terms_miss@3:40, expected_evidence_dropped_by_token_budget:40, page_miss@3:40 |

### reranker_input

| 变体 | 已完成 | 计划项 | page@1 | page@3 | answer_terms@3 | MRR page | MAP page | 失败桶 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rerank_original_query_candidate` | 100 | 0 | 0.850 | 1.000 | 1.000 | 0.925 | 0.925 | - |
| `rerank_current_unit_candidate` | 100 | 0 | 0.850 | 1.000 | 1.000 | 0.925 | 0.925 | - |
| `rerank_local_terms_candidate` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| `rerank_full_plan_summary_candidate` | 100 | 0 | 0.850 | 1.000 | 1.000 | 0.925 | 0.925 | - |
| `rerank_full_plan_all_units_candidate` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |

## 说明

filter_must_terms_sparse_boost 使用 repeat(must_have_terms, 3) 的 sparse input 重复词策略，不改 BM25 底层公式。

本报告补充 MAP_doc / MAP_page / MAP_answer_terms，但 synthetic case 每条只有一个预期 evidence 目标，所以 MAP 在这里主要等价于“正确证据排得有多靠前”；完整 MAP 结论仍要看 FinanceBench 150 条真实 eval。

这是离线 synthetic smoke，用来检查消融维度、trace 字段和排序/过滤取舍是否可解释；它不是 FinanceBench 全量质量结论，也不是 generated-answer reliability 结论。
