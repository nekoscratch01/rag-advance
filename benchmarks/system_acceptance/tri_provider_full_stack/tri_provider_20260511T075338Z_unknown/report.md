# Atlas Tri-Provider Full-Stack Acceptance

- Run ID: `tri_provider_20260511T075338Z_unknown`
- Generated: `2026-05-11T07:56:01.167321+00:00`
- Success: `True`
- Artifact dir: `/Users/neko_wen/my/代码/project/rag-advance/benchmarks/system_acceptance/tri_provider_full_stack/tri_provider_20260511T075338Z_unknown`
- Acceptance query: `For Acme Robotics' FY2024 Vision Sensor launch, use three independent evidence sources: a hybrid text evidence branch for the exact launch constraint wording; a graph relationship context branch for the Acme Robotics -> PhotonWorks supplier dependency relationship; and a SQL revenue table branch for the FY2024 Vision Sensor total revenue. What total revenue does the table report, and what supplier dependency risk explains the launch constraint?`

## Scope / Non-Claims

- Scope: synthetic fixture acceptance for contract wiring, provider isolation, evidence coverage, and citation trace behavior.
- Non-claims: this is not a FinanceBench benchmark, GraphRAG retrieval eval, Text-to-SQL benchmark, multi-table SQL proof, or general answer reliability proof.
- Structured table mode: `synthetic_structured_table_fixture`; this report does not claim Postgres/Qdrant live ingestion proof.
- Canonical full provider order: `hybrid+graph+sql`.

## Combos

| Combo | Executable providers | Expected failures | Observed failures | Passed |
| --- | --- | --- | --- | --- |
| `hybrid` | `hybrid` | `missing_sql_result,missing_graph_relationship_context` | `missing_sql_result,missing_graph_relationship_context` | `True` |
| `graph` | `graph` | `missing_sql_result,missing_hybrid_text_coverage` | `missing_sql_result,missing_hybrid_text_coverage` | `True` |
| `sql` | `sql` | `missing_supplier_risk_text_evidence` | `missing_supplier_risk_text_evidence` | `True` |
| `hybrid+graph` | `hybrid,graph` | `missing_sql_result` | `missing_sql_result` | `True` |
| `hybrid+sql` | `hybrid,sql` | `missing_graph_relationship_provenance` | `missing_graph_relationship_provenance` | `True` |
| `graph+sql` | `graph,sql` | `missing_hybrid_text_coverage` | `missing_hybrid_text_coverage` | `True` |
| `hybrid+graph+sql` | `hybrid,graph,sql` | `none` | `none` | `True` |

## Secret Scan

- Status: `passed`
