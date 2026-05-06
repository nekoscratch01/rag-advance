def build_query_planner_instructions(enabled_providers: tuple[str, ...]) -> str:
    provider_list = ", ".join(enabled_providers)
    return f"""You produce conservative retrieval plans for Atlas V1.

Rules:
- Available retriever providers for this runtime: [{provider_list}].
- The only executable V1 provider is usually `hybrid`. If SQL or Graph are not listed,
  do not output them. Use `hybrid` to retrieve financial text, serialized tables, and
  source wording. If evidence is insufficient, the answer stage must say so.
- `sql` and `graph` are future provider names, not internal hybrid lanes.
- Never output internal lanes such as dense, bm25, table, metric_alias, or section as retrievers.
- Every retrieval unit must have exactly one retriever provider. Compound units like
  [sql, hybrid] are forbidden; split them into separate single-purpose unit_proposals.
- Use metadata_filter for document, filing, section, page, or table constraints.
- Never output `filters`; the V1 provider contract accepts `metadata_filter` only.
- Only extract companies, periods, metrics, and metadata_filter values grounded in the user query.
- Do not invent companies, years, filing types, metrics, or table fields.
- Use metric aliases only when they are present in the supplied ontology excerpt.
- Keep retrieval_units small and targeted.
- Never use HyDE or Query2Doc for hybrid sparse retrieval.
- Return JSON only.
"""


QUERY_PLANNER_INSTRUCTIONS = build_query_planner_instructions(("hybrid",))


def build_query_planner_input(
    query: str,
    ontology_excerpt: str,
    max_units: int,
    *,
    validation_feedback: str | None = None,
) -> str:
    feedback_section = []
    if validation_feedback:
        feedback_section = [
            "",
            "Previous plan validation error:",
            validation_feedback,
            "",
            "Revise the plan to satisfy the schema and validation rules.",
        ]
    return "\n".join(
        [
            "User query:",
            query,
            "",
            "Finance metric ontology excerpt:",
            ontology_excerpt,
            "",
            f"Maximum retrieval units: {max_units}",
            *feedback_section,
            "",
            "Return a query plan JSON object matching the schema.",
        ]
    )
