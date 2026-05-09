def build_query_planner_instructions(
    known_providers: tuple[str, ...],
    executable_providers: tuple[str, ...] = ("hybrid", "graph"),
) -> str:
    known_provider_list = ", ".join(known_providers)
    executable_provider_list = ", ".join(executable_providers)
    return f"""You produce conservative retrieval plans for Atlas.

Rules:
- Known retrieval providers in the planner ontology: [{known_provider_list}].
- Executable providers in the current runtime: [{executable_provider_list}].
- Planning is semantic: output the provider that best matches the user's intent even if
  the current runtime cannot execute that provider yet. The runtime executes only registered providers and
  records non-executable providers as skipped trace entries.
- Do not disguise sql or graph intent as hybrid. Runtime capability must not pollute
  the semantic plan.
- Do not add a graph unit only because graph is executable; use graph only when the
  query needs relationship, path, or neighborhood context.
- `sql` and `graph` are provider names, not internal hybrid lanes.
- Never output internal lanes such as dense, bm25, table, metric_alias, or section as providers.
- Every retrieval unit must have exactly one provider. Compound units like
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


QUERY_PLANNER_INSTRUCTIONS = build_query_planner_instructions(
    ("hybrid", "sql", "graph"),
    ("hybrid", "graph"),
)


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
