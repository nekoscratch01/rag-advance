QUERY_PLANNER_INSTRUCTIONS = """You produce conservative retrieval plans for Atlas V1.

Rules:
- Only extract companies, periods, metrics, and filters grounded in the user query.
- Do not invent companies, years, filing types, metrics, or table fields.
- Use metric aliases only when they are present in the supplied ontology excerpt.
- Keep retrieval_units small and targeted.
- Never use HyDE or Query2Doc for BM25 exact retrieval.
- Return JSON only.
"""


def build_query_planner_input(query: str, ontology_excerpt: str, max_units: int) -> str:
    return "\n".join(
        [
            "User query:",
            query,
            "",
            "Finance metric ontology excerpt:",
            ontology_excerpt,
            "",
            f"Maximum retrieval units: {max_units}",
            "",
            "Return a query plan JSON object matching the schema.",
        ]
    )
