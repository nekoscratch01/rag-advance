import json

from atlas.retrieval.models.evidence import Evidence


ANSWER_INSTRUCTIONS = """你是 Atlas Evidence Kernel 的证据优先回答生成器。

必须遵守：
1. 只能使用用户提供的 evidence。
2. 如果 evidence 不足，confidence 必须是 "insufficient"，answer 必须明确说明证据不足。
3. 不要使用 evidence 外的信息。
4. 事实性结论必须带 citation marker，例如 [c1]、[c2]。
5. 不要引用无法支撑该结论的 evidence。
6. 回答要简洁。

证据是否足够的判断规则：
- 如果 evidence 中有直接句子或明确同义表述可以回答用户问题，必须回答，confidence 使用 "supported"。
- 不要因为 evidence 是片段、样本文档或不是完整报告而拒答。
- 只有在 evidence 为空、明显无关、互相冲突且无法判断，或缺少回答所需的关键事实时，才使用 "insufficient" 或 "conflicted"。
- 如果回答是“证据不足”，也要引用你用来判断不足的相关 evidence。

你必须只输出 JSON，不要输出 Markdown，不要包代码块。JSON schema：
{
  "confidence": "supported | insufficient | conflicted",
  "answer": "回答正文，必要时包含 [c1] 这样的引用"
}
"""


def build_answer_input(*, query: str, evidence: list[Evidence]) -> str:
    evidence_blocks = []
    for item in evidence:
        evidence_blocks.append(
            {
                "citation_id": item.evidence_id,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "source_title": item.source_title,
                "source_uri": item.source_uri,
                "section_title": item.section_title,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "retrieval_score": item.retrieval_score,
                "text": item.text,
            }
        )
    return json.dumps(
        {
            "user_query": query,
            "evidence": evidence_blocks,
        },
        ensure_ascii=False,
        indent=2,
    )
