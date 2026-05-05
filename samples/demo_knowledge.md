# Atlas V0.0 示例知识库

## V0.0 目标

Atlas V0.0 的目标是跑通一个证据优先的 RAG 内核。系统应该能够导入
PDF、Markdown 或 TXT 文档，把文档切成 chunks，使用本地 BGE 模型生成
embeddings，然后把文本元数据写入 Postgres，把向量写入 Qdrant。PDF 会按页
提取文本，并把页码写入 chunk metadata，方便后续 citation 和 trace 复查。

用户查询时，系统先检索相关 chunks，再把选中的 evidence 交给 gpt-5-nano
生成答案。答案必须带 citation，并返回 trace_id，方便工程师复查系统为什么
这样回答。

## 证据原则

Atlas 不应该在证据不足时编造答案。如果当前导入的文档不能支持用户问题，
系统应该明确说明证据不足。V0.0 只承诺 chunk-level citation，不承诺
claim-level citation。

## V0.0 不做什么

V0.0 不做前端、不做 Hybrid Retrieval、不做 Reranker、不做 Research Job，
也不做 GraphRAG。那些能力会在后续版本中逐步加入。
