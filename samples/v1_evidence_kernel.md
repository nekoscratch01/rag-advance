# Atlas V1 Evidence Kernel 示例知识库

## V1 当前主路径

Atlas V1 Evidence Kernel 的当前 runtime 主路径是 Hybrid Provider。系统会先把用户问题
转换成 semantic QueryPlan，再编译成 RetrievalTask，并交给 ProviderRouter 执行。
V1 runtime 只注册并执行 `hybrid` provider；如果计划里出现 `sql` 或 `graph`，它们会被
记录为 skipped trace，不会伪装成 hybrid 证据。

TextHybridProvider 是 V1 hybrid provider 的实现名。它在内部使用 dense、BM25、table
textual lane，并用 provider-local fusion 合并候选。Dense、BM25 和 table 不是顶层
planner provider。

## 证据和引用

V1 会把候选转换成 EvidencePack，再执行 evidence evaluation、answer generation 和
citation verification。答案里的 citation 必须来自被选中的 evidence，例如 `[c1]`。
系统不会在模型漏写 citation marker 时自动补 citation。

## 可观测性

每次查询都会记录 query plan、retrieval task、provider result、candidate trace、
evidence block、evidence pack、answer、citation 和 citation verification。`source_anchor`
用于把证据指回 document、chunk、parent block 和 page；V1 只承诺文本、页码和序列化表格
文本 provenance，不承诺 SQL cell provenance 或 GraphRAG provenance。
