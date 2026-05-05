# Atlas Markdown Blueprint

这组文档把 Atlas 的实现路线拆成几个可下载的 Markdown 文件，方便单独阅读和后续继续改写。

## 文档列表

1. `00_overview.md`
   总览文档。说明 Atlas 的系统目标、分阶段实现逻辑、核心数据对象、前三版技术栈和关键调整建议。

2. `01_v0_atlas_kernel.md`
   V0 详细蓝图。描述最小可用 RAG 内核，包括 FastAPI Gateway、Ingestion Pipeline、Postgres、Qdrant、Query Runtime、Citation Builder、Trace Logger 和 Evaluation Harness。

3. `02_v1_atlas_hybrid_kernel.md`
   V1 详细蓝图。描述 Hybrid Retrieval 内核，包括 Query Runtime、Query Rewriter、Keyword Retriever、Hybrid Fusion、Reranker、Evidence Builder、Critic Lite、Cache 和 Eval。

4. `03_v2_atlas_research_runtime.md`
   V2 详细蓝图。描述异步研究任务运行时，包括 Research Job API、Job Manager、Worker Pool、Planner、Subquestion Executor、Evidence Consolidator、Critic、Report Writer、Citation Verifier 和 Artifacts。

5. `04_future_versions.md`
   V3+ 后续版本路线图。只保留 Graph Context、Structured Data、Streaming Ingestion、Memory & Skills、MCP Tool Layer、Cloud Production、Event Path 的方向，不展开实现细节。

## 阅读顺序

建议按以下顺序阅读：

```text
00_overview.md
  ↓
01_v0_atlas_kernel.md
  ↓
02_v1_atlas_hybrid_kernel.md
  ↓
03_v2_atlas_research_runtime.md
  ↓
04_future_versions.md
```

## 设计原则

这组文档遵守四个原则：

```text
1. 证据优先：所有答案都要能回到证据。
2. 先做内核：不要第一天就上完整大平台。
3. 分阶段扩展：每一版都应该独立可用。
4. 可评估：每次改动都应该能通过 eval 或 trace 验证。
```
