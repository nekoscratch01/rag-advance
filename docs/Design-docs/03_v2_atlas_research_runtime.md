# V2：Atlas Research Runtime

> V2 开始进入 Agentic RAG，但它不是一个自由乱跑的 Agent。V2 的目标是把复杂问题变成可追踪、可预算、可评估的异步研究任务。

---

## 1. V2 的目标

V1 可以回答普通问题，也能通过 hybrid retrieval 和 reranker 找到更准确的证据。但有些问题不是一次检索就能解决。比如用户问“比较三家公司最近财报里的风险因素变化”，系统需要分别查每家公司、找最近财报、提取风险因素、比较相同点和不同点、检查证据是否覆盖三家公司、最后生成结构化报告。这不是一个单步 query，而是一个研究任务。

V2 的目标就是支持这类复杂任务。它把用户问题创建成一个 research job，然后由后台 worker 执行。执行过程包括规划、拆分子问题、并行检索、整理证据、生成初稿、检查证据缺口、必要时补检索、生成最终报告、验证引用和保存 artifacts。

V2 的核心不是“Agent 自主性越高越好”，而是“Agent 的每一步都受控”。每个 research job 都应该有预算：最多几个子问题、最多几轮补检索、最多多少模型调用、最多运行多久、最多花多少钱。超过预算时，系统应该输出当前证据支持的部分结果，并明确说明限制，而不是无限循环。

---

## 2. V2 和 V1 的区别

V1 是同步 query runtime。用户发一个问题，系统在几秒内返回答案。它适合事实查询、文档内问答、概念解释、短总结。

V2 是异步 research runtime。用户提交的是研究任务，系统返回 job_id，然后后台执行，最后产出 report 和 artifacts。它适合复杂比较、多来源综合、需要多轮检索的问题。

两者的区别可以这样理解：

```text
V1：问一个问题，返回一个答案。
V2：提交一个研究任务，返回一组研究产物。
```

V2 不替代 V1，而是复用 V1。每个 subquestion 的检索仍然调用 V1 的 Hybrid Kernel。V2 只是把多个 V1 检索和生成步骤编排起来，并加入计划、证据整理、Critic 和报告生成。

---

## 3. V2 系统结构

V2 在 V1 的基础上新增 Job Manager、Redis Queue、Research Worker Pool、Artifact Store 和 Research Runtime。

```text
┌──────────────────────────────────────────────┐
│                 用户提交研究任务              │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                FastAPI Job API                │
│                                              │
│  POST /v1/research/jobs       创建任务         │
│  GET  /v1/research/jobs/{id}  查看状态         │
│  GET  /v1/research/jobs/{id}/events 查看事件   │
│  GET  /v1/research/jobs/{id}/artifacts 获取产物│
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                  Job Manager                  │
│                                              │
│  创建 job                                    │
│  分配预算                                    │
│  记录状态                                    │
│  投递到队列                                  │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                  Redis Queue                  │
│               research_job_queue              │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│              Research Worker Pool             │
│                                              │
│  Planner                                     │
│  Subquestion Executor                        │
│  Evidence Consolidator                       │
│  Critic                                      │
│  Report Writer                               │
│  Citation Verifier                           │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                  Artifact Store               │
│                                              │
│  research_plan.json                           │
│  evidence_pack.json                           │
│  trace.jsonl                                  │
│  draft_report.md                              │
│  final_report.md                              │
│  eval_report.json                             │
└──────────────────────────────────────────────┘
```

V2 不一定要第一天就上复杂的分布式队列。Redis Queue、RQ、Celery 或 arq 都可以。重点不是队列工具，而是把长任务从同步 API 请求里拆出来。普通 query 不能被 research job 拖慢，research job 也不能因为 HTTP 连接断开就丢失。

---

## 4. Research Job API 设计

V2 的 API 设计应该围绕 job 生命周期，而不是围绕聊天消息。用户提交任务后，系统立即返回 job_id。后续用户通过 job_id 查询状态、事件和产物。

核心 endpoint：

```text
POST /v1/research/jobs
  创建研究任务。

GET /v1/research/jobs/{job_id}
  查询任务状态和摘要。

GET /v1/research/jobs/{job_id}/events
  查询任务执行事件。未来可以升级为 SSE。

GET /v1/research/jobs/{job_id}/artifacts
  获取任务产物列表。

GET /v1/research/jobs/{job_id}/artifacts/{artifact_name}
  获取具体产物内容。

POST /v1/research/jobs/{job_id}/cancel
  取消任务。
```

创建任务的请求体可以这样：

```json
{
  "query": "Compare the latest risk factors across Company A, Company B, and Company C.",
  "options": {
    "max_subquestions": 8,
    "max_reflexive_loops": 2,
    "max_runtime_seconds": 180,
    "output_format": "research_report"
  },
  "filters": {
    "source_type": "filing"
  }
}
```

创建后返回：

```json
{
  "job_id": "job_001",
  "status": "queued",
  "created_at": "2026-05-03T00:00:00Z"
}
```

这个 API 风格很重要。它把复杂任务从普通 query 中分离出来，也让后续前端可以自然展示任务进度、阶段事件和报告产物。

---

## 5. Job Manager 设计

Job Manager 负责创建 research job、分配预算、写入数据库、投递队列。它本身不执行研究逻辑。这样做可以把“任务管理”和“任务执行”分开，避免一个模块既管状态又管推理。

ResearchJob 表至少包含：

```text
job_id
user_query
status
priority
budget_json
created_at
started_at
finished_at
error_message
artifact_root
```

status 可以包括：

```text
queued
planning
retrieving
analyzing
writing
verifying
done
error
cancelled
```

Job Manager 在创建 job 时要分配预算。预算不是装饰字段，而是 Research Worker 执行时的硬约束。

预算可以包括：

```text
max_subquestions
max_parallel_subquestions
max_reflexive_loops
max_retrieval_calls
max_llm_calls
max_context_tokens
max_runtime_seconds
max_cost_usd
```

默认预算建议保守一点。例如 V2 初期可以设置最多 8 个子问题、最多 2 轮补检索、最多 20 次 LLM 调用、最多 180 秒运行时间。这样系统可以处理复杂问题，但不会失控。

---

## 6. Job Event 设计

V2 必须记录 job 执行事件。事件是异步研究系统的生命线。没有事件，用户只能看到 queued、done、error 三种状态；有事件，用户和工程师都能看到系统在做什么、卡在哪里、为什么补检索。

JobEvent 表可以这样设计：

```text
event_id
job_id
seq
event_type
event_payload
created_at
```

常见事件包括：

```text
job.created
job.started
plan.created
subquestion.started
subquestion.completed
evidence.selected
critic.supported
critic.insufficient
critic.conflicted
reflexive_loop.started
report.drafted
citation.verified
job.completed
job.failed
```

事件的 seq 是单调递增序号。这样后续前端或 CLI 可以从某个 seq 之后继续拉取事件，避免重复显示。即使 V2 暂时不做前端，事件系统也非常有价值，因为它让每个 research job 都可回放。

示例事件：

```json
{
  "seq": 7,
  "event_type": "critic.insufficient",
  "event_payload": {
    "reason": "Evidence for Company C is missing.",
    "suggested_query": "Company C latest annual report risk factors"
  }
}
```

这类事件比单纯日志更有结构，也更适合后续做可视化。

---

## 7. Research Worker 设计

Research Worker 是 V2 的执行主体。它从队列中取出 job，然后按固定流程执行。它不是一个自由 Agent，而是一个 workflow-first 的研究运行时。

推荐流程：

```text
1. 读取 job 和预算。
2. Planner 生成研究计划。
3. 拆成多个 subquestions。
4. Subquestion Executor 调用 V1 Hybrid Kernel 检索证据。
5. Evidence Consolidator 整理证据。
6. Draft Writer 生成初稿。
7. Critic 检查证据是否足够。
8. 如果证据不足，最多补检索 1-2 轮。
9. Report Writer 生成最终报告。
10. Citation Verifier 检查引用。
11. 保存 artifacts。
12. 标记 job done。
```

Research Worker 应该在每个阶段写 JobEvent，也应该周期性更新 job status。这样即使 worker 中途失败，系统也知道任务失败在什么阶段。

V2 可以使用 LangGraph 表达这个 workflow，也可以先用普通 Python workflow 实现。LangGraph 的好处是状态机和循环更清楚，后面更容易加入 checkpoint 和 interruption。但不要因为引入 LangGraph 而让 V2 变得过度复杂。核心是 workflow 清晰、状态可保存、预算可执行。

---

## 8. Planner 设计

Planner 负责把用户的复杂问题拆成几个可执行的小问题。Planner 的输出必须是结构化 JSON，而不是一段自由文本。结构化输出可以被后续 executor 直接执行，也方便 trace 和 eval。

用户问题：

```text
“比较三家公司最近财报里的风险因素变化。”
```

Planner 输出可以是：

```json
{
  "objective": "Compare recent risk factor changes across three companies.",
  "task_type": "comparison_report",
  "subquestions": [
    {
      "id": "sq1",
      "question": "What risk factors did Company A mention in its latest filing?",
      "filters": { "company": "Company A", "source_type": "filing" }
    },
    {
      "id": "sq2",
      "question": "What risk factors did Company B mention in its latest filing?",
      "filters": { "company": "Company B", "source_type": "filing" }
    },
    {
      "id": "sq3",
      "question": "What risk factors did Company C mention in its latest filing?",
      "filters": { "company": "Company C", "source_type": "filing" }
    },
    {
      "id": "sq4",
      "question": "What are the similarities and differences across the three companies?",
      "depends_on": ["sq1", "sq2", "sq3"]
    }
  ],
  "success_criteria": [
    "Each company must be covered.",
    "Each key claim must cite evidence.",
    "Missing evidence must be explicitly stated."
  ]
}
```

Planner 不能无限拆问题。V2 初期建议限制最多 8 个 subquestions。对于过大的问题，Planner 应该收敛范围，而不是生成 20 个任务。例如用户问“分析整个 AI 行业”，Planner 可以把任务限定为“基于当前文档，分析主要主题和证据不足之处”。

Planner 的质量也需要评估。后续 eval 应该检查 subquestions 是否覆盖原始问题、是否太宽、是否太窄、是否包含不可执行的任务。

---

## 9. Subquestion Executor 设计

Subquestion Executor 负责执行 Planner 生成的小问题。它不自己发明检索逻辑，而是调用 V1 的 Hybrid Kernel。

每个 subquestion 的执行流程：

```text
subquestion
  ↓
V1 Query Runtime
  ↓
query rewrite
  ↓
dense + keyword retrieval
  ↓
fusion
  ↓
reranker
  ↓
evidence builder
  ↓
返回 evidence candidates
```

多个 subquestions 可以并行执行，但并行数必须受限制。一个 research job 如果一次性启动 10 个子问题、每个子问题又做 dense+keyword+rerank+LLM，就可能把系统资源吃光。V2 初期建议：

```text
max_parallel_subquestions = 3 或 5
```

Subquestion Executor 还应该记录每个 subquestion 的结果：召回了哪些 evidence、是否 sufficient、是否出现错误、用了多少时间和成本。这样最终 report 出错时，可以追溯到具体子问题。

---

## 10. Evidence Consolidator 设计

Evidence Consolidator 是 V2 非常重要的模块。复杂研究任务会产生大量 evidence，如果直接把所有证据塞给 Report Writer，模型会被噪音淹没，报告质量也会下降。

Evidence Consolidator 负责把多个 subquestions 的证据整理成一个 Evidence Pack。它做以下事情：

```text
1. 去重：同一个 chunk 被多个 subquestion 找到，只保留一次。
2. 合并：同一文档相邻 chunks 合并成更完整的 evidence block。
3. 分组：按 subquestion、主题、公司、时间或来源分组。
4. 排序：按 rerank_score、source relevance、时间排序。
5. 标记冲突：不同 evidence 对同一问题说法不一致时记录 conflict。
6. 控制长度：确保进入 Report Writer 的上下文不超过 token budget。
```

Evidence Pack 示例：

```json
{
  "evidence_blocks": [
    {
      "evidence_id": "ev_001",
      "source_title": "Company A Annual Report",
      "document_id": "doc_a",
      "chunk_ids": ["chunk_a12", "chunk_a13"],
      "page_range": [12, 13],
      "section_title": "Risk Factors",
      "text": "...",
      "supports": ["sq1"],
      "relevance_score": 0.91
    }
  ],
  "coverage": {
    "sq1": ["ev_001", "ev_002"],
    "sq2": ["ev_003"],
    "sq3": []
  }
}
```

这里的 coverage 很关键。Critic 可以通过 coverage 发现某个子问题没有证据，比如 sq3 为空，就说明 Company C 还没有被覆盖。

---

## 11. Draft Writer 设计

Draft Writer 生成初稿。它的作用不是生成最终报告，而是把当前 evidence 整合成一个初步答案，供 Critic 检查。这个阶段可以使用中等模型，不一定用最强模型。

Draft Writer 的输入包括：

```text
original query
research plan
subquestions
evidence pack
output format instructions
```

输出可以是结构化草稿：

```text
Executive summary draft
Key findings draft
Evidence-backed analysis draft
Known gaps
Potential conflicts
```

Draft Writer 必须遵守证据边界。它不能使用 evidence 外的信息。它也不能把没有证据的地方写成确定结论。V2 的 prompt 应该明确要求：如果 evidence pack 中某个子问题没有证据，就写“当前证据不足”，而不是推测。

Draft 的价值在于给 Critic 一个具体对象来检查。Critic 不只是看 evidence 是否够，还可以检查 draft 中有没有 unsupported claims。

---

## 12. Critic 和 Reflexive Loop 设计

V2 的 Critic 比 V1 的 Critic Lite 更强。V1 的 Critic Lite 只判断当前 evidence 是否足够。V2 的 Critic 还要判断 research plan 是否覆盖原问题、draft 是否使用了证据外信息、是否存在缺口、是否需要补检索。

Critic 的输入：

```text
original query
research plan
subquestions
evidence pack
draft report
budget remaining
```

Critic 的输出：

```json
{
  "judgment": "insufficient",
  "reason": "Company C is not covered by the current evidence pack.",
  "missing_evidence": [
    {
      "gap": "Company C latest filing risk factors",
      "suggested_query": "Company C latest annual report risk factors",
      "filters": { "company": "Company C", "source_type": "filing" }
    }
  ],
  "conflicts": [],
  "next_action": "retrieve_more"
}
```

如果 Critic 输出 supported，系统进入 Final Report Writer。若输出 conflicted，系统进入 Conflict Handling，把冲突写进最终报告。若输出 insufficient，系统触发 Reflexive Loop，也就是根据 missing_evidence 再补一轮检索。

Reflexive Loop 必须有硬限制。V2 初期建议最多 2 轮。如果两轮后仍然 insufficient，系统不要继续跑，而是生成一份部分报告，并明确写出证据缺口。这是 bounded autonomy 的核心。

```text
Critic insufficient
  ↓
生成补充 subquestions
  ↓
调用 V1 Hybrid Kernel 补检索
  ↓
更新 Evidence Pack
  ↓
重新 Draft / Critic
  ↓
最多重复 2 次
```

这个 loop 是 V2 的亮点，但也是风险来源。一定要记录每轮 loop 的原因、补检索 query、结果是否改善、花费了多少成本。

---

## 13. Conflict Handling 设计

复杂研究任务经常会遇到证据冲突。比如一个来源说某个事件发生在 2024 年，另一个来源说是 2025 年。Atlas 不应该擅自选择一个看起来更合理的说法，而应该根据证据和 metadata 判断。如果无法判断，就把冲突展示出来。

Conflict Handling 可以按以下规则处理：

```text
1. 优先使用更新的来源。
2. 优先使用一手资料。
3. 优先使用用户指定或系统标记为更权威的来源。
4. 如果无法判断，报告中明确展示冲突。
```

最终报告可以写：

```text
当前证据存在冲突。Source A 将该事件日期标记为 2024 年，而 Source B 标记为 2025 年。当前系统无法基于现有证据确认哪一项更可靠，因此不将该日期作为确定结论。
```

这种表达比强行给答案更可信。对于金融、法律、医疗等领域，能正确处理不确定性和冲突，是系统成熟度的重要体现。

---

## 14. Final Report Writer 设计

Final Report Writer 负责生成最终报告。V2 的最终输出不应该只是一个长段落，而应该是一份结构化 report artifact。

推荐报告结构：

```text
# Executive Summary

# Key Findings

# Evidence-backed Analysis

# Comparison / Synthesis

# Conflicts and Uncertainty

# Source Table

# Limitations
```

每个主要事实性结论都应该引用 evidence_id。例如：

```text
Company A emphasizes supply chain concentration as a key risk. [ev_001]
```

Report Writer 的 prompt 应该明确限制：

```text
1. 不使用 evidence 外的信息。
2. 每个主要 claim 必须有 evidence citation。
3. 证据不足时必须说明不足。
4. 证据冲突时必须展示冲突。
5. 不要把推测写成事实。
```

V2 的报告不是越长越好。更重要的是结构清楚、引用准确、限制明确。如果当前证据只能支持部分结论，报告就应该是部分报告，而不是伪装成完整研究。

---

## 15. Citation Verifier 设计

Citation Verifier 是最终报告发布前的检查步骤。它负责检查报告中的 citations 是否真的支持对应 claim。

它至少检查四件事：

```text
1. 每个主要 claim 是否有 citation。
2. citation 指向的 evidence 是否存在。
3. evidence 文本是否支持 claim。
4. 是否有 claim 使用了 evidence 外的信息。
```

如果发现轻微问题，例如某个 claim 没有引用，但 evidence pack 中有支持证据，系统可以修正 citation。如果发现严重问题，例如报告中有多个 unsupported claims，系统应该降低 confidence，甚至把 report 标记为 insufficient。

Citation verification 可以先用 LLM-as-judge，也可以结合规则。比如检查 evidence_id 是否存在是规则；检查 claim 是否被 evidence 支持可以用模型判断。无论用什么方法，verification 的结果都要保存成 artifact。

---

## 16. Artifact Store 设计

V2 的 research job 必须保存 artifacts。最终报告只是一个产物，完整执行过程才是真正的工程价值。

每个 job 可以保存以下文件：

```text
artifacts/{job_id}/
  research_plan.json
  subquestions.json
  retrieval_trace.jsonl
  evidence_pack.json
  draft_report.md
  critic_report.json
  final_report.md
  citation_verification.json
  eval_report.json
```

这些 artifacts 有多个用途。它们可以用于 debug，可以用于 eval，可以用于后续生成前端展示，可以用于复盘失败案例，也可以被 V6 的 Memory & Skills 系统复用。例如某个 research job 质量很高，后面可以把它的 plan 和 report template 提炼成一个 skill。

V2 可以先把 artifacts 存在本地文件系统或 MinIO。生产化以后再接 S3。关键是 artifact_uri 要写入 ResearchJob 表，系统可以通过 API 获取。

---

## 17. V2 并发和资源预算设计

V2 必须把普通 query 和 research job 分开。普通 query 是低延迟请求，用户希望几秒内拿到结果。Research job 是长任务，可以跑几十秒到几分钟。如果两者使用同一个 worker pool，复杂任务会拖慢普通查询。

推荐队列划分：

```text
online_query_queue      普通查询，短任务
research_job_queue      研究任务，长任务
eval_job_queue          评估任务，后台任务
```

V2 初期可以只实现 research_job_queue，但架构上要认识到这几类任务资源不同。

Research job 的并发建议：

```text
max_running_research_jobs = 5 - 20
max_parallel_subquestions_per_job = 3 - 5
max_reflexive_loops = 2
max_llm_calls_per_job = 20
max_runtime_seconds = 180 - 300
```

这些限制不是保守，而是必要。没有限制的 Agentic 系统很容易在复杂 query 下无限扩张成本。V2 的设计理念是 bounded autonomy：允许系统自己计划和补检索，但必须在预算内完成。

---

## 18. V2 Trace 设计

V2 的 trace 是 job-level trace，而不只是 query-level trace。它应该记录整个研究任务从 planning 到 final report 的路径。

Job trace 至少包括：

```text
job_id
original_query
budget
research_plan
subquestions
每个 subquestion 的 retrieval trace
evidence consolidation 结果
critic 判断
reflexive loop 次数
每轮补检索的 query 和结果
draft report
final report
citation verification
latency and cost breakdown
```

Trace 可以分两层保存。结构化事件保存在 research_job_events 表；详细中间产物保存在 artifacts。这样查询 job 状态很快，查看完整 trace 时再读取 artifact 文件。

V2 的 trace 不只是运维工具，也会成为简历和 demo 的亮点。它能展示系统不是黑盒，而是一个有计划、有证据、有检查、有产物的研究运行时。

---

## 19. V2 Evaluation 设计

V2 的 eval 不再只是问答 eval，而是 workflow eval。它需要评估整个研究过程。

核心指标包括：

```text
Plan coverage：Planner 是否覆盖了原始问题的关键方面。
Subquestion quality：子问题是否具体、可执行、不重复。
Evidence coverage：证据是否覆盖所有关键子问题。
Critic accuracy：Critic 是否能发现证据缺口和冲突。
Loop effectiveness：补检索是否真的改善了 evidence coverage。
Citation correctness：最终报告引用是否准确。
Faithfulness：报告是否忠实于 evidence。
Runtime and cost：单个 job 的耗时和成本是否在预算内。
```

V2 eval report 示例：

```text
Research Eval Run: 2026-xx-xx
Total jobs: 30

Plan coverage: 0.81
Subquestion quality: 0.78
Evidence coverage: 0.76
Critic gap detection: 0.72
Citation correctness: 0.83
Faithfulness: 0.86
Average runtime: 92s
Average cost: $0.11/job
Average reflexive loops: 1.2

Top failure modes:
1. Planner generated overly broad subquestions.
2. Critic missed missing evidence for one entity.
3. Final report compressed uncertainty too aggressively.
4. Citation verifier passed weak evidence in 3 cases.
```

这类评估能证明 V2 是真正的 research runtime，而不是简单地让 LLM 写一篇长文。

---

## 20. V2 容量规划

V2 的容量要分普通 query 和 research job 两类。

| 指标 | V2 目标 |
|---|---:|
| 文档数 | 50万 - 200万 |
| Chunk 数 | 500万 - 2000万 |
| 普通 hybrid query QPS | 10 - 50 |
| 普通 query 延迟 | p95 5 - 10 秒 |
| Research job 并发 | 5 - 20 |
| 单个 research job 时间 | 30 秒 - 5 分钟 |
| 单个 job subquestions | 4 - 12 |
| 单个 job reflexive loops | 0 - 2 |
| Eval cases | 200 - 500 |

这个量级已经足够强。V2 不需要承诺 500 QPS 的 full agentic research。普通 query 可以扩到更高，但 complex research 必须按 job 并发来衡量，而不是按 QPS 来衡量。

---

## 21. V2 失败模式与降级

V2 的失败模式比 V1 多，因为它是异步、多阶段、长任务。

如果 Planner 失败，可以使用简单 fallback plan。例如把用户问题拆成 3 个通用步骤：背景检索、关键证据检索、总结。Fallback plan 不一定优秀，但比 job 直接失败更好。

如果某个 subquestion 检索失败，系统应该记录该 subquestion failed，并继续执行其他 subquestions。最终 Critic 会发现 evidence gap，并在报告里说明。

如果 Critic 失败，可以跳过 reflexive loop，直接进入 Final Report Writer，但 confidence 应该更保守。

如果 Report Writer 失败，可以返回 draft_report，并标记 job partial_failed。不要丢掉已经收集到的 evidence 和 plan。

如果 Citation Verifier 失败，可以返回 final_report，但标记 verification_status=failed，并提醒引用未完全验证。

如果 worker 中途宕机，job 状态应该从 running 变成 error 或 retrying。V2 初期可以简单重试整个 job。后续可以用 checkpoint 从中间步骤恢复。

---

## 22. V2 完成标准

V2 完成应达到：

```text
1. 支持创建异步 research job。
2. 支持查询 job status。
3. 支持记录和查询 job events。
4. 支持 Planner 生成结构化 research plan。
5. 支持 subquestions 并行执行。
6. 每个 subquestion 复用 V1 Hybrid Kernel。
7. 支持 Evidence Consolidator。
8. 支持 Critic 判断 supported / insufficient / conflicted。
9. 支持最多 1-2 轮 reflexive retrieval。
10. 支持生成 final_report.md。
11. 支持 citation_verification.json。
12. 支持完整 artifacts 保存。
13. 支持 job-level eval。
14. 支持并发和预算限制。
15. 支持失败降级和 partial result。
```

V2 做完后，Atlas 就不再只是一个 RAG 后端，而是一个 cloud-ready 的 Agentic Research Runtime。它仍然没有 GraphRAG、Kafka、ClickHouse、MCP、Memory，但已经具备一个高级 RAG/Agent 项目最重要的东西：可控复杂任务、证据链、trace、eval 和异步执行。

---

## 23. V2 不做什么

V2 不做 GraphRAG。关系型检索放到 V3。

V2 不做结构化数据分析。ClickHouse、DuckDB、Text-to-SQL 放到 V4。

V2 不做持续数据流入。Kafka/Flink 放到 V5。

V2 不做长期记忆和技能系统。Memory & Skills 放到 V6。

V2 不做 MCP 工具层。Tool standardization 放到 V7。

V2 不做完整前端。可以通过 API、CLI 和 artifacts 展示系统能力。

V2 的一句话目标是：

```text
把复杂问题变成可预算、可追踪、可评估、可产出报告的异步研究任务。
```
