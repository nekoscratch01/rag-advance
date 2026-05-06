# V2 — Atlas Research Runtime

> 核心目标：支持复杂、异步、可追踪、可审计的研究任务。
> 关键边界：V2 不重新实现 retrieval；V2 Planner 只编排任务，每个 subquestion 调用 V1 Evidence Kernel。

---

## 1. V2 的定位

V1 解决的是：

```text
单次 query 如何找到高质量证据并回答。
```

V2 解决的是：

```text
复杂研究任务如何拆解、执行、保存过程、生成报告、审计结论。
```

V2 不是把 V1 变复杂，而是在 V1 之上加一层异步任务运行时。

正确关系：

```text
Research Planner
 -> subquestion
 -> V1 Query Orchestrator + TextHybridProvider + Evidence Kernel
 -> evidence pack
 -> synthesis/report/audit
```

错误关系：

```text
Research Planner 自己直接 dense search / BM25 / Graph search / SQL search
```

---

## 2. V2 总体架构

```text
                           ┌──────────────────────┐
                           │   Research Request    │
                           │ user objective/scope  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Research API        │
                           │ create job / status   │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Job Store + Queue   │
                           │ status/events/budget  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Research Worker     │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Research Planner    │
                           │ plan/subquestions/DAG │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │     Task DAG          │
                           │ dependencies/budgets  │
                           └──────────┬───────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌────────────────┐            ┌────────────────┐            ┌────────────────┐
│ Subquestion A  │            │ Subquestion B  │            │ Subquestion C  │
└───────┬────────┘            └───────┬────────┘            └───────┬────────┘
        │                             │                             │
        ▼                             ▼                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                  V1 Advanced Hybrid Evidence Kernel                        │
│        Query Orchestrator -> Provider -> Rerank -> Evidence Builder         │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │ Subquestion Evidence  │
                         │ packs + evaluations   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Gap / Conflict Check │
                         │ follow-up if needed   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Synthesis Planner    │
                         │ outline/report logic  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Report Writer      │
                         │ cited report artifact │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Claim Audit         │
                         │ support verification  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Artifacts        │
                         │ report/events/trace   │
                         └──────────────────────┘
```

---

## 3. V2 与 V1 的边界

### V1 负责

```text
- Query Orchestrator
- QueryPlan
- Provider Router
- TextHybridProvider
- Fusion / Rerank
- Evidence Builder
- Evidence Evaluator
- Citation Verifier
- Single-answer generation
```

### V2 负责

```text
- ResearchJob 生命周期
- 异步队列 / Worker
- Planner
- Subquestion DAG
- Budget management
- Gap detection
- Report synthesis
- Claim-level audit
- Artifact management
- Event log
```

一句话：

```text
V2 只编排，不重写 retrieval。
```

---

## 4. ResearchJob 数据模型

```python
class ResearchJob:
    job_id: str
    user_id: str | None
    workspace_id: str | None
    objective: str
    status: str
    priority: str
    budget: ResearchBudget
    plan_id: str | None
    artifact_root: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
```

### 4.1 Job status

```text
queued
planning
running
waiting_for_retrieval
synthesizing
auditing
completed
failed
cancelled
budget_exhausted
```

### 4.2 ResearchBudget

```python
class ResearchBudget:
    max_wall_clock_seconds: int
    max_subquestions: int
    max_retrieval_calls: int
    max_llm_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_cost_usd: float | None
    max_retries_per_task: int
```

---

## 5. ResearchPlan

```python
class ResearchPlan:
    plan_id: str
    job_id: str
    objective: str
    scope: str | None
    assumptions: list[str]
    subquestions: list[Subquestion]
    dependencies: list[Dependency]
    expected_artifacts: list[str]
    risk_flags: list[str]
```

### 5.1 Subquestion

```python
class Subquestion:
    subquestion_id: str
    text: str
    purpose: str
    expected_evidence_type: str
    provider_preferences: list[str]
    depends_on: list[str]
    budget: RetrievalBudget
    status: str
```

### 5.2 Subquestion purpose

```text
fact_lookup
background_context
comparison_operand
calculation_input
causal_explanation
counterevidence_search
definition
summary_support
```

---

## 6. Planner 设计

V2 Planner 可以用 LLM，但必须输出结构化计划。

输入：

```text
- user objective
- optional constraints
- available providers
- corpus metadata summary
- budget
- desired artifact type
```

输出：

```text
ResearchPlan JSON
```

### 6.1 Planner Guardrails

```text
- subquestions 数量上限
- 不允许生成超出 corpus 范围的任务
- 每个 subquestion 必须有 purpose
- 每个 subquestion 必须能映射到 V1 QueryPlan
- 需要计算的问题必须显式标为 calculation_input 或 calculation
- 对报告型任务必须规划 report outline
```

### 6.2 Planner 不是 Agent 自由行动

V2 不做无限循环 Agent。

它做的是：

```text
Plan -> Execute -> Evaluate -> Optional Follow-up -> Synthesize -> Audit
```

---

## 7. Task DAG

V2 的任务不是线性列表，而是 DAG。

```text
Objective: Compare cash flow quality of Company A and Company B in FY2022.

DAG:
  q1: Company A FY2022 operating cash flow
  q2: Company A FY2022 capital expenditure
  q3: Company B FY2022 operating cash flow
  q4: Company B FY2022 capital expenditure
  c1: calculate Company A FCF = q1 - q2
  c2: calculate Company B FCF = q3 - q4
  s1: synthesize comparison = c1 + c2 + evidence
```

ASCII：

```text
      q1 ─┐
          ├── c1 ─┐
      q2 ─┘       │
                  ├── s1
      q3 ─┐       │
          ├── c2 ─┘
      q4 ─┘
```

---

## 8. Task Runner

每个检索型 subquestion 调用 V1：

```python
def run_subquestion(subquestion: Subquestion) -> SubquestionResult:
    query_request = QueryRequest(
        query=subquestion.text,
        mode="retrieve_and_evaluate",
        provider_preferences=subquestion.provider_preferences,
        budget=subquestion.budget,
        return_trace=True,
    )
    return v1_kernel.run(query_request)
```

不要让 Task Runner 自己做 dense/BM25。

---

## 9. Evidence Pack

V2 需要把每个 subquestion 的证据保存成独立包。

```python
class SubquestionEvidencePack:
    subquestion_id: str
    query_run_id: str
    evidence_blocks: list[EvidenceBlock]
    verification_result: VerificationResult
    coverage: dict
    gaps: list[str]
    conflicts: list[str]
```

ResearchJob 最终有一个总 Evidence Pack：

```python
class ResearchEvidencePack:
    job_id: str
    subquestion_packs: list[SubquestionEvidencePack]
    merged_evidence_blocks: list[EvidenceBlock]
    coverage_matrix: dict
    conflict_sets: list[ConflictSet]
```

---

## 10. Gap Detection

V2 必须知道证据是否足够。

Gap 类型：

```text
missing_entity
missing_period
missing_metric
missing_counterevidence
low_source_diversity
conflicting_sources
retrieval_failed
packing_failed
citation_failed
```

处理策略：

```text
missing evidence:
  generate follow-up subquestion

conflict:
  gather counterevidence or mark conflict

low confidence:
  widen retrieval budget or route additional provider

budget exhausted:
  report limitation explicitly
```

---

## 11. Corrective Loop

V2 可以有有限纠错循环。

```text
for each subquestion:
  run V1 retrieval
  evaluate evidence
  if sufficient:
      accept
  elif budget remains:
      create follow-up retrieval task
  else:
      mark insufficient
```

必须限制：

```text
max_followups_per_subquestion
max_total_followups
max_wall_clock
max_cost
```

---

## 12. Report Writer

报告生成不是直接把所有 evidence 塞给 LLM。

流程：

```text
ResearchEvidencePack
 -> report outline
 -> section evidence allocation
 -> section drafts with citations
 -> global consistency pass
 -> citation verification
 -> final report.md
```

### 12.1 Report artifacts

```text
report.md
report.json
claims.json
evidence_map.json
citation_map.json
audit_report.md
```

### 12.2 Report 要求

```text
- 每个关键结论必须有 citation。
- 证据不足的地方必须标注 limitation。
- 冲突证据必须显式列出。
- 计算结论必须列出公式和输入证据。
```

---

## 13. Claim Audit

V2 的 Claim Audit 用于检查最终答案/报告。

流程：

```text
report/answer
 -> claim extraction
 -> per-claim evidence mapping
 -> support verification
 -> failure attribution
```

### 13.1 Claim 标签

```text
supported
unsupported
insufficient
contradicted
partially_supported
```

### 13.2 Failure attribution

```text
corpus_missing:
  corpus 中根本没有证据

retrieval_failure:
  证据在 corpus 中，但没有被 V1 找到

fusion_failure:
  证据被某 lane 找到，但 fusion 后排名太低

packing_failure:
  证据进入候选，但没进 evidence pack

generation_failure:
  证据进了 prompt，但模型没用好

citation_failure:
  citation 指向的 evidence 不支持 claim

verifier_failure:
  verifier 没拦住错误
```

---

## 14. Artifact Store

V2 的价值之一是可复盘。

每个 job 产出：

```text
artifacts/{job_id}/
  job.json
  plan.json
  events.jsonl
  task_dag.json
  subquestions.json
  query_runs/
    {query_run_id}.json
  evidence_packs/
    {subquestion_id}.json
  synthesis/
    outline.json
    draft_sections.json
  report.md
  report.json
  claims.json
  audit_report.md
  metrics.json
```

---

## 15. Events

事件用于实时状态、debug 和 replay。

```json
{
  "event_id": "evt_001",
  "job_id": "job_123",
  "type": "subquestion_started",
  "timestamp": "...",
  "payload": {
    "subquestion_id": "sq_001",
    "text": "3M FY2018 capital expenditure"
  }
}
```

事件类型：

```text
job_created
planning_started
plan_created
subquestion_started
subquestion_completed
retrieval_failed
evidence_insufficient
followup_created
synthesis_started
report_drafted
claim_audit_started
claim_audit_completed
job_completed
job_failed
```

---

## 16. API

### 16.1 Create job

```http
POST /v2/research/jobs
```

```json
{
  "objective": "Compare 3M's FY2018 and FY2017 capital expenditures and explain the change.",
  "mode": "research_report",
  "budget": {
    "max_subquestions": 6,
    "max_retrieval_calls": 12,
    "max_wall_clock_seconds": 180
  },
  "return_artifacts": true
}
```

### 16.2 Get status

```http
GET /v2/research/jobs/{job_id}
```

### 16.3 Stream events

```http
GET /v2/research/jobs/{job_id}/events
```

### 16.4 Get artifacts

```http
GET /v2/research/jobs/{job_id}/artifacts/report.md
```

### 16.5 Audit existing answer

```http
POST /v2/audit/claims
```

---

## 17. Eval

### 17.1 Research quality

```text
subquestion_coverage
subquestion_redundancy_rate
evidence_pack_coverage
report_claim_support_rate
citation_support_rate
conflict_detection_rate
limitation_correctness
```

### 17.2 Runtime quality

```text
job_success_rate
job_failure_rate
budget_exhaustion_rate
avg_subquestions_per_job
avg_retrieval_calls_per_job
latency_p50/p95
total_cost_per_job
```

### 17.3 Audit quality

```text
claim_extraction_recall
claim_support_label_accuracy
false_supported_rate
false_insufficient_rate
failure_attribution_accuracy
```

---

## 18. Implementation Plan

### V2.0 Job Runtime

```text
- ResearchJob model
- Job API
- Queue + Worker
- events.jsonl
- artifact directory
```

### V2.1 Research Planner

```text
- structured planner prompt
- ResearchPlan schema
- Subquestion schema
- DAG builder
- budget manager
```

### V2.2 V1 Kernel Integration

```text
- subquestion -> V1 QueryRequest
- evidence pack collection
- subquestion-level evidence evaluation
```

### V2.3 Gap / Follow-up Loop

```text
- gap detector
- corrective retrieval
- conflict handling
- budget-aware retry
```

### V2.4 Report Artifact

```text
- outline generator
- section writer
- cited report.md
- citation map
```

### V2.5 Claim Audit

```text
- claim extraction
- per-claim evidence verification
- failure attribution
- audit_report.md
```

---

## 19. Definition of Done

V2 完成时必须满足：

```text
- 能创建异步 ResearchJob。
- Planner 能生成结构化 ResearchPlan 和 Subquestion DAG。
- 每个 subquestion 都通过 V1 Kernel 获取 evidence。
- Evidence Pack 能按 subquestion 保存。
- 能检测 evidence gap 和 conflict。
- 能生成带 citation 的 report artifact。
- Claim Audit 能判断 report claims 是否 supported。
- events.jsonl 能复盘完整执行过程。
- job 有预算控制和失败状态。
```

---

## 20. V2 结论

V2 的价值不是“加一个 Agent”。

V2 的价值是：

```text
把复杂问题变成可计划、可执行、可审计、可复盘的研究流程。
```

V2 必须建立在 V1 Evidence Kernel 上。
