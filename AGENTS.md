# Atlas 智能体地图

这个仓库把 `docs/` 当作记录系统。本文是地图，不是手册；不要把它扩写成大型说明书。

## 先读这里

```text
docs/README.md
```

如果任务涉及架构、版本规划、实现状态或测评，先读这份文档。

## 文档层级

```text
docs/Design-docs/
  设计意图和路线图。这里是计划，不是实现证明。

docs/exec-plans/milestone/
  版本执行记录。这里说明实际做了什么、相对设计改了什么、还缺什么。

docs/exec-plans/version-arch/
  实际实现架构。改 runtime 代码前先读这里。

benchmarks/
  实验证据。质量、指标、失败案例和结论必须从这里找依据。
```

## 当前版本事实

```text
V0 Atlas Kernel：已实现的基础 RAG kernel。
V1 Atlas Advanced Hybrid Kernel：检索质量主路径已实现并完成测评。
V2 Atlas Research Runtime：仍是设计阶段，尚未实现。
```

当前 V1 记录：

```text
docs/exec-plans/milestone/already_implemented.md
docs/exec-plans/milestone/v1_advanced_hybrid_kernel_milestone.md
docs/exec-plans/version-arch/v1_advanced_hybrid_kernel_arch.md
benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506/report.md
benchmarks/rag_quality/v1_hybrid_provider_reset/report.md
```

## 必须记住的事实

```text
FinanceBench JSONL 文件是冻结测评产物，不是运行时存储。
运行时存储是 Postgres + Qdrant。
V1 cache 已实现，后端是 Postgres query_cache，不是 Redis。
Redis Queue 属于 V2 Research Runtime。
V1 有检索测评证据，但还没有完整生成式答案可靠性测评。
```

## 改代码前

如果任务涉及 V0/V1 query、retrieval、evidence、cache 或测评：

```text
1. 读 docs/exec-plans/version-arch/v1_advanced_hybrid_kernel_arch.md。
2. 读 docs/exec-plans/milestone/v1_advanced_hybrid_kernel_milestone.md 的未完成项。
3. 如果会影响检索质量，读对应测评报告。
```

如果任务涉及 V2：

```text
1. 读 docs/Design-docs/02_V2_RESEARCH_RUNTIME.md。
2. 读 docs/exec-plans/milestone/v1_advanced_hybrid_kernel_milestone.md。
3. 先做 research job 脚手架，不要直接承诺 planner / 报告生成器质量。
```

## 更新规则

行为变化时：

```text
实际架构变化 -> 更新 version-arch。
状态、取舍、未完成项变化 -> 更新 milestone。
质量结论变化 -> 更新测评文档。
不要把长实现手册塞进 AGENTS.md。
```

## 常用命令

基础语法检查：

```bash
python -m compileall -q src/atlas scripts
```

FinanceBench 仅检索测评：

```bash
ATLAS_RETRIEVAL_MODE=hybrid \
ATLAS_BM25_ENABLED=true \
ATLAS_QDRANT_COLLECTION=atlas_financebench_v1 \
python -m atlas.benchmark.financebench_retrieval \
  --cases evals/financebench_cases.yaml \
  --modes dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker \
  --top-k 10
```

## 护栏

```text
不要把 Design-docs 当作实现事实。
不要声称 V1 实现了 Redis。
不要声称 V2 已实现，除非 ResearchJob / API / worker / artifacts 已存在。
不要用 retrieval-only 指标证明答案可靠性。
```
