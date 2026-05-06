# Atlas 文档地图

`docs/` 是这个仓库的记录系统。这里不追求把所有信息塞进一个巨大文件，而是让每份文档有清楚的职责、边界和新鲜度。

## 文档层级

```text
docs/Design-docs/
  设计意图和版本路线图。
  用来看每个版本原本要证明什么。

docs/exec-plans/milestone/
  执行里程碑。
  用来看某个版本实际推进了什么、相对设计改了什么、还缺什么证据。

docs/exec-plans/version-arch/
  实际实现架构。
  当代码现实和 Design-docs 不一致时，看这里。这里解释真实组件、架构图、
  存储边界和实现取舍。

benchmarks/
  实验记录和测量结果。
  用来看指标、消融、失败案例和实验时间线。
```

## 当前版本状态

```text
V0 Atlas Kernel
  状态：已实现

V1 Atlas Advanced Hybrid Kernel
  状态：Design 主链路已实现；检索质量已有测评证据；生成式答案可靠性已有 runner，仍需全量报告
  设计：docs/Design-docs/01_V1_ADVANCED_HYBRID_KERNEL.md
  执行里程碑：docs/exec-plans/milestone/v1_advanced_hybrid_kernel_milestone.md
  实际架构：docs/exec-plans/version-arch/v1_advanced_hybrid_kernel_arch.md
  FinanceBench 检索实验：benchmarks/rag_quality/financebench/retrieval_runs/full_v1_retrieval_20260506/report.md
  Provider reset 消融：benchmarks/rag_quality/v1_hybrid_provider_reset/report.md

V2 Atlas Research Runtime
  状态：仍是设计阶段
  设计：docs/Design-docs/02_V2_RESEARCH_RUNTIME.md
```


## 阅读顺序

建议按这个顺序读：

```text
1. docs/README.md
2. AGENTS.md
3. 相关版本的 Design-docs
4. 相关版本的 milestone 文档
5. 相关版本的 version-arch 文档
6. 如果涉及质量判断，再读测评报告
```

一句话原则：

```text
Design-docs 记录意图。
exec-plans/milestone 记录执行状态。
exec-plans/version-arch 记录实际架构。
benchmarks 记录实验事实。
```
