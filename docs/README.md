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
  设计：docs/Design-docs/01_v0_atlas_kernel.md

V1 Atlas Hybrid Kernel
  状态：检索质量主路径已实现并完成仅检索测评
  当前索引：docs/exec-plans/milestone/already_implemented.md
  里程碑：docs/exec-plans/milestone/v1_hybrid_kernel_milestone.md
  实际架构：docs/exec-plans/version-arch/v1_hybrid_kernel_arch.md
  实验报告：benchmarks/rag_quality/financebench/reports/full_v1_retrieval_experiment.md

V2 Atlas Research Runtime
  状态：仍是设计阶段
  设计：docs/Design-docs/03_v2_atlas_research_runtime.md
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
