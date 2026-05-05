# Atlas Kernel 架构路线图 Overview

> 本文档是 Atlas 分阶段实现路线的总览。它不把 Atlas 一开始就设计成一个完整的大型平台，而是把系统拆成几个可以独立落地、独立验证、逐步扩展的版本。

---

## 1. Atlas 要做的事情

Atlas 的目标不是做一个普通的“文档问答机器人”。普通 RAG 系统通常是把文档切块、写入向量库，然后用户一问，系统检索几个 chunk 交给大模型回答。这种系统可以做 demo，但很难进入更真实的工程场景，因为它通常无法解释为什么这样回答，也无法稳定判断证据是否足够，更无法在复杂任务中保存过程、复盘错误、评估质量。

Atlas 更适合被定义为一个**证据优先的 RAG / Agent 后端运行时**。它接收用户的问题，先寻找证据，再生成答案；如果证据不足，它应该明确说不足，而不是让模型自由发挥；如果证据冲突，它应该展示冲突，而不是擅自选择一边；如果问题复杂，它应该拆成多个子问题，异步执行，并保存完整的执行记录。

Atlas 的核心价值可以概括成一句话：

```text
先找到证据，再生成结论；
结论必须能回到证据；
每一步都应该能被追踪、评估和复现。
```

这个原则决定了整个系统的设计方向。Atlas 不是先追求“Agent 看起来很聪明”，而是先追求“系统回答是否可靠”。如果系统不能知道自己用了哪些证据、为什么用了这些证据、答案是否真的被证据支持，那么后面加再多 Agent、GraphRAG、MCP、Memory、Skills，也只是在一个不可靠的底座上堆功能。

---

## 2. 为什么不直接实现完整 Atlas

原始 Atlas 设计中包含 Kafka、Flink、Qdrant、Neo4j、ClickHouse、Redis、LangGraph、FastAPI Gateway、事件路径、订阅系统、GitOps、灾备和大规模容量规划。这些设计方向本身是合理的，但它们更像是 Atlas V3/V4 之后的目标架构，而不是第一版应该实现的内容。

如果第一天就把 Kafka、Flink、Neo4j、ClickHouse 全部接进来，系统会很快变成基础设施项目，而不是 RAG / Agent 项目。团队会花大量时间处理 Topic、Consumer Lag、Flink Checkpoint、Graph Schema、ClickHouse 分区、Kubernetes 配置，却还没有证明最重要的问题：系统到底能不能可靠地基于证据回答问题。

因此，更合理的路线是先建立一个“读路径内核”。所谓读路径，就是用户提问以后系统如何检索、整理证据、生成答案、输出引用、记录 trace、运行 eval。这条路径是 Atlas 的核心价值所在，也是后续所有能力的基础。

前三版应该聚焦在以下事情上：

```text
V0：先让系统能基于文档可靠问答。
V1：再把检索质量、引用质量、置信度判断做强。
V2：再让复杂问题变成异步研究任务。
```

等 V0/V1/V2 稳定以后，再去做 GraphRAG、结构化数据、流式写入、事件触发、Memory、Skills、MCP 和云端大规模部署，系统才会自然生长，而不是一开始被复杂度压垮。

---

## 3. 版本路线总览

| 版本 | 名称 | 核心目标 | 主要产物 |
|---|---|---|---|
| V0 | Atlas Kernel | 最小可用 RAG 内核 | 文档入库、向量检索、带引用回答、trace、基础 eval |
| V1 | Atlas Hybrid Kernel | 提升检索质量和答案可靠性 | hybrid retrieval、reranker、cache、Critic Lite、质量对比 |
| V2 | Atlas Research Runtime | 支持复杂异步研究任务 | research job、planner、subquestions、evidence pack、report artifact |
| V3 | Atlas Graph Context | 加入轻量 GraphRAG | entity、relationship、graph retrieval、文本证据融合 |
| V4 | Structured Data Context | 加入结构化数据查询 | DuckDB/ClickHouse、Text-to-SQL、数值分析 |
| V5 | Streaming Ingestion | 加入持续写入路径 | Kafka/Flink、collector、indexing SLA、DLQ |
| V6 | Memory & Skills | 加入长期记忆和技能系统 | workspace memory、semantic memory、skill directory |
| V7 | MCP Tool Layer | 工具层标准化 | MCP-ready database/file/browser/code tools |
| V8 | Cloud Production | 生产级云部署 | autoscaling、rate limit、observability、multi-worker |
| V9 | Event Path | 主动事件响应 | notification queue、trigger engine、subscriptions |

这个版本顺序的重点是：每一版都能独立提供价值，并且不会把下一版的风险提前压到当前版本。V0 不需要 V1 才能跑，V1 不需要 V2 才有意义，V2 不需要 GraphRAG 才能展示复杂任务能力。这样做可以避免“大爆炸式架构”，也更适合作为一个可以写进简历、可以部署、可以持续迭代的工程项目。

---

## 4. 三条能力主线

Atlas 的前三版虽然看起来是在做不同模块，但背后其实有三条主线。

第一条主线是**证据链**。系统必须知道每个答案来自哪些文档、哪些 chunk、哪些 evidence。证据链不是最终展示时才补上去的东西，而应该从数据模型阶段就存在。Document、Chunk、Evidence、Claim、Citation、QueryRun 都应该在 V0 就定义清楚。后面 V1 的 reranker、V2 的 research report，都是在这条证据链上继续增强。

第二条主线是**执行记录**。每次查询都应该记录系统做了什么：用了什么 query rewrite，走了哪些 retriever，召回了哪些 chunk，reranker 怎么排序，Critic 怎么判断，最后 Synthesizer 用了哪些证据。没有执行记录，就无法 debug；无法 debug，就无法做 eval；无法 eval，就无法迭代质量。

第三条主线是**预算控制**。Atlas 不能让 Agent 无限搜索，也不能让复杂 query 把所有资源吃光。每次查询和每个 research job 都应该有时间预算、token 预算、检索次数预算、模型调用预算和成本预算。预算不是运维阶段才加的限制，而是 Agentic 系统的核心设计之一。

这三条主线会贯穿 V0、V1、V2：

```text
V0：建立证据链和基础 trace。
V1：让证据链更准，让 trace 更细。
V2：让复杂研究任务也遵守证据链、trace 和预算。
```

---

## 5. 系统的高层结构

前三版可以先采用一个相对简单的后端结构。它不需要完整 Kubernetes，也不需要复杂流处理。一个合理的起点是 FastAPI + Postgres + Qdrant + Redis + Worker。

```text
┌──────────────────────────────────────────────┐
│                Client / CLI / API             │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                FastAPI Gateway                │
│  接收查询、导入文档、提交 research job、返回结果 │
└───────────────────────┬──────────────────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Ingestion     │ │ Query Runtime │ │ Research Job  │
│ 文档解析/切块  │ │ 检索/回答      │ │ 异步研究任务   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Postgres      │ │ Qdrant        │ │ Cache/Queue  │
│ 元数据/trace   │ │ 向量检索       │ │ V1 cache / V2 queue │
└──────────────┘ └──────────────┘ └──────────────┘
```

在 V0 中，Research Job 还不存在，Redis 也可以不是必须。到了 V1，系统必须有 cache 能力，但本地实现可以先用 Postgres exact cache；Redis cache / rate limit 是可替换的性能后端，不是 V1 语义本身。到了 V2，Redis Queue 或类似队列系统会成为异步 research job 的核心。这个结构的好处是简单、可部署、可扩展，而且后面可以自然接入 Neo4j、ClickHouse、Kafka、Flink，而不用推倒重来。

---

## 6. 核心数据对象

Atlas 的数据模型应该从第一版就围绕“证据”设计，而不是围绕“聊天记录”设计。下面这些对象是系统的基本骨架。

### Document

Document 表示一份进入系统的原始资料。它可以是 PDF、Markdown、TXT、HTML、研报、公告、新闻，也可以是后续版本里的结构化数据源。Document 本身不一定直接用于检索，因为一份文档通常太长。系统会把 Document 切成多个 Chunk。

Document 至少需要包含：

```text
document_id
title
source_uri
file_type
content_hash
language
created_at
metadata
```

其中 content_hash 很重要。它用于去重，也用于判断同一份文档是否已经导入过。metadata 则保留来源、作者、发布日期、source_type 等信息。后面做 metadata filter、source filtering、citation 展示都会依赖这些字段。

### Chunk

Chunk 是文档被切分后的检索单位。系统真正写入向量库、参与检索的不是 Document，而是 Chunk。一个 100 页的 PDF 可能会变成几百个 Chunk。

Chunk 至少需要包含：

```text
chunk_id
document_id
chunk_index
text
page_start
page_end
section_title
token_count
metadata
```

page_start、page_end、section_title 不只是展示字段，它们决定 citation 是否可信。一个回答如果只能说“来自 report.pdf”，可信度远远不如说“来自 report.pdf 第 12-13 页 Risk Factors 部分”。

### Evidence

Evidence 是一次查询中被选中的证据。Chunk 是库存里的片段，Evidence 是这次回答实际使用的片段。

Evidence 至少需要包含：

```text
evidence_id
chunk_id
document_id
text
source_title
page_range
retrieval_score
rerank_score
```

V0 可以直接把 top chunks 当 Evidence。V1 之后，Evidence 应该来自 hybrid retrieval + reranker 的结果。V2 中，一个 research job 会产生多个 subquestion，每个 subquestion 都会产生 Evidence，最后再合并成 Evidence Pack。

### Claim

Claim 是答案中的事实性结论。它应该能映射到一个或多个 Evidence。

```text
claim_id
text
evidence_ids
confidence
```

V0 不一定要做到严格 claim-level citation，但数据模型上应该为它留好空间。到了 V2 的报告生成阶段，claim-level citation 会变得非常重要，因为报告中每个主要结论都应该能追溯到证据。

### QueryRun

QueryRun 记录一次普通查询的完整过程。它不是聊天记录，而是一次系统执行记录。

```text
query_id
user_query
normalized_query
retrieved_chunks
selected_evidence
answer
citations
confidence
latency_ms
model_name
created_at
```

QueryRun 是 eval、debug、trace replay 的核心。如果答案错了，工程师应该能通过 QueryRun 看到：到底是检索没找到证据，还是 reranker 排错了，还是 Synthesizer 写歪了。

### ResearchJob

ResearchJob 是 V2 引入的对象，用来表示一个异步研究任务。

```text
job_id
user_query
status
budget
plan
subquestions
artifact_uri
created_at
started_at
finished_at
```

ResearchJob 和 QueryRun 的区别在于：QueryRun 是短查询，ResearchJob 是长任务。一个 ResearchJob 可能包含多个 QueryRun 或 RetrievalRun，并且会产出完整的 artifacts。

---

## 7. 推荐的前三版技术栈

前三版建议保持技术栈克制，但不要太玩具化。推荐如下：

```text
Backend API：FastAPI
Metadata Store：Postgres
Vector Store：Qdrant
Cache：V1 可用 Postgres exact cache 起步，Redis 可作为性能后端
Queue：V2 用 Redis Queue / RQ / Celery / arq 之一
Worker：V2 用 Celery / RQ / arq 任选其一
Agent Orchestration：V2 再引入 LangGraph
Observability：先自建 trace 表，后面接 Langfuse / OpenTelemetry
Evaluation：自定义 eval + RAGAS / DeepEval 可选
Deployment：Docker Compose 起步，后续 Cloud Run / ECS / Kubernetes
```

这套组合的好处是：V0 可以很快跑起来，V1 可以自然加入 cache 和 reranker，V2 可以自然加 queue、worker pool 和 LangGraph。等系统真的进入 V3/V4 后，再接 Neo4j、ClickHouse、Kafka、Flink，就不是凭空加组件，而是为已验证的能力扩容。

---

## 8. 容量目标总览

前三版的容量目标应该现实一点。不要一开始用“500 QPS、1 亿向量”作为 V0/V1 的完成标准。那是长期生产目标，不是内核验证目标。

| 版本 | 文档数 | Chunk 数 | 查询能力 | 复杂任务能力 |
|---|---:|---:|---:|---:|
| V0 | 1万 - 10万 | 10万 - 100万 | 1 - 5 QPS | 不支持 |
| V1 | 10万 - 50万 | 100万 - 500万 | 5 - 20 QPS | 实验性支持 |
| V2 | 50万 - 200万 | 500万 - 2000万 | 10 - 50 QPS | 5 - 20 个并发 research jobs |
| V3+ | 逐步扩大 | 逐步扩大 | 根据部署资源扩展 | Graph / Structured / Event Path |

这个表不是硬限制，而是设计目标。它表达的是：每一版应该证明的能力不同。V0 证明基本链路，V1 证明检索质量，V2 证明异步复杂任务。等这三个阶段都稳定以后，再谈千万文档、亿级向量和数百 QPS 才更可信。

---

## 9. 从原始设计到分阶段路线的关键调整

这里把主要调整明确列出来，避免实现时混乱。

第一，Kafka/Flink 后移。原始设计中写路径非常完整，但前三版不应该先实现完整流处理。V0-V2 可以先用批量导入和普通 ingestion pipeline。等系统证明读路径有价值以后，再用 Kafka/Flink 支撑持续数据流入。

第二，Neo4j 后移。GraphRAG 很有价值，但它应该是增强项，不应该成为第一版的主路径。V0/V1 先证明文本证据检索和引用可靠。V3 再加入轻量图谱。

第三，ClickHouse 后移。结构化数据查询可以作为 V4 能力。前三版先聚焦文档、chunk、证据、报告。否则系统会同时处理非结构化和结构化两个难题，复杂度过高。

第四，Agent 后移到 V2。V0 不要做 Agent。V1 也不要做复杂 Agent。V2 再做 workflow-first 的研究型 Agent。这样可以避免 Agent 在基础检索还不稳定时放大错误。

第五，eval 前置。评估不是后续增强，而是 V0 就必须有。哪怕一开始只有 50 条 eval cases，也比没有 eval 强得多。后续每次改 chunking、retriever、reranker、prompt，都应该通过 eval 比较效果。

第六，同步 query 和异步 research 分开。普通问答应该走低延迟路径，复杂研究任务应该走 job 队列。不要让所有请求都走完整 Agentic pipeline，否则延迟、成本和并发都会失控。

---

## 10. 最终设计态度

Atlas 的合理路线不是一开始就做一个很大的系统，而是先做一个可信的系统。可信的意思是：它知道自己为什么回答，知道自己引用了什么，知道什么时候证据不足，知道自己的执行过程在哪里，知道如何评估自己变好了还是变差了。

因此，前三版的核心不是“把所有现代 AI Infra 名词都用上”，而是建立一套可以向外解释清楚的工程逻辑：

```text
V0：我能可靠地基于文档回答问题。
V1：我能更准确地找到证据，并判断证据是否足够。
V2：我能把复杂任务拆成可追踪的异步研究流程。
```

做到这三件事以后，Atlas 才真正有资格继续扩展成 GraphRAG、Deep Research、Memory、Skills、MCP 和云端生产系统。
