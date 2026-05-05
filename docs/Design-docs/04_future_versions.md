# V3+：后续版本路线图

> 本文档只保留后续版本的方向，不展开完整实现细节。V3 之后的能力都应该建立在 V0/V1/V2 已经稳定的前提上。

---

## 1. 为什么后续版本先不展开

Atlas 后续可以扩展得非常大：GraphRAG、结构化数据查询、Kafka/Flink 流式写入、Memory、Skills、MCP、云端生产部署、事件订阅系统都可以做。但这些能力不应该在第一阶段同时展开。原因很简单：它们每一个都是一个独立工程问题，如果和核心 RAG/Agent 内核同时做，会让系统很快变成难以落地的大型工程蓝图。

更合理的方式是先完成前三版：V0 证明系统能基于证据回答，V1 证明系统能更准确地检索和判断证据，V2 证明系统能处理复杂异步研究任务。只有这三件事稳定以后，后续版本才有清晰落点。

后续版本的顺序建议如下：

```text
V3：Graph Context
V4：Structured Data Context
V5：Streaming Ingestion
V6：Memory & Skills
V7：MCP Tool Layer
V8：Cloud Production
V9：Event Path
```

这个顺序不是唯一答案，但它比较符合工程依赖关系。Graph 和结构化数据是上下文能力；Kafka/Flink 是写入能力；Memory/Skills 是复用能力；MCP 是工具标准化；Cloud Production 是部署和运维；Event Path 是主动推送能力。

---

## 2. V3：Atlas Graph Context

V3 的目标是加入轻量 GraphRAG，让系统能处理关系型问题。V0/V1/V2 主要围绕文本证据工作，擅长回答“文档里说了什么”。但有些问题天然是关系型的，例如“哪些公司和这家公司有供应链关系”、“这个风险会影响哪些实体”、“这些文档里出现的人、公司和产品之间是什么关系”。

V3 不应该一开始就做一个巨大知识图谱。更合理的做法是先做 Graph Context，也就是从已有 chunks 中抽取实体和关系，并把这些关系和原文证据绑定起来。系统回答关系问题时，先用图找到相关实体和关系，再回到原文 chunk 找证据，最后生成带引用的答案。

V3 初期可以先用 Postgres graph tables，而不是立刻上 Neo4j。比如：

```text
graph_nodes
  node_id
  name
  type
  aliases
  metadata

graph_edges
  edge_id
  source_node_id
  target_node_id
  relation
  confidence
  source_chunk_id
  metadata
```

等关系数量变大、查询复杂度提高，再迁移到 Neo4j。这样可以避免一开始就引入图数据库运维复杂度。

V3 的完成标准可以是：

```text
1. 能从 chunk 中抽取实体。
2. 能抽取基础关系。
3. 每条关系能追溯到 source_chunk_id。
4. 能回答简单 1-hop / 2-hop 关系问题。
5. 图检索结果能和 V1 Hybrid Retrieval 的文本证据融合。
6. Graph 检索失败时，系统能退回普通 Hybrid RAG。
```

V3 的原则是：Graph 是增强项，不是主路径。不要让 GraphRAG 影响 V1/V2 的稳定性。

---

## 3. V4：Structured Data Context

V4 的目标是让系统能处理结构化数据和数值问题。文档型 RAG 擅长回答“文本里怎么说”，但它不擅长计算。例如用户问“过去四个季度收入增长率是多少”、“三家公司毛利率怎么比较”、“最近三个月哪个指标变化最大”，这种问题不能只靠向量检索。

V4 可以引入 DuckDB 或 ClickHouse。DuckDB 更适合本地和中小规模分析，ClickHouse 更适合高吞吐、时序和大规模 OLAP。V4 不一定一开始就上 ClickHouse，可以先用 DuckDB 证明 Text-to-SQL 和数值分析能力，再根据数据规模升级。

V4 的核心能力包括：

```text
1. 表格数据导入。
2. 指标 schema 管理。
3. Text-to-SQL 或模板化 SQL 生成。
4. SQL 执行和结果验证。
5. 数值结果与文本证据融合。
6. 对计算结果给出引用或数据来源。
```

V4 需要特别注意 SQL 生成安全。系统不能让模型随意生成任意 SQL 执行。初期可以使用模板化 SQL，把模型的任务限制为填参数，而不是从零写 SQL。这样更安全，也更容易评估。

---

## 4. V5：Streaming Ingestion

V5 才开始引入 Kafka/Flink 或类似流处理系统。它的目标是让 Atlas 支持持续数据流入，而不是只靠批量导入。

在 V0-V2 中，文档导入可以是手动或批量的。到了 V5，如果系统需要持续接收新闻、公告、财报、API 更新、Webhook，那么就需要写路径。写路径的职责是把外部数据稳定地转化成系统内部的 documents、chunks、vectors、entities 和 events。

V5 的核心组件可以包括：

```text
Collector：接入外部数据源。
Kafka：事件总线和持久化日志。
Flink：流式清洗、去重、chunking、embedding、分发。
DLQ：失败消息队列。
Reconciliation Job：定期对账修复不一致。
```

V5 需要明确 indexing SLA。例如普通文档进入系统后 1-5 分钟可检索，高优先级新闻 30 秒内可检索。不要模糊地说“实时”，因为实时会带来很高成本。不同数据源应该有不同新鲜度目标。

V5 的完成标准可以是：

```text
1. 支持至少一个持续数据源。
2. 新数据进入后能自动产生 chunks 和 embeddings。
3. 写入失败进入 DLQ。
4. 系统能监控 ingest lag。
5. 读路径能检索到新数据。
6. 有基本 reconciliation 机制。
```

---

## 5. V6：Memory & Skills

V6 的目标是让系统开始积累经验。V0-V2 的系统每次任务基本从零开始。V6 开始，Atlas 应该能记住工作区上下文、用户偏好、历史研究报告、常用模板和成功工作流。

Memory 可以分成两类。第一类是结构化 memory，例如用户偏好、工作区配置、常用数据源、报告格式偏好。第二类是语义 memory，例如历史 research jobs、过往报告、常见失败案例、成功的 retrieval traces。

Memory 表可以包含：

```text
memory_id
workspace_id
memory_type
content
embedding
importance
source
created_at
expires_at
```

Skills 则是可复用的工作流程。一个 skill 不应该只是一个 prompt，而应该是一个目录，包括说明、输入输出格式、工具列表、示例、评估用例和失败案例。

示例：

```text
skills/
  market_research/
    SKILL.md
    prompt.md
    tools.json
    eval_cases.yaml
    examples/

  citation_audit/
    SKILL.md
    verify_rules.md
    eval_cases.yaml
```

V6 的完成标准可以是：

```text
1. 支持 workspace-level memory。
2. 支持 semantic memory retrieval。
3. 支持 skill directory。
4. Research job 可以选择 skill。
5. Skill 有独立 eval cases。
6. 失败案例可以写入 memory，供后续规避。
```

---

## 6. V7：MCP Tool Layer

V7 的目标是标准化工具层。前面的版本里，系统可能已经有 database query、retrieval、citation verification、file access、code execution 等工具。V7 需要把这些工具整理成统一接口，并逐步支持 MCP 或 MCP-ready 方式。

工具层的核心不是“用了 MCP”这个标签，而是工具调用是否可控、可追踪、可评估。每个工具都应该有明确的 input schema、output schema、权限范围、超时、重试策略和 trace。

内部工具接口可以先这样设计：

```text
Tool
  name
  description
  input_schema
  output_schema
  run(input) -> output
```

常见工具包括：

```text
hybrid_search
graph_lookup
sql_query
citation_verify
python_exec
file_lookup
web_search
```

V7 的完成标准可以是：

```text
1. 所有工具统一注册。
2. 工具调用有 trace。
3. 工具有权限和超时控制。
4. Research Runtime 可以选择工具。
5. 支持 MCP adapter 或 MCP-compatible server。
```

---

## 7. V8：Cloud Production

V8 的目标是生产化部署。前面的版本可以通过 Docker Compose 或简单云服务跑起来，但真正生产需要更多能力：autoscaling、observability、rate limit、worker pool、backup、security、deployment pipeline、cost dashboard。

V8 可以引入 Kubernetes，也可以先用 ECS、Cloud Run、Render、Fly.io 等平台。是否上 Kubernetes 取决于系统规模，不要为了炫技而过早上 K8s。

V8 的核心能力包括：

```text
1. API service 和 worker service 分离。
2. Redis queue 支持 worker pool。
3. Postgres / Qdrant / Redis 使用 managed service 或高可用部署。
4. OpenTelemetry 或 Langfuse 记录 trace。
5. Prometheus / Grafana 监控 QPS、延迟、错误率、成本。
6. Rate limit 和 basic auth。
7. Backup 和 restore 流程。
8. CI/CD 和环境隔离。
```

V8 的完成标准不是“部署上云”，而是系统在云上具备基本运维能力。能够看到它是否健康，能定位错误，能扩容 worker，能限制用户滥用，能从备份恢复。

---

## 8. V9：Event Path

V9 的目标是让 Atlas 从被动问答系统升级为主动事件响应系统。也就是说，系统不只是等用户问问题，还能在新数据进入时判断是否值得通知用户。

事件路径可以这样设计：

```text
新数据进入系统
  ↓
Event Processor 判断是否重要
  ↓
Trigger Engine 匹配用户订阅
  ↓
Importance Classifier 过滤低价值事件
  ↓
Reasoner 生成“为什么重要”的解释
  ↓
Notification Dispatcher 推送给用户
```

V9 需要依赖 V5 的 streaming ingestion，也需要依赖 V6 的用户偏好和 memory。否则系统不知道用户关心什么，也不知道新事件对谁重要。

V9 的完成标准可以是：

```text
1. 支持用户订阅条件。
2. 新事件能触发匹配。
3. 事件经过重要性过滤。
4. 系统能生成简短解释。
5. 支持至少一种推送方式，例如 email 或 SSE。
6. 有频率限制和去重机制。
```

---

## 9. 后续版本的实施顺序建议

后续版本不一定严格顺序实现，但建议遵守依赖关系。

```text
如果你的目标是展示 RAG/Agent 技术深度：
  优先 V3 Graph Context 和 V6 Memory & Skills。

如果你的目标是展示数据工程和云部署能力：
  优先 V5 Streaming Ingestion 和 V8 Cloud Production。

如果你的目标是展示金融/数据分析能力：
  优先 V4 Structured Data Context。

如果你的目标是展示现代 Agent 工具生态：
  优先 V7 MCP Tool Layer。

如果你的目标是展示产品化能力：
  优先 V9 Event Path。
```

但无论选择哪条路线，都不建议跳过 V0/V1/V2。因为后续所有版本都依赖可靠的证据链、检索质量、trace、eval 和异步任务系统。

---

## 10. 总结

后续版本的方向很多，但核心判断标准只有一个：这个能力是否增强了 Atlas 的证据能力、上下文能力、执行能力或生产能力。

GraphRAG 增强关系上下文；结构化数据增强数值上下文；Kafka/Flink 增强持续写入；Memory/Skills 增强复用能力；MCP 增强工具调用；Cloud Production 增强部署和运维；Event Path 增强主动响应。

不要为了技术名词而加技术。每个版本都应该能回答：

```text
它解决了什么真实问题？
它依赖前面哪些能力？
它失败时能不能降级？
它有没有 eval 或 metrics 证明价值？
```

只要这个原则不变，Atlas 可以稳步从一个 RAG Kernel 演进成一个真正的 Agentic Knowledge Platform。
