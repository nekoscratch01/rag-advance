# FinanceBench RAG Quality Benchmarks

这个目录只放 FinanceBench RAG 质量测评相关的人类可读报告。

```text
reports/     实验报告、结论、观察日志
```

机器生成的 raw run artifacts 当前保留在：

```text
benchmarks/financebench/retrieval_runs/
benchmarks/financebench/runs/
```

后续如果增加系统吞吐、并发压测、服务稳定性等 benchmark，应放到独立测评项目目录中，
不要和 RAG quality 报告混在一起。
