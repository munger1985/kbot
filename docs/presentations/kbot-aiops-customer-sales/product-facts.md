# KBot AIOps 客户演示稿事实基线

核验日期：2026-10-12。

- 产品定位：面向数据库运维的智能诊断与受控处置平台，不是通用监控替代品，也不是绕过审批的自动化脚本平台。
- 业务入口：对话诊断、监控告警、定时巡检，共享同一条证据、诊断、处置、验证与报告链路。
- 运维目标：公开契约支持 Oracle 19c/26ai、MySQL 8.4、PostgreSQL 16。
- 监控与证据：支持 Prometheus、Zabbix、OEM，以及日志、数据库只读诊断、用户上传材料和知识引用。
- 诊断目录：当前三类数据库共有 154 个版本化只读诊断工具，其中 Oracle 66、MySQL 31、PostgreSQL 57。
- 处置目录：当前 Oracle 与 MySQL 共有 34 个 ACTIVE、`EXECUTABLE_AFTER_APPROVAL` 的动作模板；所有实际执行仍受 Agent-Target 授权、策略、部署开关、逐条人工审批、一次性授权、Executor 复核和处理后验证约束。PostgreSQL 当前以只读诊断为主。
- 证据等级：系统验证、用户提供、知识引用、模型推断四类来源；根因结论区分已确认、很可能、可能、无法判断。
- 报告：故障、性能、巡检、工作负载和处理前后对比等报告均从已保存的事实与证据生成，可追溯版本。
- 部署事实：Linux x86_64、Python 3.10+；KBot 运行数据使用 Oracle Database 26ai Schema；Portal/APEX 只访问 Main API；生产服务可以拆分部署。单机一体化规格是本演示稿给出的项目建议，不是代码声明的最低硬件要求。
- 模型目录：模型目录默认不预置业务模型，实际启用项由部署后的 Model Catalog 决定。代码支持 LLM、VLM、OCR、文本 Embedding、视觉 Embedding 五类进程及相应 Provider。
- AIOps 生产模型约束：规划/诊断 LLM 绑定要求 `local_deepseek`，即客户近端 OpenAI 兼容推理服务；这不等同于在 KBot Python 进程内直接加载大模型权重。

主要核验来源：

- `docs/product/aiops-agent.md`
- `docs/architecture/aiops-agent.md`
- `docs/architecture/model-serving.md`
- `docs/operations/deployment.md`
- `packages/platform_core/src/platform_core/contracts/aiops/configuration.py`
- `services/aiops_agent/src/aiops_agent/diagnostics/catalog/*/manifest.json`
- `services/aiops_agent/src/aiops_agent/actions/catalog/*/manifest.json`
- `services/model_serving/src/model_serving/common/provider_catalog.py`
