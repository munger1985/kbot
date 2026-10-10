# KBot 4.0 文档

本目录只描述当前 KBot 4.0。历史设计、3.x 改造过程和已完成的逐步实施记录由
Git 历史保存，不再作为有效文档。

## 架构

- [系统架构](architecture/overview.md)：服务边界、运行拓扑和依赖规则。
- [Knowledge Core](architecture/knowledge-core.md)：文件入库、解析、索引和二阶段检索。
- [Agent Runtime](architecture/agent-runtime.md)：Execution Spec、Skill、记忆、Artifact 和 SSE。
- [AIOps Agent](architecture/aiops-agent.md)：监控、诊断、HITL、审批执行和报告。
- [AIOps 跨项目能力对齐与双向演进基线](architecture/aiops-cross-project-alignment.md)：KBot4 与 Ammolite 当前差异、双向吸收顺序和共同验收门禁。
- [AIOps Agent 专业 DBA 对话诊断详细设计](architecture/aiops-agent-chat-diagnosis.md)：Turn、Skill、证据、表结构、API 与 SSE。
- [AIOps 智能运维功能入口技术设计](architecture/aiops-conversation-starters.md)：新会话功能菜单、参数表单和自由提问入口。
- [AIOps 数据库实施方案 Runbook 技术设计](architecture/aiops-implementation-runbooks.md)：当前 ADG/DGPDB 契约、编译和导出边界。
- [核心授权策略](architecture/core-authorization-policy.md)：公开和内部资源的统一授权规则。
- [Model Serving](architecture/model-serving.md)：模型注册、托管进程和功能模型绑定。
- [身份与 API](architecture/security-and-api.md)：Domain、用户 Token、App API Key 和内部 AuthContext。
- [App API Key 安全设计](architecture/app-api-key-security.md)：App 绑定、Scope、Agent 白名单、轮换与撤销。
- [仓库结构](architecture/repository-layout.md)：代码、DDL、配置、测试和工具的归属。

## 产品说明

- [Agent 完整聊天流程](product/agent-chat.md)
- [知识入库、解析与检索](product/knowledge-lifecycle.md)
- [AIOps Agent 产品能力](product/aiops-agent.md)
- [AIOps Agent 专业 DBA 对话诊断设计](product/aiops-agent-chat-diagnosis.md)
- [智能运维新会话功能引导](product/aiops-conversation-starters.md)
- [AIOps 数据库实施方案 Runbook](product/aiops-implementation-runbooks.md)
- [AIOps 正式报告与导出设计](product/aiops-reporting.md)
- [AIOps 产品路线图](product/aiops-roadmap.md)
- [AIOps 功能介绍与 PPT 生成说明](product/aiops-ppt-brief.md)
- [Slack 集成](product/slack-integration.md)

这些文档面向产品演示和 PPT 编写；精确接口仍以 OpenAPI 快照为准。

## 部署与运维

- [部署指南](operations/deployment.md)
- [AIOps 网络边界](operations/aiops-network-boundary.md)
- [AIOps观测栈生产自动化部署](operations/aiops-observability-production-deployment.md)
- [AIOps Oracle观测栈人工安装与运维](operations/aiops-observability-manual-deployment.md)
- [配置指南](../configuration/README.md)
- [Oracle 初始化](../database/oracle/README.md)
- [脚本说明](../scripts/README.md)
- [开发日志页面](../tools/dev_console/README.md)

## 详细设计与实施方案

- [AIOps 跨项目统一实施方案](proposals/aiops-cross-project-unified-implementation-plan.md)：按七个批次完成双向能力吸收、实时监控、双前端和真实环境联合验收。
- [AIOps 实时监控 Dashboard 集成详细设计](proposals/aiops-monitoring-dashboard-integration-detailed-design.md)：第一阶段对齐 Prometheus/Zabbix 的 Oracle、MySQL、PostgreSQL 多实例图表及诊断闭环；Zabbix只保留外部接入，OEM暂缓，并要求与Ammolite Cube同步实施。
- [AIOps 数据库实施文档中心详细设计](proposals/aiops-database-implementation-library-detailed-design.md)：已实施的 Runbook v3、RAC、RMAN、补丁、升级、迁移、克隆和 ADG 演练基准。
- [AIOps 运维知识库详细设计](proposals/aiops-operations-knowledge-base-detailed-design.md)：用户手册提炼、诊断案例治理、KC 内部索引、Agent 检索与迁移方案。

## 契约

`openapi/` 保存公开和内部 API 的冻结快照。快照由应用生成，不手工维护字段。
代码中的 Pydantic DTO、Oracle DDL 和配置模型分别是 API、数据和配置的最终事实源。

## 维护规则

1. 当前行为变化时更新对应主题文档，不新增“步骤 N”“最终版”或重复方案。
2. 尚未实现的设想写入 Issue，不混入当前架构说明。
3. 精确字段、索引和路由链接到源码或快照，不在多份 Markdown 中复制。
4. 过期文档直接删除，需要时从 Git 历史恢复。
