# KBot 4.0 文档

本目录只描述当前 KBot 4.0。历史设计、3.x 改造过程和已完成的逐步实施记录由
Git 历史保存，不再作为有效文档。

## 架构

- [系统架构](architecture/overview.md)：服务边界、运行拓扑和依赖规则。
- [Knowledge Core](architecture/knowledge-core.md)：文件入库、解析、索引和二阶段检索。
- [Agent Runtime](architecture/agent-runtime.md)：Execution Spec、Skill、记忆、Artifact 和 SSE。
- [AIOps Agent](architecture/aiops-agent.md)：监控、诊断、HITL、审批执行和报告。
- [AIOps Agent 专业 DBA 对话诊断详细设计](architecture/aiops-agent-chat-diagnosis.md)：Turn、Skill、证据、表结构、API 与 SSE。
- [Model Serving](architecture/model-serving.md)：模型注册、托管进程和功能模型绑定。
- [身份与 API](architecture/security-and-api.md)：Domain、用户 Token、App API Key 和内部 AuthContext。
- [App API Key 安全设计](architecture/app-api-key-security.md)：App 绑定、Scope、Agent 白名单、轮换与撤销。
- [仓库结构](architecture/repository-layout.md)：代码、DDL、配置、测试和工具的归属。

## 产品说明

- [Agent 完整聊天流程](product/agent-chat.md)
- [知识入库、解析与检索](product/knowledge-lifecycle.md)
- [AIOps Agent 产品能力](product/aiops-agent.md)
- [AIOps Agent 专业 DBA 对话诊断设计](product/aiops-agent-chat-diagnosis.md)
- [AIOps 功能介绍与 PPT 生成说明](product/aiops-ppt-brief.md)
- [Slack 集成](product/slack-integration.md)

这些文档面向产品演示和 PPT 编写；精确接口仍以 OpenAPI 快照为准。

## 部署与运维

- [部署指南](operations/deployment.md)
- [AIOps观测栈生产自动化部署](operations/aiops-observability-production-deployment.md)
- [AIOps Oracle观测栈人工安装与运维](operations/aiops-observability-manual-deployment.md)
- [配置指南](../configuration/README.md)
- [Oracle 初始化](../database/oracle/README.md)
- [脚本说明](../scripts/README.md)
- [开发日志页面](../tools/dev_console/README.md)

## 契约

`openapi/` 保存公开和内部 API 的冻结快照。快照由应用生成，不手工维护字段。
代码中的 Pydantic DTO、Oracle DDL 和配置模型分别是 API、数据和配置的最终事实源。

## 维护规则

1. 当前行为变化时更新对应主题文档，不新增“步骤 N”“最终版”或重复方案。
2. 尚未实现的设想写入 Issue，不混入当前架构说明。
3. 精确字段、索引和路由链接到源码或快照，不在多份 Markdown 中复制。
4. 过期文档直接删除，需要时从 Git 历史恢复。
