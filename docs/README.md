# KBot 文档

本目录区分当前实现说明与明确标注状态的目标设计。历史设计、3.x 改造过程和已完成的逐步实施记录由
Git 历史保存，不再作为有效目标。知识基础组件已按下列新方案编码，正式 App 业务入口尚未切换。

## 知识基础能力（基础组件已编码，产品接入待完成）

- [基础能力设计](architecture/knowledge-answering-capability.md)：所有权、App 独占数据、Runtime 统一问答与文档优先级。
- [解析与召回算法详细设计](proposals/app-owned-knowledge-algorithm-design.md)：多模态解析、独立召回、RRF、覆盖/预算和图文双向关联；不实现独立模型重排。
- [App 接入与持久化详细设计](proposals/app-owned-knowledge-integration-detailed-design.md)：固定端口、委托鉴权、流式接收、App 物理表及实际模型交付。
- [实施计划](operations/app-owned-knowledge-implementation-plan.md)：K0–K7 工作包、依赖门禁、双平台工件和批准后的切换/回退。
- [首批编码记录](operations/app-owned-knowledge-k0-k1-progress.md)：K0/K1 已实现组件、验证证据和未完成范围；不代表现网已切换。
- [知识库 App 整体设计](product/knowledge-retrieval-ui-alignment.md)：七页分工、三档解析及资料治理流程。
- [按库绑定模型与三档文档解析](product/knowledge-library-parse-strategies.md)：创建和编辑只配模型，按文件版本审批选档，任务与重试读取冻结规格；适用于知识库 App 和 AIOps 运维资料库。
- [K2 实施记录](operations/app-owned-knowledge-k2-progress.md)：App 独占存储、计算协议、接收恢复与索引准备；知识检索 App 的 Oracle/PostgreSQL 定向验收各 18 项通过，AIOps Oracle 实库验收未执行。
- [基础组件编码与离线接入记录](operations/app-owned-knowledge-foundation-progress.md)：可复用 K3/K4 基础实现、解析器/模型端口、可选 rg、离线闭环与正式 App 配套/真实验收的剩余边界。
- [可选 ripgrep 组件](proposals/optional-ripgrep-text-scan-design.md)：R1 独立可选扩展，不是核心部署依赖。

## 架构

- [系统架构](architecture/overview.md)：服务边界、运行拓扑和依赖规则。
- [Knowledge Core](architecture/knowledge-core.md)：切换前的当前实现说明，不作为新知识能力的目标存储边界。
- [Agent Runtime](architecture/agent-runtime.md)：Execution Spec、Skill、记忆、Artifact 和 SSE。
- [AIOps Agent](architecture/aiops-agent.md)：监控、诊断、HITL、审批执行和报告。
- [DBA Agent 能力、回答、报告与案例统一设计（首批实现、整体待验收）](proposals/aiops-dba-agent-capability-redesign.md)：问题驱动调查、统一证据与判断、环境化方案、效果验证及知识回流。
- [DBA Agent 实施状态与发布检查](operations/aiops-dba-agent-implementation.md)：调查记录、回答/报告/案例一致性、指标发现授权及未交付范围。
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
- [AIOps 数据库备份与空库恢复演练设计（待确认）](proposals/aiops-data-protection-design.md)
- [AIOps 数据保护实施计划（待确认）](operations/aiops-data-protection-implementation-plan.md)
- [AIOps Agent 专业 DBA 对话诊断设计](product/aiops-agent-chat-diagnosis.md)
- [智能运维新会话功能引导](product/aiops-conversation-starters.md)
- [AIOps 数据库实施方案 Runbook](product/aiops-implementation-runbooks.md)
- [AIOps 正式报告与导出设计](product/aiops-reporting.md)
- [AIOps 产品路线图](product/aiops-roadmap.md)
- [AIOps 功能介绍与 PPT 生成说明](product/aiops-ppt-brief.md)

这些文档面向产品演示和 PPT 编写；精确接口仍以 OpenAPI 快照为准。

## 部署与运维

- [部署指南](operations/deployment.md)
- [KBot 1.0 完全离线安装手册](operations/kbot-1.0-complete-offline-installation.md)
- [AIOps 网络边界](operations/aiops-network-boundary.md)
- [AIOps 通知第一阶段基础修复](operations/aiops-notification-foundation-phase-1.md)：实施范围、部署前提、验证结果和未执行事项。
- [AIOps观测栈生产自动化部署](operations/aiops-observability-production-deployment.md)
- [AIOps Oracle观测栈人工安装与运维](operations/aiops-observability-manual-deployment.md)
- [配置指南](../configuration/README.md)
- [Oracle 初始化](../database/oracle/README.md)
- [脚本说明](../scripts/README.md)
- [开发日志页面](../tools/dev_console/README.md)

## 详细设计与实施方案

- [AIOps 飞书通知接入详细设计（代码已实现，待真实验收）](proposals/aiops-feishu-notification-detailed-design.md)：平台通知模块、独立 Worker、身份绑定、可靠投递及原任务入口。
- [飞书通知部署与验证](operations/aiops-feishu-notification-delivery.md)：默认关闭、无损数据库工件、权限、真实测试企业启用步骤与验收边界。
- [KBot 1.0 OCIR 生产部署与初始化方案](proposals/ocir-1.0-production-deployment-plan.md)：正式镜像的 OCI 基础设施、Digest 锁定门禁、空库初始化、验收与回滚方案。
- [AIOps 跨项目统一实施方案](proposals/aiops-cross-project-unified-implementation-plan.md)：按七个批次完成双向能力吸收、实时监控、双前端和真实环境联合验收。
- [AIOps SQL 诊断与窗口分析详细设计（核心实现、待真实验收）](proposals/aiops-sql-diagnosis-detailed-design.md)：三库统一身份与窗口、多源假设驱动调查、诊断解决方案及真实运维闭环验收。
- [SQL 诊断实施与发布检查](operations/aiops-sql-diagnosis-implementation.md)：本轮代码、剩余切片、验证入口和受控 Prompt 发布要求。
- [AIOps 实时监控 Dashboard 集成详细设计](proposals/aiops-monitoring-dashboard-integration-detailed-design.md)：Prometheus/Zabbix 多实例图表与统一 Profile；OEM 接入已实现，指标与硬件扩展待后续讨论。
- [OEM 服务接入与业务系统数据库绑定](operations/aiops-oem-onboarding.md)：Source 仅连接 OEM，数据库按引擎发现、选择并保存精确 GUID；本地与只读联调已验证，尚未部署。
- [AIOps 数据库实施文档中心详细设计](proposals/aiops-database-implementation-library-detailed-design.md)：已实施的 Runbook v3、RAC、RMAN、补丁、升级、迁移、克隆和 ADG 演练基准。
- [AIOps 运维知识库详细设计](proposals/aiops-operations-knowledge-base-detailed-design.md)：用户手册与诊断案例业务治理；其旧 KC 存储/固定助手实施假设由知识基础能力新方案替代。

## 契约

`openapi/` 保存公开和内部 API 的冻结快照。快照由应用生成，不手工维护字段。
代码中的 Pydantic DTO、Oracle DDL 和配置模型分别是 API、数据和配置的最终事实源。

## 维护规则

1. 当前行为变化时更新对应主题文档，不新增“步骤 N”“最终版”或重复方案。
2. 未实施的批准目标单独标明状态和生效范围，并链接详细设计/实施计划，不冒充当前实现。
3. 精确字段、索引和路由链接到源码或快照，不在多份 Markdown 中复制。
4. 过期文档直接删除，需要时从 Git 历史恢复。
