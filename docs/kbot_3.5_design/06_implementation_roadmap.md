# 3.5 实施路线图

本文定义实施顺序，不替代各模块的详细设计。V1 与 V2 可在过渡期并行运行，但 V2 的每一步均不允许读取或回退旧 `File/Chunk` 链路。

## 0. 冻结范围与建立基线

**目标：** 固定以 Domain 为顶层隔离边界的首期 Collection、KM Asset 来源、支持文件类型、权限等级、可用模型与 V1/V2 双轨边界。

**交付物：** App/Domain→Collection 清单、`base.toml` 的 app 标识注入规则、APEX 直连读视图/索引约定、Collection/Binding 管理 API 契约、来源字段映射、样本 Bundle 集、目标指标基线、旧表保留策略、V1/V2 路由与接口命名/错误码约定。

**完成门槛：** 产品、Portal 与后端确认 V2 不兼容旧表模型但 V1 暂时保留；选定不少于 20 个含多附件、失败附件和 Excel/PDF 的验收样本。

## 1. 建立 KC 数据与生命周期骨架

**目标：** 落地 `KBOT_KC_*` DDL、迁移脚本、Entity/Repository 和 Bundle/Document/Version/Job 状态机。

**交付物：** Collection、Collection Binding、Bundle、Document、Document Version、Parse View、Evidence、Discovery、Relation、Job 表及约束、索引和状态转换图；`COLLECTION_PURGE` 清除状态机。

**完成门槛：** 可通过内部测试创建、幂等更新、切换 current Version、撤销可见性和租约重领；ACTIVE Binding 阻止删除，未绑定 Collection 可异步物理清除全部内容；没有任何旧 File/Chunk Repository 依赖。

## 2. 交付 Knowledge Core 服务与入库 API

**目标：** 建立独立进程、服务认证、对象存储边界与 Bundle multipart 接收能力。

**交付物：** `kbot_app_knowledge.py`、健康检查、Domain 内 Collection 管理 API、KM Asset 专用 `POST /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}/ingestions/km-assets`、文件原子发布、Manifest 生成和 Job 投递。

**完成门槛：** 本地可启动；同一来源修订重复提交幂等；附件失败显式记录；API 不暴露数据库会话或旧上传契约。

## 3. 改造 KM Portal 为 Bundle Adapter

**目标：** 将一个来源对象及其附件作为一次独立投递，而不是多次文件上传。

**交付物：** 完整 Metadb 字段映射、稳定 `source_revision`/`external_document_id`、一次 multipart 调用、失败附件报告与状态查询。

**完成门槛：** Portal 不再调用 `/api/kb/upload`、不生成手工 Markdown、不直连 KBot 数据库；`processed=Y` 仅在 KC 接收成功后写入。

## 4. 改造 Parser 为任务 Worker

**目标：** 复用现有解析能力，但使其通过 KC 任务协议运行。

**交付物：** claim/heartbeat/result Client、Worker 能力注册、Docling/OCR/VLM handler、结果分批/临时 URI、失败分类与重试策略。

**完成门槛：** Parser 不轮询旧表、不写 KC 表；Worker 中断后任务可租约重领；可得到 Version、Parse View 与原始 Evidence 输出。

## 5. 构建检索投影与质量链路

**目标：** 形成可用的 Discovery、Evidence、Profile、Relation 与 Excel 结构化解析能力。

**交付物：** Oracle Text/Vector 索引、Discovery/Evidence API、主视图选择、章节邻接、确定性 Relation、`SPREADSHEET` View 与 `structured_artifact_uri`。

**完成门槛：** 样本集可按自然语言发现正确 Bundle/Document，并在限定范围返回可定位 Evidence；Excel 可定位子表且输出规范化表格工件。

## 6. 重构问文 Skill 与引用输出

**目标：** 新建 V2 问文链路，不适配 `TxtBaseSearch`/`DocService` 的旧检索模型，改为 KC 原生两阶段问文流程。

**交付物：** `KnowledgeRetrievalSkillV2`（或等价模块）、检索计划、Citation Pack DTO、回答前覆盖校验、V2 Doc Orchestrator/Root Agent/SSE 的新引用结构与显式路由开关。

**完成门槛：** 被路由到 V2 的问文请求只走 `Discovery → Evidence → Citation Pack`；答案可回溯 Bundle、Document、Version、View 与页段；V2 内不存在旧表查询或双后端分支，V1 请求不进入 KC。

## 7. 端到端评测、重新入库与直接上线

**目标：** 用数据证明 V2 链路质量与可靠性，完成来源重新入库后直接将 Portal 与 Agent 问文路由切到 V2。

**交付物：** 标注集、离线评测报告、压测/故障演练、存量知识重新入库工具、直接上线检查表、V2 稳定期观测方案，以及 V1 接口/Skill/旧表的清理清单。

**完成门槛：** 达成预先冻结的 Discovery/Evidence Recall@K、引用定位、跨附件覆盖、解析时延和失败率目标；Portal 与 Agent 问文路由直接使用 V2，且稳定期观测通过。V1 在稳定期内不路由、不回退；稳定验收后删除旧 `File/Chunk` 表、V1 接口、Skill 和相关实现。

## 后续但不阻塞 3.5

- Data Query 的 File Dataset 注册、NL2SQL 与问数权限模型。
- 新来源 Adapter：普通 KB、项目、工单、产品资料。
- 多 Agent、Skill 注册治理、复杂关系推理和跨租户策略。

这些事项只能消费 KC 稳定 API/引用模型，不能反向改变 3.5 的领域所有权。
