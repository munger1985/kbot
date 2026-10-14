# AIOps 业务系统资源图谱改造计划

版本：1.0

状态：待实施

日期：2026-10-14

适用仓库：`/home/chris/ammolite_cube`、`/home/chris/kbot4`

## 1. 实施目标

把当前“业务系统 → 组件 → Target → 二元关系”的实现一次性收敛到“业务系统 → 资源 + 多参与者关系”，交付三步 Wizard、自动发现、分层列表、局部拓扑和安全迁移。改造完成后不保留旧组件合同、四步 Wizard、全量拓扑 PUT 或用户可编辑技术键。

本计划不表示代码、数据库或部署已经完成。每批只有在 Ammolite 与 KBot4 都通过各自门禁后，才能标记为跨项目完成。

## 2. 实施原则

1. 先冻结合同和数据映射，再改持久化与 UI；不得边写页面边发明第三套模型。
2. AIOps 业务语义、公共/内部合同、错误码和测试场景在两个仓库同步；PostgreSQL/Oracle 和 Vue/静态 App 只做适配。
3. 使用目标态直接切换，不增加兼容路由、双读写、旧字段兜底或长期 Feature Flag。
4. 所有正式写入通过应用服务、UoW 和 Repository；API、Graph 查询和前端不得绕过边界。
5. 数据升级先预检和备份，再停写、升级、核对、启动和 Smoke；失败按备份恢复，不在生产库中临时修表。
6. 保留现有 Target ID、监控 Binding、凭据引用、诊断、巡检、会话和证据链；用户入口改为 Resource，不重建历史技术端点。
7. 保护两个仓库当前无关工作树改动，按改造批次分别提交中文 Conventional Commit。

## 3. 批次依赖

```text
批次 0：基线审计与合同冻结
  ↓
批次 1：共享合同、目录与领域规则
  ↓
批次 2：PostgreSQL/Oracle 规范模型与 Repository
  ↓
批次 3：草稿、发现、资源与关系应用服务
  ↓
批次 4：Graph 查询与现有 AIOps 消费方切换
  ↓
批次 5：Ammolite/KBot 前端三步 Wizard 与详情
  ↓
批次 6：迁移演练、数据库升级、部署和在线验收
```

批次 1～3 可以在功能分支连续开发，但上一批合同测试未通过时不得开始真实数据库升级。前端可使用冻结的 Stub 并行制作，但不得先于合同自行定义字段。

## 4. 批次 0：基线审计与冻结

### 工作项

- 记录两个仓库 HEAD、分支、工作树、OpenAPI、Schema/manifest 和前端构建基线。
- 统计当前业务系统、组件、成员、关系、Recovery Profile、Target、Binding 及孤儿数量。
- 逐个列出当前组件合同、四步 Wizard、全量 PUT 和所有 `component_id/component_key/member_key/relation_key/site_code` 调用方。
- 盘点 Target 被监控、诊断、巡检、Agent、告警、恢复、报告和工作项引用的位置，确认必须保留的 ID 和外键。
- 冻结新资源类型、关系类型、架构层、错误码和 API DTO；生成 Ammolite/KBot 合同差异检查清单。
- 准备最少四套黄金场景：单实例、RAC、DG、RAC + DG；另准备共享主机和混合版本场景。

### 交付物

- 基线报告；
- 旧字段调用方清单；
- 数据迁移映射报告模板；
- 新旧 OpenAPI 差异预期；
- 可重复的孤儿与行数检查脚本。

### 门禁

任何现有 Target 找不到明确迁移去向、任何历史引用必须换 ID、或 Oracle 实际 Schema 版本与“未执行 44”前提不一致时，停止后续数据库工作并修订迁移方案。

## 5. 批次 1：共享合同、目录与领域规则

### 代码范围

- 两仓 `packages/platform_core/src/platform_core/contracts/aiops/`；
- 两仓 AIOps `domain/`、`resources/` 或现有等价目录；
- 公共/内部 OpenAPI 快照和合同测试。

### 工作项

1. 新增 `ManagedResource`、`BusinessSystemResource`、`ResourceRelation`、`RelationParticipant`、`TopologySubgraph`、`DiscoveryJob` 和 Draft 合同。
2. `BusinessSystemCreate/Patch` 删除可写 `system_code`；摘要改为 `resource_count/resource_type_summary`，删除 `component_count/target_count` 的用户语义。
3. 建立版本化资源类型、关系类型和默认架构层目录，增加加载与结构校验。
4. 实现关系参与者数量、角色、资源类型、方向和环路规则；实现架构标签推导。
5. 定义稳定错误码，例如资源身份冲突、关系角色无效、跨系统参与者、草稿修订冲突、拓扑预算截断。
6. 删除目标合同中的 Component、Membership、Site 和二元 source/target 模型；不保留别名 DTO。

### 测试

- 目录 Schema 与稳定排序；
- RAC/DG/复制/RUNS_ON 的有效和无效参与者组合；
- 单实例到 RAC + DG 的标签推导；
- 混合版本警告；
- 合同序列化和 Ammolite/KBot 黄金 JSON 一致。

### 门禁

两个仓库的公共字段、枚举、错误码和黄金响应一致；平台专属 Scope 字段除外。任何前端都不能再依赖用户输入的技术键。

## 6. 批次 2：规范持久化与 Repository

### Ammolite PostgreSQL

- 更新 `services/aiops_agent/src/aiops_agent/entities/` 中业务系统模型；
- 增加 Resource、类型化 Profile、System Resource、Relation、Participant、Draft、Draft Change 和 Discovery Job 实体；
- 更新 Repository/UoW、`scripts/deployment/init_postgresql.py` 及 Schema acceptance；
- 不新增或长期保留本地 PostgreSQL 迁移链；实施时使用经评审的一次性受控对齐程序修改本地与 kbotdev，规范 Entity 和初始化器始终是唯一 Schema 来源。

### KBot4 Oracle

- 同步更新 Oracle Entity、Repository/UoW；
- 更新 `database/oracle/aiops_agent/` 规范 DDL、FK/索引/视图、`schema_manifest.json`；
- 重新生成 rebuild 与 validation 工件；
- 在确认旧 44 未投产后，重写 `database/oracle/operations/apply_aiops_schema_44.sql`，直接从 Schema 43 单 Target 模型升级到资源图谱目标态。

### Repository 能力

- 资源按 Scope、类型、编码、身份摘要和生命周期查询；
- 系统资源按层/类型/责任组/状态游标分页；
- 关系和 Participant 增量增删改；
- 草稿变更追加、修订冲突与幂等提交；
- Target 与 Resource 双向内部解析；
- 所有方法接收可信 Scope，Repository 不 `commit()`。

### 数据库测试

- 实体与规范 Schema 精确对齐；
- FK、必要索引、审计字段和乐观锁；
- 共享资源加入两个业务系统；
- 跨系统 Participant 被应用层拒绝；
- 并发编码分配无重复；
- 事务失败不留下系统资源、关系或孤儿 Resource；
- PostgreSQL 与 Oracle 的相同行为夹具得到等价结果。

### 门禁

- Ammolite：通过架构边界与 PostgreSQL Schema acceptance；
- KBot4：通过 `check_4_0_boundaries.py`、Oracle Schema acceptance 和 manifest/生成工件检查；
- readiness 仍只做连接与真实依赖健康检查，不加入运行时 Schema 验证或自动 DDL。

## 7. 批次 3：草稿、发现和应用服务

### 工作项

1. 实现新建/编辑 Draft、自动保存、过期和显式放弃；编辑使用变更集叠加，不复制全量拓扑。
2. 实现业务编码服务和 Tenant 配置接口；创建事务中分配 `SYS/DB/SRV` 等编码。
3. 把现有数据库连接测试与能力探测编排为 Discovery Job，新增 Oracle RAC/DG、MySQL 复制、PostgreSQL Streaming 和承载环境发现适配器。
4. 实现资源身份匹配器，严格区分 `EXACT/POSSIBLE/NEW/CONFLICT`。
5. 实现默认归层器和“专属/共享”判断，所有结果均可由用户确认调整。
6. 实现发现确认命令，一次把数据库、Host/VM/Container/Pod 候选、关系和监控 Binding 写入草稿。
7. 实现原子 Commit：重放并校验变更、锁定基础版本、分配编码、写正式聚合、记录审计、发布 Outbox。
8. Target 创建/复用成为 Database Resource 的内部步骤；公共服务只返回 Resource。

### 安全

- Discovery 只使用 Managed Credential 引用；
- 严格限制目标协议、地址和超时，沿用 AIOps 网络边界策略；
- 日志和错误统一脱敏；
- 候选结果不返回完整 DSN、用户名或 Provider 原始 Secret；
- 放弃草稿不删除被其他对象引用的 Credential。

### 测试

- 单入口完整发现与部分发现；
- 追加入口补齐 DG/RAC 成员；
- 已有资源精确复用、疑似重复人工选择、冲突阻塞；
- 连接成功但监控 Binding 失败的独立状态；
- 草稿修订并发、重复提交幂等和中途回滚；
- 1000 资源变更集的分页与提交预算。

## 8. 批次 4：Graph 查询与消费方切换

### Graph

- PostgreSQL 实现以 System Resource—Relation—Participant 为基础的递归 CTE；
- Oracle 26ai 在同一关系表上创建 Property Graph 和对应查询适配器；
- 统一邻居、上下游、最短路径、影响范围和焦点子图接口；
- 强制业务系统边界、深度、节点、路径和超时预算；
- Graph 适配器只读，不新增 Graph 写入或同步任务。

### 现有消费方

- 实时监控选择与返回模型从 Target 映射为 Resource；
- 智能诊断、巡检、告警、Agent、报告和工作项增加业务系统/资源上下文投影，内部继续解析 Target；
- 数据保护从组件作用域切到数据库 Resource 集合，并重算真实覆盖文案；
- 目录卡片和详情统计改为资源口径；
- 删除 `find_component_for_target` 等组件专属服务，替换为资源上下文查询。

### 测试

- 从任一数据库找到 RAC、DG、承载主机和上游应用；
- 共享资源查询不会越过当前业务系统；
- 环路关系有预算且不会无限递归；
- Graph 截断明确返回，不伪装全量；
- 既有 Target 的监控、诊断、巡检、告警和证据仍可通过 Resource 入口访问；
- 数据保护四种真实状态文案正确。

## 9. 批次 5：前端改造

### Ammolite Portal

主要范围：

- `apps/portal-web/src/features/aiops/management/BusinessSystemWorkspace.vue`；
- 业务系统目录、详情、Topology、Data Protection 和实时监控选择器；
- 路由、API Service、Pinia 状态、共享组件与全局 Token。

实施项：

1. 四步 Wizard 改为“业务信息—系统构成—拓扑确认”三步独立页面。
2. Step 1 移除系统编码输入，按基本信息/责任与编码分区。
3. Step 2 实现架构层导航、服务端资源列表、接入任务、发现进度、结果确认和变更摘要；所有用户文案使用“资源/数据库/服务器”，不出现“Target”。
4. Step 3 实现白底/深色底分层拓扑、焦点下钻、关系编辑、层拖动、属性面板和可访问列表替代。
5. 删除组件、成员、站点和技术键字段；大型列表使用游标分页，画布只加载摘要或子图。
6. 更新目录卡片：资源总量与类型摘要替代组件数/Target 数，新增入口保持同尺寸加号卡片。
7. 更新详情标签和实时监控选择器；数据保护保持独立模块。

### KBot4 AIOps App

在 KBot4 正式 AIOps 静态前端目录实现相同任务、字段、状态、权限和错误语义；不修改 `integrations/apex/**`。不要求视觉像素一致，但三步顺序、发现确认、分层拓扑、列表规模能力和文案必须一致。

### 前端测试

- Step 1 编码只读；
- 数据库一次接入同时出现数据库和承载资源；
- 发现部分失败、身份冲突、草稿冲突和离开保护；
- 列表与画布双向同步；
- 亮/暗主题、键盘操作、颜色非唯一状态提示；
- 1000 资源 Stub 下不全量渲染；
- Ammolite 生产构建、KBot 静态资源/CSP/缓存版本检查；
- 浏览器控制台无错误，所有网络请求走 Main API。

## 10. 批次 6：迁移与部署

### 10.1 迁移预检

- 确认服务版本、数据库版本、当前 Schema 版本和工作树；
- 备份数据库并验证备份可读取；
- 停止 AIOps API、Worker、Scheduler 和所有写入入口；
- 导出现有 Target、组件、成员、关系、Binding、Recovery Profile 数量和逐 Target 映射；
- 检查重复 Target 身份、悬空外键、未知关系、无责任组和孤儿 Target；
- 对不确定关系生成待人工确认清单，不在升级脚本中按名称猜测。

### 10.2 Ammolite PostgreSQL

1. 先在一次性副本演练升级和回滚；
2. 本地 PostgreSQL 执行受控升级，核对后启动并做 Smoke；
3. kbotdev PostgreSQL 重复备份、停写、升级、校验和启动流程；
4. 校验资源数至少覆盖全部 Target，Database Resource 与 Target 一一映射，系统资源无孤儿，所有 Participant 同系统；
5. 保留升级前备份直到在线验收完成。

### 10.3 KBot4 Oracle

1. 在隔离 Schema 使用 Schema 43 数据副本执行最终 `apply_aiops_schema_44.sql`；
2. 使用 SQL Developer F5 的真实执行模式验证，不依赖 `@/@@` 或本地工作目录；
3. 核对 manifest、对象、行数、Target ID、Binding、责任组、资源关系和保护范围；
4. 用户在正式 Oracle 执行前提供脚本、预检 SQL、执行说明和回滚说明；
5. 正式库升级结果未由用户反馈前，不声称 Oracle 已升级。

### 10.4 启动与 Smoke

- 只启动本批涉及的服务；验证进程、端口、readiness 和最近日志；
- 创建单实例业务系统并增加 RAC/DG 关系；
- 从三个既有 Target 对应业务系统进入资源和拓扑；
- 验证实时监控、智能诊断、巡检、告警上下文和数据保护跳转；
- 验证跨 Tenant/Domain、跨业务系统资源和路径访问被拒绝；
- 记录实际执行的测试、数据库、提交、推送、重启与未执行项。

## 11. 迁移校验不变量

升级前后的下列事实必须成立：

- Target 主键、Endpoint、凭据引用、Source Binding 和生命周期状态不变；
- 每个活动 Target 恰好映射一个 Database Resource；
- 每个 Database Resource 至少加入一个业务系统；
- 当前三个既有 Target 不得成为孤儿；
- 业务系统数量不因脚本重复执行而增加；
- 关系参与者全部属于关系所在业务系统；
- Recovery Profile 的覆盖数据库集合与升级前组件成员一致；
- 用户、责任组、Agent、计划、运行、报告、对话、证据和审计行数不因本次模型切换意外减少；
- Secret 和完整 DSN 不出现在迁移日志或校验输出。

任一不变量失败即不启动应用，按备份恢复或保留现场分析。

## 12. 删除清单

目标态验证通过后，同一批删除：

- Component/Membership/Site 共享合同和实体；
- 旧 `aiops_managed_component`、`business_system_component`、`component_membership`、`business_system_site` 表及 Oracle 对应对象；
- 二元关系 `source_component/source_member/target_*` 字段；
- `/business-systems/{id}/topology` 全量 PUT；
- 前端“组件与实例”“资源关系”步骤和所有技术键输入；
- Target 独立菜单和“添加 Target”文案；
- 组件级数据保护绑定和兼容转换代码；
- 已被新合同替代的测试夹具、OpenAPI 快照和文档说明。

不得把旧对象移动到 `legacy/`，也不得留下双读作为回滚方式；回滚依赖数据库备份和代码版本。

## 13. 提交组织

建议按下列逻辑分别在两仓形成可审计提交：

1. `docs(aiops): 冻结业务系统资源图谱设计`；
2. `refactor(aiops): 引入资源与多参与者关系合同`；
3. `refactor(aiops): 切换业务系统资源持久化模型`；
4. `feat(aiops): 增加资源发现和草稿提交`；
5. `feat(aiops): 增加受限拓扑遍历`；
6. `refactor(aiops): 通过资源接入监控诊断和保护`；
7. `feat(aiops): 重构业务系统三步向导`；
8. `chore(aiops): 交付数据库无损升级工件`。

提交数量可按实际依赖调整，但不得把数据库、后端、前端和无关工作树改动混成一个不可审计提交。

## 14. 完成定义

只有同时满足以下条件才算改造完成：

- 两仓领域合同和用户语义一致；
- PostgreSQL 与 Oracle 规范 Schema、实体、Repository、初始化/升级/校验工件同步；
- 三步 Wizard、列表和拓扑符合总体设计，不再出现旧组件模型；
- 现有 Target 数据与全部引用无损迁移且无孤儿；
- Graph 能从任一节点在同一业务系统内安全下钻；
- 数据保护已独立，覆盖文案基于真实证据；
- 单元、合同、架构、Schema、前端构建、迁移演练和在线 Smoke 均有实际记录；
- 代码已分别提交并按用户要求推送，服务重启和数据库执行状态如实报告。
