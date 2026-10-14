# AIOps 业务系统资源图谱详细设计

版本：1.0

状态：设计冻结，待实施

日期：2026-10-14

适用：Ammolite Cube、KBot4

## 1. 文档定位

本文把《AIOps 业务系统资源拓扑总体设计》落实为领域模型、持久化模型、公共合同、Graph 查询、自动发现和前端交互。本文定义目标态，不要求保留当前“组件、成员、站点、二元关系”的兼容接口。

两套产品必须保持相同的业务语义和合同；Ammolite 使用 PostgreSQL 与 Vue Portal，KBot4 使用 Oracle 26ai 与静态 AIOps App。数据库能力和前端技术差异只能位于适配层。

## 2. 领域边界与术语

| 术语 | 定义 | 是否用户可见 |
| --- | --- | --- |
| 业务系统 | 共同承载一项业务能力的运维范围 | 是 |
| 资源 | 可管理、可观测或用于解释拓扑的实体 | 是 |
| 系统资源 | 资源加入某业务系统后的上下文，保存架构层、排序和责任覆盖 | 概念可见，技术名不可见 |
| 资源关系 | 有类型、多个参与者及参与角色的拓扑事实 | 是 |
| 架构层 | 资源在当前业务系统中的展示泳道 | 是 |
| Target | 数据库连接、诊断和受控执行端点 | 否，仅内部适配使用 |
| 接入批次 | Wizard 中一次发现和确认的临时操作上下文 | 否，不形成持久层级 |
| 观测事实 | 自动发现或运行时看到的事实 | 只以证据和偏差形式可见 |

明确取消：`Managed Component`、`System Component`、`Component Membership`、`cluster_group_key` 和独立站点层级。RAC、DG、承载和依赖均由关系表达。

## 3. 聚合与一致性边界

```text
BusinessSystem 聚合
├─ BusinessSystemResource（引用全局 ManagedResource）
├─ ResourceRelation
│  └─ RelationParticipant（引用本系统的 BusinessSystemResource）
└─ 派生：架构标签、资源统计、拓扑校验、保护覆盖摘要

ManagedResource 聚合
├─ 稳定身份、资源类型、业务编码、技术责任组
├─ 类型化 Profile
└─ 内部 Endpoint / Target / Monitoring Binding
```

业务系统资源和关系在同一事务中修改。全局资源可以被多个业务系统引用，因此删除系统只删除系统资源关联和本系统关系，不删除全局资源。资源本身的身份、版本和 Endpoint 修改使用资源行版本控制；系统内的架构层和排序使用系统资源行版本控制。

## 4. 规范持久化模型

### 4.1 核心表

#### `aiops_business_system`

保留业务系统主表，调整规则：

- `business_system_id`：UUIDv7 主键；
- `system_code`：后端生成、不可修改；
- `display_name`、`environment`、`importance_level`、`responsibility_group_id`、`description`；
- `status`：`DRAFT / ENABLED / DISABLED`；
- `row_version` 与审计字段。

创建和 Patch 合同均不接受 `system_code`。列表中的架构标签、资源数和保护状态全部派生，不在主表重复存储。

#### `aiops_managed_resource`

| 字段 | 说明 |
| --- | --- |
| `resource_id` | UUIDv7 主键 |
| `tenant_id/domain_id` | 可信隔离范围；KBot 仅 Domain |
| `resource_code` | 稳定资源的业务编码；发现态短生命周期资源可空 |
| `resource_type_key` | 资源类型目录中的稳定键 |
| `display_name` | 用户可读名称 |
| `identity_kind` | `STABLE / DISCOVERED / EPHEMERAL` |
| `lifecycle_status` | `DISCOVERED / ACTIVE / INACTIVE / RETIRED` |
| `owner_group_id` | 默认技术责任组 |
| `location_name` | 可选位置名称，不要求用户输入站点代码 |
| `identity_fingerprint` | 规范身份摘要的不可逆散列，不保存密码或完整连接串 |
| `row_version` | 乐观锁 |
| 审计字段 | 创建、更新人和时间 |

`resource_code` 在作用域内由应用层保证唯一。KBot Oracle 遵循现有约束政策，不新增业务 `UNIQUE` 或 `CHECK`；Ammolite PostgreSQL 的规范实体按平台现行完整性规范实现。两边的应用错误语义必须一致。

#### 类型化 Profile

公共身份放在资源主表，类型专属字段进入一对一 Profile，避免把所有资源压成 EAV 或无限扩张主表：

- `aiops_database_resource`：`resource_id` 与内部 `target_id` 的一对一映射；引擎、版本、凭据引用、数据库连接、诊断和受控执行能力仍以 Target 为唯一技术事实；
- `aiops_compute_resource`：物理机、VM、容器、Pod 的稳定身份与基础平台信息；
- 后续按批次增加 `application_resource`、`middleware_resource`、`network_resource`、`storage_resource`。

一期只完整实现数据库 Profile 和自动发现所需的最小计算 Profile。公共 API 以 Resource 合同为准，不暴露 `target_id`。数据库资源的版本修改由资源应用服务写入内部 Target，Resource Read Model 通过关联投影该版本，不保存第二份版本事实。

#### `aiops_business_system_resource`

| 字段 | 说明 |
| --- | --- |
| `system_resource_id` | UUIDv7 主键，也是图中的资源顶点 ID |
| `business_system_id` | 所属业务系统 |
| `resource_id` | 引用全局资源 |
| `architecture_layer_key` | 当前系统中的主要展示层 |
| `responsibility_group_id` | 可选责任覆盖；为空时继承资源技术责任组 |
| `display_order` | 同层列表顺序 |
| `status` | `ACTIVE / REMOVED`，历史审计可追踪 |
| `row_version` | 乐观锁 |

应用层保证同一资源在同一业务系统中只有一个活动关联。关系参与者只引用 `system_resource_id`，从结构上避免把未加入本系统的资源连入拓扑。

#### `aiops_resource_relation`

| 字段 | 说明 |
| --- | --- |
| `relation_id` | UUIDv7 主键，也是图中的关系顶点 ID |
| `business_system_id` | 强制拓扑边界 |
| `relation_type_key` | 稳定关系类型 |
| `display_name` | 可选用户名称，如“生产 RAC”“异地 DG” |
| `responsibility_group_id` | 可选关系责任组 |
| `origin` | `MANUAL / DISCOVERY / IMPORT` |
| `confirmation_status` | `SUGGESTED / CONFIRMED / REJECTED` |
| `verification_status` | `UNKNOWN / MATCHED / DRIFTED / UNREACHABLE` |
| `attributes` | 按关系类型 Schema 校验的 JSON，如同步模式；不得存放 Secret |
| `row_version` | 乐观锁 |
| 审计字段 | 创建、确认和更新时间 |

#### `aiops_relation_participant`

| 字段 | 说明 |
| --- | --- |
| `participant_id` | UUIDv7 主键 |
| `relation_id` | 所属关系 |
| `system_resource_id` | 本业务系统中的资源顶点 |
| `participant_role` | 在该关系中的角色 |
| `display_order` | 稳定展示顺序 |

应用层校验关系与系统资源属于同一 `business_system_id`、参与者不重复、角色数量和资源类型满足目录定义。任何拓扑查询都从可信 Scope 加 `business_system_id`，不能先做全域路径查询再过滤结果。

### 4.2 配置目录

#### 资源类型目录

目录项包含 `resource_type_key`、资源域、中文名称、图标键、稳定/临时属性、支持的接入器和 Profile Schema 版本。目录由版本化资源文件初始化，前端从 API 读取，不硬编码平铺卡片。

#### 关系类型目录

| 关系类型 | 参与角色与数量 | 方向语义 |
| --- | --- | --- |
| `ORACLE_RAC` | `MEMBER` 至少 2 个 Oracle 数据库 | 对等 |
| `DATA_GUARD` | `PRIMARY` 至少 1 个，`STANDBY` 至少 1 个 | 主到备 |
| `MYSQL_REPLICATION` | `SOURCE` 1 个，`REPLICA` 至少 1 个 | 源到副本 |
| `POSTGRES_STREAMING` | `PRIMARY` 1 个，`STANDBY` 至少 1 个 | 主到备 |
| `RUNS_ON` | `WORKLOAD` 1 个，`HOST` 1 个 | 工作负载到承载资源 |
| `USES_STORAGE` | `CONSUMER` 至少 1 个，`PROVIDER` 至少 1 个 | 使用方到存储 |
| `DEPENDS_ON` | `CONSUMER` 至少 1 个，`PROVIDER` 至少 1 个 | 依赖方到提供方 |
| `LOAD_BALANCES` | `BALANCER` 至少 1 个，`BACKEND` 至少 1 个 | 接入到后端 |

目录还定义允许的资源类型、是否允许环路、是否允许跨架构层、属性 Schema 和拓扑线型。关系类型扩展不要求修改二元 source/target 列。

#### 架构层目录

Tenant 控制台维护层名称、顺序、启用状态和默认归层规则，初始值为 `ACCESS / APPLICATION / MIDDLEWARE / DATA / INFRASTRUCTURE`。稳定键不随中文名称变化。业务系统只使用启用层的子集；`UNCLASSIFIED` 由查询计算，不写入正式关联。

#### 业务编码策略

Tenant 控制台按对象/资源类型维护前缀、数字长度和下一序号。默认 `SYS-`、`DB-`、`SRV-`。分配必须在事务中锁定策略行并递增，失败事务不复用已对外展示的编码。编码创建后不可修改，用户无编码输入权限。

### 4.3 Wizard 草稿

大型系统编辑不能复制并全量 PUT 整张拓扑。使用服务端草稿和增量变更：

- `aiops_business_system_draft`：创建/编辑模式、基础系统与行版本、当前步骤、修订号、过期时间、创建人；
- `aiops_business_system_draft_change`：顺序记录类型化命令及经 Schema 校验的 Payload；
- `aiops_resource_discovery_job`：草稿、接入器、状态、凭据引用、开始/结束时间、脱敏错误和结果；
- 发现结果作为草稿候选保存，最终提交前不成为正式资源或关系。

编辑查询把正式快照与草稿变更叠加为当前视图。每次修改携带草稿修订号，冲突返回新修订和冲突对象。提交使用幂等键，在一个 UoW 中校验基础行版本、分配编码、创建或复用资源、写入系统关联与关系并发布监控绑定；失败不留下孤儿资源。

## 5. 资源身份与复用

自动匹配优先使用数据库系统标识、DB Unique Name、Oracle DBID、PostgreSQL system identifier、MySQL server UUID、云实例 ID、机器 ID、容器或 Kubernetes UID 等稳定事实。Endpoint、IP、显示名称和主机名只能作为辅助证据，不能单独触发自动合并。

匹配结果分为：

- `EXACT`：稳定身份唯一匹配，建议复用；
- `POSSIBLE`：存在多个或弱证据匹配，必须人工选择；
- `NEW`：没有候选，创建新资源；
- `CONFLICT`：稳定事实互相矛盾，阻塞提交。

身份摘要仅保存不可逆散列；原始发现证据按最小范围和保留期保存，日志、错误和审计不得输出用户名、密码、完整 DSN、Token 或私钥。

## 6. 自动发现流程

### 6.1 数据库一期

1. 用户选择数据库引擎、入口、访问模式和 Managed Credential；也可选择已有 Target 或监控源候选。
2. Main API 创建 discovery job，AIOps 服务在服务端测试连接并读取允许的数据库事实。
3. 引擎接入器返回数据库资源、RAC/DG/复制候选、承载链和监控候选；不直接写正式拓扑。
4. 身份匹配器查找已有资源并生成 `EXACT/POSSIBLE/NEW/CONFLICT` 决策。
5. 归层器根据接入场景、专属/共享判断给出架构层建议。
6. 用户在结果页确认资源、角色、关系、责任组和层；未发现的远端成员可以追加入口继续发现。
7. 确认结果写入草稿变更；Step 3 再做全局拓扑校验。

连接成功不等于监控映射成功，数据库事实、关系发现和监控 Binding 分别显示状态。部分失败必须保留已成功证据并给出补救动作。

### 6.2 承载链

发现器可以产生如下链路：

```text
数据库 → RUNS_ON → 进程/容器/Pod → RUNS_ON → VM/物理机
                                      ↓
                                  USES_STORAGE
```

一期至少建立能被可靠识别的数据库到 Host/VM/Container/Pod 关系。无法可靠区分物理机与 VM 时显示“服务器（待确认）”，不伪造类型。专属承载资源默认跟随数据库进入数据层；已被多个业务构成或系统共享时建议放入基础设施层。

## 7. 架构推导与校验

### 7.1 架构标签

- 一个数据库资源且不存在 HA/复制关系：`单实例`；
- 存在 `ORACLE_RAC`：`RAC`；
- 存在 `DATA_GUARD`：`Data Guard`；
- 同时存在 RAC 和 DG 且参与者连通：`RAC + Data Guard`；
- 存在 MySQL/PostgreSQL 对应关系：`主从复制`或`流复制`；
- 同一关系中的数据库版本不一致：附加`混合版本`和`待兼容性确认`。

标签是只读投影，禁止持久化为用户可改的部署形态字段。

### 7.2 阻塞项

- 系统没有任何资源；
- 稳定资源缺少技术责任组；
- 正式资源仍处于身份冲突；
- 关系参与者不属于当前业务系统；
- 关系角色、数量或资源类型不满足目录；
- `RUNS_ON` 等禁止环路关系形成环；
- 新数据库资源连接验证失败，或要求启用监控但 Binding 不完整；
- 草稿基础版本已被其他编辑提交。

### 7.3 警告项

- 资源尚未归类；
- 发现角色与用户确认角色不一致；
- 高可用资源全部位于同一位置；
- 数据库版本混合但未记录确认说明；
- 承载环境只发现到部分链路；
- 数据保护尚未配置或真实覆盖不足。

警告不自动变成成功状态；带警告提交时保存确认人和说明。

## 8. Graph 与拓扑查询

### 8.1 图映射

规范关系表构成二部图：

```text
[System Resource 顶点] --(participant_role)--> [Relation 顶点]
```

多参与者关系本身是关系顶点，Participant 是边。这样无需把一个 RAC 展开成多条互相矛盾的两两边，也无需生成隐藏集群资源。

Oracle 26ai Property Graph 直接映射 `BUSINESS_SYSTEM_RESOURCE`、`RESOURCE_RELATION` 和 `RELATION_PARTICIPANT`。PostgreSQL 使用相同三表做递归 CTE。两种实现返回统一的节点、关系和参与者合同；Graph 只负责读和遍历，写入始终经过 Repository/UoW。

### 8.2 查询能力

- 一跳邻居；
- 指定方向的上游/下游；
- 两资源之间的最短路径；
- 按关系类型、资源类型、架构层和状态过滤；
- 以焦点资源为中心、按业务深度展开子图；
- 影响范围：某资源不可用时沿 `DEPENDS_ON/RUNS_ON/LOAD_BALANCES` 规则展开；
- HA 同伴：通过 RAC、DG 或复制关系查找关联资源。

用户所说的一“跳”指资源经过一个关系到另一资源；底层二部图的两条物理边按一个业务深度计算。

### 8.3 安全与预算

查询入口必须先解析并校验 `business_system_id`，随后只使用该系统的 `system_resource_id/relation_id` 集合。默认深度 1，最大业务深度 5；默认节点预算 100，最大 500；路径数量和执行时间也必须有上限。超预算返回截断标志、聚合计数和继续下钻游标，不返回不完整结果冒充全量结果。

## 9. API 合同

公共路径继续位于 `/api/v1/apps/aiops/domains/{domain_code}/ops`，内部服务使用 `/internal/v1` 对应合同。建议资源如下：

### 9.1 业务系统与草稿

- `GET /business-systems`：游标分页、名称/编码/环境/责任组筛选；
- `GET /business-systems/{id}`：业务摘要，不内嵌全量资源；
- `POST /business-system-drafts`：创建新建或编辑草稿；
- `GET /business-system-drafts/{draft_id}`：步骤、修订、校验摘要；
- `PATCH /business-system-drafts/{draft_id}/business-info`：Step 1；
- `POST /business-system-drafts/{draft_id}:validate`；
- `POST /business-system-drafts/{draft_id}:commit`：幂等原子提交；
- `DELETE /business-system-drafts/{draft_id}`：放弃草稿，不删除正式资源。

### 9.2 草稿资源与发现

- `GET /business-system-drafts/{draft_id}/resources`：服务端筛选和游标分页；
- `POST /business-system-drafts/{draft_id}/discoveries`：启动场景化发现；
- `GET /business-system-drafts/{draft_id}/discoveries/{job_id}`：状态与候选；
- `POST /business-system-drafts/{draft_id}/discoveries/{job_id}:confirm`：确认选择；
- `POST /business-system-drafts/{draft_id}/resources:attach`：复用已有资源；
- `PATCH /business-system-drafts/{draft_id}/resources/{system_resource_id}`：层、责任覆盖和资源允许修改字段；
- `DELETE /business-system-drafts/{draft_id}/resources/{system_resource_id}`：草稿移除并返回关系影响。

### 9.3 关系与图查询

- `POST /business-system-drafts/{draft_id}/relations`；
- `PATCH /business-system-drafts/{draft_id}/relations/{relation_id}`；
- `DELETE /business-system-drafts/{draft_id}/relations/{relation_id}`；
- `GET /business-systems/{id}/topology`：摘要或焦点子图；
- `GET /business-systems/{id}/resources/{resource_id}/neighbors`；
- `GET /business-systems/{id}/paths?from=...&to=...`。

不再提供全量拓扑 PUT。用户提交显示名称、资源 ID、关系类型和参与角色；内部键、Target ID 和数据库主键不进入可编辑字段。列表响应不内嵌完整图，所有集合支持游标分页。

## 10. 前端详细交互

### 10.1 公共工作区框架

页面自上而下为：返回与标题区、三步步骤条、当前步骤内容、固定底部操作栏。底部操作为“保存草稿”“上一步”“下一步/确认创建”，离开有未保存变更时提示。创建与编辑复用同一路由骨架；编辑可从详情页直接进入 Step 2 或 Step 3，但三步状态始终可见。

### 10.2 Step 1：业务信息

内容分为两块而非控件堆叠：

- 基本信息：名称、环境、重要级、说明、标签；
- 责任与编码：业务责任组、只读编码规则和预计编码。

名称失焦后做同 Scope 近似重名提示，但不把显示名当主键。预计编码只能提示，实际编码在提交事务中分配。

### 10.3 Step 2：系统构成

页面使用三栏响应式布局：

- 左侧“架构层导航”：各层资源数、错误数、未归类；可折叠但不占全局导航；
- 中间“资源工作区”：搜索、类型/责任组/状态筛选、列表/紧凑卡片切换、批量调整层和责任组；
- 右侧“当前摘要”：资源数、关系数、待确认发现、阻塞项；选中资源时变为就地详情面板。窄屏改为页面内上下布局，不使用高密度覆盖抽屉。

新增入口打开页面内“接入任务”，先选择可搜索的资源类型，再显示该类型支持的方式。数据库一期提供“新接入”“复用已有数据库”“从监控源导入”。二期类型保留说明和计划状态，不显示可提交的空表单。

数据库新接入分四个连续区段：

1. 连接入口：引擎、Host/Service、端口、访问模式、Managed Credential；
2. 发现进度：连接、数据库事实、HA/复制、承载环境、监控映射分别显示；
3. 识别结果：按资源展示名称、类型、版本/平台、角色、身份匹配、建议层和证据；
4. 确认调整：选择复用或新建、责任组、层、关系参与角色，并可“补充另一个入口继续发现”。

确认后回到资源工作区，并以一次性变更摘要提示“新增 N 个资源、复用 N 个资源、建立 N 个关系”。不存在要求用户再添加一遍服务器的后续步骤。

### 10.4 Step 3：拓扑确认

顶部为架构摘要、校验结果和搜索；中间为分层拓扑；右侧为选中对象的资源或关系属性；底部提供可访问的资源/关系列表。画布规则：

- 亮色白底、暗色深色底，全部颜色使用全局 Token；
- 默认按架构层泳道与关系聚合，空层隐藏；
- 搜索结果居中并加载一跳邻居；
- 双击节点进入局部下钻，面包屑返回系统全景；
- 拖动跨泳道只修改架构层；连线必须经过“选择关系类型 → 分配参与角色 → 校验”确认，不能直接生成无语义边；
- 删除节点先显示本系统关系、监控、巡检和保护影响；共享资源只从本系统移除；
- 画布与列表具有相同能力，键盘和读屏用户无需依赖拖拽。

### 10.5 详情和目录

目录卡片显示业务身份、环境/重要级、业务责任组、资源总量与类型摘要、推导架构、真实健康和更新时间。新增使用同尺寸“＋”卡片。详情页标签建议为：概览、资源、拓扑、监控、数据保护、审计。

资源页默认按架构层分组，可切换按类型、责任组或位置分组。任何分组都是查询视图，不生成新的持久化父对象。

## 11. 与现有 AIOps 能力的衔接

### 11.1 Target

Target 保留为数据库诊断和执行的内部端点，由 Database Resource 一对一引用。巡检、对话、告警、受控动作和监控底层仍可使用 `target_id`，公共页面通过 `resource_id` 选择，应用服务做授权映射。不得在前端继续出现“添加 Target”或独立 Target 菜单。

### 11.2 实时监控与诊断

选择顺序为业务系统、可选架构层/类型、资源。服务端校验系统资源关联后解析内部 Target 和 Source Binding。告警、Finding 和工作项同时投影 `business_system_id/resource_id`，以便从资源节点下钻；既有 Target 证据链不丢失。

### 11.3 数据保护

恢复目标、自动备份、人工恢复和演练继续独立。保护计划通过关联表覆盖一个或多个数据库资源；选择 RAC/DG 中任一数据库时，系统可根据已确认关系建议完整范围和候选执行节点，但必须由用户确认。保护覆盖率由策略、成功备份时效、可恢复点和恢复验证共同计算，不能由已填写 RTO/RPO 代替。

### 11.4 责任与权限

业务责任组用于业务影响和总体工作项，资源责任组用于技术处置，关系责任组用于复制链路、网络或数据集成。权限仍由可信 Tenant/Domain/App 上下文和平台授权决定，责任组不等于凭据访问权或执行授权。

## 12. 迁移映射

| 当前对象 | 目标对象/处理 |
| --- | --- |
| `business_system` | 保留；编码改为只读生成 |
| `managed_component` | 移除；责任组迁移到资源或系统资源关联 |
| `business_system_component` | 由 `business_system_resource` 取代 |
| `component_membership` | 每个 Target 生成/关联一个 Database Resource，再加入业务系统 |
| `business_system_site` | `display_name` 迁移为资源位置；不再暴露 `site_code` |
| `cluster_group_key` | 同组成员生成一个 `ORACLE_RAC` 关系 |
| 二元 `resource_relation` | 转为关系实体和两个或多个 Participant |
| 组件级 Recovery Profile | 转为保护计划与数据库资源范围 |
| 孤立 Target | 各自生成业务系统、数据库资源和数据层关联，禁止迁移后仍为孤儿 |

迁移前输出逐 Target 映射报告和关系合并候选。只有稳定证据足够时才把多条二元 DG/RAC 记录合并为一个多参与者关系；不确定记录保持独立并标记待确认，不能靠名称猜测。

Oracle 当前 Schema 44 尚未投产的前提下，直接重写 `database/oracle/operations/apply_aiops_schema_44.sql`，从原始单 Target Schema 一次升级到本文目标态，跳过已否定的组件模型；canonical DDL、manifest、生成重建脚本和校验脚本同步更新。若实施前发现目标库已执行旧 44，必须先重新评审版本号和升级路径，不能假装仍是 43。

Ammolite PostgreSQL 同步更新规范 Entity 和初始化器；本地与 kbotdev 使用经评审的一次性受控对齐程序，先备份、预检、执行、核对行数与孤儿数，再启动服务。该程序不形成长期迁移链，运行时代码和 readiness 也不承担 Schema 自检或补表。

## 13. 非功能要求

- 列表接口 P95 目标在 500 ms 内返回一页，拓扑一跳查询 P95 目标在 1 s 内返回预算内子图；具体基线由实施前压测冻结。
- 1000 资源/5000 Participant 的系统不得触发全量前端渲染。
- 所有写命令幂等并带乐观锁；Repository 不提交事务。
- Secret 只经 Managed Credential 使用；草稿、Graph 属性、日志和前端响应不得包含明文 Secret 或完整 DSN。
- 创建、复用、合并、拆分、关系修改和层调整均写审计记录。
- 禁止 API/Application 直接访问 Session 或执行 SQL；所有持久化通过 UoW/Repository。
- 不增加兼容路由、双读、双写或运行时 Schema 迁移。

## 14. 验收场景

1. 从一个 Oracle 入口发现两节点 RAC、一个 DG 备库及三台主机，用户一次确认后得到资源和两个多参与者关系。
2. 给单实例新增 RAC 节点，架构标签自动从“单实例”变为“RAC”；再增加 DG 后变为“RAC + Data Guard”。
3. Oracle 主备版本不同，资源分别保留真实版本，系统显示“混合版本/待确认”，不阻止合法的 OGG 异构场景。
4. 同一共享 VM 加入两个业务系统，在两个系统中使用不同架构层；修改一边不影响另一边。
5. 从 `db01` 下钻可在当前 MES 系统内找到 RAC 同伴、DG 备库、承载主机和依赖应用，不返回其他系统的共享资源邻居。
6. 1000 个资源的系统首先展示分层摘要，通过搜索与焦点下钻加载局部图，不冻结浏览器。
7. 删除共享服务器只解除当前系统关联；存在关系时先阻塞并展示影响。
8. 未配置备份、只有 RTO/RPO、备份已成功、恢复已验证四种状态显示不同且真实的保护文案。
9. PostgreSQL 与 Oracle 对同一领域场景返回等价合同和错误码。
10. 迁移后三个既有 Target 均能从业务系统进入、映射到数据库资源，监控、诊断和巡检引用不丢失。
