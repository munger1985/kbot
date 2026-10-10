# AIOps 跨项目统一实施方案

版本：1.0

状态：实施中

基准日期：2026-10-09

适用仓库：`/home/chris/kbot4`、`/home/chris/ammolite_cube`

## 1. 目的与依据

本方案把后续 AIOps 建设组织成一个跨项目交付计划，同时完成两件事：

1. KBot4 与 Ammolite Cube 按已确认差异双向吸收能力；
2. 严格按既有详细设计交付 Prometheus/Zabbix 实时监控、诊断和恢复闭环。

本方案只规定实施顺序、代码归属、交付物、测试和门禁，不重新定义产品或接口。实现时以下文档按
优先级共同构成约束：

1. [AIOps 实时监控 Dashboard 集成详细设计](aiops-monitoring-dashboard-integration-detailed-design.md)：
   实时监控的范围、合同、安全、Provider 行为和 27 项验收标准；
2. [AIOps 跨项目能力对齐与双向演进基线](../architecture/aiops-cross-project-alignment.md)：
   A-01～A-07 的差异、统一方向和平台适配边界；
3. 当前代码、规范 Schema、OpenAPI 快照和自动化测试：用于证明现状和实现结果，不得反向放宽前两份
   文档已经冻结的产品要求。

若文档之间出现冲突，实时监控事项以实时监控详细设计为准，双向吸收事项以对齐基线为准；无法按此
规则消解时先更新设计并经评审确认，不在代码中自行创造第三套语义。

## 2. 实施基线与非目标

### 2.1 起始基线

- Ammolite Cube：`dev` 分支，双向对齐基线提交 `087aa0f`；此前 AIOps 功能变更已提交，实施开始时
  工作树必须继续保持可审计；
- KBot4：`kbot4.0` 分支，双向对齐基线提交 `06368bb6`；实施前必须记录并保护既有工作树改动；
- 每一批开始时重新记录两个仓库的实际 HEAD、工作树、Schema 版本和 OpenAPI 摘要，不能长期沿用本
  文档中的历史哈希；
- 两边共同的 AIOps 业务语义以同一批次推进，Meta Database、认证入口和前端形态按平台适配。

### 2.2 固定范围

- 双向吸收：A-01～A-07；
- 实时监控一期：Prometheus、外部 Zabbix，覆盖 Oracle、MySQL、PostgreSQL；
- 原生多实例监控页为主，Prometheus Grafana 只做满足安全条件后的受控高级下钻；
- 时间窗固定为 `15m`、`1h`、`6h`、`24h`，单请求最多 12 个实例；
- “完整接入”必须同时满足 `monitoring_readiness=READY` 与
  `diagnostic_readiness=READY`。

### 2.3 非目标

- OEM 不进入一期页面、API Profile、选源和验收；
- 不部署、托管、升级或自动配置 Zabbix，不向 `aiops-stack` 增加 Zabbix 模块、容器或凭据；
- 不在浏览器暴露监控凭据、Locator、任意 PromQL、Zabbix Item 查询或内部 API；
- 不合并 Oracle/PostgreSQL Meta Database，不统一两个前端技术栈和品牌；
- 不整目录覆盖，不新增兼容路由、双读写、回填脚本或长期 Feature Flag；
- 不在本实施中顺带决定“巡检计划删除”这类仍待产品确认的事项。

## 3. 总体依赖与交付策略

实施顺序固定为：

```text
批次 0：差异门禁和合同冻结
  ↓
批次 1：A-01 最小动作授权 + A-02 完整 Binding
  ↓
批次 2：实时监控共享内核、Provider、Catalog、Readiness
  ↓
批次 3：内部/公共 API、Platform Client、持久化适配
  ↓
批次 4：Ammolite Vue Portal + KBot4 静态 AIOps 页面
  ↓
批次 5：A-03～A-07 剩余双向吸收
  ↓
批次 6：双 Meta Database、双 Provider、三数据库真实 E2E
```

批次 1 是监控建设的安全前置：监控实例目录、指标查询和自动诊断都依赖精确 Binding，动作闭环又
依赖 Target 最小授权。批次 2～4 可以在同一功能分支内连续开发，但前一批合同测试未通过时，不得
把下一批标记为完成。批次 5 不得破坏已冻结的监控合同。批次 6 通过前，只能表述为“代码实现或
配置完成”，不能表述为“双项目已交付”或“监控完整接入”。

每个业务变更使用同一变更编号在两个仓库分别提交。允许提交数量不同，不允许只有单边提交。任一
单边因平台问题延期时，另一边可以保留已验证提交，但整个变更编号保持未交付状态。

## 4. 代码与文档归属

### 4.1 两边必须同构的服务层

| 工作域 | 主要归属 | 统一要求 |
| --- | --- | --- |
| 监控合同 | `services/aiops_agent/src/aiops_agent/contracts/` | DTO、枚举、Gap、错误码、字段必填性一致 |
| Profile 与指标 | `resources/monitoring/`、`resources/metrics/` | Profile ID、Metric Code、单位、数据库类型和稳定排序一致 |
| Provider | `adapters/diagnostic_sources/`、`ports/diagnostic_source.py` | Prometheus、Alertmanager、Zabbix 的能力与错误归一化一致 |
| 应用用例 | `application/`、`monitoring/` | 目录、发现映射、View、缓存、预算、部分成功、Readiness 一致 |
| Target/Binding | `entities/monitoring.py`、`repositories/monitoring.py`、配置应用服务 | 聚合关系、并发、状态和验证语义一致 |
| 事件诊断 | Webhook、SignalEvent、Situation、诊断、恢复验证、报告投影 | 故障/恢复、幂等、Evidence、Gap、Provenance 一致 |
| API | `api/`、Main API AIOps BFF、`packages/platform_clients/.../aiops.py` | 内部/公共路径、DTO、权限和 OpenAPI 语义一致 |
| 测试 | `tests/unit/aiops_agent/`、Main API/合同/Smoke 测试 | 相同业务场景和断言，平台差异仅限夹具与适配层 |

新建模块时优先采用清晰的监控子域，例如 `contracts/monitoring.py`、
`application/monitoring/`、`api/monitoring.py` 和
`resources/monitoring/dashboard_profiles.v1.json`；最终文件名可以按现有包结构微调，但不得把业务
逻辑塞入 API 路由、Repository 或前端。

### 4.2 Ammolite Cube 平台适配

| 层 | 实施位置 | 要求 |
| --- | --- | --- |
| Meta Database | 规范 ORM/实体、Repository/UoW、`scripts/init_postgresql.py` | PostgreSQL、UUIDv7、Tenant/Domain 约束；Schema 与初始化器同批更新 |
| Main API | `services/main_api/src/main_api/apps/aiops/` | 可信 Tenant/Domain/App 上下文、BFF 授权与错误投影 |
| Client | `packages/platform_clients/src/platform_clients/aiops.py` | 只调用内部 AIOps 合同，不绕过应用层 |
| Portal | `apps/portal-web/src/features/aiops/` 及既有路由/导航/服务 | Vue 3、共享组件、设计 Token、本地 ECharts、可访问数据表 |
| 测试 | PostgreSQL 集成测试、Portal 合同测试和生产构建 | 验证 Tenant/Domain 隔离与当前 Portal 交互规范 |

### 4.3 KBot4 平台适配

| 层 | 实施位置 | 要求 |
| --- | --- | --- |
| Meta Database | `database/oracle/aiops_agent/`、Oracle 实体、Repository/UoW 和初始化/校验工具 | 更新规范 DDL 和 manifest；保持 Domain 约束与 Oracle 事务语义 |
| Main API | `services/main_api/src/main_api/` 中既有 AIOps BFF | 可信 Domain/KBot App 上下文、BFF 授权与错误投影 |
| Client | `packages/platform_clients/src/platform_clients/aiops.py` | 与 Ammolite 保持相同方法和业务合同 |
| 前端 | `ui/aiops/` 中维护的静态页面、样式和脚本 | 不修改 `integrations/apex/**`；页面任务、状态和权限语义与 Portal 一致 |
| 测试 | Oracle 集成测试、静态页面合同和 OpenAPI 检查 | 验证 Domain 隔离、CSP、缓存版本和静态资源本地化 |

### 4.4 外部接入和部署资产

- Prometheus/Grafana 资产继续位于 `scripts/deployment/aiops_observability/`，入口仍是无参数
  `scripts/aiops-stack` 和唯一可编辑 `aiops-stack.ini`；
- Zabbix 参考 Template、Metric Mapping 与接入说明位于 `configuration/zabbix/`，不进入观测栈
  Compose，也不由部署脚本自动导入；
- 两边可以保留品牌化 Dashboard UID、数据库账号和 Header，但对应场景、Profile 和安全门禁必须
  一致；
- 部署资产变更必须有“未新增 Zabbix 服务”的结构测试。

## 5. 分批实施

### 5.1 批次 0：差异门禁和合同冻结

目标：把后续同步从人工记忆变成可重复检查，并冻结所有破坏性切换的目标合同。

实施项：

1. 记录两个仓库 HEAD、工作树、Schema/manifest、OpenAPI 和前端构建基线；
2. 建立差异清单，覆盖 Tool、Action、Starter、Check、Metric、Profile、路由、DTO、错误码和测试场景；
3. 每个差异标记 `DIRECT_COPY`、`META_DATABASE_ADAPTATION` 或 `FRONTEND_ADAPTATION`；
4. 冻结 A-01～A-07 目标合同，以及详细设计第 9 节 Monitoring API/DTO；
5. 建立双仓批次清单和交付报告模板，记录提交、测试、数据库、重启、Smoke 与未执行项；
6. 为双方文档、资源目录和 OpenAPI 增加布局/链接/差异检查入口。

交付物：基线报告、差异机读快照、合同变更清单、批次验收模板。

完成条件：所有当前差异都已分类；没有未知单边路由、字段或资源；后续每项改动都有对应编号。

### 5.2 批次 1：动作授权与 Binding 安全收敛

目标：完成 A-01、A-02，为实时监控和诊断执行建立最小授权与精确实例映射。

实施项：

1. Ammolite 引入 KBot4 的 `controlled_action_execution`：按 Target 明确选择动作，并限制 Schema、
   动态参数、Resource Manager Plan、授权用户和权限白名单；
2. 能力探测只生成候选动作，管理员确认并发布版本后才生效；运行时、审批时和执行时均重新校验；
3. KBot4 引入 Ammolite 的完整 Target×Source Binding 校验；Agent 启用时所有已选 Target 和 Source
   组合都必须存在 ACTIVE Binding；
4. 若未来允许来源子集，必须增加显式策略并冻结进 Agent Version，本批不保留“至少命中一个”的
   隐含语义；
5. 同步内部/公共 API、OpenAPI、两个管理前端和权限测试；
6. 使用 Oracle Target 验证候选发现、最小授权、越权拒绝、审批、执行和执行后验证。

依赖：批次 0 合同已冻结。

测试：策略解析、对象范围、未授权动作拒绝、版本并发、完整矩阵缺边、禁用 Binding、跨 Domain、
API/前端合同和真实 Oracle 受控动作。

完成条件：A-01、A-02 在两个仓库的业务语义一致，且监控目录不能看到未精确绑定的实例。

### 5.3 批次 2：实时监控共享内核与 Provider 对齐

目标：完成与 HTTP/数据库/前端无关的统一监控业务内核。

实施项：

1. 新增并校验 `MonitoringProfileSummary`、`MonitoringView`、Source/Instance、Window、Panel、
   Series、Point、Gap、Readiness 等合同；
2. 建立 `dashboard_profiles.v1.json`，复用唯一 Metric Catalog，不创建第二套指标定义；
3. 实现 Source/Instance 目录、实例发现候选、批量映射、Profile 过滤、Snapshot Builder 和 View
   Builder；
4. 实施 15 秒短缓存、Domain/版本缓存键、最多 12 实例/8 指标/96 Series/240 点、字节和超时预算；
5. 支持实例/指标级部分成功，并区分未配置、来源禁用、连接失败、无 Binding、无采样、过期和部分
   数据；
6. 收口 Prometheus `query_range`、Target Scope、活动事件和 Alertmanager 故障/恢复路径；
7. 加固 Zabbix 专用 JSON-RPC 健康检查、认证模式、`item.get.value_type`、`history.get`、精确 Item
   映射、Problem 查询、错误和 Gap；
8. 为 Zabbix 补齐 Oracle/MySQL/PostgreSQL Provider Definition 和 PostgreSQL 六项参考 Template；
9. 统一 SignalEvent、Provider Event ID、幂等、严重度、`event_class`、Evidence、Provenance、自动
   诊断和恢复验证；
10. OEM 保持现状，并通过测试保证不进入一期 Profile 和选源。

依赖：批次 1 精确 Binding 语义已经生效。

测试：详细设计第 17.1 节全部单元场景；两个仓库使用同一业务夹具和黄金结果，Provider 网络行为用
各自受控 Fake/Stub 隔离。

完成条件：同一标准输入在两边产生等价 Monitoring View、Readiness、Gap 和诊断事件；
Prometheus/Zabbix 对三种数据库的 Profile 与 Metric Code 集合一致。

### 5.4 批次 3：API、Client 与持久化适配

目标：按详细设计第 9、10 节把共享内核接入服务边界，不让前端或路由承担业务逻辑。

实施项：

1. 实现公共 Main API：Source/Instance/Profile 目录、多实例 View、单实例 View、实例发现和映射；
2. 实现对应 `/internal/v1/aiops/monitoring/...` 路由和 Platform Client 方法；
3. 公共请求只接受 `source_id`、`instance_id`、Profile 和受控时间窗；服务端重新验证 Domain、Target
   与 ACTIVE Binding；
4. API 不返回 Endpoint、凭据、Locator、查询文本和 Provider 原始 Payload；
5. 通过 UoW/Repository 读取 Source、Target 和 Binding，API/Application 不持有 Session 或执行 SQL；
6. Ammolite 在 PostgreSQL 规范 ORM/初始化器中实现需要的字段和索引；KBot4 在 Oracle canonical
   DDL/manifest 中实现同等结构；若设计确认无需新表，则不得为了缓存结果而新增表；
7. 生成并比较内部/公共 OpenAPI，删除被新合同替代的旧调用方，不保留兼容路由。

依赖：批次 2 合同和应用用例稳定。

测试：详细设计第 17.2 节、Repository/UoW、真实 PostgreSQL/Oracle 持久化、权限篡改和 OpenAPI
差异检查。

完成条件：公共 API 和内部 API 在两个项目具有相同业务合同；数据库差异只存在于适配层。

### 5.5 批次 4：两个前端与 Grafana 受控下钻

目标：以两套技术栈实现同一页面任务、状态和安全语义。

共同交互：

1. AIOps 导航新增“实时监控”，不把曲线查询并入现有工作台首屏；
2. 先选来源，再选最多 12 个已授权实例和固定时间窗；
3. 提供状态矩阵、多实例曲线、单实例详情、来源对比、Gap、数据质量和可访问数据表；
4. 显式区分无配置、禁用、连接失败、无映射、无采样和部分成功；无数据不得显示为 `0` 或健康；
5. 从工作台、Situation、Target 进入监控，并把 Source、Target、时间窗和 Metric Code 带入智能诊断；
6. 浏览器只调用 Main API，不按 Provider 类型拼查询，不持有 Locator 或凭据；
7. 图表库本地加载，保留许可证，并遵守各自 CSP 和 Design System。

Ammolite 实现 `AIOpsMonitoringView.vue` 及复用组件、Token、路由、导航和 API Service；KBot4 在
`ui/aiops/` 实现等价静态页面/脚本/样式并更新资源版本，禁止修改 APEX 集成目录。

Grafana 仅允许详细设计列出的固定 UID。启用 Embed 前必须完成短期 HttpOnly Session、非管理员身份、
禁用 Explore/编辑/数据源配置、Target 锁定、Domain 隔离、CSP 和审计。任一条件不满足时只提供受控
`LINK` 或 `DISABLED`，不得降级成匿名、共享管理员或 URL Token。

依赖：批次 3 公共 API 和错误语义通过合同测试。

测试：组件/静态页面合同、权限、空状态、部分成功、响应式/可访问性、本地资源/CSP、Portal 生产
构建、KBot4 静态资源检查和 Grafana 安全场景。

完成条件：相同用户任务在两个前端得到相同状态、筛选、图表、错误、权限和深链结果。

### 5.6 批次 5：剩余双向吸收

目标：完成 A-03～A-07，并保证新增监控链路使用统一后的能力。

| 编号 | 实施内容 | 主要接收方 | 验收重点 |
| --- | --- | --- | --- |
| A-03 | 对齐恢复失败、实测 RPO/RTO 缺失或超标、策略变化、过期和来源覆盖 Gap | KBot4 | Oracle RMAN、PG WAL、MySQL Binlog 的通过/失败/超标/过期/缺源 |
| A-04 | 报告新增独立 `*_display` 投影，保留原始枚举 | Ammolite | API、预览、PDF、领导简报中文一致，持久化枚举不变 |
| A-05 | 检查项显式 `supported_db_types` 并在编译阶段过滤 | KBot4 | 三数据库目录校验和跨方言误选拒绝 |
| A-06 | Diagnostic Source → Target Binding 只读反向查询 | KBot4 | 权限、稳定排序、禁用状态和监控接入映射矩阵 |
| A-07 | 工作负载下载统一为不可变 Artifact ID、所有权和安全 Header | 两边 | 调用方/OpenAPI/前端同批切换，旧专用路由删除 |

依赖：批次 4 已证明 Monitoring Context 可以进入诊断和报告；A-07 切换前完成调用方清单。

测试：对齐基线第 5 节全部门禁，并对实时告警触发的诊断、恢复、报告和 Artifact 做回归。

完成条件：A-03～A-07 从差异表移入共同基线，附两个仓库的提交和验收证据。

### 5.7 批次 6：真实环境联合验收与发布

目标：证明两个项目在各自平台边界内均形成真实闭环。

环境矩阵至少包含：

| 项目 | Meta Database | Provider | Target Database | 必验链路 |
| --- | --- | --- | --- | --- |
| Ammolite | PostgreSQL | Prometheus/Alertmanager | Oracle、MySQL、PostgreSQL | 目录→图表→故障→诊断→恢复→报告 |
| Ammolite | PostgreSQL | 外部 Zabbix | Oracle、MySQL、PostgreSQL | 发现→映射→图表→Problem/Webhook→恢复 |
| KBot4 | Oracle | Prometheus/Alertmanager | Oracle、MySQL、PostgreSQL | 目录→图表→故障→诊断→恢复→报告 |
| KBot4 | Oracle | 外部 Zabbix | Oracle、MySQL、PostgreSQL | 发现→映射→图表→Problem/Webhook→恢复 |

每个组合验证正常、认证失败、端点不可达、无采样、部分指标失败、重复事件、恢复事件、跨 Domain
越权和未映射实例。Zabbix 必须真实调用 `host.get`、`problem.get`、`item.get`、`history.get`；
Prometheus 必须真实调用 `query_range`。Grafana 单独记录 `EMBED`、`LINK` 或 `DISABLED` 以及证据，
不能从 Prometheus 成功推断 Grafana 安全验收成功。

依赖：批次 1～5 在两个仓库均通过静态、单元、合同和集成测试。

完成条件：第 7 节验收矩阵全部有证据或明确阻断；27 项中任一项失败，实时监控一期不得发布为
完成。

## 6. 每批次固定门禁

每个批次按以下顺序执行，不能用后一步替代前一步：

1. **变更前审计**：HEAD、工作树、关联编号、调用方和 Schema 影响；
2. **服务层对齐**：符号、DTO、资源、错误码、状态机和测试场景差异归零或分类；
3. **平台适配**：Oracle/PostgreSQL、Domain/Tenant 和两套前端分别实现；
4. **静态门禁**：编译、资源解析、文档链接、Schema manifest、OpenAPI 和边界检查；
5. **自动化门禁**：单元、合同、数据库集成和前端构建；
6. **运行门禁**：仅在该批涉及运行时执行进程、端口、Readiness、日志和受控 Smoke；
7. **提交门禁**：两个仓库分别提交相关文件，不夹带既有工作树改动；
8. **报告门禁**：列出真实执行结果、未执行项、数据库变更、重启/部署和在线状态。

推荐提交顺序为“合同/资源 → 应用服务/Adapter → 持久化/API → 前端 → 测试/文档”，但每个可合并
提交必须独立通过其声明的测试。提交信息使用中文 Conventional Commits，并在正文记录对应批次和
A/M 编号。

## 7. 验收追踪矩阵

### 7.1 双向吸收

| 编号 | 批次 | 完成证据 |
| --- | --- | --- |
| A-01 | 1 | Ammolite Target 动作配置、服务端多层校验、Portal、越权测试、Oracle Smoke |
| A-02 | 1 | KBot4 完整矩阵校验、管理页面、缺边/禁用/跨 Domain 测试 |
| A-03 | 5 | KBot4 恢复 Gap/状态、三数据库恢复场景与报告证据 |
| A-04 | 5 | Ammolite 原始值和展示值并存，预览/PDF/简报回归 |
| A-05 | 5 | KBot4 显式 DB 元数据、编译过滤和跨方言回归 |
| A-06 | 5 | KBot4 Source 反向 Binding API/UI、权限与排序测试 |
| A-07 | 5 | 双仓 Artifact ID 下载、所有权/Header 测试、旧路由与调用方清理 |

### 7.2 实时监控详细设计 27 项验收

| M 编号 | 详细设计验收摘要 | 主要批次 | 关键证据 |
| --- | --- | --- | --- |
| M-01 | 新增实时监控导航，不改变工作台职责/性能 | 4、6 | 前端合同、工作台负载回归 |
| M-02 | 无可用来源时只显示接入引导 | 2、4 | 门禁单测、页面测试 |
| M-03 | 禁用、连接失败、无映射、无采样四态分离 | 2、4 | DTO/Gap 与页面状态测试 |
| M-04 | 一源映射多个授权实例 | 1～3 | Binding、目录、权限测试 |
| M-05 | 最多 12 个同源实例和独立曲线 | 2、4 | 预算、Series、图例测试 |
| M-06 | 单实例下钻保留上下文 | 3、4 | 路由和深链测试 |
| M-07 | Prometheus 覆盖三种数据库指标 | 2、6 | Catalog 检查、真实查询 |
| M-08 | Zabbix 与 Prometheus 的 Profile/Metric 对齐 | 2、6 | 资源差异门禁、真实查询 |
| M-09 | OEM 不进入一期 | 2～4 | Profile、API、UI 排除测试 |
| M-10 | 固定四档时间窗 | 2～4 | 合同和 UI 测试 |
| M-11 | Panel 展示来源、实例、采样、覆盖率、单位、Gap | 2、4 | 投影和前端测试 |
| M-12 | 部分成功且无数据不伪装为零/健康 | 2、4、6 | 部分失败与真实无数据场景 |
| M-13 | 只查用户所选来源/实例，不污染工作台 | 2～4 | 查询捕获、工作台回归 |
| M-14 | 篡改 Source/Instance 不越权 | 1、3、6 | 授权单测和跨 Domain Smoke |
| M-15 | 浏览器无凭据、Locator、任意查询和内部路由 | 3、4 | DTO、网络请求和安全测试 |
| M-16 | 监控上下文可进入智能运维 | 3、4、6 | 深链与诊断 Run 证据 |
| M-17 | Grafana 不安全时不内嵌/匿名暴露 | 4、6 | 安全门禁和部署检查 |
| M-18 | 两类来源查询事件并触发相同诊断 | 2、6 | 事件查询/Webhook E2E |
| M-19 | 故障、恢复、幂等、严重度、Target、证据语义一致 | 2、6 | 黄金合同和双 Provider E2E |
| M-20 | PostgreSQL/Zabbix 图表到恢复闭环 | 2、6 | 真实专项 E2E |
| M-21 | Zabbix 专用健康检查及真实 Smoke | 2、6 | 单测和 JSON-RPC 证据 |
| M-22 | 双 Readiness 才能标记完整接入 | 2～4、6 | 投影、UI、故障场景 |
| M-23 | 观测栈不含 Zabbix 部署资产 | 0、2、6 | 配置/Compose 结构测试 |
| M-24 | Zabbix 只从 AIOps App 外部接入 | 3、4、6 | 配置合同和接入审计 |
| M-25 | 不自动改动客户 Zabbix | 2、4、6 | 只读 API 捕获和操作记录 |
| M-26 | OpenAPI、前端、单元和边界检查全通过 | 0～6 | 双仓测试报告 |
| M-27 | 分别如实报告四类 Provider 状态 | 6 | 最终实施报告 |

矩阵中的“主要批次”表示实现或验收落点，不改变详细设计原文。最终交付报告必须逐条引用 M-01～
M-27 的证据，不能只写“测试通过”。

## 8. 失败处理与回退

1. 每批次提交前保留两个仓库的基准提交、Schema 版本、OpenAPI 和资源摘要；
2. 应用代码失败时回退整个变更编号的对应提交，不恢复被替代的兼容路由或双读写；
3. 合同切换失败时前端、Main API、Client 和 AIOps API 作为一个发布单元回退，禁止新旧合同混跑；
4. Schema 变更只通过各自规范 Schema/初始化流程重建或重新发布；不得临时创建影子表、兼容列或
   回填路径；执行任何破坏性数据库操作前必须另行审批和备份；
5. Provider 故障通过 Readiness、Gap 和部分成功降级，不回退为浏览器直连或放宽 Target Scope；
6. Grafana 安全条件失败时回退到 `LINK` 或 `DISABLED`，不影响原生监控页；
7. Zabbix 真实验收失败时保留外部接入配置和稳定错误证据，但产品状态不得显示“完整接入”；
8. 单边失败时另一边不得自行演进合同；修复仍须同步进入两个仓库并重新执行差异门禁。

## 9. 最终完成定义

只有同时满足以下条件，工作才可以整体关闭：

- A-01～A-07 均已在两个仓库形成共同基线，没有未分类业务差异；
- M-01～M-27 逐条通过，Prometheus 与外部 Zabbix 均完成三数据库真实环境验收；
- 两个项目的内部/公共 OpenAPI、Metric/Profile 资源、错误码和状态机等价；
- Ammolite PostgreSQL 与 KBot4 Oracle 的规范 Schema、初始化和持久化验证均通过；
- 两个前端具有等价任务、状态、权限和深链，且构建/静态检查通过；
- Grafana 状态被如实标记为 `EMBED`、`LINK` 或 `DISABLED`，OEM 明确标记为一期未实施；
- 两个仓库均有对应提交和完整实施报告，报告明确列出未执行项、部署、重启和在线验证事实。

文档、编译、HTTP 200、配置保存成功或单一 Provider 有数据，都不能单独作为整体完成证据。
