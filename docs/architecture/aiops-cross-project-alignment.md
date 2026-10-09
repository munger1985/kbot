# AIOps 跨项目能力对齐与双向演进基线

## 1. 文档目的与审计基准

本文用于约束 KBot4 与 Ammolite Cube 后续 AIOps 演进。目标不是把任一仓库整体复制到另一个仓库，
而是在保持各自平台边界的前提下，将已验证的业务能力双向合并，避免再次形成长期分叉。

本次静态审计日期为 2026-10-09，代码基准为：

- KBot4：`/home/chris/kbot4`，分支 `kbot4.0`，提交 `1eb3e635`；工作树中的文档改动不作为已交付代码基准。
- Ammolite Cube：`/home/chris/ammolite_cube`，分支 `dev`，提交 `a4fad94`。

本次结论来自当前 Python 服务、版本化目录、API 路由、契约、OpenAPI、Portal/静态页面和测试，
不是根据旧 Roadmap 的数量描述推断。旧文档中的 24 或 33 个巡检项已经过时，当前两个仓库实际均为
36 个检查项。

本文是差异与后续实施基线，不表示尚未迁移的能力已经交付，也不替代真实数据库、服务栈和在线 E2E
验收。

## 2. 已经对齐的核心能力

以下能力在产品语义上已经基本一致，后续必须保持同步：

| 能力 | 当前共同基线 |
| --- | --- |
| 业务入口 | 智能诊断、告警诊断、日常巡检，共用同一调查内核 |
| 调查闭环 | Turn、计划、只读取证、Finding、证据缺口、HITL、回答、报告 |
| 数据库诊断 | 154 个相同 Tool ID：Oracle 66、MySQL 31、PostgreSQL 57 |
| 受控动作目录 | 34 个相同 Action ID：Oracle 26、MySQL 8；PostgreSQL 当前无自动变更目录 |
| 新会话入口 | 24 个相同 Conversation Starter ID |
| 巡检目录 | 36 个相同 Check ID，其中 31 个 `READY`、5 个 `PLANNED` |
| 报告与知识 | 正式报告、模板、原生数据库报告、运维知识库、实施 Runbook |
| 诊断治理 | Target、Diagnostic Source、精确 Source Binding、私有 Agent、审批和执行验证 |
| 新增公共能力 | 场景化图表、跨数据库恢复策略与恢复演练证据 |

目录 ID 一致只证明能力清单对齐，不证明 SQL 内容、权限、状态机、错误语义、展示合同和真实运行结果
已经完全等价。每个后续批次仍须进行符号级、契约级和运行级验证。

## 3. 当前差异与双向吸收决策

### 3.1 必须双向合并的业务差异

| 编号 | 当前差异 | 优势来源 | 统一目标 | 优先级 |
| --- | --- | --- | --- | --- |
| A-01 | KBot4 支持按 Target 显式选择受控动作，并限制 Schema、动态参数、Resource Manager Plan、授权用户和权限白名单；Ammolite 当前按 Target 能力自动开放全部兼容动作 | KBot4 | Ammolite 引入同等粒度的 `controlled_action_execution` 配置、能力探测选项、服务端校验和 Portal 表单；禁止仅靠审批代替授权范围 | P0 |
| A-02 | Agent 启用时，Ammolite 要求每个 Target 与全部选中 Diagnostic Source 建立有效 Binding；KBot4 只要求至少命中一个来源 | Ammolite | 统一为完整 Target×Source 矩阵校验；若产品需要“任一来源即可”，必须改成显式策略字段，不能继续使用隐含的集合交集语义 | P0 |
| A-03 | Ammolite 的恢复保障快照区分最近演练失败、实测 RPO/RTO 缺失或超标、策略变更、演练过期和必需备份来源未覆盖；KBot4 当前只覆盖较粗的恢复验证、过期、策略变更和外部备份平台缺失 | Ammolite | KBot4 对齐稳定 Gap Code、`assurance_status`、实测目标比较和来源覆盖；两边保持 Oracle/PG/MySQL 恢复坐标差异 | P1 |
| A-04 | KBot4 的正式报告展示会翻译状态、风险等级和稳定错误/缺口枚举；Ammolite 仍有原始枚举直接进入预览或 PDF | KBot4 | Ammolite 对齐报告展示投影；保留原始合同值，新增独立 `*_display`，不修改持久化枚举 | P1 |
| A-05 | Ammolite 的 PostgreSQL 巡检项显式声明 `supported_db_types`，并在编译检查步骤时按数据库类型过滤；KBot4 对部分 PG 检查仍依赖 ID/调用上下文 | Ammolite | KBot4 补齐显式方言元数据，两个仓库使用相同的目录校验和跨方言误选回归测试 | P1 |
| A-06 | Ammolite 提供 Diagnostic Source 反向查询 Target Binding；KBot4 主要从 Target 方向管理 Binding | Ammolite | KBot4 增加同等只读反向查询，供监控接入页展示完整映射矩阵；创建、修改和删除仍归 Target×Source Binding 聚合管理 | P2 |
| A-07 | KBot4 的会话原生工作负载下载使用 `tool_id + action_id` 专用路径；Ammolite 使用 Conversation/Turn 下的通用不可变 Artifact ID | Ammolite 为主，保留 KBot4 下载安全约束 | 统一为 Artifact ID、归属校验、附件响应和安全 Header；调用方、OpenAPI、前端和测试同批切换，不保留双路由兼容 | P2 |

### 3.2 需要产品决策、不能直接互抄的差异

| 差异 | 当前状态 | 决策原则 |
| --- | --- | --- |
| 巡检计划删除 | Ammolite 有删除接口；KBot4 采用启用、暂停、禁用生命周期 | 已产生 Fire、报告或审计引用的计划不得物理删除。仅当双方确认“从未执行的草稿可删除”时统一增加受限删除，否则 Ammolite 收敛为禁用/归档 |
| Agent 自动开放动作 | Ammolite 自动生成兼容动作策略，KBot4 要求用户明确选择 | 安全基线采用 KBot4 的显式最小授权。自动探测只能生成候选项，不能直接形成生效授权 |
| 监控源完整矩阵 | Ammolite 全包含，KBot4 至少一个 | 默认采用完整矩阵；如需支持按 Target 选择来源子集，应将子集冻结进 Agent Version，而不是放宽校验 |

### 3.3 应长期保留的平台适配差异

以下差异不属于功能落后，不应通过复制目录消除：

| 边界 | KBot4 | Ammolite Cube | 共同语义要求 |
| --- | --- | --- | --- |
| Meta Database | Oracle DDL、实体、Repository、UoW | PostgreSQL ORM、UUIDv7、Repository、UoW | 实体关系、事务、排序、租约和错误语义一致 |
| 身份与隔离 | Domain 与 KBot App 认证 | Tenant、Domain、App 授权与可信内部上下文 | 进入 AIOps 应用层前形成可信 Scope；不得信任调用方自报身份 |
| 前端 | 独立 AIOps 静态 App | Vue Portal 中的 AIOps App | 页面任务、状态、权限、错误、筛选、审批和报告语义一致 |
| 品牌与协议 | KBot 名称和 Header | Ammolite 名称和 `X-Ammolite-*` | 外部已发布的监控指标名按兼容性审慎处理，内部品牌不得串用 |
| Schema 交付 | Oracle canonical DDL 与受控升级脚本 | PostgreSQL canonical ORM 与初始化器 | 各自更新唯一正式 Schema；不得新增双读写或兼容表 |

## 4. 分批实施顺序

### 批次 1：动作授权与 Binding 安全收敛

1. 将 KBot4 的显式受控动作选择、对象范围、能力探测候选和服务端校验移植到 Ammolite。
2. 将 Ammolite 的完整 Target×Source Binding 校验及反向查询移植到 KBot4。
3. 同步应用服务、API、Platform Client、OpenAPI、两个前端和测试。
4. 使用 Oracle Target 验证候选动作发现、最小授权、越权拒绝、逐条审批、执行和同口径验证。

### 批次 2：恢复保障与报告呈现

1. 将 Ammolite 的细粒度恢复保障状态和 Gap Code 合并到 KBot4。
2. 将 KBot4 的正式报告中文展示投影合并到 Ammolite。
3. 对 Oracle RMAN、PostgreSQL WAL、MySQL Binlog 各执行一组通过、失败、过期、目标超标和来源未覆盖测试。
4. 验证报告预览、PDF、领导简报和 API 同时保留原始合同值与用户可读展示值。

### 批次 3：巡检元数据与 Artifact 合同

1. 对齐检查项方言元数据和编译过滤，阻止跨数据库误选。
2. 统一会话 Artifact 下载为 ID 与所有权驱动的合同，删除被替代的专用入口及调用方。
3. 对原生 HTML、PDF、Markdown、ZIP、导入报告等媒体类型执行权限、文件名、缓存和内容安全测试。

### 批次 4：持续同步门禁

1. 建立可重复的目录 ID、路由、合同字段、状态机和测试场景差异检查。
2. 每个 AIOps 业务变更必须同时记录两个仓库的基准提交、适配分类和验证结果。
3. 同步更新 Oracle canonical DDL 与 PostgreSQL canonical ORM/初始化器。
4. 任一仓库未完成真实数据库和在线链路验收时，整体能力不得标记为双项目已交付。

## 5. 每批次强制验收

- 目录：Tool、Action、Starter、Check ID 和版本化资源差异均有明确分类。
- 业务：相同输入在两个项目产生等价状态、Finding、Gap、审批和报告语义。
- 安全：Tenant/Domain 隔离、Target 授权、Binding、动作白名单、四眼审批和 Secret 不泄漏。
- 持久化：分别在真实 Oracle 与 PostgreSQL 验证写入、并发、租约、排序和回滚。
- 合同：生成并比较 OpenAPI，验证 Main API、AIOps API、Platform Client 和前端调用方。
- 运行：以同一受控场景完成 Portal/App → Main API → AIOps → Worker/Executor → 数据库/监控源 → Evidence/Report。
- 交付：分别记录提交、测试、未执行项、服务重启和在线验证状态；不得用编译或 HTTP 200 代替业务成功。

## 6. 维护规则

1. 本文件在两个仓库中保持镜像；修改差异、优先级或统一决策时必须同步更新两份。
2. 已完成差异从“当前差异”移入共同基线，并附两个仓库的提交与验收结果。
3. 新差异必须标明是业务差异、产品决策还是平台适配，不能以“平台不同”为由长期保留业务分叉。
4. 不整目录覆盖，不引入兼容路由、双读写、历史回填或临时字符串身份字段。
5. 文档对齐不等于实现完成；只有两个仓库均通过各自门禁后才关闭对应编号。
