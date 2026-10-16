# Oracle DBA 操作资产

本目录保存必须由 DBA 显式确认后执行的重置、导出等 Oracle SQL。它们不属于空 Schema
初始化序列，也不会由应用启动或部署入口自动执行。

每个操作必须在文件头说明影响范围、前置条件、是否包含敏感数据以及失败后的恢复方式。

`apply_aiops_schema_20.sql` 用于已有测试数据不能重建时，将 AIOps Schema 19 原地升级到
Schema 20。执行前必须停止 AIOps API、Worker 和 DB Executor，并完成 Schema 备份；脚本保留
历史数据，但会关闭历史 Agent–Target 的受控执行，升级后需要在 Agent 页面重新明确授权。

`apply_aiops_schema_22.sql` 用于将 AIOps Schema 21 原地升级到 Schema 22，扩展附件受控
检索所需的工具调用分类约束。执行前必须停止 AIOps API、Worker、Scheduler 和 DB Executor，
并完成 Schema 备份；脚本不删除表、不删除业务行，也不修改既有诊断或审计数据。

`apply_aiops_schema_23.sql` 用于将 AIOps Schema 22 原地升级到 Schema 23，增加 Oracle
CDB/PDB 预期范围与连接实测字段。脚本保留全部业务行、Endpoint 和凭据引用；由于 Service
Name 不能可靠判断容器类型，既有 Oracle 直连 Target 会被安全停用为仅监控模式，升级后须在
Target 页面明确选择 `CDB Root`、`PDB` 或 `Non-CDB`，重新测试连接后再启用。

`apply_aiops_schema_26.sql` 用于将 AIOps Schema 23 / `aiops-oracle-v13` 或 Schema 24 /
`aiops-oracle-v14` 原地升级到 Schema 26 / `aiops-oracle-v16`。脚本给巡检计划补
`SELECTED_CHECKS_JSON`，并把缺少检查项快照的计划回填为当前 Check Catalog 的 READY
默认项；同时新增 `KBOT_OPS_TARGET_FACT`。执行前必须停止 AIOps API、Worker、Scheduler
和 DB Executor，并完成 Schema 备份。若当前仍是 Schema 23，先执行
`converge_core_authorization.sql` 删除废弃的 `KBOT_OPS_AGENT_GRANT`。脚本不删除业务行，
也不改写历史运行、诊断、报告、附件或审计数据。

`apply_aiops_schema_27.sql` 用于将 AIOps Schema 26 / `aiops-oracle-v16` 原地升级到
Schema 27 / `aiops-oracle-v17`。脚本仅扩展回答块类型约束以允许
`IMPLEMENTATION_RUNBOOK`，并更新 Schema 版本视图；不删除或改写任何业务行。

`apply_aiops_schema_28.sql` 用于将 AIOps Schema 27 / `aiops-oracle-v17` 原地升级到
Schema 28 / `aiops-oracle-v18`。脚本移除全部 `KBOT_OPS_%` 表上的用户命名 `CHECK`
约束，把枚举、状态和范围规则收回应用层合同；不删除表，也不删除或改写业务行。执行前
必须停止 AIOps API、Worker、Scheduler 和 DB Executor，并完成 Schema 备份。脚本设置
60 秒 DDL 锁等待；若仍遇到活动事务，会跳过该表并以 `-20064` 结束。此前成功删除的约束
不会回滚，停止持锁服务后重新执行同一脚本即可从剩余约束继续。

`apply_aiops_schema_29.sql` 用于将 AIOps Schema 28 / `aiops-oracle-v18` 原地升级到
Schema 29 / `aiops-oracle-v19`。脚本为 `KBOT_OPS_TARGET` 新增
`IMPORTANCE_LEVEL`，既有 Target 回填为 3（普通），不增加业务 `CHECK` 或 `UNIQUE`
约束；同时更新 Target 查询视图和 Schema 版本视图。执行前必须停止 AIOps API、Worker、
Scheduler 和 DB Executor，并完成 Schema 备份。

`apply_aiops_schema_30.sql` 用于将 AIOps Schema 29 / `aiops-oracle-v19` 原地升级到
Schema 30 / `aiops-oracle-v20`。脚本新增独立的巡检模板及不可变版本表，把每个既有
巡检计划的勾选项和固定取证步骤迁移为一个模板版本，并将 Plan 与历史 Fire 改为引用该
版本；随后移除旧模板字符串、计划内勾选项、计划解析器版本和本次涉及的业务 `UNIQUE`
约束。执行前必须停止 AIOps API、Worker、Scheduler 和 DB Executor，等待运行中或排队中
的巡检 Fire 全部结束，并完成 Schema 备份。脚本不删除业务行。

`apply_aiops_schema_31.sql` 用于将 AIOps Schema 30 / `aiops-oracle-v20` 原地升级到
Schema 31 / `aiops-oracle-v21`。脚本为 Target 增加工作负载快照与活动采样策略和运行态，
并新增 MySQL/PostgreSQL 工作负载快照、语句、指标和活动样本表。新表只包含主键、外键
和列级 `NOT NULL`，不增加业务 `CHECK`、`UNIQUE` 或唯一索引。

`apply_aiops_schema_32.sql` 用于将 AIOps Schema 31 / `aiops-oracle-v21` 原地升级到
Schema 32 / `aiops-oracle-v22`。脚本新增运维知识资产、不可变版本、适用范围、来源、
KC 索引引用和审核记录六张表，并更新 Schema 版本视图；不删除或改写既有业务行，
不增加业务 `CHECK`、`UNIQUE` 或唯一索引。脚本可在部分升级后续跑，执行前必须停止
AIOps API、Worker、Scheduler 和 DB Executor，并完成 Schema 备份。

`apply_aiops_schema_33.sql` 用于将 AIOps Schema 32 / `aiops-oracle-v22` 原地升级到
Schema 33 / `aiops-oracle-v23`。脚本新增 Target 恢复目标和恢复演练两张表，保留现有业务行，
不增加业务 `CHECK`、`UNIQUE` 或唯一索引。脚本可在部分升级后续跑；执行前必须停止 AIOps API、
Worker、Scheduler 和 DB Executor，并完成 Schema 备份。

`apply_aiops_schema_34.sql` 用于将 AIOps Schema 33 / `aiops-oracle-v23` 原地升级到
Schema 34 / `aiops-oracle-v24`。脚本为恢复目标新增显式必需备份来源集合，并依据 Target 数据库类型
为既有 Profile 写入规范默认来源；不删除目标或演练记录。脚本可在部分升级后续跑；执行前必须停止
AIOps API、Worker、Scheduler 和 DB Executor，并完成 Schema 备份。

`apply_aiops_schema_36.sql` 用于将 AIOps Schema 35 / `aiops-oracle-v25` 原地升级到
Schema 36 / `aiops-oracle-v26`。脚本先重建 Target 只读视图，再删除 Target 上已废弃的
`SECURITY_LEVEL`，保留全部业务行。脚本可在部分升级后续跑；执行前必须停止 AIOps API、
Worker、Scheduler 和 DB Executor，并完成 Schema 备份。

`apply_aiops_schema_37.sql` 用于将 AIOps Schema 36 / `aiops-oracle-v26` 原地升级到
Schema 37 / `aiops-oracle-v27`。脚本删除废弃的 Agent–监控源关系表，但不会把旧监控源关系
猜测为 Target 授权；没有有效监控映射或引用未启用监控源的 Target 会被停用，没有 Target 或
引用未启用 Target 的 ACTIVE Agent 也会被停用。Agent、Target、监控源、Target 映射及历史
运行数据均保留。脚本可在部分升级后续跑；执行前必须停止 AIOps API、Worker、Scheduler 和
DB Executor，并完成 Schema 备份。

`apply_aiops_schema_43.sql` 用于将 AIOps Schema 42 / `aiops-oracle-v32` 原地升级到
Schema 43 / `aiops-oracle-v33`。脚本为会话报告模板增加 Domain 默认标记，为正式报告
增加来源会话引用和查询索引；既有模板默认不覆盖系统模板，既有报告不猜测来源会话。
脚本保留全部业务行，不增加业务 `CHECK`、`UNIQUE` 或唯一索引。执行前必须停止 AIOps
API、Worker、Scheduler 和 DB Executor，并完成 Schema 备份。

`apply_aiops_schema_44.sql` 用于将 AIOps Schema 43 / `aiops-oracle-v33` 直接升级到
Schema 44 / `aiops-oracle-v44-resource-graph`。脚本跳过已否定且未部署的 Database System
中间模型，直接建立业务系统、可管理资源、数据库/计算资源及资源关系，为每个既有 Target
无损补齐一套业务系统资源图谱，并把数据保护与恢复演练迁移到数据库资源。执行前必须确保
所有 Target 已有责任组，停止 AIOps 写入服务并完成 Schema 备份；脚本拒绝覆盖任何中间版
或目标表。

`apply_aiops_schema_45.sql` 用于将 AIOps Schema 44 /
`aiops-oracle-v44-resource-graph` 升级到 Schema 45 /
`aiops-oracle-v45-shared-monitor-locator`。脚本移除监控 Locator 的全局唯一约束，允许 RAC
节点等多个 Target 复用同一监控对象；不删除 Target、监控源或绑定业务行。

`../generated/kbot_41/apply_kbot_41_from_aiops_schema_42.sql` 是 SQL Developer 可直接
执行的单文件追平产物，严格从 AIOps Schema 42 / `aiops-oracle-v32` 依次升级到 Schema 45 /
`aiops-oracle-v45-shared-monitor-locator`，随后执行 KBot 4.1 的 KM Asset 退役。该脚本由
`tools/db/render_kbot_41_online_upgrade.py` 从 43、44、45 三段 AIOps 增量脚本和 KM 退役
脚本生成，不使用 `@`/`@@`。它会物理删除 KM 私有表和授权数据；执行前必须停止全部相关
写入服务、完成并验证 Data Pump 备份，继续运行 KM 4.0 的环境不得执行。

若 42→43 已经成功提交、后续阶段因前置数据检查停止，必须改用
`../generated/kbot_41/apply_kbot_41_from_aiops_schema_43.sql` 从 Schema 43 续跑；该产物只包含
43→44、44→45 和 KM Asset 退役，不会重复执行已经完成的 42→43 DDL。

`apply_knowledge_retrieval_and_media_studio.sql` 用于既有 Schema 补齐知识检索与多媒体创作
工作台表结构：把 `KBOT_KR_AGENT_VERSION` 从 `ENABLED_CAPABILITIES_JSON` 收敛为
`KNOWLEDGE_CORE_ID` + `DATA_MODEL_IDS_JSON`，新增 X Search 运行表，并创建多媒体绑定、
Run、事件、提示词版本和资产表。脚本会删除已退役的智能工作台 `KBOT_ASST_%` 表，不迁移
历史运行或图片元数据。表结构完成后依次执行
`conda run -n kbot4 python scripts/db/apply_oracle_schema.py --foundation-only`、
`database/oracle/bootstrap/knowledge_retrieval/initial_admin.sql` 和
`database/oracle/bootstrap/media_studio/initial_admin.sql`，以收敛权限目录并写入各 App
初始管理员。

`enable_model_serving_capabilities.sql` 用于为已有模型目录新增图片生成能力标志。
X Search 属于 Grok 模型能力，不再写入模型表。脚本不会把任何模型标记为可用；新增列默认
关闭，必须由受控 Canary 写入验收结果后，业务入口才可使用对应能力。

## 权限模型收敛

`purge_retired_assistant_authorization.sql` 用于物理删除已退役 `assistant` App 的
权限、角色、Domain、成员、API Client 以及 App 自有用户。若 App 自有用户曾被显式
授予其他 App，这些绑定也会一并删除。脚本保留历史审计表，新增一条物理清理事件，
支持重复执行并在异常时整体回滚。

`upgrade_app_authorization_roles.sql` 用于把既有业务 App 升级到统一的 `app_user`、
`app_manager`、`app_admin` 三层系统角色。执行前必须停止 Main API 的授权写入、完成备份，
并运行 `scripts/db/audit_app_authorization.py`。脚本保留专业角色、成员和 Domain Scope，
将专业系统角色转为 App 内自定义角色；若数据存在歧义会整体回滚。执行后必须再次运行
审计，并确认系统角色差异和异常计数全部为空。

既有 KBot 4.0 Schema 升级到统一核心权限规则时，先由 Schema Owner 执行
`converge_core_authorization.sql` 删除废弃的人类 Agent Grant 表，再运行：

```bash
conda run -n kbot4 python scripts/db/apply_oracle_schema.py --foundation-only
```

第二步会精确收敛权限目录和系统内置角色映射；自定义角色保留，但其废弃权限映射会被清理。
