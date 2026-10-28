# SQL 诊断实施记录与发布检查

日期：2026-10-28。Ammolite / KBot4 同步实现，覆盖 PG / MySQL / Oracle 的同一领域合同。
状态：核心代码已实现并通过本地回归；未部署、未完成真实模型/三库闭环验收。
不能把此状态改写为目标设计全部完成。

## 本轮交付

- 结构化结果行选择、不可变证据回查、权限/完整性/期限校验和单 Target 路由；
  两端前端提交引用，避免旧对话劫持 SQL 身份，大整数精确传输。
- 普通单 SQL 使用问题/假设驱动完整调查，数据库和监控可以同轮取证，
  不再把监控不足作为查库门禁，取消普通调查的固定 Oracle 基线快捷返回。
- SQL Evidence Bundle、结构化判断和建议、行/字段引用及作用域检查，
  方案包含条件、风险、验证和停止/回滚；逐项展示时间口径和背景作用域，
  未核验材料不能升级为已确认事实；失败才回证据摘要，标记部分完成。
- MySQL 专用长尾只读查询的 AST、签名 Grant、Executor 再校验、只读事务；
  不放开业务表/DML/变量/文件/锁；三库原有安全及 Oracle Pack 边界保持。
- 有限候选/有界可见集合设置，真实采样边界和覆盖、源版本/已识别重置处理，
  NULL Schema 与空字符串区分，PG 多维身份和精确差分、不相减极值。
- 默认交互绝对 deadline、模型和 Worker 预算、重规划汇总预留，
  已持久化事实的截止部分回答及取消连接清理回归。
- 共享合同、OpenAPI、前端及当前边界文档同步；无新增数据库列。

## 尚未交付的目标能力

1. 稳定物理实例/启动代次的完整跨源映射、自然语言同标识多候选消歧、
   无选行引用场景的有界跨轮搜索、完整计划/参数/对象关系图。
2. 新的 APM、发布变更、主机执行 Adapter；只使用已有真实接入且授权的来源，
   不把登记名称当作接入。
3. 各引擎自动取得匹配代表参数的计划、完整跨实例/容器历史计划适配；
   现有计划工具以真实权限、版本和许可为准，不虚构历史实际计划。
4. 所有单条重置/淘汰的识别、更大集合的有界分页，以及源级字节/并发/成本配额的联合治理。
5. 未建 Run 的排队截止扫描、WAITING_USER 续跑预算、底层跨进程终止与迟到结果隔离；
   默认 90 秒是代码预算目标，不是已测 P95 承诺。
6. 详细设计第 10 节的真实三库业务读写、隐藏故障真值的原因区分、
   措施前后同负载对照、真实 UI 和性能/并发/取消故障验收。

引用存在不等于机械证明因果；本地合同测试和受控模型替身不证明线上诊断有效性。
上述缺口继续作为发布门禁，不以核心回归通过替代。

## 可复跑验证

Ammolite 使用 cube 环境，KBot 使用 kbot4 环境；本地单测不连接目标生产数据库。

- python -m pytest tests/unit/aiops_agent -q
- python tests/acceptance/check_openapi_contracts.py
- python tests/acceptance/check_aiops_cross_project_alignment.py --peer-root <另一仓库>
- Ammolite：python tests/acceptance/check_architecture_boundaries.py；
  apps/portal-web 下 npm run build。
- KBot：python tests/acceptance/check_4_0_boundaries.py；
  node tests/contract/aiops_top_sql_table.mjs；
  node tests/contract/aiops_workload_snapshot.mjs；
  相关 UI 静态合同及 JS 语法检查。

新增 SQL 专项覆盖身份越权/撤权/过期/篡改、字段引用、自证拒绝、
无计划的会话锁关联、模型结构化诊断调用、三库选行冻结、MySQL AST/绑定/取消连接、
窗口偏差/源变化/正增长重置/候选完整性及预算部分结果。
2026-10-28 本地执行结果：Ammolite 1,346 项 AIOps 单测及 218 项子测试通过；
KBot4 1,275 项 AIOps 单测及 143 项子测试通过。
两端 OpenAPI、跨项目语义和架构边界检查通过；Ammolite Vue 类型检查及生产构建通过，
构建仍有大 chunk 提醒；KBot 两项 Node 合同及 57 项相关 UI 静态合同通过。
这些结果不替代真实依赖、浏览器交互或生产诊断质量验收。

## 发布顺序

1. 管理员受控发布更新后的规划、评估 Prompt，并新增 aiops_agent.sql_diagnosis；
   保留既有冻结版本。KBot 文件安装基线按仓库约定仍为 1.0.0，
   不得以全库初始化替代定向 Prompt 发布。
2. API、Worker、Executor、前端同步更新。无表列升级，不安装扩展、不改目标库配置。
3. 验证线上实际 Prompt、选行请求、证据行与部分结果，再做已授权真实库纵向闭环；
   三库分别验收，采集策略默认不变，开启/修改需发布配置。
4. 回滚应用和前端到上一受控版本，还原本轮配置/Prompt 激活选择；
   不删除证据、不重置数据库统计、不自动回填窗口。
