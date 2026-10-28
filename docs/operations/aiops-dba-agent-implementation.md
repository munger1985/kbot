# DBA Agent 实施状态与发布检查

更新日期：2026-10-28。两仓同步，基线分别为 Ammolite `f7f0e25`、KBot4 `29c4831a`。

本文件说明[统一设计](../proposals/aiops-dba-agent-capability-redesign.md)的实际落地范围。
本批为 P0 正确性核心和 P1 指标发现切片，不是整体设计验收完成；未部署或重启，
未修改线上数据库、监控绑定或 Prompt。既有 SQL 专项门禁继续适用。

## 已接通的业务链路

| 环节 | 本批代码行为 | 仍需注意 |
| --- | --- | --- |
| 普通诊断 | 模型提交严格 Claim：对象、时间、事实/解释/原因、支持/反证字段、替代解释与限制；服务端校验行/字段和信任 | 引用有效不等于自动证明因果；尚无通用身份关联和时间适配引擎 |
| SQL 诊断 | 先执行原 SQL 专项身份与证据校验，再投影为同一调查记录 | 非直接事实保守保留为解释，不从自由正文猜出“已确认原因” |
| 回答 | 先校验、冻结，再确定性渲染与推送正文；模型失败保留证据并输出部分结论 | 直接 Top N 仍用确定性列表，不被历史单 SQL 问题劫持；专项降级画像未冒充完成诊断 |
| 报告 | 从冻结记录无损投影分析、方案及引用；旧正文原样保留，不反向制造 Claim；空正文/缺引用不默认为 READY | 多轮按时间保留；没有修订关联时不取历史最高等级拼成根因 |
| 会话案例候选 | 最新结构化调查中有可追溯有效事实或支持判断即可提炼诊断参考，不再要求 CONFIRMED 或先有正式报告 | 校验记录与 Run/Target 一致；最新记录撤回、仅假设或缺失时不倒查旧肯定结论 |
| 人工收口与报告提炼 | 人工材料仍是 USER_PROVIDED/USER_REPORTED；READY、恢复勾选、不同前后引用不能自动产生 VERIFIED | 修改原因或处置摘要不继承旧验证背书；受控可比效果闭环尚未完成 |
| 权限与支持强度 | 配置只读通道不再推定 Oracle 任意字典权限、PG/MySQL 视图或变更能力；模型 DIRECT 与来源数量不自动确认根因 | 只读目录允许 UNKNOWN 访问候选；商业许可硬门禁不变，实际访问由目标库反馈 |
| 指标发现 | `monitor.discover_metrics` 从规划、冻结、任务到 Prometheus Adapter、证据和后续规划接通 | 只有显式授权的绑定可用；不是开放全局指标目录 |

## 持久化和合同边界

新增严格内部合同 `DBA_DIAGNOSTIC_ASSESSMENT.v1` 和 `DBA_INVESTIGATION_RECORD.v1`。
记录嵌入已有 `AIOPS_TURN_RESULT.v1.investigation_record`，由现有不可变 Artifact
保存，不新增表、外部路由或运行时 Schema 自检。本批每个 Turn 建立 revision=1；
`previous_revision`、execution/verification 引用字段不表示已有跨轮修订或效果写回引擎。

记录冻结原问题、Run/Task/Target、观测身份/时间/口径/信任、判断、建议和本轮目标状态。
报告再次校验证据集合与封套。人工材料不会因报告、审核或案例提炼升级为现场观测。
同源证据按目标与来源保守归组；尚未完成原生计数器级血缘和通用公式计算图。

两平台的必要实现差异保留：Ammolite UUID 与 KBot 字符串标识、各自数据库/UoW、
Portal 与静态 AIOps 页面；业务语义一致。无需本批线上 DDL。

## 指标目录发现的启用与限制

Prometheus Adapter 新增 `metric.discover`。管理员须按既有诊断源/绑定治理流程，
在来源 `declared_capabilities` 声明该能力，并在目标绑定
`capability_scope.capabilities` 中显式限定授权；
本批不自动修改旧配置，也不暗示“升级后所有绑定自动启用”。
它与 `metric.query_range` 是不同能力；后续数值查询仍需已有查询授权。

定位只来自已冻结绑定，不接受模型指定 URL、任意 selector 或其他 Target。
数据库范围使用精确的 instance/target_key 定位；主机范围只使用已绑定 host_target_key。
返回指标名与标签名，不返回标签值；类型为 UNKNOWN，帮助/单位元数据尚未核验。

每轮最多一个发现动作，各授权绑定独立限额；窗口 60–3600 秒、默认 900 秒；
默认 50、最多 100 个指标，
处理最多 1000 条系列、每指标最多 128 个标签名；生产执行响应默认上限 256 KiB。
达到目录上限或源警告标截断；字节超限、无权限或来源失败形成缺口，不当空成功。
目录证明请求窗口曾可见，不证明当前有采样，更不是性能数值。

## 发布要求

1. 两仓部署对应代码及前端，保持相同语义版本；本任务未代为部署。
2. 按既有受控 Prompt 管理发布 `aiops_agent.diagnosis_answer_compose` 的结构化内容，
   新调用不再使用 `diagnosis_answer_stream`；检查新 Run 冻结的 Prompt 内容与输出合同匹配。
   本批没有操作线上 Prompt 数据，不自动回写历史 Run。
3. 若需要指标发现，显式调整对应来源和绑定授权后重新测试连接；不要扩大为全局范围。
4. 在隔离或明确批准环境跑真实三库、模型和浏览器闭环，记录证据归属、结论准确性、
   方案针对性、报告完整性、案例治理与时延；仅通过本地测试不作为生产能力验收。
5. 回退代码/Prompt 时保留已有 Artifact；不把历史自由正文自动回填成结构化事实。

## 未交付的设计范围

- 完整跨轮 Claim 修订、反证关联、故障阶段合并，以及多目标/多子目标完成度。
- 原生计数器血缘、通用派生公式、跨源物理身份与时间关联、权限探测有效期。
- Prometheus 类型/help/单位补查，Zabbix/OEM 的同类目录发现。
- 按三库实际机制实现完整环境化 Runbook、执行关联、可比效果与护栏验证闭环。
- 四类案例严格合同、逐项效果测量、完整脱敏、更正撤回传播及权限变化后的知识治理。
- 真实三库/模型/UI验收与并发、取消、预算、P95 时延验证。

这些缺口对应设计 P0 的剩余部分及 P1/P2；不能将“已能生成带引用的回答”
等同于“所有 DBA 问题均可诊断并解决”。

## 本地验证入口

两仓分别使用其 Python 环境执行：

```bash
python -m pytest tests/unit/aiops_agent -q
python tests/acceptance/check_aiops_cross_project_alignment.py --peer-root <对端仓库>
python tests/acceptance/check_openapi_contracts.py
```

架构门禁：Ammolite 为 `tests/acceptance/check_architecture_boundaries.py`；
KBot 为 `tests/acceptance/check_4_0_boundaries.py`。
Portal 执行 `npm run build`；KBot 修改页面脚本执行 `node --check`。

新增回归覆盖：严格不可变记录、行/字段/信任校验、原因不随模型 DIRECT 升级、
回答/报告/方案/引用一致、无结构正文不制造事实、人工收口不确认根因、
Run/Target 不匹配拒绝提炼、后续撤回/假设不倒查旧结论，以及指标发现授权、
目标隔离、转义、截断/字节预算、冻结和再规划证据投影。

2026-10-28 实际本地结果：Ammolite 全部 AIOps 单测及文档/诊断页合同共
1,398 项、222 项子测试通过；KBot 对应 AIOps、文档和相关 UI 合同共
1,365 项、147 项子测试通过。两端架构边界、跨项目语义与 OpenAPI 检查通过。
Portal 类型检查和生产构建通过（保留大 chunk 提醒）；KBot 两项 Node 合同及修改
脚本语法检查通过。没有执行真实目标库、真实模型、生产浏览器和性能闭环验收。
