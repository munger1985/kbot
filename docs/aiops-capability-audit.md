# AIOps 能力审计与五阶段闭环对比报告

> 审计日期: 2026-07-16
> 审计范围: 全部 22 个 AIOps 相关 Python 文件

---

## 一、整体架构概览

AIOps 模块共 **22 个 Python 文件**，按层分布：

| 层级 | 文件 | 职责 |
|------|------|------|
| API | `ops_router.py`, `ops_controller.py`, `ops_schema.py` | 6 个 REST 端点 (CRUD 实例 / 流式Chat / Resume / Approve / Cancel) |
| Agent | `ops_agent.py` | SSE 流式入口，视觉搜索预处理，Chat / Resume / Approve 三路 |
| 编排 | `ops_orchestrator.py` | **核心流水线**：资产锁定 → Planner → Safety Gate → SkillRuntime 状态机 → HITL 挂起/恢复/审批/超时检测 |
| 规划 | `ops_planner.py` | RAG 检索运维 SOP + LLM 生成动态 ExecutionPlan，注入 Prometheus 指标清单 + 17 诊断工具清单 |
| 技能 | `db-metric-skill` | Prometheus 优先→专家SQL 兜底 (两阶段) |
| 技能 | `db-analysis-skill` | RCA 引擎 v3：融合多路数据 + HITL 多轮交互 + 数据充分性检查 |
| 技能 | `ops-heal-skill` | 多轮愈合循环：查询→执行→自愈重试 (最多5轮，每轮3次重试) |
| 上下文 | `ops_context.py` | OpsContextMemory TypedDict — 20+ 字段强类型运维总线 |
| 诊断工具 | `diagnostic_tools.py` | **17 个专家 SQL 工具**覆盖 Oracle/PostgreSQL/MySQL 五类场景 |
| 监控 | `prometheus.py`, `zabbix.py`, `registry.py` | Prometheus / Zabbix HTTP 客户端 + 指标注册中心 |
| DB 执行 | `ops.py` | 动态凭证→DB Executor 微服务（只读/变更双轨道） |
| DAO | `ops_agent_conf`, `ops_db_instance`, `ops_pending` | PostgreSQL 表映射 + OracleJSON 快照存储 |

**触发源**: 目前仅支持 `manual`（用户在前端选实例后提问）。`webhook` / `cron` 有枚举但**未实现**。

---

## 二、五阶段闭环逐项对照

### 1️⃣ 感知 (Sense) — ⚠️ 部分实现 (~60%)

**已实现**:
| 能力 | 位置 | 详情 |
|------|------|------|
| Prometheus 指标采集 | `utils/monitor/prometheus.py` | 健康检查、Instant Query、支持指标注册表 |
| Zabbix 指标采集 | `utils/monitor/zabbix.py` | 预留实现，当前返回 `NotImplementedError` |
| DB 诊断 SQL | `diagnostic_tools.py` | 17 个专家工具，三引擎 Oracle/PostgreSQL/MySQL |
| SOP 文档检索 | `ops_planner.py:114-120` | 带 `["ops", "sop"]` 标签的 RAG 检索 |
| 图片搜索 | `ops_agent.py:60-95` | 发送图片触发 visual search |
| UnifiedMetricRegistry | `utils/monitor/registry.py` | 注册中心管理可用指标及其 PromQL |

**缺口**:
- ❌ **告警被动接入 (Alert Ingestion)**：`alert_context` 字段在 `OpsContextMemory` 中已定义但无任何代码往里面写。不能接收 Prometheus AlertManager webhook、Zabbix 告警或云监控回调
- ❌ **日志采集**：`os_log_snapshots` 缓冲区已定义但从未被填充。无 DB alert.log / 系统 OOM 日志 tailing
- ❌ **定时巡检 (Cron-based)**：`trigger_type="cron"` 已枚举但无定时任务框架

---

### 2️⃣ 诊断 (Diagnose) — ✅ 较完善 (~80%)

**已实现**:
| 能力 | 位置 | 详情 |
|------|------|------|
| DBAnalysisSkill v3 RCA 引擎 | `db-analysis-skill/db_analysis_skill_core.py` | 流式输出，`<thought>`/`<answer>` 解析 |
| 数据充分性检查 | `db-analysis-skill/sufficiency_checker.py` | LLM 判断数据是否够用，不够则触发 HITL |
| 17 个专家诊断工具 | `diagnostic_tools.py` | 5 场景：锁/资源/性能/变更/高可用 |
| HITL 多轮交互 | `DBAnalysisSkill v3` | `hitl_history` 追加式 Timeline 构建 |
| 自动诊断补齐 | `DBAnalysisSkill._run_diagnostic_tools()` | 数据不足时自动执行建议工具 (最多3个) |
| 流式思考/回答分离 | `DBAnalysisSkill` | `<thought>`/`</thought>` 标签解析 |
| 多引擎覆盖 | `diagnostic_tools.py` | Oracle / PostgreSQL / MySQL 三引擎专家 SQL |

**缺口**:
- ⚠️ **无日志/链路分析**：`os_log_snapshots` 有字段无数据。不能解析 alert.log 或 OOM 日志
- ⚠️ **无时间序列异常检测**：仅做单点 PromQL 查询，没有历史趋势分析或基线对比
- ⚠️ **无跨实例关联分析**：诊断范围限于单个 `instance_id`，不支持跨实例拓扑分析

---

### 3️⃣ 决策 (Decide) — ⚠️ 部分实现 (~50%)

**已实现**:
| 能力 | 位置 | 详情 |
|------|------|------|
| 动态 LLM 规划 | `ops_planner.py:154-193` | 注入 skills + SOP + 监控指标 + 诊断工具，LLM 生成 ExecutionPlan |
| 安全熔断门禁 | `ops_orchestrator._check_safety_gate()` | Hard block + Approval block + 日频次限制 |
| HITL 审批中断 | `ops_orchestrator` 第 231-277 行 | `REQUIRE_APPROVAL` 包 → 持久化 → 等待审批 |
| 兜底降级计划 | `ops_planner.py:198-221` | LLM 规划失败 → 单步 `db-metric-skill` 安全探测 |
| 多轮上下文改写 | `ops_planner._get_recent_chat_history()` | 从对话历史做指代消解 |
| 拓扑安全注入 | `ops_orchestrator:143-171` | 资产锁定后写安全策略到变量 |

**缺口**:
- ❌ **无条件/分支执行**：`ExecutionPlan.steps` 中的 `condition` 字段已定义但**始终为 `None`**，不能实现 if-then-else 逻辑
- ❌ **无并行步骤**：全部线性串行执行
- ❌ **无结构化 Runbook**：规划输出纯自由文本，没有预定义的故障树或决策树
- ❌ **无风险评分模型**：`risk_level` 硬编码为 `"medium"`，不是基于历史数据或影响范围的自动评估

---

### 4️⃣ 执行 (Execute) — ✅ 较完善 (~75%)

**已实现**:
| 能力 | 位置 | 详情 |
|------|------|------|
| OpsHealSkill 多轮循环 | `ops-heal-skill/ops_heal_skill_core.py` | 最多 5 轮，LLM 决策 query/execute/done |
| 自动重试 + LLM 修正 | `_llm_correct()` | 执行失败时最多 3 次重试，LLM 自动修正 SQL |
| 双轨道 DB 执行 | `OpsDBExecutor` | `read_only` (SELECT) / `mutation` (DDL/DML) |
| 动态凭证管理 | `_dispatch_to_ops_service()` | 实时从 CMDB 拉取连接串和密码 |
| HITL 挂起/恢复 | `_suspend_for_approval()` | 全状态快照 → `kbot_ops_pending_request` 表 → 重建恢复 |
| 超时检测 | `check_pending_timeouts()` | 扫描超时 pending + cron 脚本 |
| 取消挂起 | `ops_controller.cancel_pending()` | 用户可取消 HITL 请求 |
| 审批恢复 | `resume_with_approval()` | 重建上下文从断点继续执行 |
| 变量传递 | `SkillRuntime.create_execution_context()` | `output_var` 机制在步骤间传递数据 |

**缺口**:
- ❌ **无自动回滚**：`rollback_sql` 在 approve 阶段被收集但**从未执行**。治愈失败后不自动 rollback
- ❌ **无执行前校验**：执行变更前不验证数据库连接活性或操作对象是否存在
- ❌ **无执行频次熔断**：`max_daily_execution` 已存储但无计数/熔断逻辑

---

### 5️⃣ 验证 (Verify) — ❌ **几乎缺失 (~10%)**

**已实现**:
- 执行结果被 `final_answer_accumulator` 收集并 persist 到 memory
- `memory_service.persist_and_reflect_memory()` 记录执行轨迹

**缺口**:
- ❌ **无结构化验证步骤** — 执行完变更后，系统**没有**：
  - 重新查询 Prometheus 指标确认指标恢复
  - 对比变更前后的监控快照 (before/after)
  - 执行健康检查 SQL 验证数据库状态
  - 发送 `VERIFICATION_RESULTS` 包给前端
  - 验证失败时自动触发回滚
- ❌ **无回滚执行** — `OpsHealSkill` 收集 `rollback_sql` 但从不执行它
- ❌ **无验证指标** — 没有定义什么是"治愈成功"的衡量标准
- ❌ **无验证阶段 PacketType** — `PacketType` 枚举中没有 `VERIFICATION_RESULTS` 或类似定义

---

## 三、偏差总结

| 阶段 | 完成度 | 关键缺口 |
|------|--------|----------|
| **1. 感知 (Sense)** | ~60% | ❸ 告警被动接入 / 日志采集 / 定时巡检 |
| **2. 诊断 (Diagnose)** | ~80% | ❷ 日志分析 / 异常检测 / 跨实例分析 |
| **3. 决策 (Decide)** | ~50% | ❸ 条件分支 / 并行执行 / Runbook / 风险评分 |
| **4. 执行 (Execute)** | ~75% | ❷ 自动回滚 / 执行前校验 / 频次熔断 |
| **5. 验证 (Verify)** | **~10%** | ❺ 结构化验证闭环完全缺失 |

### 最严重偏差

**验证 (Verify)** 是唯一几乎完全缺失的阶段。当前闭环在第 4 步（执行）结束后直接返回 DONE，没有：
1. 采集"修复后"快照
2. 对比"修复前"基线
3. 判断修复是否成功
4. 失败时执行 rollback

---

## 四、代码文件清单 (全部 22 个)

| 编号 | 文件路径 | 行数 | 核心职责 |
|------|---------|------|---------|
| 1 | `agent/agent/ops_agent.py` | ~210 | SSE 流式入口，Chat/Resume/Approve |
| 2 | `agent/orchestrator/ops_orchestrator.py` | ~1130 | 核心编排：Plan → Execute → HITL → Resume |
| 3 | `agent/planner/ops_planner.py` | ~245 | RAG+LLM 动态规划 |
| 4 | `agent/common/ops_context.py` | ~75 | OpsContextMemory TypedDict |
| 5 | `agent/common/diagnostic_tools.py` | ~1050 | 17 个专家诊断工具 |
| 6 | `skills/skill_libs/ops-heal-skill/__init__.py` | 2 | 技能包入口 |
| 7 | `skills/skill_libs/ops-heal-skill/ops_heal_skill_core.py` | ~295 | 自愈执行引擎 |
| 8 | `skills/skill_libs/db-metric-skill/__init__.py` | 3 | 技能包入口 |
| 9 | `skills/skill_libs/db-metric-skill/db_metric_skill_core.py` | ~455 | 指标采集(双阶段) |
| 10 | `skills/skill_libs/db-analysis-skill/db_analysis_skill_core.py` | ~370 | RCA 引擎 v3 |
| 11 | `skills/skill_libs/db-analysis-skill/sufficiency_checker.py` | ~85 | 数据充分性检查 |
| 12 | `api/controllers/ops_controller.py` | ~150 | REST 控制器 |
| 13 | `api/routers/ops_router.py` | ~150 | REST 路由 |
| 14 | `api/schemas/ops_schema.py` | ~105 | Pydantic 模型 |
| 15 | `utils/clients/ops.py` | ~145 | DB 执行 HTTP 客户端 |
| 16 | `utils/monitor/__init__.py` | ~13 | 监控子系统入口 |
| 17 | `utils/monitor/prometheus.py` | - | Prometheus HTTP 客户端 |
| 18 | `utils/monitor/zabbix.py` | - | Zabbix HTTP 客户端 |
| 19 | `utils/monitor/registry.py` | - | 指标注册中心 |
| 20 | `dao/entities/ops_pending.py` | ~55 | HITL 挂起实体 |
| 21 | `dao/entities/ops_db_instance.py` | ~45 | CMDB 实例实体 |
| 22 | `dao/entities/ops_agent_conf.py` | ~35 | Agent-实例绑定实体 |
| 23 | `kbot_hitl_timeout_check.py` | ~45 | 超时检测 CLI 工具 |
| 24 | `tests/test_ops_agent.py` | ~355 | 集成测试 |

---

## 五、架构图 (数据流)

```
用户/告警
    │
    ▼
[OpsAgent.chat()] ─→ 视觉搜索 (可选)
    │
    ▼
[OpsOrchestrator.execute_ops_stream_pipeline()]
    │
    ├── 1. 资产锁定 (OpsAgentConfService)
    ├── 2. OpsPlanner LLM 规划
    │       ├── RAG 检索 SOP (doc_orchestrator)
    │       └── LLM → ExecutionPlan
    ├── 3. Safety Gate (hard/approval block)
    ├── 4. SkillRuntime 状态机执行
    │       ├── db-metric-skill  (Prometheus → 诊断SQL)
    │       ├── db-analysis-skill (RCA + HITL)
    │       └── ops-heal-skill   (查询→执行→重试)
    └── 5. Memory Persist
         │
    [HITL 断点] ───→ kbot_ops_pending_request
         │                │
         ▼                ▼
    resume_ops_stream_pipeline() / resume_with_approval()
```

> **文件**: `docs/aiops-capability-audit.md`
> **生成工具**: Reasonix 代码审计
