# kbot3 AIOps 五阶段闭环改造计划

> 生成日期: 2026-07-16
> 范围: 感知 → 诊断 → 决策 → 执行 → 验证 全链路
> 目标数据库: Oracle / PostgreSQL / MySQL
> 架构: 纯后端 (无前端)，SSE 流式输出

---

## 当前状态总览

| 阶段 | 完成度 | 关键缺口 |
|------|--------|---------|
| 感知 (Sense) | ~60% | 告警被动接入 / 日志采集 / 定时巡检 |
| 诊断 (Diagnose) | ~80% | 日志分析 / 异常检测 / 跨实例分析 |
| 决策 (Decide) | ~50% | 条件分支 / 并行执行 / 风险评分 / 频次熔断 |
| 执行 (Execute) | ~75% | 自动回滚 / 执行前校验 |
| 验证 (Verify) | **~10%** | 结构化验证闭环完全缺失 |

---

## 一、感知 (Sense): ~60% → ~90%

### 1.1 告警被动接入

**现状**: `ops_context.py:39` 定义了 `alert_context: dict[str, Any] | None`，但 `ops_orchestrator.py:117` 始终初始化为 `None`，没有代码往里面写数据。`ops_agent.py:103` 硬编码 `trigger_type="manual"`。

**改造**:

#### 新增告警 Webhook 端点

`api/routers/ops_router.py` — 新增:

```python
@router.post("/alert-webhook")
async def alert_webhook(payload: AlertWebhookRequest):
    """
    接收 Prometheus AlertManager / Zabbix Action 回调。
    解析告警 → 构建 alert_context → 自动触发 OpsAgent.chat()
    """
```

#### AlertContext 解析器

`agent/common/alert_parser.py` (新文件):

```python
class AlertParser:
    """将异构告警统一为 OpsContextMemory.alert_context"""
    def parse_prometheus_alertmanager(self, payload: dict) -> dict
    def parse_zabbix_action(self, payload: dict) -> dict
```

#### 告警驱动自动诊断

`agent/agent/ops_agent.py` — 新增:

```python
async def chat_from_alert(self, alert_context: dict, instance_id: str):
    """告警驱动的自动诊断入口，trigger_type="webhook\""""
```

### 1.2 日志采集

**现状**: `ops_context.py:64` 定义了 `os_log_snapshots: list[str]`，但 `ops_orchestrator.py:130` 初始化为 `[]` 后从未追加。Prompt (`default_prompt.py:683`) 引用了 `{os_log_snapshots}` 但始终收到空数组。

**改造**:

#### 日志采集器

`agent/common/log_collector.py` (新文件):

```python
class LogCollector:
    """通过 DB Executor 微服务向 target 实例发送诊断 SQL 采集日志。
    target 实例可能是 Oracle / PostgreSQL / MySQL 中的任意一种，
    由 instance 配置中的 db_type 决定，LogCollector 据此派发不同 SQL。
    """

    async def collect_db_logs(
        self, instance_id: str, db_type: str, lines: int = 200
    ) -> list[str]:
        if db_type == "Oracle":
            return await self._collect_oracle_logs(instance_id, lines)
        elif db_type == "PostgreSQL":
            return await self._collect_postgresql_logs(instance_id, lines)
        elif db_type == "MySQL":
            return await self._collect_mysql_logs(instance_id, lines)
        else:
            return []

    async def _collect_oracle_logs(self, instance_id, lines) -> list[str]:
        """✅ V$DIAG_ALERT_EXT (ADR 自动诊断仓库, 11g+ 默认启用)
           SELECT ORIGINATING_TIMESTAMP, MESSAGE_TEXT
           FROM V$DIAG_ALERT_EXT
           ORDER BY ORIGINATING_TIMESTAMP DESC FETCH FIRST :n ROWS ONLY"""

    async def _collect_postgresql_logs(self, instance_id, lines) -> list[str]:
        """⚠️ pg_read_file() 在 systemd 部署下不可用 (日志进 journald 非文件)。
           改为采集替代数据源:
           1. pg_stat_activity: 当前连接/等待事件/活跃查询
           2. pg_stat_statements: 最近慢查询统计 (需扩展)
           3. pg_locks + pg_stat_activity: 锁等待链 (比日志更能定位阻塞根因)
           ⚠️ 如需 error log，需在 target PG 的 postgresql.conf 中开启
           log_destination='csvlog' + 文件路径可读权限，否则无法通过 SQL 获取。"""

    async def _collect_mysql_logs(self, instance_id, lines) -> list[str]:
        """✅ MySQL 5.7+: SELECT * FROM performance_schema.error_log LIMIT :n
           ⚠️ MySQL 5.6: 无 SQL 接口读取 error log，需降级为:
              SHOW ENGINE INNODB STATUS + SELECT * FROM mysql.slow_log (需 slow_query_log=ON)"""
```

#### 诊断流程集成

`agent/orchestrator/ops_orchestrator.py` — 在 `db-analysis-skill` 执行前:

```python
logs = await self.log_collector.collect_db_logs(instance_id, db_type)
ctx["os_log_snapshots"] = logs
```

### 1.3 定时巡检

**现状**: `trigger_type` 支持 `Literal["manual", "webhook", "cron"]` 但 `"cron"` 从未被使用。

**改造**:

扩展 `kbot_hitl_timeout_check.py` → 升级为 `kbot_scheduler.py`:

```python
# 原功能: 扫描超时 pending → 标记超时
# 新增功能: 定时执行预定义巡检 PromQL → 超阈值自动触发诊断
```

调度器使用 `aiops_agent/config.py` 中受校验的产品默认值。运维人员只维护
`configuration/kbot.toml`，不再为 Worker 或 Scheduler 编写独立 TOML。

---

## 二、诊断 (Diagnose): ~80% → ~90%

### 2.1 日志接入诊断分析

诊断 Prompt 已引用 `{os_log_snapshots}`，感知层补齐日志采集后诊断能力自然提升。无需额外代码改动。

### 2.2 时间序列异常检测

**现状**: `db-metric-skill` 仅做单点 PromQL `query_instant`，无历史趋势。

**改造**:

`skills/skill_libs/db-metric-skill/db_metric_skill_core.py` — 增加 range query:

```python
range_result = await monitor_provider.query_range(
    promql, start="-1h", end="now", step="5m"
)
metric_result["trend"] = self._analyze_trend(range_result)
```

### 2.3 跨实例关联分析 (P3)

`agent/common/diagnostic_tools.py` — 新增工具:

```python
async def cross_instance_lock_chain(self, cluster_id: str) -> list[dict]:
    """Oracle RAC: 通过 GV$SESSION 关联分析跨实例锁等待链"""

async def cross_instance_compare(self, cluster_id: str) -> list[dict]:
    """Oracle RAC: 对比 GV$INSTANCE 中各节点负载"""
```

---

## 三、决策 (Decide): ~50% → ~85%

### 3.1 条件分支执行

**现状**: `ops_planner.py:219` 硬编码 `"condition": None`。LLM Prompt 未引导填入 condition。

**改造**:

#### Planner Prompt 更新

`agent/prompt/default_prompt.py` — 更新 `OPS_DIAGNOSE_TASK_PLANNER_PROMPT`:

在 TaskStep 示例中增加 condition 字段:

```
"condition": "if metric_results_1 shows tablespace_usage > 90"
```

#### Orchestrator 条件评估

`agent/orchestrator/ops_orchestrator.py` — SkillRuntime 循环中增加:

```python
if step.get("condition"):
    if not self._evaluate_condition(step["condition"], ctx):
        yield {"type": PacketType.THOUGHT,
               "content": f"⏭️ 跳过步骤 {step['step_id']} (条件不满足)"}
        continue
```

### 3.2 并行步骤执行

**现状**: `ops_orchestrator.py:205` 使用 `for idx, step in enumerate(plan_steps):` 串行循环。

**改造**:

`agent/orchestrator/ops_orchestrator.py`:

```python
# ExecutionPlan 增加 wave 字段
waves = self._group_by_wave(plan_steps)
for wave in waves:
    results = await asyncio.gather(*[
        self._execute_step(step, ctx, session) for step in wave
    ], return_exceptions=True)
```

`agent/common/ops_context.py` — `TaskStep` 增加 `wave: int` 字段。

**kbot3 注意**: Oracle 连接不能跨协程共享，每个并行步骤使用独立 session:

```python
async def _execute_step(self, step, ctx):
    async with get_session() as session:
        # 每个步骤独立 session
```

### 3.3 动态风险评分

**现状**: `ops_heal_skill_core.py:132` 硬编码 `ctx_vars["pending_action_risk_level"] = "medium"`。

**改造**:

`skills/skill_libs/ops-heal-skill/ops_heal_skill_core.py`:

```python
risk_level = decision.get("risk_level", "medium")
if not self._is_valid_risk_level(risk_level):
    risk_level = self._assess_risk(sql, db_type, environment, db_role)
ctx_vars["pending_action_risk_level"] = risk_level
```

风险评估 (Oracle 环境感知):

```python
def _assess_risk(self, sql: str, db_type: str, env: str, role: str) -> str:
    if role in ("standby", "physical_standby"):
        return "low"  # 备库操作天然低风险
    if db_type == "Oracle" and env == "prod" and role == "primary":
        if any(kw in sql.upper() for kw in ["DROP", "TRUNCATE", "ALTER SYSTEM"]):
            return "critical"
    if any(kw in sql.upper() for kw in ["ALTER", "KILL"]):
        return "high" if env == "prod" else "medium"
    return "medium" if env == "prod" else "low"
```

### 3.4 执行频次熔断

**现状**: `max_daily_execution` 在 `ops_orchestrator.py:157` 存入上下文变量，`_check_safety_gate()` 无频次检查。

**改造**:

`agent/orchestrator/ops_orchestrator.py` — `_check_safety_gate()` 新增:

```python
# 4. 日频次熔断
today_count = await self._count_today_mutations(instance_id)
if today_count >= max_daily_execution:
    raise SafetyGateBlocked(
        f"今日变更操作已达上限 ({today_count}/{max_daily_execution})"
    )
```

`dao/repositories/ops_pending_repo.py`:

```python
async def count_today_mutations(self, instance_id: str) -> int:
    """Oracle: SELECT COUNT(*) FROM kbot_ops_pending_request
       WHERE instance_id = :i AND status = 'approved'
       AND TRUNC(created_at) = TRUNC(SYSDATE)"""
```

---

## 四、执行 (Execute): ~75% → ~90%

### 4.1 自动回滚

**现状**: `ops_heal_skill_core.py:124` 读取 `rollback_sql`，行 131 存入变量，行 134 展示给用户，但从未执行。`utils/clients/ops.py` 无 rollback 方法。

**改造**:

#### OpsDBExecutor 新增 rollback

`utils/clients/ops.py`:

```python
async def execute_rollback_ops_sql(
    self, instance_id: str, db_type: str, rollback_sql: str, reason: str
) -> dict:
    """执行回滚 SQL，Oracle 注意事务边界"""
```

**Oracle 特殊处理**:

- `ALTER TABLESPACE ... ADD DATAFILE` 回滚 → `ALTER TABLESPACE ... DROP DATAFILE`
- `ALTER SYSTEM KILL SESSION` → 无需回滚 (会话已终止)
- 回滚前确认 datafile 路径在 ASM/文件系统中存在

#### 验证失败自动回滚

`agent/orchestrator/ops_orchestrator.py` — 验证阶段集成 (见第五节)。

### 4.2 执行前校验

**现状**: LLM 决策后直接调用 `execute_mutation_ops_sql()`。

**改造**:

`skills/skill_libs/ops-heal-skill/ops_heal_skill_core.py`:

```python
async def _preflight(self, sql: str, instance_id: str, db_type: str) -> bool:
    """
    Oracle:
      1. 连接活性: SELECT 1 FROM DUAL
      2. 对象存在性: SELECT COUNT(*) FROM DBA_OBJECTS
         WHERE OWNER = :schema AND OBJECT_NAME = :name
      3. KILL SESSION: SELECT COUNT(*) FROM V$SESSION
         WHERE SID = :sid AND SERIAL# = :serial
    PostgreSQL:
      1. 连接活性: SELECT 1
      2. 对象存在性: SELECT COUNT(*) FROM pg_class WHERE relname = :name
      3. 锁验证: SELECT COUNT(*) FROM pg_locks WHERE pid = :pid
    MySQL:
      1. 连接活性: SELECT 1
      2. 对象存在性: SELECT COUNT(*) FROM information_schema.TABLES
         WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :name
      3. 进程验证: SELECT COUNT(*) FROM information_schema.PROCESSLIST
         WHERE ID = :id
    """
```

---

## 五、验证 (Verify): ~10% → ~85%  🔴 最大缺口

### 5.1 现状分析

`ops_orchestrator.py` 在 SkillRuntime 循环结束后直接发送 `PacketType.DONE` (行 430)。变量 `_monitor_snapshot` (行 286) 和 `_metric_snapshot` (行 287) 仅用于计算步骤间新增数据条数 (变量传递)，不作为 before/after 对比。没有 `PacketType.VERIFICATION_RESULTS`。没有验证步骤。`action_result` (行 383) 只判断 SQL 执行是否报错，不判断修复是否生效。

### 5.2 改造内容

#### 5.2a PacketType 新增

`core/dictionary.py`:

```python
class PacketType(str, Enum):
    # ... 现有类型 ...
    VERIFICATION_RESULTS = "verification_results"  # 自愈验证结果
```

#### 5.2b 新增 OpsVerifier

`agent/common/ops_verifier.py` (新文件):

```python
from dataclasses import dataclass
from enum import Enum

class VerifyStatus(str, Enum):
    VERIFIED = "verified"      # 修复成功: 核心指标恢复至正常范围
    DEGRADED = "degraded"      # 部分修复: 指标改善但未达最佳
    FAILED = "failed"          # 修复失败: 指标无改善或恶化，需回滚

@dataclass
class VerifyResult:
    status: VerifyStatus
    pre_snapshot: dict          # 修复前监控指标 {metric_name: {value, timestamp, promql}}
    post_snapshot: dict         # 修复后监控指标
    health_check_result: dict   # DB 健康检查 {check_name: {ok: bool, detail: str}}
    summary: str                # 自然语言总结

class OpsVerifier:
    """自愈后验证器 — 对比修复前后监控快照 + 执行 DB 健康检查，判断自愈是否成功。

    验证三阶段:
    Phase A: 重新查询 Prometheus/Zabbix (与 pre_snapshot 相同 PromQL)
    Phase B: 执行 DB 健康检查 SQL (连接/锁/资源)
    Phase C: 综合判定 VERIFIED / DEGRADED / FAILED
    """

    METRIC_STABILIZE_SECONDS = 5   # 变更生效等待时间
    RECOVERY_THRESHOLD = 0.5       # 指标改善阈值: 恢复 >= 50% 算 VERIFIED

    async def verify(
        self,
        ctx: OpsContextMemory,
        pre_snapshot: dict[str, dict],
        executed_sql: str,
        rollback_sql: str,
    ) -> VerifyResult:
        import asyncio

        instance_id = ctx["instance_id"]
        db_type = ctx["db_type"]
        monitor_type = ctx["variables"].get("monitor_type", "prometheus")

        # Phase A: 等待变更生效 + 重新查询监控
        await asyncio.sleep(self.METRIC_STABILIZE_SECONDS)

        post_snapshot = {}
        for metric_name, pre_data in pre_snapshot.items():
            promql = pre_data.get("promql", "")
            if promql:
                post_val = await self._query_metric(monitor_type, promql, instance_id)
                post_snapshot[metric_name] = {"value": post_val, "promql": promql}

        # Phase B: DB 健康检查
        health = await self._health_check(instance_id, db_type)

        # Phase C: 综合判定
        status = self._compare_snapshots(pre_snapshot, post_snapshot)
        summary = self._build_summary(status, pre_snapshot, post_snapshot, health)

        return VerifyResult(
            status=status,
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            health_check_result=health,
            summary=summary,
        )

    async def _query_metric(self, monitor_type: str, promql: str, instance_id: str) -> float | None:
        """查询单个监控指标当前值"""
        from utils.monitor import get_monitor_provider
        provider = get_monitor_provider(monitor_type)
        result = await provider.query_instant(promql)
        return self._extract_value(result)

    async def _health_check(self, instance_id: str, db_type: str) -> dict:
        """执行数据库健康检查 — 3 引擎自适应"""
        checks = {}
        for check_name, sql in self._get_health_check_sqls(db_type).items():
            try:
                result = await self._execute_readonly(instance_id, sql)
                checks[check_name] = {"ok": True, "detail": str(result)[:500]}
            except Exception as e:
                checks[check_name] = {"ok": False, "detail": str(e)[:500]}
        return checks

    def _get_health_check_sqls(self, db_type: str) -> dict[str, str]:
        """返回各引擎的健康检查 SQL 映射"""
        if db_type == "Oracle":
            return {
                "connection":      "SELECT 1 FROM DUAL",
                "lock_blocking":   "SELECT COUNT(*) FROM V$LOCK WHERE BLOCK=1",
                "active_sessions": "SELECT COUNT(*) FROM V$SESSION WHERE STATUS='ACTIVE'",
                "tablespace":      "SELECT TABLESPACE_NAME, ROUND(100-100*FREE_SPACE/TOTAL_SPACE,1) AS PCT_USED FROM (SELECT ...)",
                "archive_dest":    "SELECT COUNT(*) FROM V$ARCHIVE_DEST_STATUS WHERE STATUS!='VALID'",
            }
        elif db_type == "PostgreSQL":
            return {
                "connection":      "SELECT 1",
                "lock_wait":       "SELECT COUNT(*) FROM pg_locks WHERE NOT GRANTED",
                "active_queries":  "SELECT COUNT(*) FROM pg_stat_activity WHERE state='active' AND wait_event IS NOT NULL",
                "idle_in_trans":   "SELECT COUNT(*) FROM pg_stat_activity WHERE state='idle in transaction'",
                "replication_lag": "SELECT pg_is_in_recovery(), pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()",
            }
        elif db_type == "MySQL":
            return {
                "connection":      "SELECT 1",
                "innodb_trx":      "SELECT COUNT(*) FROM information_schema.INNODB_TRX WHERE trx_state='RUNNING'",
                "active_process":  "SELECT COUNT(*) FROM information_schema.PROCESSLIST WHERE COMMAND!='Sleep'",
                "replication":     "SHOW SLAVE STATUS",
            }
        return {}

    def _compare_snapshots(self, pre: dict, post: dict) -> VerifyStatus:
        """
        对比修复前后快照，判定恢复等级:
        - VERIFIED: 所有异常指标恢复到健康范围 (改善 >= 50%)
        - DEGRADED: 部分指标改善但未完全恢复 (改善 >= 20% 但 < 50%)
        - FAILED: 指标无改善或恶化
        """
        if not pre or not post:
            return VerifyStatus.FAILED
        recovered = degraded = failed = 0
        for metric_name, pre_data in pre.items():
            pre_val = pre_data.get("value")
            post_val = post.get(metric_name, {}).get("value")
            if pre_val is None or post_val is None:
                failed += 1; continue
            if pre_val == 0:
                recovered += 1; continue
            change = abs(post_val - pre_val) / abs(pre_val)
            if change >= self.RECOVERY_THRESHOLD:      recovered += 1
            elif change >= self.RECOVERY_THRESHOLD*0.4: degraded += 1
            else:                                        failed += 1
        if (total := recovered + degraded + failed) == 0:
            return VerifyStatus.FAILED
        if failed > 0:                      return VerifyStatus.FAILED
        if degraded > 0 or recovered < total: return VerifyStatus.DEGRADED
        return VerifyStatus.VERIFIED

    def _build_summary(self, status, pre, post, health) -> str:
        items = [f"  {n}: {d.get('value','?')} → {post.get(n,{}).get('value','?')}"
                 for n, d in pre.items()]
        failed = [k for k, v in health.items() if not v["ok"]]
        health_line = "健康检查全部通过" if not failed else f"异常: {failed}"
        return f"验证状态: {status.value}\n指标变化:\n" + "\n".join(items) + f"\n{health_line}"
```

#### 5.2c 编排器集成

`agent/orchestrator/ops_orchestrator.py` — 核心修改: 将原来 `SkillRuntime循环 → DONE` 改为 `SkillRuntime循环 → Verify → DONE`。

**Pre-snapshot 采集** (在 OpsHealSkill 的 mutation 执行前):

```python
# 在 _check_safety_gate 通过后、实际调用 OpsHealSkill 前:
if skill_name == "ops-heal-skill":
    related_metrics = ctx["variables"].get("related_metrics", [])
    pre_snapshot = {}
    for metric_def in related_metrics:
        promql = self._render_promql(metric_def, ctx["variables"])
        val = await self._query_monitor_instant(ctx, promql)
        pre_snapshot[metric_def["name"]] = {"value": val, "promql": promql}
    ctx["variables"]["_pre_snapshot"] = pre_snapshot
```

**验证阶段插入** (替换当前行 382-430 的 `闭环落库 → DONE`):

```python
        # --- 4. 验证阶段 (Verify) ---
        pre_snapshot = ctx["variables"].get("_pre_snapshot", {})
        pending_sql = ctx["variables"].get("pending_action_sql", "")
        pending_rollback = ctx["variables"].get("pending_action_rollback", "")

        if pending_sql and pre_snapshot:
            yield {"type": PacketType.THOUGHT, "content": "🔍 开始验证自愈效果..."}

            verifier = OpsVerifier()
            verify_result = await verifier.verify(
                ctx=ctx,
                pre_snapshot=pre_snapshot,
                executed_sql=pending_sql,
                rollback_sql=pending_rollback,
            )

            yield {
                "type": PacketType.VERIFICATION_RESULTS,
                "content": {
                    "status": verify_result.status.value,
                    "pre_snapshot": verify_result.pre_snapshot,
                    "post_snapshot": verify_result.post_snapshot,
                    "health_check": verify_result.health_check_result,
                    "summary": verify_result.summary,
                },
            }

            if verify_result.status == VerifyStatus.FAILED:
                if pending_rollback:
                    yield {"type": PacketType.WARNING,
                           "content": f"❌ 验证失败，执行自动回滚..."}
                    rollback_result = await self.db_executor.execute_rollback_ops_sql(
                        instance_id=ctx["instance_id"],
                        db_type=ctx["db_type"],
                        rollback_sql=pending_rollback,
                        reason=f"自愈验证失败: {verify_result.summary}",
                    )
                    final_answer_accumulator = (
                        f"❌ 自愈失败，已自动回滚。\n"
                        f"原因: {verify_result.summary}\n"
                        f"回滚结果: {rollback_result}"
                    )
            elif verify_result.status == VerifyStatus.DEGRADED:
                yield {"type": PacketType.WARNING,
                       "content": "⚠️ 部分指标已恢复但未达预期，建议人工检查"}
        # (纯诊断场景免验证，跳过)

        # --- 5. 闭环落库 (原步骤 4，不变) ---
```

**执行流程对比**:

```
改造前: SkillRuntime → action_result 检查 → memory_persist → DONE
改造后: SkillRuntime → pre_snapshot 采集 → OpsHealSkill(mutation) → wait 5s
         → post_snapshot 采集 → health_check → _compare_snapshots
         → VERIFIED → memory_persist → DONE
         → FAILED   → rollback → memory_persist → ERROR
        ctx=ctx,
        pre_snapshot=pre_snapshot,
        executed_sql=ctx["variables"]["pending_action_sql"],
        rollback_sql=ctx["variables"].get("pending_action_rollback", ""),
    )

    yield {
        "type": PacketType.VERIFICATION_RESULTS,
        "content": {
            "status": verify_result.status.value,
            "pre_snapshot": verify_result.pre_snapshot,
            "post_snapshot": verify_result.post_snapshot,
            "health_check": verify_result.health_check_result,
            "summary": verify_result.summary,
        }
    }

    if verify_result.status == VerifyStatus.FAILED:
        rollback_sql = ctx["variables"].get("pending_action_rollback")
        if rollback_sql:
            yield {"type": PacketType.WARNING, "content": "验证失败，执行自动回滚..."}
            await self._execute_rollback(ctx, rollback_sql)
    elif verify_result.status == VerifyStatus.DEGRADED:
        yield {"type": PacketType.WARNING,
               "content": "⚠️ 部分指标已恢复，未达预期水平，建议人工检查"}
```

#### 5.2d Pre-snapshot 采集

在执行 mutation 之前采集基线:

```python
# 在 OpsHealSkill 执行前采集监控快照
pre_snapshot = {}
for metric_name in ctx["variables"].get("related_metrics", []):
    promql = render_promql(metric_name, ctx["variables"])
    pre_snapshot[metric_name] = await monitor_provider.query_instant(promql)
```

### 5.3 执行报告生成与持久化

验证完成后，生成一份专业的执行报告，告知用户做了什么操作、效果如何，并持久化到数据库供后续审计和分析。kbot3 无前端，报告通过 SSE 的 `answer` 包以 Markdown 格式输出。

#### 5.3a 报告内容结构

```python
@dataclass
class OpsExecutionReport:
    """AIOps 执行报告 — 自愈操作完成后生成"""
    entry_id: str                        # 关联 chat entry
    session_id: str                      # 会话 ID
    user_id: str | None                  # 操作用户
    agent_id: str                        # 运维 Agent ID

    instance_id: str                     # 数据库实例 ID
    instance_name: str                   # 实例名称 (展示用)
    db_type: str                         # Oracle / PostgreSQL / MySQL
    environment: str                     # prod / staging / dev

    trigger_type: str                    # manual / webhook / cron
    original_question: str               # 用户原始问题或告警摘要
    diagnosis_summary: str               # LLM 诊断结论摘要

    actions_executed: list[dict]         # [{sql, impact, risk_level, context}]

    pre_snapshot: dict | None            # 修复前监控快照
    post_snapshot: dict | None           # 修复后监控快照
    verification_status: str             # verified / degraded / failed / skipped
    health_check_result: dict | None     # DB 健康检查详情

    rollback_info: dict | None           # {rollback_sql, executed: bool, result}

    recommendations: str                 # 后续建议 (LLM 生成)

    total_duration_seconds: float        # 从诊断到验证的总耗时
    created_at: str                      # ISO 时间戳
```

#### 5.3b 新增数据库表 (kbot3 — Oracle 23ai)

DDL 文件: `docs/database/kbot_db_change_ddl_ops_report.sql`

```sql
CREATE TABLE kbot.ops_execution_report (
    id                VARCHAR2(64) DEFAULT SYS_GUID() PRIMARY KEY,
    entry_id          VARCHAR2(64)  NOT NULL,
    session_id        VARCHAR2(64)  NOT NULL,
    user_id           VARCHAR2(64),
    agent_id          VARCHAR2(64)  NOT NULL,
    instance_id       VARCHAR2(64)  NOT NULL,
    instance_name     VARCHAR2(256) NOT NULL,
    db_type           VARCHAR2(32)  NOT NULL,
    environment       VARCHAR2(32)  DEFAULT 'prod' NOT NULL,
    trigger_type      VARCHAR2(32)  DEFAULT 'manual' NOT NULL,
    original_question CLOB          DEFAULT EMPTY_CLOB(),
    diagnosis_summary CLOB          DEFAULT EMPTY_CLOB(),

    -- JSON 字段 (Oracle 23ai JSON 类型, 支持 JSON 查询索引)
    actions_executed     JSON DEFAULT '[]' NOT NULL,
    pre_snapshot         JSON,
    post_snapshot        JSON,
    health_check_result  JSON,
    rollback_info        JSON,

    verification_status  VARCHAR2(32)  DEFAULT 'skipped' NOT NULL,
    report_content       CLOB          DEFAULT EMPTY_CLOB(),
    recommendations      CLOB          DEFAULT EMPTY_CLOB(),

    total_duration_seconds NUMBER      DEFAULT 0 NOT NULL,
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,

    CONSTRAINT fk_ops_report_entry
        FOREIGN KEY (entry_id) REFERENCES kbot.chat_history(entry_id)
);

-- 索引 (Oracle Text 可选, 此处用普通 B-tree 即可)
CREATE INDEX idx_ops_rep_entry    ON kbot.ops_execution_report(entry_id);
CREATE INDEX idx_ops_rep_instance ON kbot.ops_execution_report(instance_id);
CREATE INDEX idx_ops_rep_created  ON kbot.ops_execution_report(created_at DESC);
CREATE INDEX idx_ops_rep_status   ON kbot.ops_execution_report(verification_status);

-- Oracle JSON 查询索引 (加速 JSON 字段内查询)
CREATE SEARCH INDEX idx_ops_rep_json ON kbot.ops_execution_report(actions_executed)
    FOR JSON;

COMMENT ON TABLE kbot.ops_execution_report IS 'AIOps 自愈执行报告';
COMMENT ON COLUMN kbot.ops_execution_report.verification_status IS 'verified/degraded/failed/skipped';
COMMENT ON COLUMN kbot.ops_execution_report.report_content IS 'Markdown 格式完整报告';
```

#### 5.3c 报告生成器

`agent/common/ops_reporter.py` (新文件):

```python
class OpsReporter:
    """AIOps 执行报告生成器 — 将诊断+执行+验证结果整合为专业报告"""

    async def generate_report(
        self,
        ctx: OpsContextMemory,
        verify_result: VerifyResult | None,
        actions: list[dict],
        total_duration: float,
    ) -> str:
        """
        生成 Markdown 格式的专业运维报告。

        报告模板:
        # 数据库自愈执行报告

        ## 1. 概览
        | 项目 | 详情 |
        |------|------|
        | 目标实例 | {instance_name} ({db_type}) — {environment} |
        | 触发方式 | {trigger_type} |
        | 执行耗时 | {duration}s |
        | 验证状态 | {status_icon} {verification_status} |
        | 报告时间 | {created_at} |

        ## 2. 问题诊断
        > {original_question}
        {diagnosis_summary}

        ## 3. 执行动作
        | # | SQL | 影响 | 风险等级 |
        |---|-----|------|---------|
        {actions_table}

        ## 4. 效果验证
        ### 4.1 指标变化
        | 指标 | 修复前 | 修复后 | 变化幅度 | 判定 |
        |------|--------|--------|---------|------|
        {metrics_table}

        ### 4.2 健康检查
        | 检查项 | 状态 | 详情 |
        |--------|------|------|
        {health_table}

        ## 5. 回滚信息
        {rollback_section}  (如有)

        ## 6. 后续建议
        {recommendations}

        ---
        *本报告由 Nexus AIOps 自动生成*
        """

    async def generate_recommendations(
        self, verify_result: VerifyResult, actions: list[dict], db_type: str
    ) -> str:
        """调用 LLM 基于验证结果和数据库类型生成后续优化建议

        Prompt 要点:
        - VERIFIED:  建议定期巡检 + 预防性配置
        - DEGRADED:  建议人工复核 + 进一步诊断方向
        - FAILED:    建议升级处理 + 手动介入步骤
        - Oracle:    AWR 报告分析 / SQL Profile 绑定
        - PostgreSQL: VACUUM / ANALYZE / 索引优化
        - MySQL:      InnoDB 调优 / 查询缓存
        """
```

#### 5.3d DAO 层

`dao/entities/ops_execution_report.py` (新文件):

```python
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.types import CLOB
from dao.entities.base import Base

class OpsExecutionReportEntity(Base):
    __tablename__ = "ops_execution_report"
    __table_args__ = {"schema": "kbot"}

    id = Column(String(64), primary_key=True, default=text("SYS_GUID()"))
    entry_id = Column(String(64), ForeignKey("kbot.chat_history.entry_id"), nullable=False)
    session_id = Column(String(64), nullable=False)
    user_id = Column(String(64))
    agent_id = Column(String(64), nullable=False)
    instance_id = Column(String(64), nullable=False)
    instance_name = Column(String(256), nullable=False, default="")
    db_type = Column(String(32), nullable=False)
    environment = Column(String(32), nullable=False, default="prod")
    trigger_type = Column(String(32), nullable=False, default="manual")
    original_question = Column(CLOB, nullable=False, default="")
    diagnosis_summary = Column(CLOB, nullable=False, default="")

    # Oracle JSON 列 (23ai 原生支持)
    actions_executed = Column(Text, nullable=False, default="[]")
    pre_snapshot = Column(Text)
    post_snapshot = Column(Text)
    health_check_result = Column(Text)
    rollback_info = Column(Text)

    verification_status = Column(String(32), nullable=False, default="skipped")
    report_content = Column(CLOB, nullable=False, default="")
    recommendations = Column(CLOB, nullable=False, default="")

    total_duration_seconds = Column(Float, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("SYSTIMESTAMP"))
```

`dao/repositories/ops_execution_report_repo.py` (新文件):

```python
class OpsExecutionReportRepository:
    async def create(self, report: dict) -> OpsExecutionReportEntity: ...
    async def get_by_entry_id(self, entry_id: str) -> OpsExecutionReportEntity | None: ...
    async def list_by_instance(self, instance_id: str, limit: int = 20): ...
    async def get_recent_by_days(self, instance_id: str, days: int = 7): ...
```

#### 5.3e 编排器集成

`agent/orchestrator/ops_orchestrator.py` — 在验证完成后、memory_persist 前:

```python
        # --- 4b. 生成执行报告 ---
        reporter = OpsReporter()
        report_md = await reporter.generate_report(
            ctx=ctx,
            verify_result=verify_result if pending_sql else None,
            actions=executed_actions,
            total_duration=(time.time() - start_time),
        )
        recommendations = await reporter.generate_recommendations(
            verify_result=verify_result, actions=executed_actions, db_type=ctx["db_type"],
        ) if verify_result else ""

        # 持久化报告到 Oracle
        await self.report_repo.create({
            "entry_id": entry_id,
            "session_id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "instance_id": ctx["instance_id"],
            "instance_name": ctx["variables"].get("instance_name", ""),
            "db_type": ctx["db_type"],
            "environment": ctx["variables"].get("environment", "prod"),
            "trigger_type": trigger_type,
            "original_question": question,
            "diagnosis_summary": ctx.get("diagnosis_summary", ""),
            "actions_executed": json.dumps(executed_actions),
            "pre_snapshot": json.dumps(verify_result.pre_snapshot) if verify_result else None,
            "post_snapshot": json.dumps(verify_result.post_snapshot) if verify_result else None,
            "verification_status": verify_result.status.value if verify_result else "skipped",
            "health_check_result": json.dumps(verify_result.health_check_result) if verify_result else None,
            "rollback_info": json.dumps(rollback_info) if rollback_info else None,
            "report_content": report_md,
            "recommendations": recommendations,
            "total_duration_seconds": total_duration,
        })

        # SSE 推送 Markdown 报告
        yield {"type": PacketType.ANSWER, "content": report_md}
```

#### 5.3f 报告查询 API

`api/routers/ops_router.py` — 新增:

```python
@router.get("/reports/{entry_id}")
async def get_execution_report(entry_id: str):
    """查询单次自愈执行报告 (返回 Markdown)"""

@router.get("/instances/{instance_id}/reports")
async def list_execution_reports(instance_id: str, days: int = 7):
    """查询某实例近 N 天的执行报告列表"""
```

### 5.4 SSE 输出格式

kbot3 无前端，验证结果通过 SSE 流输出。客户端 (curl/Postman/SDK) 消费 `verification_results` 包:

```json
{
  "type": "verification_results",
  "content": {
    "status": "verified",
    "pre_snapshot": {
      "tablespace_usage_pct": 94.2,
      "active_sessions": 45
    },
    "post_snapshot": {
      "tablespace_usage_pct": 63.8,
      "active_sessions": 12
    },
    "health_check": {
      "connection_ok": true,
      "no_lock_wait": true,
      "archive_log_ok": true
    },
    "summary": "表空间使用率从 94.2% 降至 63.8%，恢复健康水平。"
  }
}
```

---

## 六、实施路线图

| 阶段 | 内容 | 工期 | 依赖 |
|------|------|------|------|
| **M1** | 验证闭环 + 自动回滚 | 1-2 周 | 无 |
| **M2** | 告警 Webhook + 日志采集 | 2-3 周 | M1 |
| **M3** | 条件分支 + 风险评分 + 频次熔断 | 2-3 周 | M2 |
| **M4** | 并行执行 + 跨实例分析 + Cron 巡检 | 3-4 周 | M3 |

### 每个里程碑的验收标准

**M1 — 验证闭环**:
- `PacketType.VERIFICATION_RESULTS` 可被 SSE 客户端消费
- 变更执行后自动采集 post-snapshot，对比 pre-snapshot
- FAILED 状态自动执行 rollback_sql (Oracle 特别注意 datafile 回滚路径)
- 3 种数据库健康检查 SQL 均可用:
  - Oracle: `SELECT 1 FROM DUAL` + `V$LOCK` + `V$SESSION` + `DBA_DATA_FILES`
  - PostgreSQL: `SELECT 1` + `pg_locks` + `pg_stat_activity`
  - MySQL: `SELECT 1` + `INNODB_TRX` + `PROCESSLIST`

**M2 — 感知补齐**:
- AlertManager webhook → 自动解析 → 触发 OpsAgent.chat_from_alert()
- 诊断前自动采集数据库日志填入 `os_log_snapshots`
- `kbot_scheduler.py` 可独立运行巡检

**M3 — 决策安全**:
- 条件分支可跳过不满足条件的步骤
- 风险等级不再硬编码为 "medium"; Oracle RAC primary + prod 自动升级
- 日变更频次超限被熔断拒绝 (Oracle `TRUNC(SYSDATE)` 日期比较)

**M4 — 高级能力**:
- 同 wave 步骤并行执行 (每个步骤独立 Oracle session)
- 跨实例 RAC 关联分析可用 (`GV$` 视图)
- kbot_scheduler 定时巡检自动发现异常

---

## 七、涉及文件清单

| 文件 | 改动类型 | 所属阶段 |
|------|---------|---------|
| `core/dictionary.py` | 修改: 新增 PacketType.VERIFICATION_RESULTS | Verify |
| `agent/common/ops_verifier.py` | **新增** | Verify |
| `agent/common/ops_reporter.py` | **新增** | Verify |
| `agent/common/ops_context.py` | 修改: TaskStep 增加 wave 字段 | Decide |
| `agent/common/alert_parser.py` | **新增** | Sense |
| `agent/common/log_collector.py` | **新增** | Sense |
| `agent/common/diagnostic_tools.py` | 修改: 新增跨实例分析 + Oracle RAC 工具 | Diagnose |
| `agent/orchestrator/ops_orchestrator.py` | 修改: 验证阶段 + 回滚 + 报告生成 + 条件 + 并行 + 频次熔断 | Verify/Decide/Execute |
| `agent/agent/ops_agent.py` | 修改: 新增 chat_from_alert | Sense |
| `agent/planner/ops_planner.py` | 修改: Prompt 引导 condition/wave | Decide |
| `api/routers/ops_router.py` | 修改: 新增 alert-webhook + report 查询端点 | Sense/Verify |
| `api/schemas/ops_schema.py` | 修改: 新增 AlertWebhookRequest | Sense |
| `skills/skill_libs/ops-heal-skill/ops_heal_skill_core.py` | 修改: 风险评分 + 执行前校验 | Decide/Execute |
| `skills/skill_libs/db-metric-skill/db_metric_skill_core.py` | 修改: 增加 range query 趋势分析 | Diagnose |
| `utils/clients/ops.py` | 修改: 新增 execute_rollback_ops_sql | Execute |
| `dao/entities/ops_execution_report.py` | **新增** | Verify |
| `dao/repositories/ops_execution_report_repo.py` | **新增** | Verify |
| `dao/repositories/ops_pending_repo.py` | 修改: 新增 count_today_mutations | Decide |
| `kbot_hitl_timeout_check.py` → `kbot_scheduler.py` | 重构: 扩展为巡检 + 超时检测 | Sense |
| `aiops_agent/config.py` | 修改：新增受校验的 Scheduler 产品默认值 | Sense |
| `docs/database/kbot_db_change_ddl_ops_report.sql` | **新增** DDL | Verify |
| `agent/prompt/default_prompt.py` | 修改: Planner Prompt 增加 condition/wave 示例 | Decide |
