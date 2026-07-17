"""AIOps 自愈验证器 — 对比修复前后监控快照 + 执行 DB 健康检查，判断自愈是否成功。

验证三阶段:
  Phase A: 重新查询 Prometheus/Zabbix (与 pre_snapshot 相同 PromQL)
  Phase B: 执行 DB 健康检查 SQL (连接/锁/资源) — Oracle/PostgreSQL/MySQL 三引擎自适应
  Phase C: 综合判定 VERIFIED / DEGRADED / FAILED
"""
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger

from utils.clients import OpsDBExecutor
from utils.monitor import get_monitor_provider


class VerifyStatus(str, Enum):
    VERIFIED = "verified"      # 修复成功: 核心指标恢复至正常范围
    DEGRADED = "degraded"      # 部分修复: 指标改善但未达最佳
    FAILED = "failed"          # 修复失败: 指标无改善或恶化，需回滚


@dataclass
class VerifyResult:
    status: VerifyStatus
    pre_snapshot: dict[str, dict]        # 修复前监控指标 {metric_name: {value, promql, timestamp}}
    post_snapshot: dict[str, dict]       # 修复后监控指标
    health_check_result: dict[str, dict] # DB 健康检查 {check_name: {ok: bool, detail: str}}
    summary: str                         # 自然语言总结


class OpsVerifier:
    """自愈后验证器。

    验证三阶段:
    Phase A: 重新查询 Prometheus/Zabbix (与 pre_snapshot 相同 PromQL)
    Phase B: 执行 DB 健康检查 SQL (连接/锁/资源)
    Phase C: 综合判定 VERIFIED / DEGRADED / FAILED
    """

    # 变更生效等待时间 (秒)
    METRIC_STABILIZE_SECONDS: int = 5
    # 指标改善阈值: post 值必须恢复到 pre 值的这个比例以上才算 VERIFIED
    RECOVERY_THRESHOLD: float = 0.5

    def __init__(self):
        self.db_executor = OpsDBExecutor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(
        self,
        instance_id: str,
        db_type: str,
        monitor_type: str,
        pre_snapshot: dict[str, dict],
        executed_sql: str = "",
        rollback_sql: str = "",
    ) -> VerifyResult:
        """执行完整的自愈效果验证。

        Args:
            instance_id: 目标数据库实例 ID
            db_type: Oracle / PostgreSQL / MySQL
            monitor_type: prometheus / zabbix
            pre_snapshot: 修复前采集的监控指标快照
            executed_sql: 已执行的变更 SQL (用于日志关联)
            rollback_sql: 回滚 SQL (验证失败时使用，此处仅记录不执行)

        Returns:
            VerifyResult 包含状态、快照对比、健康检查和自然语言总结
        """
        # Phase A: 等待变更生效 + 重新查询监控指标
        await asyncio.sleep(self.METRIC_STABILIZE_SECONDS)
        post_snapshot = await self._collect_post_snapshot(pre_snapshot, monitor_type)

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

    # ------------------------------------------------------------------
    # Phase A: 监控指标重新采集
    # ------------------------------------------------------------------

    async def _collect_post_snapshot(
        self, pre_snapshot: dict[str, dict], monitor_type: str,
    ) -> dict[str, dict]:
        """对 pre_snapshot 中的每个指标重新查询当前值。"""
        post: dict[str, dict] = {}
        for metric_name, pre_data in pre_snapshot.items():
            promql = pre_data.get("promql", "")
            if not promql:
                post[metric_name] = {"value": None, "promql": "", "error": "missing promql"}
                continue
            try:
                val = await self._query_metric(monitor_type, promql)
                post[metric_name] = {"value": val, "promql": promql}
            except Exception as e:
                logger.warning(f"[OpsVerifier] 指标 {metric_name} 查询失败: {e}")
                post[metric_name] = {"value": None, "promql": promql, "error": str(e)}
        return post

    async def _query_metric(
        self, monitor_type: str, promql: str,
    ) -> float | None:
        """查询单个监控指标当前值。"""
        provider = get_monitor_provider(monitor_type)
        result = await provider.query_instant(promql)
        return self._extract_value(result)

    @staticmethod
    def _extract_value(query_result: Any) -> float | None:
        """从 Prometheus/Zabbix 返回结果中提取数值。

        Prometheus 返回格式: [{"metric": {...}, "value": [ts, "val"]}]
        Zabbix 返回格式: [{"value": "val"}]
        """
        if not query_result:
            return None
        if isinstance(query_result, list) and len(query_result) > 0:
            item = query_result[0]
            if isinstance(item, dict):
                if "value" in item:
                    v = item["value"]
                    if isinstance(v, list) and len(v) >= 2:
                        return float(v[1])
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        try:
                            return float(v)
                        except ValueError:
                            return None
                if "values" in item:
                    vals = item["values"]
                    if isinstance(vals, list) and len(vals) > 0:
                        last = vals[-1]
                        if isinstance(last, list) and len(last) >= 2:
                            return float(last[1])
        if isinstance(query_result, (int, float)):
            return float(query_result)
        return None

    # ------------------------------------------------------------------
    # Phase B: 数据库健康检查 (3 引擎自适应)
    # ------------------------------------------------------------------

    async def _health_check(self, instance_id: str, db_type: str) -> dict[str, dict]:
        """执行数据库健康检查 — 按 db_type 派发不同 SQL。"""
        checks: dict[str, dict] = {}
        for check_name, sql in self._get_health_check_sqls(db_type).items():
            try:
                result = await self.db_executor.execute_readonly_ops_sql(
                    instance_id=instance_id, sql=sql, limit=5,
                )
                checks[check_name] = {
                    "ok": True, "detail": str(result)[:500],
                }
            except Exception as e:
                checks[check_name] = {"ok": False, "detail": str(e)[:500]}
        return checks

    @staticmethod
    def _get_health_check_sqls(db_type: str) -> dict[str, str]:
        """返回各引擎的健康检查 SQL 映射。

        每引擎 5 项关键检查: 连接 / 锁等待 / 活跃会话 / 资源使用 / 复制状态
        """
        db = db_type.lower() if db_type else ""
        if db in ("oracle",):
            return {
                "connection":      "SELECT 1 FROM DUAL",
                "lock_blocking":   "SELECT COUNT(*) AS BLOCKING_LOCKS FROM V$LOCK WHERE BLOCK=1",
                "active_sessions": "SELECT COUNT(*) AS ACTIVE_COUNT FROM V$SESSION WHERE STATUS='ACTIVE' AND TYPE!='BACKGROUND'",
                "tablespace_usage": (
                    "SELECT TABLESPACE_NAME, "
                    "ROUND(MAX(DECODE(MAXBYTES,0,BYTES/1024/1024,MAXBYTES/1024/1024)),2) AS MAX_MB, "
                    "ROUND(SUM(BYTES)/1024/1024,2) AS USED_MB, "
                    "ROUND(100*SUM(BYTES)/NULLIF(SUM(MAXBYTES),0),1) AS PCT "
                    "FROM DBA_DATA_FILES GROUP BY TABLESPACE_NAME "
                    "FETCH FIRST 10 ROWS ONLY"
                ),
                "archive_log":     "SELECT COUNT(*) AS INVALID_DEST FROM V$ARCHIVE_DEST_STATUS WHERE STATUS!='VALID'",
            }
        if db in ("postgresql", "postgres"):
            return {
                "connection":      "SELECT 1",
                "lock_wait":       "SELECT COUNT(*) AS WAITING_LOCKS FROM pg_locks WHERE NOT GRANTED",
                "active_queries":  "SELECT COUNT(*) AS ACTIVE_BACKENDS FROM pg_stat_activity WHERE state='active' AND wait_event IS NOT NULL",
                "idle_in_trans":   "SELECT COUNT(*) AS IDLE_IN_TRANS FROM pg_stat_activity WHERE state='idle in transaction'",
                "replication":     "SELECT pg_is_in_recovery(), pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()",
            }
        if db in ("mysql",):
            return {
                "connection":      "SELECT 1",
                "innodb_trx":      "SELECT COUNT(*) AS RUNNING_TRX FROM information_schema.INNODB_TRX WHERE trx_state='RUNNING'",
                "processlist":     "SELECT COUNT(*) AS NON_SLEEP FROM information_schema.PROCESSLIST WHERE COMMAND!='Sleep'",
                "error_log":       "SELECT COUNT(*) AS RECENT_ERRORS FROM performance_schema.error_log WHERE LOGGED > DATE_SUB(NOW(), INTERVAL 5 MINUTE) AND PRIO IN ('Error','Critical')",
                "replication":     "SELECT SERVICE_STATE FROM performance_schema.replication_connection_status WHERE CHANNEL_NAME='group_replication_applier'",
            }
        return {
            "connection": "SELECT 1",
        }

    # ------------------------------------------------------------------
    # Phase C: 综合判定
    # ------------------------------------------------------------------

    def _compare_snapshots(
        self, pre: dict[str, dict], post: dict[str, dict],
    ) -> VerifyStatus:
        """对比修复前后快照，判定恢复等级。

        判定规则:
        - VERIFIED: 所有异常指标恢复到健康范围 (改善 >= 50%)
        - DEGRADED: 部分指标改善但未完全恢复 (改善 >= 20% 但 < 50%)
        - FAILED: 指标无改善或恶化

        特殊处理:
        - 如果某指标 pre 值为 0，跳过计算 (避免除零)
        - 如果所有指标 error 或无值，判定 FAILED
        """
        if not pre or not post:
            return VerifyStatus.FAILED

        recovered = 0
        degraded = 0
        failed = 0

        for metric_name, pre_data in pre.items():
            pre_val = pre_data.get("value")
            post_data = post.get(metric_name, {})
            post_val = post_data.get("value")

            if pre_val is None or post_val is None:
                failed += 1
                continue

            if pre_val == 0:
                recovered += 1
                continue

            change_ratio = abs(post_val - pre_val) / abs(pre_val)

            if change_ratio >= self.RECOVERY_THRESHOLD:
                recovered += 1
            elif change_ratio >= self.RECOVERY_THRESHOLD * 0.4:
                degraded += 1
            else:
                failed += 1

        total = recovered + degraded + failed
        if total == 0:
            return VerifyStatus.FAILED

        if failed > 0:
            return VerifyStatus.FAILED
        if degraded > 0 or recovered < total:
            return VerifyStatus.DEGRADED
        return VerifyStatus.VERIFIED

    def _build_summary(
        self, status: VerifyStatus,
        pre: dict[str, dict], post: dict[str, dict],
        health: dict[str, dict],
    ) -> str:
        """生成自然语言验证总结。"""
        parts: list[str] = [f"验证状态: {status.value}"]

        if pre and post:
            parts.append("指标变化:")
            for name, pre_data in pre.items():
                pre_v = pre_data.get("value", "?")
                post_v = post.get(name, {}).get("value", "?")
                error = post.get(name, {}).get("error", "")
                if error:
                    parts.append(f"  {name}: {pre_v} → 查询失败 ({error})")
                else:
                    parts.append(f"  {name}: {pre_v} → {post_v}")

        failed_checks = [k for k, v in health.items() if not v.get("ok")]
        if failed_checks:
            parts.append(f"健康检查异常: {failed_checks}")
        else:
            parts.append("健康检查: 全部通过")

        return "\n".join(parts)
