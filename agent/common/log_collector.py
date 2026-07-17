"""日志采集器 — 通过 DB Executor 微服务向 target 实例发送诊断 SQL 采集日志。

target 实例可能是 Oracle / PostgreSQL / MySQL 中的任意一种，
由 instance 配置中的 db_type 决定，LogCollector 据此派发不同 SQL。
"""
from loguru import logger
from utils.clients import OpsDBExecutor


class LogCollector:
    """通过 DB Executor 微服务采集 target 数据库实例的日志。

    三引擎自适应:
    - Oracle:  V$DIAG_ALERT_EXT (ADR, 11g+ 默认启用)
    - PostgreSQL: pg_stat_activity + pg_stat_statements (替代方案)
    - MySQL:    performance_schema.error_log (5.7+)
    """

    def __init__(self):
        self.db_executor = OpsDBExecutor()

    async def collect_db_logs(
        self, instance_id: str, db_type: str, lines: int = 200,
    ) -> list[str]:
        """统一入口 — 按 db_type 派发。"""
        db = db_type.lower() if db_type else ""
        if db in ("oracle",):
            return await self._collect_oracle_logs(instance_id, lines)
        elif db in ("postgresql", "postgres"):
            return await self._collect_postgresql_logs(instance_id, lines)
        elif db in ("mysql",):
            return await self._collect_mysql_logs(instance_id, lines)
        else:
            logger.warning(f"[LogCollector] 未知 db_type={db_type}, 跳过日志采集")
            return []

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------

    async def _collect_oracle_logs(self, instance_id: str, lines: int) -> list[str]:
        """Oracle ADR 告警日志 (11g+ 默认启用, 需 SELECT ON V$DIAG_ALERT_EXT 权限)。"""
        sql = (
            "SELECT TO_CHAR(ORIGINATING_TIMESTAMP,'YYYY-MM-DD HH24:MI:SS') AS TS, "
            "MESSAGE_TEXT "
            "FROM V$DIAG_ALERT_EXT "
            "ORDER BY ORIGINATING_TIMESTAMP DESC "
            f"FETCH FIRST {lines} ROWS ONLY"
        )
        try:
            rows = await self.db_executor.execute_readonly_ops_sql(
                instance_id=instance_id, sql=sql, limit=lines,
            )
            return [f"[{r.get('TS', '')}] {r.get('MESSAGE_TEXT', '')}" for r in rows]
        except Exception as e:
            logger.warning(f"[LogCollector] Oracle 日志采集失败: {e}")
            return [f"Oracle 日志采集失败: {e}"]

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------

    async def _collect_postgresql_logs(self, instance_id: str, lines: int) -> list[str]:
        """PG 日志采集 — pg_read_file() 在 systemd 部署下不可用，改用活动查询/锁视图。

        注意: 纯 error log 无法通过 SQL 获取 (需 SSH 读 journald)。
        此方法采集 pg_stat_activity + pg_locks 作为替代诊断数据。
        """
        results: list[str] = []

        # 1. 活跃查询 (含等待事件)
        try:
            rows = await self.db_executor.execute_readonly_ops_sql(
                instance_id=instance_id,
                sql=(
                    "SELECT pid, state, wait_event, LEFT(query,200) AS query "
                    "FROM pg_stat_activity "
                    "WHERE state != 'idle' AND pid != pg_backend_pid() "
                    f"ORDER BY query_start DESC NULLS LAST LIMIT {lines}"
                ),
                limit=lines,
            )
            for r in rows:
                results.append(
                    f"[pid={r.get('pid')}, state={r.get('state')}, "
                    f"wait={r.get('wait_event')}] {r.get('query', '')}"
                )
        except Exception as e:
            results.append(f"pg_stat_activity 采集失败: {e}")

        # 2. 锁等待
        try:
            rows = await self.db_executor.execute_readonly_ops_sql(
                instance_id=instance_id,
                sql="SELECT COUNT(*) AS waiting FROM pg_locks WHERE NOT GRANTED",
                limit=1,
            )
            if rows:
                waiting = rows[0].get("waiting", 0)
                if waiting > 0:
                    results.insert(0, f"⚠️ 当前存在 {waiting} 个锁等待")
        except Exception:
            pass

        return results if results else ["pg_stat_activity 查询为空 (无活跃后端)"]

    # ------------------------------------------------------------------
    # MySQL
    # ------------------------------------------------------------------

    async def _collect_mysql_logs(self, instance_id: str, lines: int) -> list[str]:
        """MySQL 日志采集 — 5.7+ 使用 performance_schema.error_log。"""
        results: list[str] = []

        # 1. 错误日志 (MySQL 5.7+)
        try:
            rows = await self.db_executor.execute_readonly_ops_sql(
                instance_id=instance_id,
                sql=(
                    "SELECT LOGGED, PRIO, ERROR_CODE, DATA "
                    "FROM performance_schema.error_log "
                    "ORDER BY LOGGED DESC "
                    f"LIMIT {lines}"
                ),
                limit=lines,
            )
            for r in rows:
                results.append(
                    f"[{r.get('LOGGED', '')}] [{r.get('PRIO', '')}] "
                    f"code={r.get('ERROR_CODE', '')} {r.get('DATA', '')}"
                )
            return results
        except Exception:
            pass

        # 2. 降级: InnoDB 引擎状态 + 慢查询日志
        try:
            rows = await self.db_executor.execute_readonly_ops_sql(
                instance_id=instance_id, sql="SHOW ENGINE INNODB STATUS", limit=1,
            )
            if rows:
                results.append(f"InnoDB Status: {str(rows[0])[:500]}")
        except Exception as e:
            results.append(f"InnoDB 状态采集失败: {e}")

        try:
            rows = await self.db_executor.execute_readonly_ops_sql(
                instance_id=instance_id,
                sql=f"SELECT * FROM mysql.slow_log ORDER BY start_time DESC LIMIT {lines}",
                limit=lines,
            )
            for r in rows:
                results.append(f"[slow] {r.get('sql_text', '')[:200]}")
        except Exception:
            pass

        return results if results else ["MySQL 日志采集: error_log 不可用 (需 5.7+)"]
