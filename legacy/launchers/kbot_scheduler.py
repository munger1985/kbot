#!/usr/bin/env python
"""kbot3 AIOps 定时巡检调度器 (扩展自 kbot_hitl_timeout_check.py)。

功能:
  1. HITL 超时 pending 扫描 (原有功能)
  2. 扫描所有 active 状态的数据库实例
  3. 对每个实例执行预定义 PromQL 阈值巡检
  4. 超阈值指标 → 自动调用 /api/ops/chat/alert-webhook 触发诊断

用法:
  python kbot_scheduler.py                     # 单次巡检
  python kbot_scheduler.py --interval 1800     # 每 1800s 循环

配置 (configuration/development.toml):
  [scheduler]
  enabled = true
  patrol_interval_minutes = 30
"""
import asyncio
import argparse
import os
import sys
from datetime import datetime, timezone, timedelta

import aiohttp
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from platform_core.config import get_settings
from utils.monitor import PrometheusClient, UnifiedMetricRegistry

DEFAULT_PATROL_RULES: list[dict] = [
    {"metric": "tablespace_usage_pct", "threshold": 90, "severity": "warning"},
    {"metric": "active_sessions", "threshold": 100, "severity": "warning"},
    {"metric": "lock_wait_count", "threshold": 1, "severity": "critical"},
]


class PatrolScheduler:
    """定时巡检 + HITL 超时扫描"""

    def __init__(self, interval_seconds: int = 0):
        self.interval = interval_seconds
        self.prometheus = PrometheusClient()
        self.registry = UnifiedMetricRegistry()
        self._api_base = "http://127.0.0.1:18090/api/ops"

    # ------------------------------------------------------------------
    # HITL 超时扫描 (原 kbot_hitl_timeout_check.py 逻辑)
    # ------------------------------------------------------------------

    async def scan_hitl_timeouts(self):
        """扫描并标记超时的 HITL pending 请求。"""
        try:
            from platform_core.database import get_session
            from sqlalchemy import text

            async with get_session() as session:
                timeout_threshold = datetime.now(timezone.utc) - timedelta(minutes=30)
                result = await session.execute(
                    text(
                        "UPDATE kbot_ops_pending_request "
                        "SET status = 'timeout' "
                        "WHERE status = 'pending' AND timeout_at <= :now"
                    ),
                    {"now": timeout_threshold},
                )
                await session.commit()
                if result.rowcount and result.rowcount > 0:
                    logger.info(
                        f"[Scheduler] HITL 超时扫描: {result.rowcount} 条标记为 timeout"
                    )
        except Exception as e:
            logger.error(f"[Scheduler] HITL 超时扫描失败: {e}")

    # ------------------------------------------------------------------
    # 指标巡检
    # ------------------------------------------------------------------

    async def run_once(self) -> list[dict]:
        """执行一次完整巡检，返回触发的告警列表。"""
        # 1. HITL 超时扫描
        await self.scan_hitl_timeouts()

        # 2. 指标巡检
        triggered: list[dict] = []
        instances = await self._get_active_instances()
        if not instances:
            logger.debug("[Scheduler] 无活跃实例")
            return triggered

        for inst in instances:
            instance_id = inst["instance_id"]
            instance_name = inst.get("instance_name", instance_id)
            db_type = inst.get("db_type", "")
            agent_id = inst.get("agent_id", "")

            if not agent_id:
                continue

            for rule in DEFAULT_PATROL_RULES:
                try:
                    promql = self.registry.render_query(
                        rule["metric"], db_type=db_type, instance_id=instance_id,
                    )
                    result = await self.prometheus.query_instant(promql)
                    value = self._extract_value(result)

                    if value is not None and value > rule["threshold"]:
                        logger.warning(
                            f"[Scheduler] 🚨 {instance_name} {rule['metric']}={value}"
                        )
                        alert_payload = {
                            "source": "cron_patrol",
                            "instance_id": instance_id,
                            "agent_id": str(agent_id),
                            "payload": {
                                "alerts": [{
                                    "status": "firing",
                                    "labels": {
                                        "alertname": f"Patrol_{rule['metric']}",
                                        "severity": rule["severity"],
                                        "instance": instance_name,
                                    },
                                    "annotations": {
                                        "summary": f"巡检 {rule['metric']}={value} > {rule['threshold']}",
                                        "description": f"{instance_name}({db_type}) {rule['metric']}={value}",
                                    },
                                }],
                            },
                        }
                        await self._trigger_diagnosis(alert_payload)
                        triggered.append({
                            "instance_id": instance_id,
                            "instance_name": instance_name,
                            "metric": rule["metric"],
                            "value": value,
                            "threshold": rule["threshold"],
                        })
                except Exception as e:
                    logger.debug(f"[Scheduler] {instance_name}/{rule['metric']}: {e}")

        return triggered

    async def run_loop(self):
        """循环执行。"""
        logger.info(f"[Scheduler] 启动, 间隔={self.interval}s")
        while True:
            try:
                triggered = await self.run_once()
                if triggered:
                    logger.info(f"[Scheduler] 触发 {len(triggered)} 条告警")
            except Exception as e:
                logger.error(f"[Scheduler] 异常: {e}")
            if self.interval <= 0:
                break
            await asyncio.sleep(self.interval)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_active_instances(self) -> list[dict]:
        try:
            from platform_core.database import get_session
            from sqlalchemy import text
            async with get_session() as session:
                result = await session.execute(text(
                    "SELECT i.instance_id, i.instance_name, i.db_type, i.monitor_type, "
                    "       c.agent_id "
                    "FROM kbot.ops_db_instance i "
                    "LEFT JOIN kbot.ops_agent_conf c ON i.instance_id = c.instance_id "
                    "WHERE i.status = 'active'"
                ))
                rows = result.fetchall()
                return [
                    {"instance_id": r[0], "instance_name": r[1],
                     "db_type": r[2], "monitor_type": r[3], "agent_id": r[4]}
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"[Scheduler] 获取实例失败: {e}")
            return []

    async def _trigger_diagnosis(self, alert_payload: dict):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._api_base}/chat/alert-webhook",
                    json=alert_payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"[Scheduler] webhook HTTP {resp.status}")
        except Exception as e:
            logger.error(f"[Scheduler] webhook 调用失败: {e}")

    @staticmethod
    def _extract_value(result) -> float | None:
        if not result:
            return None
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                v = item.get("value")
                if isinstance(v, list) and len(v) >= 2:
                    return float(v[1])
                if isinstance(v, (int, float)):
                    return float(v)
        return None


async def main():
    parser = argparse.ArgumentParser(description="kbot3 AIOps 定时巡检 + HITL 超时扫描")
    parser.add_argument("--interval", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config = get_settings()
    sched_cfg = getattr(config, "scheduler", None)
    if sched_cfg and not getattr(sched_cfg, "enabled", True):
        logger.warning("[Scheduler] 未启用")
        return

    if args.once:
        args.interval = 0

    scheduler = PatrolScheduler(interval_seconds=args.interval)
    if args.interval > 0:
        await scheduler.run_loop()
    else:
        triggered = await scheduler.run_once()
        print(f"\n{'='*50}")
        print(f"巡检完成: {datetime.now(timezone.utc).isoformat()}")
        print(f"触发告警: {len(triggered)} 条")
        for t in triggered:
            print(f"  - {t['instance_name']}: {t['metric']}={t['value']} (阈值 {t['threshold']})")
        print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
