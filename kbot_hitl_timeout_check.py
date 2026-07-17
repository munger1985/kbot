#!/usr/bin/env python3
"""
HITL 超时检测 CLI 工具 — 供 cron 定时调用。

用法:
  python kbot_hitl_timeout_check.py

Cron 示例 (每 5 分钟):
  */5 * * * * cd /home/chris/kbot3 && python kbot_hitl_timeout_check.py >> logs/hitl_timeout.log 2>&1
"""

import asyncio
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from agent.orchestrator.ops_orchestrator import OpsOrchestrator


async def main():
    logger.info("HITL 超时检测任务启动")
    orchestrator = OpsOrchestrator()
    timed_out = await orchestrator.check_pending_timeouts()

    if timed_out:
        logger.warning(f"发现 {len(timed_out)} 个超时挂起请求:")
        for req in timed_out:
            logger.warning(
                f"  - request_id={req['request_id']} "
                f"session={req['session_id']} "
                f"instance={req['instance_id']}"
            )
    else:
        logger.info("本轮未发现超时挂起请求")

    logger.info("HITL 超时检测任务完成")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
