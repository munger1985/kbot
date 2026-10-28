"""Driver 只向本次租约收集回执；回传失败不能改写取消语义或宣称终止成功。"""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from loguru import logger

from platform_core.contracts.aiops.executor import ReadonlyTerminationRequest
from platform_core.contracts.aiops.readonly_lifecycle import ReadonlyTerminationObservation


_current = ContextVar("aiops_readonly_termination", default=None)


@contextmanager
def collect_termination(db_type):
    # MySQL 子任务继承相同容器；不同请求的 ContextVar 不共享容器。
    collector = {"db_type": db_type, "observation": None}
    token = _current.set(collector)
    try:
        yield collector
    finally:
        _current.reset(token)


def observe_termination(db_type, status, method):
    collector = _current.get()
    if collector is None:
        return
    if collector["db_type"] != db_type:
        raise RuntimeError("数据库终止回执与当前租约不匹配")
    collector["observation"] = ReadonlyTerminationObservation(
        db_type=db_type, status=status, method=method, observed_at=datetime.now(UTC))


async def report_termination(control_plane, *, receipt, collector, trace_id):
    observation = collector["observation"]
    if observation is None:
        return
    request = ReadonlyTerminationRequest(resource_lease=receipt, observation=observation)
    try:
        # 相同内容可重放，覆盖“已提交但响应丢失”；不重试任何数据库取消命令。
        async with asyncio.timeout(3):
            for attempt in range(2):
                try:
                    await control_plane.report_readonly_termination(request, trace_id=trace_id)
                    return
                except Exception:
                    if attempt == 1:
                        raise
                    await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        logger.warning("只读查询终止回执回传被取消；控制面租约保留未知状态")
        raise
    except Exception:
        logger.warning("只读查询终止回执未确认持久化；不能宣称数据库已停止")
