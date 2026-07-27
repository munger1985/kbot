"""KC 异步任务唤醒端口。"""

from collections.abc import Set
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession


PARSE_WAKEUP_CHANNEL = "KBOT_KC_PARSE_READY"
PROJECTION_WAKEUP_CHANNEL = "KBOT_KC_PROJECTION_READY"


def wakeup_channel_for_job(job_type: str) -> str:
    """将持久化任务类型映射为粗粒度 Worker 唤醒通道。"""
    return (
        PARSE_WAKEUP_CHANNEL
        if job_type == "PARSE"
        else PROJECTION_WAKEUP_CHANNEL
    )


class JobWakeupPublisher(Protocol):
    """在任务事务提交前登记提交后生效的唤醒信号。"""

    async def signal(
        self,
        session: AsyncSession,
        channels: Set[str],
    ) -> None: ...


class JobWakeupListener(Protocol):
    """等待任务提示；提示不是任务事实来源。"""

    async def wait(self, timeout_seconds: float) -> bool: ...

    async def close(self) -> None: ...
