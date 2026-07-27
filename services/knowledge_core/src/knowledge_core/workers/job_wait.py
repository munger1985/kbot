"""通知优先、轮询兜底的 KC Worker 等待策略。"""

import asyncio
import random

from loguru import logger

from knowledge_core.ports.job_wakeup import JobWakeupListener


class AdaptiveJobWait:
    """收到通知立即返回；通知故障时使用指数退避轮询。"""

    def __init__(
        self,
        *,
        listener: JobWakeupListener | None,
        notification_timeout_seconds: float,
        fallback_min_seconds: float,
        fallback_max_seconds: float,
        fallback_multiplier: float,
        jitter_ratio: float,
    ):
        self._listener = listener
        self._notification_timeout = notification_timeout_seconds
        self._fallback_min = fallback_min_seconds
        self._fallback_max = fallback_max_seconds
        self._multiplier = fallback_multiplier
        self._jitter_ratio = jitter_ratio
        self._fallback_delay = fallback_min_seconds
        self._listener_failed = False

    def reset(self) -> None:
        self._fallback_delay = self._fallback_min

    async def wait(self) -> None:
        if self._listener is not None:
            try:
                await self._listener.wait(self._notification_timeout)
                if self._listener_failed:
                    logger.info("KC 任务通知连接已恢复")
                self._listener_failed = False
                self.reset()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._listener_failed:
                    logger.warning(
                        "KC 任务通知不可用，切换为自适应轮询：{}",
                        exc,
                    )
                self._listener_failed = True

        jitter = self._fallback_delay * self._jitter_ratio
        delay = max(
            0.05,
            self._fallback_delay + random.uniform(-jitter, jitter),
        )
        await asyncio.sleep(delay)
        self._fallback_delay = min(
            self._fallback_max,
            self._fallback_delay * self._multiplier,
        )

    async def close(self) -> None:
        if self._listener is not None:
            await self._listener.close()
