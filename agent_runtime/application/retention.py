"""Conversation 保留策略清理 Worker。"""

import asyncio

from loguru import logger


class ConversationRetentionWorker:
    def __init__(
        self,
        *,
        conversation_service,
        poll_interval_seconds: float = 60,
    ):
        self._conversation_service = conversation_service
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    async def run_forever(self) -> None:
        logger.info("Conversation 保留策略 Worker 已启动")
        while not self._stop_event.is_set():
            worked = await self.run_once()
            if not worked:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._poll_interval_seconds,
                    )
                except TimeoutError:
                    pass
        logger.info("Conversation 保留策略 Worker 已停止")

    async def run_once(self) -> bool:
        try:
            return await self._conversation_service.purge_one_due()
        except Exception:
            logger.exception("清理到期 Conversation 失败")
            return False
