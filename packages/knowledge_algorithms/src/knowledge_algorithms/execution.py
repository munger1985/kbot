"""协作式预算、取消和超时；转换器的进程硬限制仍由执行服务负责。"""

import asyncio
import math
import time
from typing import Awaitable, TypeVar

from .contracts.errors import AlgorithmError, ErrorCode, require


T = TypeVar("T")


class Deadline:
    def __init__(self, seconds: float):
        require(math.isfinite(seconds) and seconds > 0, "超时必须为有限正数")
        self.end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self.end - time.monotonic())

    async def run(self, operation: Awaitable[T], *, stage: str) -> T:
        try:
            return await asyncio.wait_for(operation, self.remaining())
        except asyncio.TimeoutError as exc:
            raise AlgorithmError(ErrorCode.TIMEOUT, "阶段执行超时", stage=stage) from exc

    async def checkpoint(self, *, stage: str) -> None:
        if self.remaining() <= 0:
            raise AlgorithmError(ErrorCode.TIMEOUT, "阶段执行超时", stage=stage)
        await asyncio.sleep(0)
