"""AIOps 结构化模型调用端口。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt


OutputModel = TypeVar("OutputModel", bound=BaseModel)


@dataclass(frozen=True)
class StructuredModelResult:
    output: BaseModel
    receipt: ModelInvocationReceipt


@dataclass(frozen=True)
class ModelTextDelta:
    """模型原文增量；重试时明确撤回上次尚未核验的草稿。"""

    delta: str = ""
    reset: bool = False


TextDeltaCallback = Callable[[ModelTextDelta], Awaitable[None]]


class AIOpsModelPort(Protocol):
    async def generate_structured(
        self,
        *,
        purpose: str,
        output_model: type[OutputModel],
        model_snapshot: dict[str, Any],
        prompt_ref: dict[str, str],
        input_payload: dict[str, Any],
        deadline: datetime | None,
        idempotency_key: str,
        text_field: str | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> StructuredModelResult: ...

    def stream_text(
        self,
        *,
        purpose: str,
        model_snapshot: dict[str, Any],
        prompt_ref: dict[str, str],
        input_payload: dict[str, Any],
        deadline: datetime | None,
        idempotency_key: str,
    ) -> AsyncIterator[str]: ...
