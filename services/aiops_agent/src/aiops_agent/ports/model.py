"""AIOps 结构化模型调用端口。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections.abc import AsyncIterator
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt


OutputModel = TypeVar("OutputModel", bound=BaseModel)


@dataclass(frozen=True)
class StructuredModelResult:
    output: BaseModel
    receipt: ModelInvocationReceipt


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
