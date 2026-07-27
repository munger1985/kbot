"""通过模型托管内部 API 执行严格结构化推理。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any, TypeVar

import aiohttp
from pydantic import BaseModel, ValidationError

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.ports.model import StructuredModelResult
from platform_core.security import build_internal_auth_headers


OutputModel = TypeVar("OutputModel", bound=BaseModel)


class AIOpsModelError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


class AIOpsStructuredModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        audience: str,
        caller_service: str,
        timeout_seconds: int,
        session: aiohttp.ClientSession,
    ):
        self._url = (
            base_url.rstrip("/") + "/internal/v1/chat/completions"
        )
        self._audience = audience
        self._caller = caller_service
        self._timeout_seconds = timeout_seconds
        self._session = session

    async def generate_structured(
        self,
        *,
        purpose: str,
        output_model: type[OutputModel],
        model_snapshot: dict[str, Any],
        prompt_ref: dict[str, str],
        input_payload: dict[str, Any],
        max_output_tokens: int,
        deadline: datetime | None,
        idempotency_key: str,
    ) -> StructuredModelResult:
        del idempotency_key
        technical_name = str(model_snapshot["technical_name"])
        prompt_content = str(prompt_ref["content"])
        if (
            hashlib.sha256(prompt_content.encode()).hexdigest()
            != prompt_ref["prompt_sha256"]
        ):
            raise AIOpsModelError("PROMPT_HASH_MISMATCH")
        request_payload = {
            "served_model_name": technical_name,
            "messages": [
                {"role": "system", "content": prompt_content},
                {
                    "role": "user",
                    "content": json.dumps(
                        input_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ),
                },
            ],
            "stream": False,
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__,
                    "strict": True,
                    "schema": output_model.model_json_schema(),
                },
            },
        }
        timeout = self._timeout_seconds
        if deadline is not None:
            remaining = int(
                (deadline.astimezone(UTC) - datetime.now(UTC)).total_seconds()
            )
            if remaining <= 1:
                raise AIOpsModelError("MODEL_DEADLINE_EXCEEDED")
            timeout = min(timeout, remaining)
        started = time.monotonic()
        headers = build_internal_auth_headers(
            audience=self._audience,
            caller_service=self._caller,
        )
        try:
            async with self._session.post(
                self._url,
                headers=headers,
                json=request_payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status != 200:
                    raise AIOpsModelError("MODEL_SERVICE_UNAVAILABLE")
                envelope = await response.json()
        except AIOpsModelError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise AIOpsModelError("MODEL_SERVICE_UNAVAILABLE") from exc
        try:
            choice = envelope["choices"][0]
            content = choice["message"]["content"]
            raw_output = json.loads(content)
            output = output_model.model_validate(raw_output)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise AIOpsModelError("MODEL_OUTPUT_INVALID") from exc
        usage = envelope.get("usage") or {}
        duration_ms = int((time.monotonic() - started) * 1000)
        receipt = ModelInvocationReceipt(
            purpose=purpose,
            schema_id=output_model.model_fields[
                "schema_version"
            ].default,
            model_technical_name=technical_name,
            model_revision=str(model_snapshot["revision"]),
            prompt_id=prompt_ref["prompt_id"],
            prompt_version=prompt_ref["prompt_version"],
            prompt_sha256=prompt_ref["prompt_sha256"],
            input_sha256=_canonical_hash(input_payload),
            output_sha256=_canonical_hash(raw_output),
            provider_request_id=envelope.get("id"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            duration_ms=duration_ms,
            finish_reason=choice.get("finish_reason"),
        )
        return StructuredModelResult(output=output, receipt=receipt)
