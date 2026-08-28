"""通过模型托管内部 API 执行严格结构化推理。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any, TypeVar

import aiohttp
from loguru import logger
from pydantic import BaseModel, ValidationError

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.ports.model import StructuredModelResult
from platform_core.security import build_internal_auth_headers


OutputModel = TypeVar("OutputModel", bound=BaseModel)


class AIOpsModelError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


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


def _output_schema_id(output_model: type[BaseModel]) -> str:
    """返回稳定输出契约标识，组合契约可不声明顶层版本字段。"""
    schema_field = output_model.model_fields.get("schema_version")
    if (
        schema_field is not None
        and isinstance(schema_field.default, str)
        and schema_field.default.strip()
    ):
        return schema_field.default
    return output_model.__name__


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
        output_schema = output_model.model_json_schema()
        structured_prompt = (
            f"{prompt_content.rstrip()}\n\n"
            "必须只返回一个 JSON 对象，不得添加 Markdown 代码块或解释文字。"
            "返回对象必须严格满足以下 JSON Schema：\n"
            f"{json.dumps(output_schema, ensure_ascii=False, separators=(',', ':'))}"
        )
        request_payload = {
            "served_model_name": technical_name,
            "messages": [
                {"role": "system", "content": structured_prompt},
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
            # OpenAI 兼容厂商对 json_schema 的支持并不一致。统一使用
            # JSON Mode，并在本服务内执行同一份 Schema 的严格校验。
            "response_format": {"type": "json_object"},
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
                    response_text = (await response.text())[:1000]
                    logger.error(
                        "AIOps 结构化模型调用失败：purpose={} model={} "
                        "status={} response={}",
                        purpose,
                        technical_name,
                        response.status,
                        response_text,
                    )
                    raise AIOpsModelError(
                        "MODEL_SERVICE_UNAVAILABLE",
                        f"模型服务返回 HTTP {response.status}",
                    )
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
        except (
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            logger.warning(
                "AIOps 模型结构化输出校验失败：purpose={} model={} "
                "schema={} error={}",
                purpose,
                technical_name,
                output_model.__name__,
                str(exc),
            )
            raise AIOpsModelError("MODEL_OUTPUT_INVALID", str(exc)) from exc
        usage = envelope.get("usage") or {}
        duration_ms = int((time.monotonic() - started) * 1000)
        receipt = ModelInvocationReceipt(
            purpose=purpose,
            schema_id=_output_schema_id(output_model),
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

    async def stream_text(
        self,
        *,
        purpose: str,
        model_snapshot: dict[str, Any],
        prompt_ref: dict[str, str],
        input_payload: dict[str, Any],
        deadline: datetime | None,
        idempotency_key: str,
    ):
        """把模型服务SSE归一为最终回答正文增量。"""
        del idempotency_key
        prompt_content = str(prompt_ref["content"])
        if (
            hashlib.sha256(prompt_content.encode()).hexdigest()
            != prompt_ref["prompt_sha256"]
        ):
            raise AIOpsModelError("PROMPT_HASH_MISMATCH")
        technical_name = str(model_snapshot["technical_name"])
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
            "stream": True,
            "temperature": 0,
        }
        timeout = self._timeout_seconds
        if deadline is not None:
            remaining = int(
                (deadline.astimezone(UTC) - datetime.now(UTC)).total_seconds()
            )
            if remaining <= 1:
                raise AIOpsModelError("MODEL_DEADLINE_EXCEEDED")
            timeout = min(timeout, remaining)
        headers = build_internal_auth_headers(
            audience=self._audience,
            caller_service=self._caller,
        )
        pending = ""
        try:
            async with self._session.post(
                self._url,
                headers=headers,
                json=request_payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status != 200:
                    response_text = (await response.text())[:1000]
                    logger.error(
                        "AIOps流式模型调用失败：purpose={} model={} "
                        "status={} response={}",
                        purpose,
                        technical_name,
                        response.status,
                        response_text,
                    )
                    raise AIOpsModelError(
                        "MODEL_SERVICE_UNAVAILABLE",
                        f"模型服务返回 HTTP {response.status}",
                    )
                async for raw in response.content.iter_any():
                    pending += raw.decode("utf-8")
                    lines = pending.splitlines(keepends=True)
                    pending = ""
                    if lines and not lines[-1].endswith(("\n", "\r")):
                        pending = lines.pop()
                    for line in lines:
                        content = self._decode_stream_line(line)
                        if content:
                            yield content
                if pending.strip():
                    content = self._decode_stream_line(pending)
                    if content:
                        yield content
        except AIOpsModelError:
            raise
        except (aiohttp.ClientError, TimeoutError, UnicodeDecodeError) as exc:
            raise AIOpsModelError("MODEL_SERVICE_UNAVAILABLE") from exc

    @staticmethod
    def _decode_stream_line(line: str) -> str | None:
        value = line.strip()
        if not value or value in {"data: [DONE]", "[DONE]"}:
            return None
        if value.startswith("data:"):
            value = value[5:].strip()
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return value
        if payload.get("error"):
            raise AIOpsModelError(
                "MODEL_SERVICE_UNAVAILABLE", str(payload["error"])[:1000]
            )
        choices = payload.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta") or choice.get("message") or {}
        return str(delta.get("content") or "") or None
