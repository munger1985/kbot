"""通过模型托管内部 API 执行严格结构化推理。"""

from __future__ import annotations

import codecs
import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any, TypeVar

import aiohttp
from loguru import logger
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.ports.model import ModelTextDelta, StructuredModelResult, TextDeltaCallback
from platform_core.security import build_internal_auth_headers


OutputModel = TypeVar("OutputModel", bound=BaseModel)


class AIOpsModelError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None, *, answer_complete: bool = False):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail
        self.answer_complete = answer_complete
        self.retryable = code != "MODEL_DEADLINE_EXCEEDED"


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
        text_field: str | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> StructuredModelResult:
        technical_name = str(model_snapshot["technical_name"])
        prompt_content = str(prompt_ref["content"])
        if (
            hashlib.sha256(prompt_content.encode()).hexdigest()
            != prompt_ref["prompt_sha256"]
        ):
            raise AIOpsModelError("PROMPT_HASH_MISMATCH")
        output_schema = output_model.model_json_schema()
        if on_text_delta is not None:
            if not text_field or text_field not in output_model.model_fields:
                raise ValueError("流式正文必须对应输出合同中的字符串字段")
            # 继承合同把新增正文字段排在最后；模型所见 Schema 与正文优先指令必须一致。
            properties = output_schema["properties"]
            output_schema["properties"] = {
                text_field: properties[text_field],
                **{key: value for key, value in properties.items() if key != text_field},
            }
            if text_field in output_schema.get("required", ()):
                output_schema["required"] = [text_field, *(key for key in output_schema["required"] if key != text_field)]
        structured_prompt = (
            f"{prompt_content.rstrip()}\n\n"
            "必须只返回一个 JSON 对象，不得添加 Markdown 代码块或解释文字。"
            "返回对象必须严格满足以下 JSON Schema：\n"
            f"{json.dumps(output_schema, ensure_ascii=False, separators=(',', ':'))}"
        )
        if on_text_delta is not None:
            structured_prompt += f"\n先输出 {text_field} 字段中的完整 Markdown 正文，再输出其余诊断字段；正文与结构化判断必须一致。"
        messages = [
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
        ]
        started = time.monotonic()
        logger.info("AIOps 模型输入分布：purpose={} request_id={} field_chars={}",
                    purpose, idempotency_key,
                    {key: len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))
                     for key, value in input_payload.items()})
        prompt_tokens = 0
        completion_tokens = 0
        published_chars = 0
        original_callback = on_text_delta

        async def publish_text(delta):
            nonlocal published_chars
            published_chars += len(delta.delta)
            await original_callback(delta)

        if original_callback is not None:
            on_text_delta = publish_text
        for attempt in range(2):
            if on_text_delta is not None:
                await on_text_delta(ModelTextDelta(reset=True))
            request_payload = {
                "served_model_name": technical_name,
                "messages": messages,
                "stream": on_text_delta is not None,
                "temperature": 0,
                # OpenAI 兼容厂商对 json_schema 的支持并不一致。统一使用
                # JSON Mode，并在本服务内执行同一份 Schema 的严格校验。
                "response_format": {"type": "json_object"},
            }
            timeout = self._timeout_seconds
            if deadline is not None:
                remaining = int(
                    (
                        deadline.astimezone(UTC) - datetime.now(UTC)
                    ).total_seconds()
                )
                if remaining <= 1:
                    raise AIOpsModelError("MODEL_DEADLINE_EXCEEDED")
                timeout = min(timeout, remaining)
            logger.info(
                "AIOps 结构化模型阶段开始：purpose={} request_id={} attempt={} timeout_seconds={} input_chars={} schema_chars={}",
                purpose, idempotency_key, attempt + 1, timeout,
                sum(len(item["content"]) for item in messages), len(json.dumps(output_schema)),
            )
            # 把本次剩余预算传到模型托管，避免调用方截止后 Provider 仍继续生成长报告。
            request_payload["timeout"] = timeout
            # 上一次生成可能已耗尽短期令牌寿命，每次 HTTP 请求独立签发。
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
                        if response.status == 408:
                            code = "MODEL_DEADLINE_EXCEEDED" if deadline is not None and (
                                deadline.astimezone(UTC) - datetime.now(UTC)
                            ).total_seconds() <= 1 else "MODEL_SERVICE_UNAVAILABLE"
                            raise AIOpsModelError(code)
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
                    envelope = (
                        await self._read_structured_stream(response, text_field=text_field, on_text_delta=on_text_delta,
                            purpose=purpose, request_id=idempotency_key, attempt=attempt + 1)
                        if on_text_delta is not None else await response.json()
                    )
            except AIOpsModelError:
                raise
            except TimeoutError as exc:
                code = "MODEL_DEADLINE_EXCEEDED" if deadline is not None and (
                    deadline.astimezone(UTC) - datetime.now(UTC)
                ).total_seconds() <= 1 else "MODEL_SERVICE_UNAVAILABLE"
                logger.warning("AIOps 模型阶段超时：purpose={} request_id={} code={} elapsed_ms={} timeout_seconds={}",
                               purpose, idempotency_key, code, int((time.monotonic() - started) * 1000), timeout)
                raise AIOpsModelError(code) from exc
            except (aiohttp.ClientError, ValueError) as exc:
                raise AIOpsModelError("MODEL_SERVICE_UNAVAILABLE") from exc
            usage = envelope.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            content = ""
            try:
                choice = envelope["choices"][0]
                content = choice["message"]["content"]
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    raise ValueError("模型输出未正常完成")
                raw_output = json.loads(content)
                output = output_model.model_validate(raw_output)
            except (
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
                ValidationError,
                ValueError,
            ) as exc:
                error_detail = json.dumps(exc.errors(include_input=False, include_context=False), ensure_ascii=False) if isinstance(exc, ValidationError) else type(exc).__name__
                logger.warning(
                    "AIOps 模型结构化输出校验失败：purpose={} model={} "
                    "schema={} attempt={}/2 error={}",
                    purpose,
                    technical_name,
                    output_model.__name__,
                    attempt + 1,
                    error_detail,
                )
                # 用户已经看见的正文是本轮唯一回复；结构核验不能重新生成并清空它。
                if published_chars:
                    raise AIOpsModelError("MODEL_OUTPUT_INVALID", error_detail,
                        answer_complete=choice.get("finish_reason") == "stop") from exc
                if attempt == 0:
                    messages = [
                        *messages,
                        {
                            "role": "assistant",
                            "content": str(content)[:20000],
                        },
                        {
                            "role": "user",
                            "content": (
                                "上一份 JSON 未通过既定输出合同。请保持原任务"
                                "和事实不变，只依据以下校验错误修正 JSON，并重新"
                                "返回满足同一 JSON Schema 的完整对象：\n"
                                + error_detail[:6000]
                            ),
                        },
                    ]
                    continue
                raise AIOpsModelError(
                    "MODEL_OUTPUT_INVALID", error_detail
                ) from exc
            break
        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "AIOps 结构化模型阶段完成：purpose={} request_id={} duration_ms={} prompt_tokens={} completion_tokens={}",
            purpose, idempotency_key, duration_ms, prompt_tokens, completion_tokens,
        )
        receipt = ModelInvocationReceipt(
            purpose=purpose,
            schema_id=_output_schema_id(output_model),
            model_technical_name=technical_name,
            model_revision=str(model_snapshot["revision"]),
            prompt_id=prompt_ref["prompt_id"],
            prompt_version=prompt_ref["prompt_version"],
            prompt_sha256=prompt_ref["prompt_sha256"],
            prompt_version_id=prompt_ref.get("prompt_version_id"),
            prompt_source=prompt_ref.get("prompt_source"),
            input_sha256=_canonical_hash(input_payload),
            output_sha256=_canonical_hash(raw_output),
            provider_request_id=envelope.get("id"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            finish_reason=choice.get("finish_reason"),
        )
        return StructuredModelResult(output=output, receipt=receipt)

    async def _read_structured_stream(self, response, *, text_field, on_text_delta, purpose, request_id, attempt):
        """边接收边解码正文，不暴露 JSON 元数据、推理内容或未完成的转义。"""
        decoder = codecs.getincrementaldecoder("utf-8")()
        pending, content, emitted = "", "", ""
        provider_id, finish_reason, usage = None, None, {}
        done = False
        started = time.monotonic()
        frame_count, reasoning_chars = 0, 0
        first_content_ms, first_answer_ms = None, None
        stream_ended = False

        async def consume(frame):
            nonlocal content, emitted, provider_id, finish_reason, usage, done
            nonlocal frame_count, reasoning_chars, first_content_ms, first_answer_ms
            data = "\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
            if not data:
                return
            if data == "[DONE]":
                done = True
                return
            item = json.loads(data)
            frame_count += 1
            if item.get("error"):
                raise AIOpsModelError("MODEL_SERVICE_UNAVAILABLE")
            provider_id = item.get("id") or provider_id
            usage = item.get("usage") or usage
            for choice in item.get("choices") or ():
                if choice.get("index", 0) != 0:
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = dict(choice.get("delta") or {})
                # 只累计长度，不保留或记录推理正文。
                reasoning_chars += len(str(delta.get("reasoning_content") or ""))
                chunk = str(delta.get("content") or "")
                if chunk and first_content_ms is None:
                    first_content_ms = int((time.monotonic() - started) * 1000)
                content += chunk
            try:
                partial = from_json(content, allow_partial="trailing-strings")
            except ValueError:
                return
            text = partial.get(text_field, "") if isinstance(partial, dict) else ""
            if not isinstance(text, str) or not text.startswith(emitted):
                raise AIOpsModelError("MODEL_OUTPUT_INVALID")
            delta = text[len(emitted):]
            # 小幅合并网络碎片以限制进度事务数量，不等待完整正文或完整诊断。
            if len(delta) >= 32 or "\n" in delta:
                if first_answer_ms is None:
                    first_answer_ms = int((time.monotonic() - started) * 1000)
                for offset in range(0, len(delta), 2000):
                    await on_text_delta(ModelTextDelta(delta=delta[offset:offset + 2000]))
                emitted = text

        try:
            async for raw in response.content.iter_any():
                pending = (pending + decoder.decode(raw)).replace("\r\n", "\n")
                frames = pending.split("\n\n")
                pending = frames.pop()
                for frame in frames:
                    await consume(frame)
                    if done:
                        break
                if done:
                    break
            if not done:
                pending += decoder.decode(b"", final=True)
                if pending.strip():
                    await consume(pending)
            if not done and finish_reason is None:
                raise AIOpsModelError("MODEL_STREAM_INTERRUPTED")
            try:
                text = json.loads(content).get(text_field, "")
            except (ValueError, AttributeError):
                text = emitted
            if isinstance(text, str) and text.startswith(emitted) and len(text) > len(emitted):
                if first_answer_ms is None:
                    first_answer_ms = int((time.monotonic() - started) * 1000)
                for offset in range(len(emitted), len(text), 2000):
                    await on_text_delta(ModelTextDelta(delta=text[offset:offset + 2000]))
                emitted = text
            stream_ended = True
            return {"id": provider_id, "usage": usage,
                    "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
        finally:
            logger.info(
                "AIOps 模型流阶段统计：purpose={} request_id={} attempt={} stream_ended={} "
                "elapsed_ms={} frames={} content_chars={} reasoning_chars={} answer_chars={} "
                "first_content_ms={} first_answer_ms={} finish_reason={}",
                purpose, request_id, attempt, stream_ended, int((time.monotonic() - started) * 1000),
                frame_count, len(content), reasoning_chars, len(emitted), first_content_ms, first_answer_ms, finish_reason,
            )

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
        except TimeoutError as exc:
            code = "MODEL_DEADLINE_EXCEEDED" if deadline is not None and (
                deadline.astimezone(UTC) - datetime.now(UTC)
            ).total_seconds() <= 1 else "MODEL_SERVICE_UNAVAILABLE"
            raise AIOpsModelError(code) from exc
        except (aiohttp.ClientError, UnicodeDecodeError) as exc:
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
