"""通过 OCI 签名调用 Grok 4.6 Responses，不走 Generic Chat。"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import requests
from loguru import logger

from model_serving.common.oci_auth import validated_oci_config
from model_serving.llm.responses.errors import GenerativeAdapterError
from platform_core.contracts import (
    GeneratedImageArtifact,
    ImageGenerationRequest,
    ImageGenerationResult,
    ModelUsage,
    ResearchCitation,
    ResearchRequest,
    ResearchResult,
)


_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"RIFF", "image/webp"),
)


class OciGrokResponsesAdapter:
    """生产路径：对配置的 OCI Responses 端点签发 POST，认证走 IAM。"""

    def __init__(self, *, timeout_seconds: int = 300):
        self._timeout_seconds = timeout_seconds

    async def research(
        self, request: ResearchRequest, material: dict[str, Any],
    ) -> ResearchResult:
        body = {
            "model": material["provider_model_name"],
            "input": request.input,
            "tools": [self._x_search_tool(request)],
        }
        payload = await self._invoke(material, body)
        return parse_research_response(payload)

    async def generate_image(
        self, request: ImageGenerationRequest, material: dict[str, Any],
    ) -> ImageGenerationResult:
        body: dict[str, Any] = {
            "model": material["provider_model_name"],
            "input": request.prompt,
            "tools": [{
                "type": "image_generation",
                "aspect_ratio": request.aspect_ratio,
            }],
        }
        if request.count > 1:
            body["instructions"] = f"Generate exactly {request.count} images."
        payload = await self._invoke(material, body)
        return parse_image_response(payload)

    async def _invoke(
        self, material: dict[str, Any], body: dict[str, Any],
    ) -> dict[str, Any]:
        provider = str(material.get("provider") or "").strip().upper()
        endpoint = str(material.get("api_endpoint") or "").strip().rstrip("/")
        if provider != "OCI" or not endpoint:
            raise GenerativeAdapterError(
                "PROVIDER_UNAVAILABLE", "当前模型缺少可用的 OCI Responses 端点",
                status_code=503,
            )
        project = oci_generative_ai_project(material.get("model_params"))
        try:
            signer = self._signer(material)
        except ValueError as exc:
            raise GenerativeAdapterError(
                "PROVIDER_UNAVAILABLE", "当前模型的 OCI 认证材料不完整",
                status_code=503,
            ) from exc
        url = build_oci_responses_url(endpoint)

        def _post() -> requests.Response:
            return requests.post(
                url,
                json=body,
                auth=signer,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "OpenAI-Project": project,
                },
                timeout=self._timeout_seconds,
            )

        try:
            response = await asyncio.to_thread(_post)
        except requests.Timeout as exc:
            raise GenerativeAdapterError(
                "PROVIDER_TIMEOUT", "上游 Responses 调用超时", status_code=504,
            ) from exc
        except requests.RequestException as exc:
            logger.warning("OCI Responses 网络失败：{}", type(exc).__name__)
            raise GenerativeAdapterError(
                "PROVIDER_UNAVAILABLE", "上游 Responses 暂时不可用", status_code=503,
            ) from exc
        return self._parse_http(response)

    @staticmethod
    def _signer(material: dict[str, Any]):
        import oci

        config = validated_oci_config((material.get("model_params") or {}).get("config_file"))
        return oci.signer.Signer(
            tenancy=config["tenancy"],
            user=config["user"],
            fingerprint=config["fingerprint"],
            private_key_file_location=config.get("key_file"),
            pass_phrase=config.get("pass_phrase"),
            private_key_content=config.get("key_content"),
        )

    @staticmethod
    def _x_search_tool(request: ResearchRequest) -> dict[str, Any]:
        tool: dict[str, Any] = {"type": "x_search"}
        if request.from_date is not None:
            tool["from_date"] = request.from_date.isoformat()
        if request.to_date is not None:
            tool["to_date"] = request.to_date.isoformat()
        if request.allowed_x_handles:
            tool["allowed_x_handles"] = list(request.allowed_x_handles)
        if request.excluded_x_handles:
            tool["excluded_x_handles"] = list(request.excluded_x_handles)
        if request.enable_image_understanding:
            tool["enable_image_understanding"] = True
        if request.enable_video_understanding:
            tool["enable_video_understanding"] = True
        return tool

    @staticmethod
    def _parse_http(response: requests.Response) -> dict[str, Any]:
        status = int(response.status_code)
        try:
            payload = response.json()
        except ValueError:
            payload = {"message": (response.text or "")[:500]}
        if status < 400:
            if not isinstance(payload, dict):
                raise GenerativeAdapterError(
                    "PROVIDER_UNAVAILABLE", "上游 Responses 返回了无法解析的正文",
                    status_code=502,
                )
            return payload
        code, message = provider_error_fields(payload)
        logger.warning(
            "OCI Responses 调用失败：status={} code={} message={}",
            status, code or "-", message or "-",
        )
        raise classify_provider_http_error(status, payload)


def build_oci_responses_url(endpoint: str) -> str:
    """把配置的 inference 地址规范成实际 POST 的 Responses URL。"""
    base = str(endpoint or "").strip().rstrip("/")
    if not base:
        raise GenerativeAdapterError(
            "PROVIDER_UNAVAILABLE", "当前模型缺少可用的 OCI Responses 端点",
            status_code=503,
        )
    lowered = base.lower()
    if lowered.endswith("/openai/v1/responses"):
        return base
    if lowered.endswith("/openai/v1"):
        return f"{base}/responses"
    if lowered.endswith("/openai"):
        return f"{base}/v1/responses"
    path = (urlparse(base).path or "").rstrip("/")
    if path in {"", "/"}:
        return f"{base}/openai/v1/responses"
    if path == "/v1":
        # 旧 Chat host 误带了 /v1，才改写到 OpenAI 兼容路径。
        return f"{base[:-3]}/openai/v1/responses"
    # /20231130/actions/v1 本身就是 Responses 地址，不得再拼 /openai/v1/responses。
    return base


def oci_generative_ai_project(model_params: Any) -> str:
    """读取 Responses 调用所需的 Generative AI project OCID。"""
    if not isinstance(model_params, dict):
        raise GenerativeAdapterError(
            "PROVIDER_UNAVAILABLE", "当前模型缺少 Generative AI project",
            status_code=503,
        )
    project = str(model_params.get("project") or "").strip()
    if not project:
        raise GenerativeAdapterError(
            "PROVIDER_UNAVAILABLE", "当前模型缺少 Generative AI project",
            status_code=503,
        )
    return project


def provider_error_fields(payload: Any) -> tuple[str | None, str | None]:
    """只取出上游错误码和短消息，避免把完整正文或凭据写入日志。"""
    if isinstance(payload, str):
        text = payload.strip()
        return None, text[:200] if text else None
    if not isinstance(payload, dict):
        return None, None
    error = payload.get("error") if isinstance(payload.get("error"), dict) else payload
    if not isinstance(error, dict):
        return None, None
    code = error.get("code") or error.get("error_code") or payload.get("code")
    message = error.get("message") or payload.get("message")
    code_text = str(code).strip() if code is not None else ""
    message_text = str(message).strip() if message is not None else ""
    return (
        code_text[:64] or None,
        message_text[:200] or None,
    )


def classify_provider_http_error(status: int, payload: Any) -> GenerativeAdapterError:
    text = json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload
    lowered = text.lower()
    if status in {408, 504}:
        return GenerativeAdapterError("PROVIDER_TIMEOUT", "上游 Responses 调用超时", status_code=504)
    if status == 429:
        return GenerativeAdapterError("PROVIDER_QUOTA_EXHAUSTED", "上游配额已耗尽", status_code=429)
    if "content" in lowered and any(token in lowered for token in ("reject", "safety", "moderation", "policy")):
        return GenerativeAdapterError("CONTENT_REJECTED", "内容安全策略拒绝了本次请求", status_code=422)
    if any(token in lowered for token in ("unsupported", "unknown tool", "invalid tool", "x_search", "image_generation")) and status in {400, 404, 422}:
        if "tool" in lowered or "x_search" in lowered or "image_generation" in lowered:
            return GenerativeAdapterError(
                "PROVIDER_UNSUPPORTED_TOOL", "上游拒绝 x_search 或 image_generation 工具",
                status_code=422,
            )
    if status in {401, 403, 404, 502, 503}:
        return GenerativeAdapterError("PROVIDER_UNAVAILABLE", "上游 Responses 暂时不可用", status_code=503)
    return GenerativeAdapterError("PROVIDER_UNAVAILABLE", "上游 Responses 调用失败", status_code=502)


def parse_research_response(payload: dict[str, Any]) -> ResearchResult:
    status = _terminal_status(payload, image=False)
    error_code = None
    if status == "FAILED":
        error_code = "PROVIDER_UNAVAILABLE"
    answer = _extract_output_text(payload)
    citations = tuple(_extract_citations(payload))
    return ResearchResult(
        status=status,
        answer=answer or None,
        citations=citations,
        usage=_extract_usage(payload),
        provider_request_id=_provider_request_id(payload),
        error_code=error_code,
    )


def parse_image_response(payload: dict[str, Any]) -> ImageGenerationResult:
    artifacts = tuple(_extract_images(payload))
    refusal = _is_content_rejected(payload)
    if refusal:
        return ImageGenerationResult(
            status="REJECTED",
            artifacts=(),
            usage=_extract_usage(payload),
            provider_request_id=_provider_request_id(payload),
            error_code="CONTENT_REJECTED",
        )
    if _terminal_status(payload, image=True) == "FAILED" or not artifacts:
        return ImageGenerationResult(
            status="FAILED",
            artifacts=(),
            usage=_extract_usage(payload),
            provider_request_id=_provider_request_id(payload),
            error_code="PROVIDER_UNAVAILABLE",
        )
    return ImageGenerationResult(
        status="COMPLETED",
        artifacts=artifacts,
        usage=_extract_usage(payload),
        provider_request_id=_provider_request_id(payload),
    )


def _provider_request_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("id") or payload.get("request_id")
    return str(value) if value else None


def _extract_usage(payload: dict[str, Any]) -> ModelUsage | None:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return None
    provider_usage = {
        str(key): value
        for key, value in raw.items()
        if isinstance(value, (int, float, str))
    }
    return ModelUsage(
        input_tokens=_optional_int(raw.get("input_tokens") or raw.get("prompt_tokens")),
        output_tokens=_optional_int(raw.get("output_tokens") or raw.get("completion_tokens")),
        total_tokens=_optional_int(raw.get("total_tokens")),
        provider_usage=provider_usage,
    )


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _extract_output_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                        text = part.get("text")
                        if text:
                            chunks.append(str(text))
            text = item.get("text")
            if item.get("type") in {"message", "output_text"} and isinstance(text, str) and text:
                chunks.append(text)
    if not chunks and isinstance(payload.get("output_text"), str):
        chunks.append(payload["output_text"])
    return "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip())


def _extract_citations(payload: dict[str, Any]) -> list[ResearchCitation]:
    rows: list[ResearchCitation] = []
    seen: set[str] = set()
    raw_items = []
    if isinstance(payload.get("citations"), list):
        raw_items.extend(payload["citations"])
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("annotations"), list):
                        raw_items.extend(part["annotations"])
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("canonical_url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        citation_id = str(
            item.get("id") or item.get("provider_citation_id") or f"x-citation-{index}"
        )
        published = item.get("published_at") or item.get("created_at")
        rows.append(ResearchCitation(
            provider_citation_id=citation_id[:256],
            canonical_url=url[:2048],
            title=_optional_str(item.get("title"), 512),
            excerpt=_optional_str(item.get("excerpt") or item.get("snippet") or item.get("text"), 4000),
            author_handle=_optional_str(
                item.get("author_handle") or item.get("username") or item.get("author"),
                128,
            ),
            published_at=published if hasattr(published, "tzinfo") else None,
        ))
    return rows


def _extract_images(payload: dict[str, Any]) -> list[GeneratedImageArtifact]:
    artifacts: list[GeneratedImageArtifact] = []
    output = payload.get("output")
    if not isinstance(output, list):
        return artifacts
    for item in output:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "image_generation_call":
            continue
        raw = item.get("result")
        content = _decode_image_bytes(raw)
        if not content:
            continue
        mime_type, content = _detect_image(content)
        artifacts.append(GeneratedImageArtifact(
            provider_artifact_id=_optional_str(item.get("id"), 256),
            mime_type=mime_type,
            content=content,
            width=_optional_int(item.get("width")),
            height=_optional_int(item.get("height")),
        ))
    return artifacts


def _decode_image_bytes(raw: Any) -> bytes:
    if isinstance(raw, bytes) and raw:
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return b""
    text = raw.strip()
    if "base64," in text:
        text = text.split("base64,", 1)[1]
    try:
        return base64.b64decode(text, validate=False)
    except (ValueError, TypeError):
        return b""


def _detect_image(content: bytes) -> tuple[str, bytes]:
    if content.startswith(b"RIFF") and b"WEBP" in content[:16]:
        return "image/webp", content
    for magic, mime_type in _IMAGE_MAGIC:
        if magic != b"RIFF" and content.startswith(magic):
            return mime_type, content
    return "image/png", content


def _is_content_rejected(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or "").lower()
    if status in {"rejected", "cancelled"}:
        return True
    incomplete = payload.get("incomplete_details")
    if isinstance(incomplete, dict):
        reason = str(incomplete.get("reason") or "").lower()
        if any(token in reason for token in ("content", "safety", "moderation", "policy")):
            return True
    text = json.dumps(payload, ensure_ascii=False).lower()
    return "content_filter" in text or "content policy" in text


def _terminal_status(payload: dict[str, Any], *, image: bool) -> str:
    status = str(payload.get("status") or "completed").lower()
    if status in {"failed", "error", "cancelled"}:
        return "FAILED"
    if image and status in {"rejected"}:
        return "REJECTED"
    return "COMPLETED"


def _optional_str(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def new_local_request_id() -> str:
    return f"resp-{uuid4()}"
