"""通过 OCI 签名调用 Grok 4.6 Responses，不走 Generic Chat。"""

from __future__ import annotations

import asyncio
import base64
import json
import re
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
        # Studio 有明确 prompt 和比例，优先走 Images API；工具路径没有尺寸字段。
        images_result = await self._try_images_api(request, material)
        if images_result is not None:
            return images_result
        prompt = request.prompt.strip()
        if request.aspect_ratio:
            prompt = f"{prompt}\n\nGenerate the image in a {request.aspect_ratio} aspect ratio."
        if request.count > 1:
            prompt = f"{prompt}\nGenerate exactly {request.count} images."
        body: dict[str, Any] = {
            "model": material["provider_model_name"],
            "input": prompt,
            "tools": [{"type": "image_generation"}],
        }
        payload = await self._invoke(material, body)
        return await self._image_result_from_payload(payload)

    async def _try_images_api(
        self, request: ImageGenerationRequest, material: dict[str, Any],
    ) -> ImageGenerationResult | None:
        """xAI 对精确比例的文生图应走 images/generations，而不是 Responses 工具。"""
        endpoint = str(material.get("api_endpoint") or "").strip().rstrip("/")
        if str(material.get("provider") or "").strip().upper() != "OCI" or not endpoint:
            return None
        body: dict[str, Any] = {
            "model": material["provider_model_name"],
            "prompt": request.prompt.strip(),
            "n": request.count,
            "response_format": "b64_json",
        }
        if request.aspect_ratio:
            body["aspect_ratio"] = request.aspect_ratio
        for url in build_oci_images_urls(endpoint):
            path = urlparse(url).path or url
            try:
                payload = await self._invoke(material, body, url=url)
            except GenerativeAdapterError as exc:
                logger.warning(
                    "OCI Images API 不可用：path={} code={} status={}",
                    path, exc.code, exc.status_code,
                )
                if exc.code in {"CONTENT_REJECTED", "PROVIDER_QUOTA_EXHAUSTED"}:
                    raise
                continue
            result = await self._image_result_from_payload(payload)
            if result.status == "COMPLETED" and result.artifacts:
                logger.info("OCI Images API 已返回图片：path={} count={}", path, len(result.artifacts))
                return result
            logger.warning("OCI Images API 没有返回可保存的图片：path={}", path)
        return None

    async def _image_result_from_payload(
        self, payload: dict[str, Any],
    ) -> ImageGenerationResult:
        artifacts = _extract_images(payload)
        if not artifacts:
            artifacts = await asyncio.to_thread(
                _download_image_artifacts, _image_urls_from_payload(payload),
            )
        return build_image_result(payload, tuple(artifacts))

    async def _invoke(
        self, material: dict[str, Any], body: dict[str, Any], *, url: str | None = None,
    ) -> dict[str, Any]:
        provider = str(material.get("provider") or "").strip().upper()
        endpoint = str(material.get("api_endpoint") or "").strip().rstrip("/")
        if provider != "OCI" or not endpoint:
            raise GenerativeAdapterError(
                "PROVIDER_UNAVAILABLE", "当前模型缺少可用的 OCI Responses 端点",
                status_code=503,
            )
        model_params = material.get("model_params")
        project = oci_generative_ai_project(model_params)
        compartment_id = oci_compartment_id(model_params)
        try:
            signer = self._signer(material)
        except ValueError as exc:
            raise GenerativeAdapterError(
                "PROVIDER_UNAVAILABLE", "当前模型的 OCI 认证材料不完整",
                status_code=503,
            ) from exc
        url = url or build_oci_responses_url(endpoint)

        def _post() -> requests.Response:
            return requests.post(
                url,
                json=body,
                auth=signer,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "OpenAI-Project": project,
                    "CompartmentId": compartment_id,
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
        return self._parse_http(response, url=url)

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
    def _parse_http(response: requests.Response, *, url: str = "") -> dict[str, Any]:
        status = int(response.status_code)
        payload = decode_oci_http_payload(response)
        if status < 400:
            if not isinstance(payload, dict):
                raise GenerativeAdapterError(
                    "PROVIDER_UNAVAILABLE", "上游 Responses 返回了无法解析的正文",
                    status_code=502,
                )
            if payload_has_provider_error(payload):
                code, message = provider_error_fields(payload)
                logger.warning(
                    "OCI Responses 以成功状态返回了错误正文：url={} status={} code={} message={}",
                    url or "-", status, code or "-", message or "-",
                )
                raise classify_provider_http_error(400, payload)
            return payload
        code, message = provider_error_fields(payload)
        logger.warning(
            "OCI Responses 调用失败：url={} status={} code={} message={}",
            url or "-", status, code or "-", message or "-",
        )
        raise classify_provider_http_error(status, payload)


def decode_oci_http_payload(response: requests.Response) -> dict[str, Any]:
    """把上游正文解析成 JSON；Grok 文生图成功时也可能直接返回图片字节。"""
    content = getattr(response, "content", None)
    content_type = _response_content_type(response)
    if isinstance(content, (bytes, bytearray)) and content:
        raw = bytes(content)
        image_bytes = _image_bytes_from_body(raw)
        if image_bytes:
            mime_type, _ = _detect_image(image_bytes)
            logger.info(
                "OCI Responses 返回了图片二进制正文：mime={} bytes={}",
                mime_type, len(image_bytes),
            )
            return _payload_from_image_bytes(image_bytes)
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError, TypeError):
            logger.warning(
                "OCI Responses 正文既不是 JSON 也不是可识别图片：content_type={} bytes={} prefix={}",
                content_type or "-",
                len(raw),
                raw[:12].hex(),
            )
            try:
                message = raw.decode("utf-8")[:500]
            except UnicodeDecodeError:
                message = "上游返回了非 JSON 正文"
            return {"message": message}
        return payload if isinstance(payload, dict) else {"message": str(payload)[:500]}
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError, TypeError):
        try:
            text = response.text or ""
        except UnicodeDecodeError as exc:
            raise GenerativeAdapterError(
                "PROVIDER_UNAVAILABLE", "上游 Responses 返回了无法解析的正文",
                status_code=502,
            ) from exc
        payload = {"message": text[:500]}
    return payload if isinstance(payload, dict) else {"message": str(payload)[:500]}


def _response_content_type(response: requests.Response) -> str:
    headers = getattr(response, "headers", None) or {}
    try:
        return str(headers.get("Content-Type") or headers.get("content-type") or "")
    except Exception:
        return ""


def _image_bytes_from_body(content: bytes) -> bytes:
    """识别图片正文；允许文件头前只有空白。"""
    if _looks_like_image(content):
        return content
    stripped = content.lstrip(b"\x00 \t\r\n")
    if stripped != content and _looks_like_image(stripped):
        return stripped
    return b""


def _looks_like_image(content: bytes) -> bool:
    if content.startswith(b"RIFF") and b"WEBP" in content[:16]:
        return True
    return content.startswith(b"\x89PNG\r\n\x1a\n") or content.startswith(b"\xff\xd8\xff")


def _payload_from_image_bytes(content: bytes) -> dict[str, Any]:
    return {
        "status": "completed",
        "output": [{
            "type": "image_generation_call",
            "result": base64.b64encode(content).decode("ascii"),
        }],
    }


def build_oci_responses_url(endpoint: str) -> str:
    """把配置的 inference 地址规范成实际 POST 的 Responses URL。"""
    base = str(endpoint or "").strip().rstrip("/")
    if not base:
        raise GenerativeAdapterError(
            "PROVIDER_UNAVAILABLE", "当前模型缺少可用的 OCI Responses 端点",
            status_code=503,
        )
    lowered = base.lower()
    if lowered.endswith("/responses"):
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
    if path.endswith("/20231130/actions/v1"):
        # Codex 把该路径当 Responses base_url，实际 POST 还要加 /responses。
        return f"{base}/responses"
    return base


def build_oci_images_urls(endpoint: str) -> tuple[str, ...]:
    """从同一 inference 地址推导 Images API 候选，不猜测模型名。"""
    base = str(endpoint or "").strip().rstrip("/")
    if not base:
        return ()
    urls: list[str] = []
    try:
        responses = build_oci_responses_url(base)
    except GenerativeAdapterError:
        responses = ""
    if responses.lower().endswith("/responses"):
        urls.append(f"{responses[:-len('responses')]}images/generations")
    parsed = urlparse(base if "://" in base else f"https://{base}")
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    if origin:
        openai_url = f"{origin}/openai/v1/images/generations"
        if openai_url not in urls:
            urls.append(openai_url)
    return tuple(urls)


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


def oci_compartment_id(model_params: Any) -> str:
    """读取 Responses 调用所需的 compartment，与 Chat 使用同一 MODEL_PARAMS 字段。"""
    if not isinstance(model_params, dict):
        raise GenerativeAdapterError(
            "PROVIDER_UNAVAILABLE", "当前模型缺少 compartment",
            status_code=503,
        )
    compartment_id = str(model_params.get("compartment_id") or "").strip()
    if not compartment_id:
        raise GenerativeAdapterError(
            "PROVIDER_UNAVAILABLE", "当前模型缺少 compartment",
            status_code=503,
        )
    return compartment_id


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


def payload_has_provider_error(payload: dict[str, Any]) -> bool:
    """HTTP 200 也可能带 error 对象，不能再当成功图片响应解析。"""
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    code, message = provider_error_fields(payload)
    return bool(code or message)


def parse_image_response(payload: dict[str, Any]) -> ImageGenerationResult:
    return build_image_result(payload, tuple(_extract_images(payload)))


def build_image_result(
    payload: dict[str, Any], artifacts: tuple[GeneratedImageArtifact, ...],
) -> ImageGenerationResult:
    """把已提取的图片字节收成终态；没有字节时只记录结构。"""
    excerpt = _safe_output_excerpt(payload)
    refusal = _is_content_rejected(payload)
    if refusal:
        return ImageGenerationResult(
            status="REJECTED",
            artifacts=(),
            usage=_extract_usage(payload),
            provider_request_id=_provider_request_id(payload),
            error_code="CONTENT_REJECTED",
            error_message=excerpt or None,
        )
    if _terminal_status(payload, image=True) == "FAILED" or not artifacts:
        if _image_tool_failed(payload):
            logger.warning("OCI 文生图工具执行失败：{}", summarize_image_payload(payload))
        elif not artifacts:
            logger.warning("OCI 文生图响应没有可解析的图片：{}", summarize_image_payload(payload))
        return ImageGenerationResult(
            status="FAILED",
            artifacts=(),
            usage=_extract_usage(payload),
            provider_request_id=_provider_request_id(payload),
            error_code="PROVIDER_UNAVAILABLE",
            error_message=excerpt or None,
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


def summarize_image_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """只记录结构，不把图片字节或 URL 写入日志。"""
    output = payload.get("output")
    items = _iter_output_items(payload)
    summarized: list[dict[str, Any]] = []
    for item in items[:12]:
        content = item.get("content")
        content_types: list[str] = []
        if isinstance(content, list):
            for part in content[:8]:
                if isinstance(part, dict):
                    content_types.append(str(part.get("type") or "")[:64] or type(part).__name__)
                else:
                    content_types.append(type(part).__name__)
        summarized.append({
            "type": str(item.get("type") or "")[:64],
            "status": str(item.get("status") or "")[:64],
            "keys": [str(key) for key in item.keys()][:20],
            "fields": {
                str(key)[:64]: _value_shape(value)
                for key, value in list(item.items())[:20]
            },
            "result_kind": _result_kind(item.get("result")),
            "content_types": content_types,
            "has_prompt": bool(str(item.get("prompt") or "").strip()),
        })
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "keys": [str(key) for key in payload.keys()][:20],
        "status": str(payload.get("status") or "")[:64],
        "truncation": str(payload.get("truncation") or "")[:64],
        "output_type": type(output).__name__,
        "item_types": [row["type"] for row in summarized],
        "result_kinds": [row["result_kind"] for row in summarized],
        "items": summarized,
        "has_error": isinstance(payload.get("error"), dict),
        "incomplete_reason": (
            str((payload.get("incomplete_details") or {}).get("reason"))[:64]
            if isinstance(payload.get("incomplete_details"), dict)
            else ""
        ),
        "usage_keys": [str(key) for key in usage.keys()][:12],
        "text_kind": _value_shape(payload.get("text")),
        "image_url_count": len(_image_urls_from_payload(payload)),
        "image_call_statuses": [
            row["status"] for row in summarized
            if row["type"].endswith("image_generation_call") or row["type"] == "image_generation"
        ],
        "output_excerpt": _safe_output_excerpt(payload),
    }


def _result_kind(raw: Any) -> str:
    return _value_shape(raw).split(":", 1)[0]


def _value_shape(raw: Any) -> str:
    """只描述字段形态和长度，不写入图片或 URL 正文。"""
    if raw is None:
        return "null"
    if isinstance(raw, bool):
        return "bool"
    if isinstance(raw, (int, float)):
        return type(raw).__name__
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith(("http://", "https://")):
            return f"url:{len(text)}"
        if text.startswith("data:image/"):
            return f"data_uri:{len(text)}"
        return f"string:{len(text)}"
    if isinstance(raw, (bytes, bytearray)):
        return f"bytes:{len(raw)}"
    if isinstance(raw, list):
        return f"list:{len(raw)}"
    if isinstance(raw, dict):
        keys = ",".join(str(key) for key in list(raw.keys())[:12])
        return f"dict:{keys}"[:96]
    return type(raw).__name__


_MARKDOWN_IMAGE_URL = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")

_NON_IMAGE_KEYS = {
    "id", "type", "status", "prompt", "revised_prompt", "action", "quality",
    "size", "role", "name", "index", "status_details", "incomplete_details",
    "error", "background", "output_format",
}


def _iter_output_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = payload.get("output")
    if isinstance(output, list):
        items = output
    elif isinstance(output, dict):
        items = [output]
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _extract_images(payload: dict[str, Any]) -> list[GeneratedImageArtifact]:
    artifacts: list[GeneratedImageArtifact] = []
    seen: set[bytes] = set()

    def add(content: bytes, item: dict[str, Any] | None = None) -> None:
        if not content or content in seen or not _looks_like_image(content):
            return
        seen.add(content)
        mime_type, content = _detect_image(content)
        artifacts.append(GeneratedImageArtifact(
            provider_artifact_id=_optional_str((item or {}).get("id"), 256),
            mime_type=mime_type,
            content=content,
            width=_optional_int((item or {}).get("width")),
            height=_optional_int((item or {}).get("height")),
        ))

    for item in _iter_image_call_items(payload):
        for raw in _image_result_candidates(item):
            add(_decode_image_bytes(raw), item)
    for item in _iter_output_items(payload):
        type_name = str(item.get("type") or "").strip().lower()
        if type_name.endswith("image_generation_call") or type_name == "image_generation":
            for key, value in item.items():
                if key in _NON_IMAGE_KEYS:
                    continue
                add(_decode_image_bytes(value), item)
        elif type_name in {"message", "output_message"}:
            add(_decode_image_bytes(item.get("content")), item)
        elif "image" in type_name:
            for key, value in item.items():
                if key in _NON_IMAGE_KEYS:
                    continue
                add(_decode_image_bytes(value), item)
    if not artifacts:
        for raw in _image_result_candidates(payload):
            add(_decode_image_bytes(raw))
    return artifacts


def _iter_image_call_items(node: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def walk(value: Any, *, depth: int) -> None:
        if depth > 6 or value is None:
            return
        if isinstance(value, list):
            for child in value:
                walk(child, depth=depth + 1)
            return
        if not isinstance(value, dict):
            return
        type_name = str(value.get("type") or "").strip().lower()
        if type_name.endswith("image_generation_call") or type_name == "image_generation":
            items.append(value)
        for key in ("output", "data", "content", "images", "body", "response"):
            if key in value:
                walk(value[key], depth=depth + 1)

    walk(node, depth=0)
    return items


def _image_result_candidates(item: dict[str, Any]) -> list[Any]:
    return [
        item[key]
        for key in ("result", "b64_json", "image", "output", "content", "data", "images", "image_url")
        if key in item
    ]


def _decode_image_bytes(raw: Any) -> bytes:
    if isinstance(raw, bytes) and raw:
        return raw
    if isinstance(raw, list):
        for item in raw:
            content = _decode_image_bytes(item)
            if content:
                return content
        return b""
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in _NON_IMAGE_KEYS:
                continue
            content = _decode_image_bytes(value)
            if content:
                return content
        return b""
    if not isinstance(raw, str) or not raw.strip():
        return b""
    text = raw.strip()
    if text.startswith(("http://", "https://")):
        return b""
    if "base64," in text:
        text = text.split("base64,", 1)[1]
        text = re.split(r"[^A-Za-z0-9+/=]+", text, maxsplit=1)[0]
    try:
        decoded = base64.b64decode(text, validate=False)
    except (ValueError, TypeError):
        return b""
    return decoded if decoded and _looks_like_image(decoded) else b""


def _image_urls_from_payload(payload: dict[str, Any]) -> list[str]:
    """收集文生图条目里的 https 图片地址，不记录完整 URL。"""
    urls: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text.startswith(("http://", "https://")) or text in seen:
            return
        seen.add(text)
        urls.append(text)

    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            add(item.get("url"))
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                add(image_url.get("url"))
            else:
                add(image_url)

    def walk(value: Any, *, depth: int, take: bool) -> None:
        if depth > 6 or value is None:
            return
        if isinstance(value, list):
            for child in value:
                walk(child, depth=depth + 1, take=take)
            return
        if isinstance(value, dict):
            type_name = str(value.get("type") or "").strip().lower()
            nested_take = take or type_name.endswith("image_generation_call") or "image" in type_name
            for key, child in value.items():
                key_take = nested_take or key in {"image_url", "url", "result", "image"}
                if key_take and isinstance(child, str):
                    add(child)
                walk(child, depth=depth + 1, take=key_take)
            return
        if take:
            add(value)
            return
        if isinstance(value, str):
            for match in _MARKDOWN_IMAGE_URL.finditer(value):
                add(match.group(1))

    for item in _iter_output_items(payload):
        type_name = str(item.get("type") or "").strip().lower()
        take = type_name.endswith("image_generation_call") or "image" in type_name
        if take:
            walk(item, depth=0, take=True)
        elif type_name in {"message", "output_message"}:
            walk(item.get("content"), depth=0, take=False)
    return urls[:4]


def _url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "-")[:128]
    except Exception:
        return "-"


def _download_image_artifacts(urls: list[str]) -> list[GeneratedImageArtifact]:
    artifacts: list[GeneratedImageArtifact] = []
    limit = 32 * 1024 * 1024
    for url in urls:
        host = _url_host(url)
        try:
            response = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            logger.warning("OCI 文生图结果 URL 下载失败：host={} error={}", host, type(exc).__name__)
            continue
        content = getattr(response, "content", b"") or b""
        content_type = ""
        headers = getattr(response, "headers", None) or {}
        try:
            content_type = str(headers.get("Content-Type") or headers.get("content-type") or "")
        except Exception:
            content_type = ""
        if getattr(response, "status_code", 0) != 200 or not content:
            logger.warning(
                "OCI 文生图结果 URL 下载失败：host={} status={} bytes={}",
                host, getattr(response, "status_code", 0), len(content),
            )
            continue
        if len(content) > limit:
            logger.warning("OCI 文生图结果 URL 超过大小限制：host={} bytes={}", host, len(content))
            continue
        if not _looks_like_image(content):
            logger.warning(
                "OCI 文生图结果 URL 不是图片：host={} content_type={} bytes={} prefix={}",
                host, content_type or "-", len(content), content[:12].hex(),
            )
            continue
        mime_type, content = _detect_image(content)
        artifacts.append(GeneratedImageArtifact(mime_type=mime_type, content=content))
    return artifacts


def _detect_image(content: bytes) -> tuple[str, bytes]:
    if content.startswith(b"RIFF") and b"WEBP" in content[:16]:
        return "image/webp", content
    for magic, mime_type in _IMAGE_MAGIC:
        if magic != b"RIFF" and content.startswith(magic):
            return mime_type, content
    return "image/png", content


_DATA_URI_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", re.I)
_HTTP_URL_RE = re.compile(r"https?://\S+", re.I)
_REJECT_TOKENS = (
    "content_filter",
    "content policy",
    "content_policy",
    "moderation",
    "safety policy",
    "not allowed",
    "prohibited",
    "拒绝生成",
    "内容安全",
)


def _safe_output_excerpt(payload: dict[str, Any], *, limit: int = 200) -> str:
    """截取模型说明，去掉 URL 和图片字节，避免把 prompt 或密钥写入日志。"""
    text = _extract_output_text(payload)
    if not text:
        return ""
    text = _DATA_URI_RE.sub("[image]", text)
    text = _HTTP_URL_RE.sub("[url]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _image_tool_failed(payload: dict[str, Any]) -> bool:
    for item in _iter_image_call_items(payload):
        if str(item.get("status") or "").strip().lower() == "failed":
            return True
    return False


def _is_content_rejected(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or "").lower()
    if status in {"rejected", "cancelled"}:
        return True
    incomplete = payload.get("incomplete_details")
    if isinstance(incomplete, dict):
        reason = str(incomplete.get("reason") or "").lower()
        if any(token in reason for token in ("content", "safety", "moderation", "policy")):
            return True
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    message = str((error or {}).get("message") or payload.get("message") or "").lower()
    code = str((error or {}).get("code") or "").lower()
    text = f"{code} {message} {_extract_output_text(payload).lower()}"
    return any(token in text for token in _REJECT_TOKENS)


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
