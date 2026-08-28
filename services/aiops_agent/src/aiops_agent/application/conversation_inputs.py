"""把对话上传文件转换为可审计的用户证据材料。"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ResolvedConversationUpload:
    item_no: int
    upload_id: str
    file_name: str
    media_type: str
    byte_size: int
    content_hash: str
    payload_uri: str
    extracted_text: str
    extraction_mode: str
    model_id: UUID | None = None
    model_revision: str | None = None
    extraction_error: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationUploadSource:
    """在任何图片模型调用前冻结的原始上传文件描述。"""

    item_no: int
    upload_id: str
    file_name: str
    media_type: str
    byte_size: int
    content_hash: str
    payload_uri: str


class ConversationInputResolver:
    """解析文本或图片附件；单个附件失败只形成可见缺口。"""

    _TEXT_MEDIA_TYPES = frozenset(
        {"text/plain", "text/csv", "application/json", "application/sql"}
    )

    def __init__(
        self, *, upload_store, image_model_client=None, max_extracted_chars: int
    ) -> None:
        self._upload_store = upload_store
        self._image_model_client = image_model_client
        self._max_extracted_chars = max_extracted_chars

    async def resolve(
        self,
        *,
        domain_id: int,
        actor_id: str,
        content: tuple[dict, ...],
        image_capabilities: dict[str, Any],
    ) -> tuple[tuple[dict, ...], tuple[ResolvedConversationUpload, ...]]:
        normalized: list[dict] = []
        uploads: list[ResolvedConversationUpload] = []
        for item_no, item in enumerate(content, start=1):
            upload_id = item.get("upload_id")
            if not upload_id:
                normalized.append(dict(item))
                continue
            stored = self._upload_store.get(
                upload_id=str(upload_id),
                domain_id=domain_id,
                actor_id=actor_id,
            )
            stored = self._upload_store.preserve(stored)
            raw = self._upload_store.read(stored)
            resolved = await self._extract(
                item_no=item_no,
                stored=stored,
                raw=raw,
                image_capabilities=image_capabilities,
            )
            uploads.append(resolved)
            normalized.append(
                {
                    "content_type": item.get("content_type", "FILE"),
                    "upload_id": stored.upload_id,
                    "file_name": stored.file_name,
                    "media_type": stored.media_type,
                    "text": resolved.extracted_text,
                    "extraction_mode": resolved.extraction_mode,
                    **(
                        {"extraction_error": resolved.extraction_error}
                        if resolved.extraction_error
                        else {}
                    ),
                }
            )
        return tuple(normalized), tuple(uploads)

    def describe_sources(
        self,
        *,
        domain_id: int,
        actor_id: str,
        content: tuple[dict, ...],
    ) -> tuple[ConversationUploadSource, ...]:
        """只读取受控上传元数据，不执行文本提取或模型调用。"""
        sources = []
        for item_no, item in enumerate(content, start=1):
            upload_id = item.get("upload_id")
            if not upload_id:
                continue
            stored = self._upload_store.get(
                upload_id=str(upload_id),
                domain_id=domain_id,
                actor_id=actor_id,
            )
            stored = self._upload_store.preserve(stored)
            sources.append(
                ConversationUploadSource(
                    item_no=item_no,
                    upload_id=stored.upload_id,
                    file_name=stored.file_name,
                    media_type=stored.media_type,
                    byte_size=stored.byte_size,
                    content_hash=stored.content_hash,
                    payload_uri=stored.payload_uri,
                )
            )
        return tuple(sources)

    async def _extract(
        self, *, item_no: int, stored, raw: bytes, image_capabilities: dict
    ) -> ResolvedConversationUpload:
        common = {
            "item_no": item_no,
            "upload_id": stored.upload_id,
            "file_name": stored.file_name,
            "media_type": stored.media_type,
            "byte_size": stored.byte_size,
            "content_hash": stored.content_hash,
            "payload_uri": stored.payload_uri,
        }
        if stored.media_type in self._TEXT_MEDIA_TYPES:
            try:
                text = self._bounded(raw.decode("utf-8-sig"))
                return ResolvedConversationUpload(
                    **common,
                    extracted_text=text,
                    extraction_mode="TEXT_DECODE",
                )
            except UnicodeDecodeError:
                return ResolvedConversationUpload(
                    **common,
                    extracted_text=(
                        f"文件 {stored.file_name} 不是有效 UTF-8 文本，未能提取正文。"
                    ),
                    extraction_mode="TEXT_DECODE",
                    extraction_error="INPUT_TEXT_ENCODING_INVALID",
                )
        mode, capability = self._image_capability(image_capabilities)
        if mode is None or capability is None or self._image_model_client is None:
            return ResolvedConversationUpload(
                **common,
                extracted_text=(
                    f"用户提供了图片 {stored.file_name}，但当前 Agent 未配置 OCR/VLM 能力。"
                ),
                extraction_mode="UNAVAILABLE",
                extraction_error="IMAGE_MODEL_UNAVAILABLE",
            )
        model_id = UUID(str(capability["default_model_id"]))
        try:
            result = await self._image_model_client.process(
                mode=mode,
                model_id=model_id,
                mime_type=stored.media_type,
                content_base64=base64.b64encode(raw).decode("ascii"),
            )
            text = self._bounded(str(result.get("text") or "").strip())
            if not text:
                raise ValueError("图片模型没有返回可用文字")
            return ResolvedConversationUpload(
                **common,
                extracted_text=text,
                extraction_mode=mode,
                model_id=model_id,
                model_revision=str(result.get("model_revision") or model_id),
            )
        except Exception as exc:
            return ResolvedConversationUpload(
                **common,
                extracted_text=f"图片 {stored.file_name} 解析失败，仍保留原始图片证据。",
                extraction_mode=mode,
                model_id=model_id,
                extraction_error=f"IMAGE_EXTRACTION_FAILED:{type(exc).__name__}",
            )

    @staticmethod
    def _image_capability(
        image_capabilities: dict[str, Any],
    ) -> tuple[str | None, dict | None]:
        for key, mode in (("vlm", "VLM"), ("ocr", "OCR")):
            capability = dict(image_capabilities.get(key) or {})
            if capability.get("default_model_id"):
                return mode, capability
        return None, None

    def _bounded(self, text: str) -> str:
        if len(text) <= self._max_extracted_chars:
            return text
        return (
            text[: self._max_extracted_chars]
            + "\n\n[附件正文已按单轮输入上限截断]"
        )


__all__ = [
    "ConversationInputResolver",
    "ConversationUploadSource",
    "ResolvedConversationUpload",
]
