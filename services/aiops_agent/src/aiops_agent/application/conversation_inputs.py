"""把对话上传文件转换为可审计的用户证据材料。"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from uuid import UUID

from loguru import logger


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
    searchable_payload_uri: str | None = None
    searchable_content_hash: str | None = None
    searchable_byte_size: int = 0
    extracted_char_count: int = 0
    line_count: int = 0
    model_id: UUID | None = None
    model_revision: str | None = None
    prompt_ref: dict[str, str] | None = None
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


class _HtmlEvidenceExtractor(HTMLParser):
    """从 AWR/ASH HTML 中提取可供调查的可见正文，不执行页面内容。"""

    _BLOCK_TAGS = frozenset(
        {
            "article", "br", "caption", "div", "h1", "h2", "h3", "h4",
            "h5", "h6", "li", "p", "pre", "table", "td", "th", "tr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and normalized in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif not self._ignored_depth and normalized in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


class ConversationInputResolver:
    """解析文本或图片附件；单个附件失败只形成可见缺口。"""

    _TEXT_MEDIA_TYPES = frozenset(
        {
            "text/plain", "text/html", "application/xhtml+xml", "text/csv",
            "application/json", "application/sql",
        }
    )

    def __init__(
        self,
        *,
        upload_store,
        image_model_client=None,
        prompt_registry=None,
        max_extracted_chars: int,
    ) -> None:
        self._upload_store = upload_store
        self._image_model_client = image_model_client
        self._prompt_registry = prompt_registry
        self._max_extracted_chars = max_extracted_chars

    async def resolve(
        self,
        *,
        domain_id: int,
        actor_id: str,
        content: tuple[dict, ...],
        image_capabilities: dict[str, Any],
        run_id: str | None = None,
        agent_id: str | None = None,
        trace_id: str | None = None,
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
                run_id=run_id,
                agent_id=agent_id,
                trace_id=trace_id,
            )
            uploads.append(resolved)
            normalized.append(
                {
                    "content_type": item.get("content_type", "FILE"),
                    "upload_id": stored.upload_id,
                    "file_name": stored.file_name,
                    "media_type": stored.media_type,
                    "text": self._manifest_text(resolved),
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
        self,
        *,
        item_no: int,
        stored,
        raw: bytes,
        image_capabilities: dict,
        run_id: str | None,
        agent_id: str | None,
        trace_id: str | None,
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
                text = self._decode_text(raw)
                extraction_mode = "TEXT_DECODE"
                if stored.media_type in {"text/html", "application/xhtml+xml"}:
                    extractor = _HtmlEvidenceExtractor()
                    extractor.feed(text)
                    extractor.close()
                    text = extractor.text()
                    extraction_mode = "HTML_TEXT_EXTRACT"
                if not text:
                    raise ValueError("文件没有可用于诊断的正文")
                searchable_uri, searchable_hash, searchable_size = (
                    self._upload_store.store_searchable_text(
                        upload_id=stored.upload_id,
                        text=text,
                    )
                )
                return ResolvedConversationUpload(
                    **common,
                    extracted_text="",
                    extraction_mode=extraction_mode,
                    searchable_payload_uri=searchable_uri,
                    searchable_content_hash=searchable_hash,
                    searchable_byte_size=searchable_size,
                    extracted_char_count=len(text),
                    line_count=text.count("\n") + 1,
                )
            except (UnicodeDecodeError, ValueError) as exc:
                return ResolvedConversationUpload(
                    **common,
                    extracted_text=(
                        f"文件 {stored.file_name} 未能提取可用文本正文。"
                    ),
                    extraction_mode="TEXT_DECODE",
                    extraction_error=(
                        "INPUT_TEXT_ENCODING_INVALID"
                        if isinstance(exc, UnicodeDecodeError)
                        else "INPUT_TEXT_EXTRACTION_EMPTY"
                    ),
                )
        mode, capability = self._image_capability(image_capabilities)
        if mode is None or capability is None or self._image_model_client is None:
            logger.warning(
                "图片证据不可解析：未取得图片模型能力 | run_id={} | agent_id={} | trace_id={} | file_name={} | media_type={} | configured_capabilities={}",
                run_id,
                agent_id,
                trace_id,
                stored.file_name,
                stored.media_type,
                ",".join(sorted(image_capabilities)),
            )
            return ResolvedConversationUpload(
                **common,
                extracted_text=(
                    f"用户提供了图片 {stored.file_name}，但当前 Agent 未配置 OCR/VLM 能力。"
                ),
                extraction_mode="UNAVAILABLE",
                extraction_error="IMAGE_MODEL_UNAVAILABLE",
            )
        model_id = UUID(str(capability["default_model_id"]))
        logger.info(
            "图片证据准备解析 | run_id={} | agent_id={} | trace_id={} | mode={} | model_id={} | file_name={} | media_type={} | byte_size={}",
            run_id,
            agent_id,
            trace_id,
            mode,
            model_id,
            stored.file_name,
            stored.media_type,
            stored.byte_size,
        )
        prompt = None
        try:
            if mode == "VLM":
                if self._prompt_registry is None:
                    raise RuntimeError("VLM 图片解析 Prompt Registry 不可用")
                prompt = await self._prompt_registry.resolve(
                    "image_evidence_extract"
                )
                prompt_ref = prompt.ref()
                logger.info(
                    "图片 VLM Prompt 已解析 | run_id={} | agent_id={} | trace_id={} | prompt_id={} | prompt_version={} | prompt_source={}",
                    run_id,
                    agent_id,
                    trace_id,
                    prompt_ref.get("prompt_id"),
                    prompt_ref.get("prompt_version"),
                    prompt_ref.get("prompt_source"),
                )
            result = await self._image_model_client.process(
                mode=mode,
                model_id=model_id,
                mime_type=stored.media_type,
                content_base64=base64.b64encode(raw).decode("ascii"),
                prompt_content=(prompt.content if prompt is not None else None),
                run_id=run_id,
                agent_id=agent_id,
                trace_id=trace_id,
            )
            text = self._bounded(str(result.get("text") or "").strip())
            if not text:
                raise ValueError("图片模型没有返回可用文字")
            logger.info(
                "图片证据解析完成 | run_id={} | agent_id={} | trace_id={} | mode={} | model_id={} | file_name={} | extracted_char_count={}",
                run_id,
                agent_id,
                trace_id,
                mode,
                model_id,
                stored.file_name,
                len(text),
            )
            return ResolvedConversationUpload(
                **common,
                extracted_text=text,
                extraction_mode=mode,
                model_id=model_id,
                model_revision=str(result.get("model_revision") or model_id),
                prompt_ref=(prompt.ref() if prompt is not None else None),
            )
        except Exception as exc:
            stage = "VLM_PROMPT_RESOLVE" if mode == "VLM" and prompt is None else "MODEL_INFERENCE"
            logger.warning(
                "图片证据解析失败 | run_id={} | agent_id={} | trace_id={} | stage={} | mode={} | model_id={} | file_name={} | error_type={}",
                run_id,
                agent_id,
                trace_id,
                stage,
                mode,
                model_id,
                stored.file_name,
                type(exc).__name__,
            )
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

    @staticmethod
    def _manifest_text(upload: ResolvedConversationUpload) -> str:
        """只向规划模型展示材料清单；文本正文仅能由受控检索读取。"""
        if upload.searchable_payload_uri is not None:
            return (
                f"[诊断材料：{upload.file_name}；类型：{upload.extraction_mode}；"
                f"可检索正文：{upload.extracted_char_count} 字符、"
                f"{upload.line_count} 行。请使用 artifact.search 查询具体证据。]"
            )
        return upload.extracted_text

    @staticmethod
    def _decode_text(raw: bytes) -> str:
        """支持数据库日志常见字符集，拒绝含 NUL 的疑似二进制输入。"""
        for encoding in ("utf-8-sig", "utf-16", "gb18030"):
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            if "\x00" not in text:
                return text
        raise UnicodeDecodeError(
            "diagnostic-text", raw, 0, min(1, len(raw)), "不支持的文本编码"
        )


__all__ = [
    "ConversationInputResolver",
    "ConversationUploadSource",
    "ResolvedConversationUpload",
]
