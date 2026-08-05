"""Bundle Revision 与源文件安全预览内部端点。"""

from dataclasses import asdict
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from knowledge_core.application.preview import KnowledgePreviewNotFoundError
from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import require_domain_match


router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/domains/{{domain_id}}",
    tags=["Knowledge Core Preview"],
)

_INLINE_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
        "text/plain",
    }
)


def _parse_range(value: str | None, byte_size: int) -> tuple[int, int] | None:
    """解析单一 HTTP bytes Range，拒绝多段或越界请求。"""
    if value is None:
        return None
    if byte_size <= 0 or not value.startswith("bytes=") or "," in value:
        raise ValueError("INVALID_RANGE")
    bounds = value[6:].strip()
    if "-" not in bounds:
        raise ValueError("INVALID_RANGE")
    start_text, end_text = bounds.split("-", 1)
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start = max(0, byte_size - suffix)
            end = byte_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else byte_size - 1
    except ValueError as exc:
        raise ValueError("INVALID_RANGE") from exc
    if start < 0 or start >= byte_size or end < start:
        raise ValueError("INVALID_RANGE")
    return start, min(end, byte_size - 1)


@router.get(
    "/collections/{collection_id}/bundles/{bundle_id}"
    "/revisions/{bundle_revision_id}/preview"
)
async def preview_bundle_revision(
    domain_id: int,
    collection_id: UUID,
    bundle_id: UUID,
    bundle_revision_id: UUID,
    request: Request,
):
    require_domain_match(request, domain_id)
    try:
        preview = await request.app.state.kc_preview_service.get_bundle_revision(
            domain_id=domain_id,
            collection_id=collection_id,
            bundle_id=bundle_id,
            bundle_revision_id=bundle_revision_id,
        )
    except KnowledgePreviewNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "KNOWLEDGE_PREVIEW_NOT_FOUND",
                "message": str(exc),
            },
        ) from exc
    return asdict(preview)


@router.get(
    "/collections/{collection_id}/bundles/{bundle_id}"
    "/revisions/{bundle_revision_id}/documents/{document_version_id}/content"
)
async def preview_source_file(
    domain_id: int,
    collection_id: UUID,
    bundle_id: UUID,
    bundle_revision_id: UUID,
    document_version_id: UUID,
    request: Request,
):
    require_domain_match(request, domain_id)
    try:
        source = await request.app.state.kc_preview_service.get_source_file(
            domain_id=domain_id,
            collection_id=collection_id,
            bundle_id=bundle_id,
            bundle_revision_id=bundle_revision_id,
            document_version_id=document_version_id,
        )
        actual_size = await request.app.state.kc_object_store.size(
            source.storage_uri
        )
    except (KnowledgePreviewNotFoundError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "KNOWLEDGE_PREVIEW_NOT_FOUND",
                "message": "源文件不存在",
            },
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "KNOWLEDGE_OBJECT_STORE_UNAVAILABLE",
                "message": "对象存储暂时不可用",
            },
        ) from exc
    if actual_size != source.byte_size:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "KNOWLEDGE_OBJECT_SIZE_MISMATCH",
                "message": "源文件大小与入库事实不一致",
            },
        )
    try:
        requested_range = _parse_range(
            request.headers.get("range"), source.byte_size
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=416,
            detail={"code": "INVALID_RANGE", "message": "Range 无效"},
            headers={"Content-Range": f"bytes */{source.byte_size}"},
        ) from exc
    start, end = (
        requested_range
        if requested_range is not None
        else (0, source.byte_size - 1)
    )
    length = max(0, end - start + 1)
    safe_mime = source.mime_type.lower().split(";", 1)[0].strip()
    inline = safe_mime in _INLINE_MIME_TYPES
    media_type = safe_mime if inline else "application/octet-stream"
    disposition = "inline" if inline else "attachment"
    encoded_name = quote(source.file_name, safe="")
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": (
            f"{disposition}; filename*=UTF-8''{encoded_name}"
        ),
        "Content-Length": str(length),
        "Cache-Control": "private, no-store",
        "Content-Security-Policy": "sandbox; default-src 'none'",
        "X-Content-Type-Options": "nosniff",
    }
    if requested_range is not None:
        headers["Content-Range"] = (
            f"bytes {start}-{end}/{source.byte_size}"
        )
    return StreamingResponse(
        request.app.state.kc_object_store.stream(
            source.storage_uri,
            offset=start,
            length=length,
        ),
        status_code=206 if requested_range is not None else 200,
        media_type=media_type,
        headers=headers,
    )
