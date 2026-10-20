"""用检测到的 MIME 类型统一确定 KC 源文件格式。"""

from pathlib import Path


_MIME_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "text/markdown": ".md",
    "text/plain": ".md",
    "text/csv": ".csv",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
}
_SUPPORTED_SUFFIXES = frozenset(_MIME_SUFFIXES.values())


def source_suffix(
    *, detected_mime_type: str | None, file_name: str | None = None
) -> str | None:
    """优先信任内容检测结果，仅在 MIME 未知时使用受支持的文件后缀。"""
    mime_type = str(detected_mime_type or "").partition(";")[0].strip().lower()
    suffix = _MIME_SUFFIXES.get(mime_type)
    if suffix is not None:
        return suffix
    if mime_type:
        return None
    declared_suffix = Path(str(file_name or "")).suffix.lower()
    return declared_suffix if declared_suffix in _SUPPORTED_SUFFIXES else None
