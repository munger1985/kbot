"""原生解析器接受的媒体类型；计算端和提交端共用。"""

OFFICE_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}
DOCUMENT_TYPES = {**OFFICE_TYPES, "application/pdf": "pdf", "text/html": "html", "text/markdown": "md"}
IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/tiff": "tif"}
IMAGE_TYPES = frozenset(IMAGE_EXTENSIONS)
NATIVE_FORMATS = {**DOCUMENT_TYPES, **IMAGE_EXTENSIONS, "text/plain": "txt"}
