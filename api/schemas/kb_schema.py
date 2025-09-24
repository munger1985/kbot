from pydantic import BaseModel
from fastapi import UploadFile


class KBUploadForm(BaseModel):
    """知识库上传表单模型"""
    files: list[UploadFile]
    app_id: int
    domain_id: int
    kb_id: int
    overwrite: bool
    batch_name: str
    batch_id: int | None = None
    biz_metadata: dict | None = None
    created_by: str | None = None


class KBDeleteForm(BaseModel):
    """知识库删除表单模型"""
    app_id: int
    domain_id: int
    kb_id: int
    batch_id: int | None = None
    batch_name: str | None = None
    file_ids: list[str] | None = None
    file_paths: list[str] | None = None

class KBReparseForm(BaseModel):
    """知识库重新解析表单模型"""
    kb_id: int
    file_ids: list[str]

class KBFilePreviewForm(BaseModel):
    """知识库文件预览表单模型"""
    file_id: str
    max_length: int | None = None
    pages: int | list[int] | None = None
    sheet_index: int | None = None
    preview_rows: int | None = None
    slide: int | None = None

class KBFileChunkEditForm(BaseModel):
    """知识库文件分片编辑表单模型"""
    kb_id: int
    file_id: str
    embed_id: str
    new_chunk: str | None = None
    action: str
    