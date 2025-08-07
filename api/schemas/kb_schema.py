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
        