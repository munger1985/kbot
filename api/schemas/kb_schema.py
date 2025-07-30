from pydantic import BaseModel
from fastapi import UploadFile


class KBUploadRequest(BaseModel):
    """知识库上传请求模型"""
    app_id: int
    domain_id: int
    kb_id: int
    overwrite: bool
    batch_name: str
    batch_id: int | None = None
    biz_metadata: dict | None = None
    created_by: str | None = None


class KBUploadForm:
    """知识库上传表单模型"""
    def __init__(
        self,
        files: list[UploadFile],
        metadata: KBUploadRequest
    ):
        self.files = files
        self.metadata = metadata

class KBDeleteRequest(BaseModel):
    """知识库删除请求模型"""
    app_id: int
    domain_id: int
    kb_id: int
    batch_id: int | None
    batch_name: str | None
    file_ids: list[int] | None
    file_paths: list[str] | None



class KBDeleteForm:
    """知识库上传表单模型"""
    def __init__(
        self,
        metadata: KBDeleteRequest
    ):
        self.metadata = metadata


        