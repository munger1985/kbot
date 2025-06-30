from typing import List, Optional
from pydantic import BaseModel
from fastapi import UploadFile


class KBUploadRequest(BaseModel):
    """知识库上传请求模型"""
    app_id: int
    domain_id: int
    kb_id: int
    overwrite: bool
    batch_name: str
    batch_id: Optional[int] = None
    security_level: Optional[str] = None
    biz_metadata: Optional[dict] = None
    created_by: Optional[str] = None


class KBUploadForm:
    """知识库上传表单模型"""
    def __init__(
        self,
        files: List[UploadFile],
        metadata: KBUploadRequest
    ):
        self.files = files
        self.metadata = metadata