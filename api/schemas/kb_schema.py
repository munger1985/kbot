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

class KBDeleteRequest(BaseModel):
    """知识库删除请求模型"""
    app_id: int
    domain_id: int
    kb_id: int
    batch_id: Optional[int]
    batch_name: Optional[str]
    file_ids: Optional[List[int]]
    file_paths: Optional[List[str]]



class KBDeleteForm:
    """知识库上传表单模型"""
    def __init__(
        self,
        metadata: KBDeleteRequest
    ):
        self.metadata = metadata


        