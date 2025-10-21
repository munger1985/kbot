from pydantic import BaseModel, Field
from fastapi import UploadFile


class KBUploadForm(BaseModel):
    """知识库文件上传表单模型"""
    files: list[UploadFile] = Field(..., description="文件列表")
    app_id: int = Field(..., description="应用ID")
    domain_id: int = Field(..., description="领域ID")
    kb_id: int = Field(..., description="知识库ID")
    overwrite: bool = Field(..., description="是否覆盖")
    batch_name: str = Field(..., description="批次名称")
    batch_id: int | None = Field(None, description="批次ID")
    biz_metadata: dict | None = Field(None, description="业务元数据")
    created_by: str | None = Field(None, description="创建人")


class KBDeleteForm(BaseModel):
    """知识库删除表单模型"""
    app_id: int = Field(..., description="应用ID")
    domain_id: int = Field(..., description="领域ID")
    kb_id: int = Field(..., description="知识库ID")
    batch_id: int | None = Field(None, description="批次ID")
    batch_name: str | None = Field(None, description="批次名称")
    file_ids: list[str] | None = Field(None, description="文件ID列表")
    file_paths: list[str] | None = Field(None, description="文件路径列表")

class KBReparseForm(BaseModel):
    """知识库重新解析表单模型"""
    kb_id: int = Field(..., description="知识库ID")
    file_ids: list[str] = Field(..., description="文件ID列表")

class KBFilePreviewForm(BaseModel):
    """知识库文件预览表单模型"""
    file_id: str = Field(..., description="文件ID")
    max_length: int | None = Field(None, description="最大长度")
    pages: int | list[int] | None = Field(None, description="页数")
    sheet_index: int | None = Field(None, description="Sheet索引")
    preview_rows: int | None = Field(None, description="预览行数")
    slide: int | None = Field(None, description="幻灯片页码")

class KBFileChunkEditForm(BaseModel):
    """知识库文件分片编辑表单模型"""
    kb_id: int = Field(..., description="知识库ID")
    file_id: str = Field(..., description="文件ID")
    embed_id: str = Field(..., description="分片ID")
    new_chunk: str | None = Field(None, description="新分片内容")
    action: str = Field(..., description="操作类型")
    