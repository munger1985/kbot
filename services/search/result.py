from pydantic import BaseModel, Field, field_validator

class TxtBaseSearchResult(BaseModel):
    """文本知识库搜索结果 - 适配分层检索架构"""
    chunk_id: str = Field(..., description="分片ID")
    file_id: str = Field(..., description="文件ID")
    kb_id: int = Field(..., description="知识库ID")
    chunk_num: int = Field(0, description="分片号")
    chunk_type: str = Field("text", description="分片类别: text, table, picture，heading")
    
    # 核心字段对齐
    content: str = Field(..., description="分片内容（可能已被上下文扩展覆盖）")
    origin_content: str | None = Field(None, description="原始分片内容（上下文扩展前），供 Rerank 判断使用")
    header: str = Field("", description="当前标题")
    doc_summary: str = Field("", description="文档摘要")
    search_helper: str = Field("", description="搜索助手")
    
    # 元数据字段 (来自 chunk_metadata)
    page_num: int = Field(0, description="页码")
    image_name: str = Field("", description="图片名称")
    bbox: list[float] = Field([], description="Chunk所在PDF坐标")

    # 层级路径 (新增)
    hierarchy_path: list[str] = Field(default_factory=list, description="层级路径 ['第一章', '1.1 背景']")
    heading_level: int = Field(0, description="标题级别 0-6")
    section_id: str | None = Field(None, description="所属 section 标识")

    # 业务元数据
    biz_metadata: dict = Field(default_factory=dict, description="业务元数据")

    # 评分与权重
    score: float = Field(0.0, description="原始搜索评分")
    weight: float = Field(0.0, description="搜索权重")
    rerank_score: float = Field(0.0, description="重排后的评分")

    def to_dict(self, **kwargs) -> dict:
        return self.model_dump(**kwargs)
