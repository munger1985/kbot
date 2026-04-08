from pydantic import BaseModel, Field

class TxtBaseSearchResult(BaseModel):
    """文本知识库搜索结果 - 适配分层检索架构"""
    chunk_id: str = Field(..., description="分片ID")
    file_id: str = Field(..., description="文件ID")
    content: str = Field(..., description="分片内容")
    
    # 核心字段对齐
    structure_level: int = Field(..., description="层级深度 (L1, L2...)")
    path_names: str = Field(..., description="章节路径基因")
    header_context: str = Field(..., description="标题上下文")
    
    # 元数据字段 (来自 chunk_metadata)
    page_num: int = Field(0, description="页码")
    chunk_num: int = Field(0, description="分片号")
    chunk_type: str = Field("text", description="分片类别: text, table, picture，heading")
    
    # 评分与权重
    score: float = Field(0.0, description="原始搜索评分")
    weight: float = Field(0.0, description="搜索权重")
    rerank_score: float = Field(0.0, description="重排后的评分")
    
    # 其他
    image_name: str = Field("", description="图片名称")
    search_type: str = Field("", description="搜索类型")

    def to_dict(self, **kwargs) -> dict:
        return self.model_dump(**kwargs)

class InternetSearchResult(BaseModel):
    """网络搜索结果"""
    title: str = ""
    url: str = ""
    content: str = ""
    snippet: str = ""
    relevance_score: float = 0.0
    weight: float = 0.0
    rerank_score: float = 0.0

class CalculatorResult(BaseModel):
    """计算器结果"""
    expression: str = ""
    result: str = ""
    steps: list[str] = []  # 计算步骤
    confidence: float = 1.0