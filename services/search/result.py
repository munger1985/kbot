from pydantic import BaseModel, Field

class TxtBaseSearchResult(BaseModel):
    """文本知识库搜索结果 - 适配分层检索架构"""
    chunk_id: str = Field(..., description="分片ID")
    file_id: str = Field(..., description="文件ID")
    content: str = Field(..., description="分片内容")
    
    # 核心字段对齐
    structure_level: int = Field(..., description="层级深度 (L1, L2...)")
    path_names: str = Field(..., description="章节路径基因")
    node_path: str = Field("", description="解析节点路径")
    
    # 元数据字段 (来自 chunk_metadata)
    page_num: int = Field(0, description="页码")
    chunk_num: int = Field(0, description="分片号")
    sub_index: int = Field(0, description="子序号")
    chunk_type: str = Field("text", description="分片类别: text, table, picture，heading")
    
    # 评分与权重
    search_type: str = Field("vector", description="搜索类型: vector, fulltext")
    score: float = Field(0.0, description="原始搜索评分")
    weight: float = Field(0.0, description="搜索权重")
    rerank_score: float = Field(0.0, description="重排后的评分")
    
    # 其他
    embedding: list[float] = Field(default_factory=list, description="分片向量")

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