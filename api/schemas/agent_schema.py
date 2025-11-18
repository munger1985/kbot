from pydantic import BaseModel, Field

class AgentChatForm(BaseModel):
    """智能体聊天表单模型"""
    session_id: str = Field(..., description="会话ID")
    by: str = Field(..., description="请求用户ID")
    agent_id: int = Field(..., description="智能体ID")
    security_level: int = Field(0, description="安全级别")
    request_time: str = Field(..., description="请求时间")
    question: str = Field(..., description="问题")
    tags: list[str] = Field([], description="标签")
    deep_mind: int = Field(0, description="是否使用深度思考, 0：不使用，1：使用")

class AgentChatFeedbackForm(BaseModel):
    """智能体聊天获取反馈表单模型"""
    session_id: str = Field(..., description="会话ID")
    question_index: int = Field(..., description="问题索引")
    feedback: int = Field(..., description="问题反馈，0：不反馈，1：赞同，-1：不赞同")

class AgentChatDifyForm(BaseModel):
    """智能体聊天Dify表单模型"""
    knowledge_id: str = Field(..., description="知识库ID")
    query: str = Field(..., description="查询文本")
    retrieval_setting: dict = Field(..., description="检索设置")
    metadata_condition: dict | None = Field(None, description="元数据条件")

