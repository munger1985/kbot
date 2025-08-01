from pydantic import BaseModel

class AgentChatForm(BaseModel):
    """智能体聊天表单模型"""
    session_id: str
    by: str
    agent_id: int
    security_level: int
    request_time: str
    question_index: int
    question: str

class AgentChatHistForm(BaseModel):
    """智能体聊天获取聊天历史表单模型"""
    session_id: str

class AgentChatFeedbackForm(BaseModel):
    """智能体聊天获取反馈表单模型"""
    session_id: str
    question_index: int
    feedback: int