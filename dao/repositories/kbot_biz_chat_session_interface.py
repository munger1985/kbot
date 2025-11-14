from abc import ABC, abstractmethod
from typing import Sequence
from dao.entities.kbot_biz_chat_session import KbotBizChatSession, QAData


class IChatSessionRepository(ABC):
    """聊天会话存储库的统一接口"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化连接"""
        pass
    
    @abstractmethod
    async def create_session(self, session_data: dict) -> bool:
        """创建聊天会话记录"""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> KbotBizChatSession | None:
        """根据会话ID获取聊天会话记录"""
        pass
    
    @abstractmethod
    async def add_qa_data(self, session_id: str, qa_data: QAData) -> bool:
        """添加问答数据到会话"""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """根据会话ID删除聊天会话记录"""
        pass

    @abstractmethod
    async def update_qa_feedback(self, session_id: str, qa_index_num: int, feedback: int) -> bool:
        """更新问答数据的反馈"""
        pass
    
    @abstractmethod
    async def get_last_qa_data(self, session_id: str) -> dict | None:
        """获取会话的最后一次问答数据"""
        pass

    @abstractmethod
    async def get_qa_data(self, session_id: str, qa_index_num: int) -> QAData | None:
        """根据会话ID和问答索引获取问答数据"""
        pass

    @abstractmethod
    async def update_last_qa_data_answer(self, session_id: str, answer: str) -> bool:
        """更新会话的最后一次问答数据的回答"""
        pass

    @abstractmethod
    async def delete_by_agent_id(self, agent_id: int) -> bool:
        """根据智能体ID删除所有会话记录"""
        pass
