from typing import Sequence
from sqlalchemy import select, delete
from sqlalchemy import func
from core.exceptions import DatabaseException
from dao.entities import ChatHistoryEntity
from .base_repo import BaseRepository  # 补充继承基础仓库类


class ChatHistoryRepository(BaseRepository[ChatHistoryEntity]):
    """Repository for KBOT_MD_CHAT_HISTORY table operations."""
    
    async def create(self, chat_history: ChatHistoryEntity) -> ChatHistoryEntity:
        """Create a new chat history record."""
        try:
            self.session.add(chat_history)
            await self.session.flush()
            await self.session.refresh(chat_history)
            return chat_history
        except Exception as e:
            raise DatabaseException("Failed to create chat history record", original_error=e)
            
    async def get_by_session_id(self, session_id: str) -> Sequence[ChatHistoryEntity]:
        """Get chat histories by session ID."""
        try:
            stmt = select(ChatHistoryEntity).where(ChatHistoryEntity.session_id == session_id)
            result = await self.session.execute(stmt)
            chat_histories = result.scalars().all()
            return chat_histories
        except Exception as e:
            raise DatabaseException("Failed to get chat histories by session ID", original_error=e)
        
    async def delete_by_agent_id(self, agent_id: int) -> None:
        """Delete chat histories by agent ID."""
        try:
            stmt = delete(ChatHistoryEntity).where(ChatHistoryEntity.agent_id == agent_id)
            await self.session.execute(stmt)
        except Exception as e:
            raise DatabaseException("Failed to delete chat histories by agent ID", original_error=e)