from typing import Sequence
from sqlalchemy import select, delete
from sqlalchemy import func
from core.exceptions import DatabaseException
from dao.entities import ChatSessionEntity
from .base_repo import BaseRepository  # 补充继承基础仓库类


class ChatSessionRepository(BaseRepository[ChatSessionEntity]):
    """Repository for KBOT_MD_CHAT_SESSION table operations."""
    
    async def create(self, chat_session: ChatSessionEntity) -> ChatSessionEntity:
        """Create a new chat session record."""
        try:
            self.session.add(chat_session)
            await self.session.flush()
            await self.session.refresh(chat_session)
            return chat_session
        except Exception as e:
            raise DatabaseException("Failed to create chat session record", original_error=e)
            
    async def get_by_session_id(self, session_id: str) -> Sequence[ChatSessionEntity]:
        """Get chat session by session ID."""
        try:
            stmt = select(ChatSessionEntity).where(ChatSessionEntity.session_id == session_id)
            result = await self.session.execute(stmt)
            sessions = result.scalars().all()
            return sessions
        except Exception as e:
            raise DatabaseException("Failed to get chat session by session ID", original_error=e)

    async def get_by_agent(self, agent_id: int) -> Sequence[ChatSessionEntity]:
        """Get chat session by agent ID."""
        try:
            stmt = select(ChatSessionEntity).where(ChatSessionEntity.agent_id == agent_id)
            result = await self.session.execute(stmt)
            sessions = result.scalars().all()
            return sessions
        except Exception as e:
            raise DatabaseException("Failed to get chat session by agent ID", original_error=e)

    async def delete(self, session_id: str) -> None:
        """Delete chat session by session ID."""
        try:
            stmt = delete(ChatSessionEntity).where(ChatSessionEntity.session_id == session_id)
            await self.session.execute(stmt)
        except Exception as e:
            raise DatabaseException("Failed to delete chat session by session ID", original_error=e)


    async def delete_by_agent_id(self, agent_id: int) -> None:
        """Delete chat session by agent ID."""
        try:
            stmt = delete(ChatSessionEntity).where(ChatSessionEntity.agent_id == agent_id)
            await self.session.execute(stmt)
        except Exception as e:
            raise DatabaseException("Failed to delete chat session by agent ID", original_error=e)