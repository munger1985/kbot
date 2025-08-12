from typing import Sequence
from sqlalchemy import select
from sqlalchemy import func
from dao.entities.kbot_md_chat_history import KbotMdChatHistory
from core.database.meta_oracle import get_session


class KbotMdChatHistoryRepository:
    """Repository for KBOT_MD_CHAT_HISTORY table operations."""
    
    async def create(self, chat_history: KbotMdChatHistory) -> bool:
        """Create a new chat history record."""
        async with get_session() as session:
            session.add(chat_history)
            await session.commit()
            return True
            
    async def get_by_session_id(self, session_id: str) -> Sequence[KbotMdChatHistory]:
        """Get chat histories by session ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdChatHistory)
                .where(KbotMdChatHistory.session_id == session_id)
            )
            return result.scalars().all()
            