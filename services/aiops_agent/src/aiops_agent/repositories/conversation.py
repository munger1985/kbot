"""AIOps Conversation 根聚合 Repository。"""

from uuid import UUID

from sqlalchemy import select

from aiops_agent.entities import OpsConversationEntity
from aiops_agent.repositories._base import AIOpsRepository


class ConversationRepository(AIOpsRepository):
    async def add_conversation(self, row): return await self._add(row)

    async def get_conversation(
        self, *, domain_id: int, conversation_id: UUID, lock: bool = False
    ):
        statement = select(OpsConversationEntity).where(
            OpsConversationEntity.domain_id == domain_id,
            OpsConversationEntity.conversation_id == conversation_id,
        )
        if lock: statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_conversations(
        self, *, domain_id: int, created_by: str,
        agent_id: UUID | None = None, limit: int = 50,
        target_id: UUID | None = None,
    ):
        statement = select(OpsConversationEntity).where(
            OpsConversationEntity.domain_id == domain_id,
            OpsConversationEntity.created_by == created_by,
            OpsConversationEntity.status != "ARCHIVED",
        )
        if agent_id is not None:
            statement = statement.where(OpsConversationEntity.agent_id == agent_id)
        if target_id is not None:
            statement = statement.where(OpsConversationEntity.target_id == target_id)
        rows = await self._session.scalars(statement.order_by(
            OpsConversationEntity.updated_at.desc(),
            OpsConversationEntity.conversation_id.desc(),
        ).limit(limit))
        return list(rows)
