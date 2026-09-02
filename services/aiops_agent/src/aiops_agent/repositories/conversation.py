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

    async def list_for_inspection_fire(self, *, inspection_fire_id: UUID):
        """返回一次巡检 Fire 已经创建的系统会话。"""
        rows = await self._session.scalars(
            select(OpsConversationEntity)
            .where(
                OpsConversationEntity.source_inspection_fire_id
                == inspection_fire_id
            )
            .order_by(OpsConversationEntity.target_id)
        )
        return list(rows)

    async def get_auto_situation_conversation(
        self,
        *,
        domain_id: int,
        situation_id: UUID,
        agent_id: UUID,
        actor_id: str,
    ):
        """返回同一告警情境已经提交给同一 Agent 的系统会话。"""
        statement = (
            select(OpsConversationEntity)
            .where(
                OpsConversationEntity.domain_id == domain_id,
                OpsConversationEntity.source_type == "SITUATION",
                OpsConversationEntity.source_situation_id == situation_id,
                OpsConversationEntity.agent_id == agent_id,
                OpsConversationEntity.created_by == actor_id,
            )
            .order_by(OpsConversationEntity.created_at)
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()
