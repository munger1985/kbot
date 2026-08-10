"""AIOps 对话聚合 Repository。"""

from uuid import UUID

from sqlalchemy import func, select

from aiops_agent.entities import (
    ActionStepEntity, EvidenceRequestEntity, ImageEvidenceProcessingEntity,
    OpsConversationEntity, OpsConversationMessageEntity,
    OpsConversationRunEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class ConversationRepository(AIOpsRepository):
    async def add_conversation(self, row): return await self._add(row)
    async def add_message(self, row): return await self._add(row)
    async def add_run(self, row): return await self._add(row)
    async def add_evidence_request(self, row): return await self._add(row)
    async def add_image_processing(self, row): return await self._add(row)
    async def add_action_step(self, row): return await self._add(row)

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
    ):
        statement = select(OpsConversationEntity).where(
            OpsConversationEntity.domain_id == domain_id,
            OpsConversationEntity.created_by == created_by,
        )
        if agent_id is not None:
            statement = statement.where(OpsConversationEntity.agent_id == agent_id)
        rows = await self._session.scalars(statement.order_by(
            OpsConversationEntity.updated_at.desc(),
            OpsConversationEntity.conversation_id.desc(),
        ).limit(limit))
        return list(rows)

    async def first_user_messages(self, *, conversation_ids: list[UUID]):
        if not conversation_ids: return {}
        rows = await self._session.scalars(
            select(OpsConversationMessageEntity).where(
                OpsConversationMessageEntity.conversation_id.in_(conversation_ids),
                OpsConversationMessageEntity.role == "USER",
                OpsConversationMessageEntity.message_type == "USER_MESSAGE",
            ).order_by(
                OpsConversationMessageEntity.conversation_id,
                OpsConversationMessageEntity.sequence_no,
            )
        )
        result = {}
        for message in rows: result.setdefault(message.conversation_id, message)
        return result

    async def next_message_sequence(self, *, conversation_id: UUID) -> int:
        value = await self._session.scalar(select(func.coalesce(func.max(
            OpsConversationMessageEntity.sequence_no), 0)).where(
                OpsConversationMessageEntity.conversation_id == conversation_id
            ))
        return int(value) + 1

    async def next_run_sequence(self, *, conversation_id: UUID) -> int:
        value = await self._session.scalar(select(func.coalesce(func.max(
            OpsConversationRunEntity.sequence_no), 0)).where(
                OpsConversationRunEntity.conversation_id == conversation_id
            ))
        return int(value) + 1

    async def list_messages(
        self, *, conversation_id: UUID, after_sequence: int = 0, limit: int = 200
    ):
        rows = await self._session.scalars(select(
            OpsConversationMessageEntity
        ).where(
            OpsConversationMessageEntity.conversation_id == conversation_id,
            OpsConversationMessageEntity.sequence_no > after_sequence,
        ).order_by(OpsConversationMessageEntity.sequence_no).limit(limit))
        return list(rows)

    async def list_runs(self, *, conversation_id: UUID):
        rows = await self._session.scalars(select(OpsConversationRunEntity).where(
            OpsConversationRunEntity.conversation_id == conversation_id
        ).order_by(OpsConversationRunEntity.sequence_no))
        return list(rows)

    async def get_evidence_request(
        self, *, conversation_id: UUID, request_id: UUID, lock: bool = False
    ):
        statement = select(EvidenceRequestEntity).where(
            EvidenceRequestEntity.conversation_id == conversation_id,
            EvidenceRequestEntity.request_id == request_id,
        )
        if lock: statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_image_processing(
        self, *, processing_id: UUID, lock: bool = False
    ):
        statement = select(ImageEvidenceProcessingEntity).where(
            ImageEvidenceProcessingEntity.processing_id == processing_id
        )
        if lock: statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_action_steps(self, *, conversation_id: UUID):
        rows = await self._session.scalars(select(ActionStepEntity).where(
            ActionStepEntity.conversation_id == conversation_id
        ).order_by(ActionStepEntity.ordinal))
        return list(rows)

    async def get_action_step(
        self, *, conversation_id: UUID, action_step_id: UUID,
        lock: bool = False,
    ):
        statement = select(ActionStepEntity).where(
            ActionStepEntity.conversation_id == conversation_id,
            ActionStepEntity.action_step_id == action_step_id,
        )
        if lock: statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()
