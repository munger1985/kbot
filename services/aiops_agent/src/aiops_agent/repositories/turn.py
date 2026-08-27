"""AIOps Conversation Turn、证据和回答投影 Repository。"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from aiops_agent.entities import (
    OpsAnswerBlockEntity,
    OpsAnswerCitationEntity,
    OpsConversationMessageEntity,
    OpsConversationTurnEntity,
    OpsSkillInvocationEntity,
    OpsTurnEventEntity,
    OpsTurnEvidenceEntity,
    OpsTurnRunEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class TurnRepository(AIOpsRepository):
    """只持久化 Turn 子聚合，不拥有事务提交。"""

    async def add_turn(
        self, row: OpsConversationTurnEntity
    ) -> OpsConversationTurnEntity:
        return await self._add(row)

    async def add_message(
        self, row: OpsConversationMessageEntity
    ) -> OpsConversationMessageEntity:
        return await self._add(row)

    async def add_run(self, row: OpsTurnRunEntity) -> OpsTurnRunEntity:
        return await self._add(row)

    async def get_run_link(
        self,
        *,
        turn_id: UUID,
        purpose: str,
    ) -> OpsTurnRunEntity | None:
        statement = select(OpsTurnRunEntity).where(
            OpsTurnRunEntity.turn_id == turn_id,
            OpsTurnRunEntity.purpose == purpose,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add_skill_invocation(
        self, row: OpsSkillInvocationEntity
    ) -> OpsSkillInvocationEntity:
        return await self._add(row)

    async def add_evidence(
        self, row: OpsTurnEvidenceEntity
    ) -> OpsTurnEvidenceEntity:
        return await self._add(row)

    async def add_answer_block(
        self, row: OpsAnswerBlockEntity
    ) -> OpsAnswerBlockEntity:
        return await self._add(row)

    async def add_answer_citation(
        self, row: OpsAnswerCitationEntity
    ) -> OpsAnswerCitationEntity:
        return await self._add(row)

    async def add_event(
        self, row: OpsTurnEventEntity
    ) -> OpsTurnEventEntity:
        return await self._add(row)

    async def get_turn(
        self,
        *,
        domain_id: int,
        turn_id: UUID,
        lock: bool = False,
    ) -> OpsConversationTurnEntity | None:
        statement = select(OpsConversationTurnEntity).where(
            OpsConversationTurnEntity.domain_id == domain_id,
            OpsConversationTurnEntity.turn_id == turn_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_idempotency(
        self,
        *,
        conversation_id: UUID,
        idempotency_key: str,
    ) -> OpsConversationTurnEntity | None:
        statement = select(OpsConversationTurnEntity).where(
            OpsConversationTurnEntity.conversation_id == conversation_id,
            OpsConversationTurnEntity.idempotency_key == idempotency_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_turns(
        self,
        *,
        conversation_id: UUID,
        after_turn_no: int = 0,
        limit: int = 50,
    ) -> list[OpsConversationTurnEntity]:
        rows = await self._session.scalars(
            select(OpsConversationTurnEntity)
            .where(
                OpsConversationTurnEntity.conversation_id == conversation_id,
                OpsConversationTurnEntity.turn_no > after_turn_no,
            )
            .order_by(OpsConversationTurnEntity.turn_no)
            .limit(limit)
        )
        return list(rows)

    async def list_messages(
        self, *, turn_id: UUID
    ) -> list[OpsConversationMessageEntity]:
        rows = await self._session.scalars(
            select(OpsConversationMessageEntity)
            .where(OpsConversationMessageEntity.turn_id == turn_id)
            .order_by(OpsConversationMessageEntity.sequence_no)
        )
        return list(rows)

    async def list_recent_conversation_messages(
        self,
        *,
        conversation_id: UUID,
        before_sequence_no: int,
        limit: int = 12,
    ) -> list[OpsConversationMessageEntity]:
        rows = list(
            await self._session.scalars(
                select(OpsConversationMessageEntity)
                .where(
                    OpsConversationMessageEntity.conversation_id
                    == conversation_id,
                    OpsConversationMessageEntity.sequence_no
                    < before_sequence_no,
                )
                .order_by(OpsConversationMessageEntity.sequence_no.desc())
                .limit(limit)
            )
        )
        rows.reverse()
        return rows

    async def list_events(
        self,
        *,
        turn_id: UUID,
        after_sequence: int = 0,
        limit: int = 200,
        user_visible_only: bool = True,
    ) -> list[OpsTurnEventEntity]:
        statement = select(OpsTurnEventEntity).where(
            OpsTurnEventEntity.turn_id == turn_id,
            OpsTurnEventEntity.sequence_no > after_sequence,
        )
        if user_visible_only:
            statement = statement.where(OpsTurnEventEntity.visibility == "USER")
        rows = await self._session.scalars(
            statement.order_by(OpsTurnEventEntity.sequence_no).limit(limit)
        )
        return list(rows)

    async def list_answer_blocks(
        self, *, turn_id: UUID
    ) -> list[OpsAnswerBlockEntity]:
        rows = await self._session.scalars(
            select(OpsAnswerBlockEntity)
            .where(
                OpsAnswerBlockEntity.turn_id == turn_id,
                OpsAnswerBlockEntity.status == "ACTIVE",
            )
            .order_by(OpsAnswerBlockEntity.block_no)
        )
        return list(rows)
