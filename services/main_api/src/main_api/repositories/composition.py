"""组合编排 Receipt Repository；事务由 Main API UoW 管理。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from main_api.entities import CompositionReceiptEntity


class CompositionReceiptRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_idempotency(
        self, *, domain_id: int, actor_id: str,
        operation: str, idempotency_key: str, lock: bool = False,
    ) -> CompositionReceiptEntity | None:
        statement = select(CompositionReceiptEntity).where(
            CompositionReceiptEntity.domain_id == domain_id,
            CompositionReceiptEntity.actor_id == actor_id,
            CompositionReceiptEntity.operation == operation,
            CompositionReceiptEntity.idempotency_key == idempotency_key,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get(
        self, *, receipt_id: UUID, domain_id: int, actor_id: str,
    ) -> CompositionReceiptEntity | None:
        return (
            await self._session.execute(
                select(CompositionReceiptEntity).where(
                    CompositionReceiptEntity.receipt_id == receipt_id,
                    CompositionReceiptEntity.domain_id == domain_id,
                    CompositionReceiptEntity.actor_id == actor_id,
                )
            )
        ).scalar_one_or_none()

    async def add(self, entity: CompositionReceiptEntity) -> None:
        self._session.add(entity)
        await self._session.flush()

    async def transition(
        self, entity: CompositionReceiptEntity, *, status: str,
        resource_id: str | None = None,
        resource_version: str | None = None,
        verification: dict | None = None,
        error_code: str | None = None,
    ) -> None:
        entity.status = status
        if resource_id is not None:
            entity.resource_id = resource_id
        if resource_version is not None:
            entity.resource_version = resource_version
        if verification is not None:
            entity.verification_json = verification
        entity.error_code = error_code
        entity.attempt_count = int(entity.attempt_count) + 1
        entity.row_version = int(entity.row_version) + 1
        entity.updated_at = datetime.now(timezone.utc)
        await self._session.flush()
