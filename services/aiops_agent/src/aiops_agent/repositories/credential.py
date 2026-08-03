"""AIOps 加密凭据 Repository。"""

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import CredentialEntity
from aiops_agent.repositories._base import AIOpsRepository


class CredentialRepository(AIOpsRepository):
    async def add(self, entity: CredentialEntity) -> CredentialEntity:
        return await self._add(entity)

    async def get_scoped(self, *, credential_id: UUID, domain_id: int,
                         credential_kind: str, active_only: bool = False,
                         lock: bool = False) -> CredentialEntity | None:
        self._check_active()
        statement = select(CredentialEntity).where(
            CredentialEntity.credential_id == credential_id,
            CredentialEntity.domain_id == domain_id,
            CredentialEntity.credential_kind == credential_kind,
        )
        if active_only:
            statement = statement.where(CredentialEntity.status == "ACTIVE")
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def revoke(self, entity: CredentialEntity, *, actor_id: str, now: datetime) -> None:
        self._check_active()
        entity.status = "REVOKED"
        entity.updated_by = actor_id
        entity.updated_at = now
        await self._session.flush()
