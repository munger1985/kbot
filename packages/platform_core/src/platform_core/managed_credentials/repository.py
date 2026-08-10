"""托管凭据 Repository。"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .entities import ManagedCredentialEntity


class ManagedCredentialRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: ManagedCredentialEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(
        self,
        *,
        domain_id: int,
        credential_id: UUID,
        lock: bool = False,
    ) -> ManagedCredentialEntity | None:
        statement = select(ManagedCredentialEntity).where(
            ManagedCredentialEntity.domain_id == domain_id,
            ManagedCredentialEntity.credential_id == credential_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find(
        self,
        *,
        domain_id: int,
        namespace: str,
        credential_kind: str,
        external_key: str,
        lock: bool = False,
    ) -> ManagedCredentialEntity | None:
        statement = select(ManagedCredentialEntity).where(
            ManagedCredentialEntity.domain_id == domain_id,
            ManagedCredentialEntity.namespace == namespace,
            ManagedCredentialEntity.credential_kind == credential_kind,
            ManagedCredentialEntity.external_key == external_key,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()


__all__ = ["ManagedCredentialRepository"]
