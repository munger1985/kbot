"""App API Client 与 Credential Repository。"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from main_api.entities.app_api_key import (
    AppApiClientAgentEntity,
    AppApiClientEntity,
    AppApiClientScopeEntity,
    AppApiCredentialEntity,
)


class AppApiKeyRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add_client(self, row: AppApiClientEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def add_credential(self, row: AppApiCredentialEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def replace_scopes(
        self, *, client_id: UUID, scopes: tuple[str, ...]
    ) -> None:
        await self._session.execute(
            delete(AppApiClientScopeEntity).where(
                AppApiClientScopeEntity.client_id == client_id
            )
        )
        self._session.add_all([
            AppApiClientScopeEntity(client_id=client_id, scope_code=scope)
            for scope in scopes
        ])
        await self._session.flush()

    async def replace_agents(
        self, *, client_id: UUID, agent_ids: tuple[UUID, ...]
    ) -> None:
        await self._session.execute(
            delete(AppApiClientAgentEntity).where(
                AppApiClientAgentEntity.client_id == client_id
            )
        )
        self._session.add_all([
            AppApiClientAgentEntity(client_id=client_id, agent_id=agent_id)
            for agent_id in agent_ids
        ])
        await self._session.flush()

    async def get_client(self, client_id: UUID) -> AppApiClientEntity | None:
        return await self._session.get(AppApiClientEntity, client_id)

    async def list_clients(
        self, *, app_id: str, domain_id: int
    ) -> list[AppApiClientEntity]:
        rows = await self._session.scalars(
            select(AppApiClientEntity)
            .where(
                AppApiClientEntity.app_id == app_id,
                AppApiClientEntity.domain_id == domain_id,
            )
            .order_by(AppApiClientEntity.created_at.desc())
        )
        return list(rows)

    async def list_scopes(self, *, client_id: UUID) -> tuple[str, ...]:
        rows = await self._session.scalars(
            select(AppApiClientScopeEntity.scope_code)
            .where(AppApiClientScopeEntity.client_id == client_id)
            .order_by(AppApiClientScopeEntity.scope_code)
        )
        return tuple(rows)

    async def list_agents(self, *, client_id: UUID) -> tuple[UUID, ...]:
        rows = await self._session.scalars(
            select(AppApiClientAgentEntity.agent_id)
            .where(AppApiClientAgentEntity.client_id == client_id)
            .order_by(AppApiClientAgentEntity.agent_id)
        )
        return tuple(rows)

    async def list_credentials(
        self, *, client_id: UUID
    ) -> list[AppApiCredentialEntity]:
        rows = await self._session.scalars(
            select(AppApiCredentialEntity)
            .where(AppApiCredentialEntity.client_id == client_id)
            .order_by(AppApiCredentialEntity.created_at.desc())
        )
        return list(rows)

    async def get_credential_by_public_id(
        self, public_key_id: str
    ) -> AppApiCredentialEntity | None:
        return await self._session.scalar(
            select(AppApiCredentialEntity).where(
                AppApiCredentialEntity.public_key_id == public_key_id
            )
        )

    async def revoke_active_credentials(
        self, *, client_id: UUID
    ) -> None:
        rows = await self._session.scalars(
            select(AppApiCredentialEntity).where(
                AppApiCredentialEntity.client_id == client_id,
                AppApiCredentialEntity.status == "ACTIVE",
            )
        )
        now = datetime.now(timezone.utc)
        for row in rows:
            row.status = "REVOKED"
            row.revoked_at = now
        await self._session.flush()

    async def touch_credential(
        self, credential: AppApiCredentialEntity
    ) -> None:
        credential.last_used_at = datetime.now(timezone.utc)
        await self._session.flush()


__all__ = ["AppApiKeyRepository"]
