"""Prompt Registry 的只读和受控发布 Repository。"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .entities import PlatformPromptEntity, PlatformPromptVersionEntity


class PlatformPromptRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_definition(
        self, *, prompt_key: str, lock: bool = False
    ) -> PlatformPromptEntity | None:
        statement = select(PlatformPromptEntity).where(
            PlatformPromptEntity.prompt_key == prompt_key
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_active(
        self, *, prompt_key: str
    ) -> tuple[PlatformPromptEntity, PlatformPromptVersionEntity] | None:
        statement = (
            select(PlatformPromptEntity, PlatformPromptVersionEntity)
            .join(
                PlatformPromptVersionEntity,
                PlatformPromptVersionEntity.prompt_version_id
                == PlatformPromptEntity.active_version_id,
            )
            .where(PlatformPromptEntity.prompt_key == prompt_key)
        )
        return (await self._session.execute(statement)).one_or_none()

    async def get_version(
        self, *, prompt_key: str, version: str
    ) -> tuple[PlatformPromptEntity, PlatformPromptVersionEntity] | None:
        statement = (
            select(PlatformPromptEntity, PlatformPromptVersionEntity)
            .join(
                PlatformPromptVersionEntity,
                PlatformPromptVersionEntity.prompt_id
                == PlatformPromptEntity.prompt_id,
            )
            .where(
                PlatformPromptEntity.prompt_key == prompt_key,
                PlatformPromptVersionEntity.version == version,
            )
        )
        return (await self._session.execute(statement)).one_or_none()

    async def get_version_by_id(
        self, *, prompt_version_id: UUID
    ) -> tuple[PlatformPromptEntity, PlatformPromptVersionEntity] | None:
        statement = (
            select(PlatformPromptEntity, PlatformPromptVersionEntity)
            .join(
                PlatformPromptVersionEntity,
                PlatformPromptVersionEntity.prompt_id
                == PlatformPromptEntity.prompt_id,
            )
            .where(
                PlatformPromptVersionEntity.prompt_version_id
                == prompt_version_id
            )
        )
        return (await self._session.execute(statement)).one_or_none()

    async def add_definition(
        self, entity: PlatformPromptEntity
    ) -> PlatformPromptEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def add_version(
        self, entity: PlatformPromptVersionEntity
    ) -> PlatformPromptVersionEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity
