"""AIOps Repository 的公共会话和生命周期护栏。"""

from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID

from sqlalchemy import bindparam, outparam, text
from sqlalchemy.dialects.oracle import RAW
from sqlalchemy.ext.asyncio import AsyncSession

from platform_core.persistence.orm import BaseEntity, UniversalTimestamp


EntityT = TypeVar("EntityT", bound=BaseEntity)


class AIOpsRepository:
    """Repository 只使用注入 Session，不拥有事务提交权。"""

    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        self._session = session
        self._assert_active = assert_active or (lambda: None)

    def _check_active(self) -> None:
        self._assert_active()

    async def _add(self, entity: EntityT) -> EntityT:
        self._check_active()
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def _add_all(self, entities: list[EntityT]) -> list[EntityT]:
        self._check_active()
        self._session.add_all(entities)
        await self._session.flush()
        return entities

    async def _claim_oracle_uuid(
        self,
        *,
        plsql: str,
        parameters: dict[str, Any],
    ) -> UUID | None:
        """由 Oracle 服务端游标只领取一行，避免驱动预取扩大锁范围。"""
        self._check_active()
        bind_parameters = [outparam("claimed_id", type_=RAW(16))]
        bind_parameters.extend(
            bindparam(name, type_=UniversalTimestamp(timezone=True))
            for name, value in parameters.items()
            if isinstance(value, datetime)
        )
        statement = text(plsql).bindparams(*bind_parameters)
        result = await self._session.execute(statement, parameters)
        value = result.out_parameters["claimed_id"]
        if value is None:
            return None
        if isinstance(value, UUID):
            return value
        return UUID(bytes=bytes(value))
