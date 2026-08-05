"""Data Query 对 KBot Agent 定义的只读归属查询。"""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class PlatformResourceAccessRepository:
    """只通过稳定表契约确认 Agent 的 Domain 归属。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def agent_domain_id(
        self, *, domain_id: int, agent_id: UUID
    ) -> int | None:
        result = await self._session.execute(
            text(
                """
                SELECT domain_id
                FROM KBOT_AGENT_DEFINITION
                WHERE agent_id = :agent_id
                  AND domain_id = :domain_id
                  AND status = 'ACTIVE'
                """
            ),
            {"agent_id": agent_id.bytes, "domain_id": domain_id},
        )
        value = result.scalar_one_or_none()
        return None if value is None else int(value)

    async def agent_data_query_mode(
        self, *, domain_id: int, agent_id: UUID
    ) -> str | None:
        """管理面允许给 DRAFT Agent 建绑定，但 Provider 必须已固定为 SEMANTIC。"""
        result = await self._session.execute(
            text(
                """
                SELECT data_query_mode
                FROM KBOT_AGENT_DEFINITION
                WHERE agent_id = :agent_id
                  AND domain_id = :domain_id
                  AND status IN ('DRAFT', 'ACTIVE')
                """
            ),
            {"agent_id": agent_id.bytes, "domain_id": domain_id},
        )
        value = result.scalar_one_or_none()
        return None if value is None else str(value)
