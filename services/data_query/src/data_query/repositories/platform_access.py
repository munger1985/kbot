"""Data Query 对 KBot Agent 定义的只读归属查询。"""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class PlatformResourceAccessRepository:
    """只通过稳定表契约确认 Agent 的 Domain 归属。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def agent_domain_id(
        self, *, domain_id: int, consumer_app_id: str,
        agent_id: UUID, agent_version_id: UUID,
    ) -> int | None:
        result = await self._session.execute(
            text(
                """
                SELECT domain_id FROM (
                    SELECT a.domain_id
                    FROM KBOT_KR_AGENT a
                    JOIN KBOT_KR_AGENT_VERSION v ON v.agent_id = a.agent_id
                    WHERE :consumer_app_id = 'knowledge_retrieval'
                      AND a.agent_id = :agent_id
                      AND v.agent_version_id = :agent_version_id
                      AND a.current_version_id = v.agent_version_id
                      AND a.domain_id = :domain_id AND a.status = 'ACTIVE'
                    UNION ALL
                    SELECT a.domain_id
                    FROM KBOT_OPS_AGENT a
                    JOIN KBOT_OPS_AGENT_VERSION v ON v.agent_id = a.agent_id
                    WHERE :consumer_app_id = 'aiops'
                      AND a.agent_id = :agent_id
                      AND v.agent_version_id = :agent_version_id
                      AND a.current_version_id = v.agent_version_id
                      AND a.domain_id = :domain_id AND a.status = 'ACTIVE'
                )
                """
            ),
            {
                "consumer_app_id": consumer_app_id,
                "agent_id": agent_id.bytes,
                "agent_version_id": agent_version_id.bytes,
                "domain_id": domain_id,
            },
        )
        value = result.scalar_one_or_none()
        return None if value is None else int(value)

    async def agent_data_query_mode(
        self, *, domain_id: int, consumer_app_id: str,
        agent_id: UUID, agent_version_id: UUID,
    ) -> str | None:
        """管理面允许给 DRAFT Agent 建绑定，但 Provider 必须已固定为 SEMANTIC。"""
        result = await self._session.execute(
            text(
                """
                SELECT data_query_mode FROM (
                    SELECT JSON_VALUE(v.config_json, '$.data_query_mode') AS data_query_mode
                    FROM KBOT_KR_AGENT a
                    JOIN KBOT_KR_AGENT_VERSION v ON v.agent_id = a.agent_id
                    WHERE :consumer_app_id = 'knowledge_retrieval'
                      AND a.agent_id = :agent_id
                      AND v.agent_version_id = :agent_version_id
                      AND a.domain_id = :domain_id
                      AND a.status IN ('DRAFT', 'ACTIVE')
                    UNION ALL
                    SELECT JSON_VALUE(v.config_json, '$.data_query_mode') AS data_query_mode
                    FROM KBOT_OPS_AGENT a
                    JOIN KBOT_OPS_AGENT_VERSION v ON v.agent_id = a.agent_id
                    WHERE :consumer_app_id = 'aiops'
                      AND a.agent_id = :agent_id
                      AND v.agent_version_id = :agent_version_id
                      AND a.domain_id = :domain_id
                      AND a.status IN ('DRAFT', 'ACTIVE')
                )
                """
            ),
            {
                "consumer_app_id": consumer_app_id,
                "agent_id": agent_id.bytes,
                "agent_version_id": agent_version_id.bytes,
                "domain_id": domain_id,
            },
        )
        value = result.scalar_one_or_none()
        return None if value is None else str(value)
