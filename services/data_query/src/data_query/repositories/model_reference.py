"""Data Query 对模型目录 UUID 的反向引用查询。"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from data_query.entities import (
    DataQueryRunEntity,
    SemanticModelGenerationJobEntity,
    SemanticModelVersionEntity,
)


def _contains_model_id(value: Any, expected: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_model_id(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_model_id(item, expected) for item in value)
    return isinstance(value, str) and value == expected


class DataQueryModelReferenceRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def list_for_model(self, *, model_id: UUID) -> list[dict[str, str]]:
        """查询配置与运行中任务，不把已完成任务永久视为阻塞引用。"""
        expected = str(model_id)
        references: list[dict[str, str]] = []
        versions = list(
            (await self._session.execute(select(SemanticModelVersionEntity))).scalars()
        )
        references.extend(
            {
                "service": "data-query",
                "resource_type": "semantic_model_version",
                "resource_id": str(row.semantic_model_version_id),
                "usage": "generation_model",
            }
            for row in versions
            if _contains_model_id(row.definition_json, expected)
        )
        jobs = list((await self._session.execute(
            select(SemanticModelGenerationJobEntity).where(
                SemanticModelGenerationJobEntity.status.in_(("QUEUED", "RUNNING"))
            )
        )).scalars())
        references.extend(
            {
                "service": "data-query",
                "resource_type": "semantic_model_generation_job",
                "resource_id": str(row.generation_job_id),
                "usage": "running_generation",
            }
            for row in jobs
            if _contains_model_id(row.request_json, expected)
        )
        runs = list((await self._session.execute(
            select(DataQueryRunEntity).where(
                DataQueryRunEntity.status.in_(
                    ("CREATED", "PLANNED", "QUEUED", "RUNNING")
                )
            )
        )).scalars())
        references.extend(
            {
                "service": "data-query",
                "resource_type": "run",
                "resource_id": str(row.data_query_run_id),
                "usage": "running_query",
            }
            for row in runs
            if any(
                _contains_model_id(payload, expected)
                for payload in (
                    row.plan_snapshot_json,
                    row.policy_snapshot_json,
                    row.semantic_model_snapshot_json,
                )
            )
        )
        return references
