"""AIOps Agent 聚合 Repository。"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    AIOpsAgentEntity,
    AIOpsAgentGrantEntity,
    AIOpsAgentVersionEntity,
    InspectionPlanEntity,
    MonitorSourceEntity,
    PolicyEntity,
    TargetEntity,
)


class AIOpsAgentRepository:
    def __init__(self, session: AsyncSession, write_guard):
        self._session = session
        self._write_guard = write_guard

    async def add_agent(self, row: AIOpsAgentEntity) -> None:
        self._write_guard()
        self._session.add(row)
        await self._session.flush()

    async def add_version(self, row: AIOpsAgentVersionEntity) -> None:
        self._write_guard()
        self._session.add(row)
        await self._session.flush()

    async def add_grant(self, row: AIOpsAgentGrantEntity) -> None:
        self._write_guard()
        self._session.add(row)
        await self._session.flush()

    async def get(
        self, *, domain_id: int, agent_id: UUID, lock: bool = False
    ) -> AIOpsAgentEntity | None:
        statement = select(AIOpsAgentEntity).where(
            AIOpsAgentEntity.domain_id == domain_id,
            AIOpsAgentEntity.agent_id == agent_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list(self, *, domain_id: int) -> list[AIOpsAgentEntity]:
        rows = await self._session.scalars(
            select(AIOpsAgentEntity)
            .where(AIOpsAgentEntity.domain_id == domain_id)
            .order_by(AIOpsAgentEntity.updated_at.desc(), AIOpsAgentEntity.agent_id)
        )
        return list(rows)

    async def version(self, *, agent_id: UUID, agent_version_id: UUID):
        statement = select(AIOpsAgentVersionEntity).where(
            AIOpsAgentVersionEntity.agent_id == agent_id,
            AIOpsAgentVersionEntity.agent_version_id == agent_version_id,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def next_version_no(self, *, agent_id: UUID) -> int:
        value = await self._session.scalar(
            select(func.max(AIOpsAgentVersionEntity.version_no)).where(
                AIOpsAgentVersionEntity.agent_id == agent_id
            )
        )
        return int(value or 0) + 1

    async def resource_states(
        self,
        *,
        domain_id: int,
        monitor_source_id: UUID,
        policy_id: UUID,
        target_id: UUID | None,
        inspection_plan_id: UUID | None,
    ) -> dict[str, str | None]:
        monitor = (
            await self._session.execute(
                select(MonitorSourceEntity.status).where(
                    MonitorSourceEntity.domain_id == domain_id,
                    MonitorSourceEntity.monitor_source_id == monitor_source_id,
                )
            )
        ).scalar_one_or_none()
        policy = (
            await self._session.execute(
                select(PolicyEntity.status).where(
                    PolicyEntity.domain_id == domain_id,
                    PolicyEntity.policy_id == policy_id,
                )
            )
        ).scalar_one_or_none()
        target = None
        if target_id is not None:
            target = (
                await self._session.execute(
                    select(TargetEntity.status).where(
                        TargetEntity.domain_id == domain_id,
                        TargetEntity.target_id == target_id,
                    )
                )
            ).scalar_one_or_none()
        plan = None
        if inspection_plan_id is not None:
            plan = (
                await self._session.execute(
                    select(InspectionPlanEntity.status).where(
                        InspectionPlanEntity.domain_id == domain_id,
                        InspectionPlanEntity.inspection_plan_id
                        == inspection_plan_id,
                    )
                )
            ).scalar_one_or_none()
        return {
            "monitor": monitor,
            "policy": policy,
            "target": target,
            "plan": plan,
        }

    async def model_references(self, *, model_id: UUID):
        rows = await self._session.execute(
            select(AIOpsAgentEntity, AIOpsAgentVersionEntity).join(
                AIOpsAgentVersionEntity,
                AIOpsAgentVersionEntity.agent_version_id
                == AIOpsAgentEntity.current_version_id,
            )
        )
        materialized = list(rows)
        expected = str(model_id)
        regular = [
            (agent, role)
            for agent, version in materialized
            for role, value in dict(version.models_json or {}).items()
            if str(value) == expected
        ]
        images = [
            (agent, f"image_{mode}")
            for agent, version in materialized
            for mode, capability in dict(
                version.image_capabilities_json or {}
            ).items()
            if expected
            in {
                str(item)
                for item in dict(capability or {}).get("allowed_model_ids", ())
            }
        ]
        return regular + images

    async def list_grants(self, *, domain_id: int):
        rows = await self._session.scalars(
            select(AIOpsAgentGrantEntity)
            .where(AIOpsAgentGrantEntity.domain_id == domain_id)
            .order_by(
                AIOpsAgentGrantEntity.updated_at.desc(),
                AIOpsAgentGrantEntity.agent_grant_id,
            )
        )
        return list(rows)

    async def find_grant(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        subject_type: str,
        subject_id: str,
        lock: bool = False,
    ):
        statement = select(AIOpsAgentGrantEntity).where(
            AIOpsAgentGrantEntity.domain_id == domain_id,
            AIOpsAgentGrantEntity.agent_id == agent_id,
            AIOpsAgentGrantEntity.subject_type == subject_type,
            AIOpsAgentGrantEntity.subject_id == subject_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_grant(
        self, *, domain_id: int, grant_id: UUID, lock: bool = False
    ):
        statement = select(AIOpsAgentGrantEntity).where(
            AIOpsAgentGrantEntity.domain_id == domain_id,
            AIOpsAgentGrantEntity.agent_grant_id == grant_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def has_active_grant(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        user_id: str,
        role_codes: tuple[str, ...],
    ) -> bool:
        subjects = [
            (AIOpsAgentGrantEntity.subject_type == "USER")
            & (AIOpsAgentGrantEntity.subject_id == user_id)
        ]
        if role_codes:
            subjects.append(
                (AIOpsAgentGrantEntity.subject_type == "ROLE")
                & AIOpsAgentGrantEntity.subject_id.in_(role_codes)
            )
        value = await self._session.scalar(
            select(AIOpsAgentGrantEntity.agent_grant_id).where(
                AIOpsAgentGrantEntity.domain_id == domain_id,
                AIOpsAgentGrantEntity.agent_id == agent_id,
                AIOpsAgentGrantEntity.status == "ACTIVE",
                or_(*subjects),
            )
        )
        return value is not None

    async def get_active(
        self, *, domain_id: int, agent_id: UUID, lock: bool = False
    ):
        statement = (
            select(AIOpsAgentEntity, AIOpsAgentVersionEntity)
            .join(
                AIOpsAgentVersionEntity,
                AIOpsAgentVersionEntity.agent_version_id
                == AIOpsAgentEntity.current_version_id,
            )
            .where(
                AIOpsAgentEntity.domain_id == domain_id,
                AIOpsAgentEntity.agent_id == agent_id,
                AIOpsAgentEntity.status == "ACTIVE",
            )
        )
        if lock:
            statement = statement.with_for_update()
        row = (await self._session.execute(statement)).one_or_none()
        return None if row is None else _execution_binding(*row)


@dataclass(frozen=True)
class AIOpsAgentExecutionBinding:
    binding_id: UUID
    agent_id: UUID
    target_id: UUID | None
    monitor_source_id: UUID
    inspection_plan_id: UUID | None
    policy_id: UUID
    status: str
    row_version: int
    allow_mutation: bool
    allowed_actions_json: list[str]
    instruction: str | None


def _execution_binding(agent: AIOpsAgentEntity, version: AIOpsAgentVersionEntity):
    config: dict[str, Any] = dict(version.config_json or {})
    return AIOpsAgentExecutionBinding(
        binding_id=version.agent_version_id,
        agent_id=agent.agent_id,
        target_id=version.target_id,
        monitor_source_id=version.monitor_source_id,
        inspection_plan_id=version.inspection_plan_id,
        policy_id=version.policy_id,
        status=agent.status,
        row_version=int(agent.row_version),
        allow_mutation=bool(config.get("allow_mutation", False)),
        allowed_actions_json=list(config.get("allowed_actions") or []),
        instruction=version.instruction,
    )
