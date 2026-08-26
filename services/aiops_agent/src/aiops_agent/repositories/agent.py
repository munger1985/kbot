"""AIOps Agent 聚合 Repository。"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    AIOpsAgentEntity,
    AIOpsAgentGrantEntity,
    AIOpsAgentVersionEntity,
    AIOpsAgentVersionSourceEntity,
    PolicyEntity,
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

    async def add_version_sources(
        self, *, version_id: UUID, source_ids: tuple[UUID, ...]
    ) -> None:
        self._write_guard()
        self._session.add_all(
            AIOpsAgentVersionSourceEntity(
                agent_version_id=version_id,
                diagnostic_source_id=source_id,
            )
            for source_id in source_ids
        )
        await self._session.flush()

    async def version_source_ids(self, *, agent_version_id: UUID) -> list[UUID]:
        rows = await self._session.scalars(
            select(AIOpsAgentVersionSourceEntity.diagnostic_source_id)
            .where(
                AIOpsAgentVersionSourceEntity.agent_version_id
                == agent_version_id
            )
            .order_by(AIOpsAgentVersionSourceEntity.diagnostic_source_id)
        )
        return list(rows)

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
        policy_id: UUID,
    ) -> dict[str, str | None]:
        policy = (
            await self._session.execute(
                select(PolicyEntity.status).where(
                    PolicyEntity.domain_id == domain_id,
                    PolicyEntity.policy_id == policy_id,
                )
            )
        ).scalar_one_or_none()
        return {"policy": policy}

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
        if row is None:
            return None
        agent, version = row
        source_ids = await self.version_source_ids(
            agent_version_id=version.agent_version_id
        )
        policy = await self._session.get(PolicyEntity, version.policy_id)
        return _execution_binding(agent, version, source_ids, policy)

    async def resolve_auto_alert(
        self, *, domain_id: int, source_id: UUID, target_id: UUID
    ):
        """选择一个明确订阅该监控源的启用 Agent，精确 Target 优先。"""
        rows = await self._session.execute(
            select(AIOpsAgentEntity, AIOpsAgentVersionEntity, PolicyEntity)
            .join(
                AIOpsAgentVersionEntity,
                AIOpsAgentVersionEntity.agent_version_id
                == AIOpsAgentEntity.current_version_id,
            )
            .join(
                AIOpsAgentVersionSourceEntity,
                AIOpsAgentVersionSourceEntity.agent_version_id
                == AIOpsAgentVersionEntity.agent_version_id,
            )
            .join(PolicyEntity, PolicyEntity.policy_id == AIOpsAgentVersionEntity.policy_id)
            .where(
                AIOpsAgentEntity.domain_id == domain_id,
                AIOpsAgentEntity.status == "ACTIVE",
                AIOpsAgentVersionSourceEntity.diagnostic_source_id == source_id,
                or_(
                    AIOpsAgentVersionEntity.target_id == target_id,
                    AIOpsAgentVersionEntity.target_id.is_(None),
                ),
                PolicyEntity.status == "ACTIVE",
            )
            .order_by(
                case(
                    (AIOpsAgentVersionEntity.target_id == target_id, 0),
                    else_=1,
                ),
                AIOpsAgentEntity.agent_id,
            )
        )
        for agent, version, policy in rows:
            if bool(policy.rules_json.get("auto_alert_enabled", True)):
                source_ids = await self.version_source_ids(
                    agent_version_id=version.agent_version_id
                )
                return _execution_binding(agent, version, source_ids, policy), policy
        return None


@dataclass(frozen=True)
class AIOpsAgentExecutionBinding:
    binding_id: UUID
    agent_id: UUID
    policy_id: UUID
    target_id: UUID | None
    diagnostic_source_ids: tuple[UUID, ...]
    status: str
    row_version: int
    allow_mutation: bool
    allowed_actions_json: list[str]
    instruction: str | None


def _execution_binding(
    agent: AIOpsAgentEntity,
    version: AIOpsAgentVersionEntity,
    source_ids: list[UUID],
    policy: PolicyEntity | None,
):
    rules: dict[str, Any] = dict(policy.rules_json or {}) if policy else {}
    return AIOpsAgentExecutionBinding(
        binding_id=version.agent_version_id,
        agent_id=agent.agent_id,
        policy_id=version.policy_id,
        target_id=version.target_id,
        diagnostic_source_ids=tuple(source_ids),
        status=agent.status,
        row_version=int(agent.row_version),
        allow_mutation=bool(rules.get("allow_agent_execution", False)),
        allowed_actions_json=list(rules.get("allowed_action_types") or [])
        if rules.get("allow_agent_execution", False) else [],
        instruction=version.instruction,
    )
