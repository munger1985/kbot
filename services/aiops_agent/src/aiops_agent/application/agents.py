"""AIOps 私有 Agent 生命周期。"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError

from aiops_agent.entities import (
    AIOpsAgentEntity,
    AIOpsAgentGrantEntity,
    AIOpsAgentVersionEntity,
)
from platform_core.identity import uuid7


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageModelCapability(_Model):
    allowed_model_ids: tuple[UUID, ...] = Field(min_length=1, max_length=16)
    default_model_id: UUID

    @model_validator(mode="after")
    def validate_default_model(self):
        if self.default_model_id not in self.allowed_model_ids:
            raise ValueError("default_model_id 必须属于 allowed_model_ids")
        if len(set(self.allowed_model_ids)) != len(self.allowed_model_ids):
            raise ValueError("allowed_model_ids 不能重复")
        return self


class AgentImageCapabilities(_Model):
    ocr: ImageModelCapability | None = None
    vlm: ImageModelCapability | None = None


class CreateAIOpsAgentCommand(_Model):
    domain_id: int = Field(ge=1)
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    policy_id: UUID
    models: dict[str, UUID] = Field(default_factory=dict)
    image_capabilities: AgentImageCapabilities = Field(
        default_factory=AgentImageCapabilities
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"
    actor_id: str = Field(min_length=1, max_length=256)


class UpdateAIOpsAgentCommand(_Model):
    domain_id: int = Field(ge=1)
    agent_id: UUID
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    policy_id: UUID | None = None
    models: dict[str, UUID] | None = None
    image_capabilities: AgentImageCapabilities | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None
    actor_id: str = Field(min_length=1, max_length=256)


class UpsertAIOpsAgentGrantCommand(_Model):
    domain_id: int = Field(ge=1)
    agent_id: UUID
    subject_type: Literal["USER", "ROLE"]
    subject_id: str = Field(min_length=1, max_length=256)
    status: Literal["ACTIVE", "DISABLED"] = "ACTIVE"
    actor_id: str = Field(min_length=1, max_length=256)


class AIOpsAgentError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class AIOpsAgentService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def create(self, command: CreateAIOpsAgentCommand) -> dict[str, Any]:
        agent_id, version_id = uuid7(), uuid7()
        async with self._uow_factory() as uow:
            states = await uow.agents.resource_states(
                domain_id=command.domain_id,
                policy_id=command.policy_id,
            )
            self._validate_resources(
                states, command.status, command.model_dump()
            )
            agent = AIOpsAgentEntity(
                agent_id=agent_id,
                domain_id=command.domain_id,
                display_name=command.display_name.strip(),
                description=command.description,
                status=command.status,
                current_version_id=None,
                created_by=command.actor_id,
                updated_by=command.actor_id,
            )
            try:
                await uow.agents.add_agent(agent)
                await uow.agents.add_version(
                    self._new_version(
                        agent_id=agent_id,
                        version_id=version_id,
                        version_no=1,
                        values=command.model_dump(),
                        actor_id=command.actor_id,
                    )
                )
                agent.current_version_id = version_id
                await uow.commit()
            except IntegrityError as exc:
                raise AIOpsAgentError(
                    "AIOPS_AGENT_CONFLICT", "AIOps Agent 名称或资源组合冲突"
                ) from exc
        return await self.get(domain_id=command.domain_id, agent_id=agent_id)

    async def list(self, *, domain_id: int) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            return [
                self._view(agent, await self._version(uow.agents, agent))
                for agent in await uow.agents.list(domain_id=domain_id)
            ]

    async def get(self, *, domain_id: int, agent_id: UUID) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if agent is None:
                self._not_found()
            return self._view(agent, await self._version(uow.agents, agent))

    async def update(self, command: UpdateAIOpsAgentCommand) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(
                domain_id=command.domain_id,
                agent_id=command.agent_id,
                lock=True,
            )
            if agent is None:
                self._not_found()
            if int(agent.row_version) != command.expected_row_version:
                raise AIOpsAgentError(
                    "STATE_VERSION_CONFLICT", "Agent 配置版本已变化"
                )
            current = await self._version(uow.agents, agent)
            changes = command.model_dump(
                exclude={"domain_id", "agent_id", "expected_row_version", "actor_id"},
                exclude_unset=True,
            )
            effective = {
                "policy_id": changes.get("policy_id", current.policy_id),
                "models": changes.get("models", current.models_json),
                "image_capabilities": changes.get(
                    "image_capabilities", current.image_capabilities_json
                ),
                "instruction": changes.get("instruction", current.instruction),
                "config": changes.get("config", current.config_json),
            }
            states = await uow.agents.resource_states(
                domain_id=command.domain_id,
                policy_id=effective["policy_id"],
            )
            self._validate_resources(
                states, str(changes.get("status", agent.status)), effective
            )
            version_fields = {
                "policy_id",
                "models",
                "image_capabilities",
                "instruction",
                "config",
            }
            if version_fields.intersection(changes):
                version_id = uuid7()
                await uow.agents.add_version(
                    self._new_version(
                        agent_id=agent.agent_id,
                        version_id=version_id,
                        version_no=await uow.agents.next_version_no(
                            agent_id=agent.agent_id
                        ),
                        values=effective,
                        actor_id=command.actor_id,
                    )
                )
                agent.current_version_id = version_id
            for field in ("display_name", "description", "status"):
                if field in changes:
                    setattr(agent, field, changes[field])
            agent.updated_by = command.actor_id
            agent.row_version = int(agent.row_version) + 1
            await uow.commit()
        return await self.get(domain_id=command.domain_id, agent_id=command.agent_id)

    async def execution_spec(self, *, domain_id: int, agent_id: UUID):
        row = await self.get(domain_id=domain_id, agent_id=agent_id)
        if row["status"] != "ACTIVE":
            raise AIOpsAgentError(
                "AGENT_NOT_ACTIVE", "Agent 未启用，不能执行", status_code=422
            )
        return {
            "schema_version": "1.0",
            "owner_app_id": "aiops",
            "domain_id": str(domain_id),
            "consumer_agent_id": row["agent_id"],
            "consumer_agent_version_id": row["agent_version_id"],
            "agent_kind": "AIOPS",
            "display_name": row["display_name"],
            "enabled_capabilities": ["aiops"],
            "models": row["models"],
            "instruction": row["instruction"],
            "resource_context": {
                "policy_id": row["policy_id"],
                "image_capabilities": row["image_capabilities"],
                **row["config"],
            },
            "runtime_policy": {},
        }

    async def model_references(self, *, model_id: UUID) -> list[dict[str, Any]]:
        """返回模型在全部 AIOps 私有 Agent 当前版本中的引用。"""

        async with self._uow_factory() as uow:
            return [
                {
                    "service": "aiops-agent",
                    "domain_id": str(agent.domain_id),
                    "resource_type": "aiops_agent",
                    "resource_id": str(agent.agent_id),
                    "display_name": agent.display_name,
                    "status": agent.status,
                    "binding_role": role,
                }
                for agent, role in await uow.agents.model_references(
                    model_id=model_id
                )
            ]

    async def list_grants(self, *, domain_id: int):
        async with self._uow_factory() as uow:
            return [
                self._grant_view(row)
                for row in await uow.agents.list_grants(domain_id=domain_id)
            ]

    async def authorize(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        user_id: str,
        role_codes: tuple[str, ...],
    ) -> dict[str, object]:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get_active(
                domain_id=domain_id, agent_id=agent_id
            )
            allowed = await uow.agents.has_active_grant(
                domain_id=domain_id,
                agent_id=agent_id,
                user_id=user_id,
                role_codes=role_codes,
            )
        if agent is None or not allowed:
            raise AIOpsAgentError(
                "AIOPS_AGENT_ACCESS_DENIED",
                "当前用户无权使用该 AIOps Agent",
                status_code=403,
            )
        return {"allowed": True, "agent_id": str(agent_id)}

    async def upsert_grant(self, command: UpsertAIOpsAgentGrantCommand):
        async with self._uow_factory() as uow:
            if await uow.agents.get(
                domain_id=command.domain_id, agent_id=command.agent_id
            ) is None:
                self._not_found()
            row = await uow.agents.find_grant(
                domain_id=command.domain_id,
                agent_id=command.agent_id,
                subject_type=command.subject_type,
                subject_id=command.subject_id,
                lock=True,
            )
            if row is None:
                row = AIOpsAgentGrantEntity(
                    agent_grant_id=uuid7(),
                    domain_id=command.domain_id,
                    agent_id=command.agent_id,
                    subject_type=command.subject_type,
                    subject_id=command.subject_id,
                    status=command.status,
                    created_by=command.actor_id,
                    updated_by=command.actor_id,
                )
                await uow.agents.add_grant(row)
            else:
                row.status = command.status
                row.updated_by = command.actor_id
                row.row_version = int(row.row_version) + 1
            await uow.commit()
            return self._grant_view(row)

    async def update_grant_status(
        self,
        *,
        domain_id: int,
        grant_id: UUID,
        status: str,
        expected_row_version: int,
        actor_id: str,
    ):
        async with self._uow_factory() as uow:
            row = await uow.agents.get_grant(
                domain_id=domain_id, grant_id=grant_id, lock=True
            )
            if row is None:
                raise AIOpsAgentError(
                    "AIOPS_AGENT_GRANT_NOT_FOUND",
                    "Agent 授权不存在",
                    status_code=404,
                )
            if int(row.row_version) != expected_row_version:
                raise AIOpsAgentError(
                    "STATE_VERSION_CONFLICT", "Agent 授权版本已变化"
                )
            row.status = status
            row.updated_by = actor_id
            row.row_version = int(row.row_version) + 1
            await uow.commit()
            return self._grant_view(row)

    @staticmethod
    def _validate_resources(states, status: str, values) -> None:
        required = ("policy",)
        if any(states[item] is None for item in required):
            raise AIOpsAgentError(
                "AIOPS_AGENT_RESOURCE_NOT_FOUND", "Agent 引用的策略不存在"
            )
        if status == "ACTIVE":
            expected = {
                "policy": "ACTIVE",
            }
            for key, active in expected.items():
                if states[key] is not None and states[key] != active:
                    raise AIOpsAgentError(
                        "AIOPS_AGENT_RESOURCE_NOT_ACTIVE",
                        f"启用 Agent 前资源 {key} 必须处于 ACTIVE",
                        status_code=422,
                    )

    @staticmethod
    def _new_version(*, agent_id, version_id, version_no, values, actor_id):
        return AIOpsAgentVersionEntity(
            agent_version_id=version_id,
            agent_id=agent_id,
            version_no=version_no,
            policy_id=values["policy_id"],
            models_json={
                key: str(value)
                for key, value in dict(values.get("models") or {}).items()
            },
            image_capabilities_json=_image_capabilities_json(
                values.get("image_capabilities")
            ),
            instruction=values.get("instruction"),
            config_json=dict(values.get("config") or {}),
            created_by=actor_id,
        )

    @staticmethod
    async def _version(repository, agent):
        if agent.current_version_id is None:
            raise AIOpsAgentError("AIOPS_AGENT_VERSION_MISSING", "Agent 缺少当前版本")
        version = await repository.version(
            agent_id=agent.agent_id,
            agent_version_id=agent.current_version_id,
        )
        if version is None:
            raise AIOpsAgentError(
                "AIOPS_AGENT_VERSION_MISSING", "Agent 当前版本不存在"
            )
        return version

    @staticmethod
    def _view(agent, version):
        return {
            "agent_id": str(agent.agent_id),
            "domain_id": str(agent.domain_id),
            "display_name": agent.display_name,
            "description": agent.description,
            "status": agent.status,
            "agent_version_id": str(version.agent_version_id),
            "version_no": int(version.version_no),
            "policy_id": str(version.policy_id),
            "models": dict(version.models_json or {}),
            "instruction": version.instruction,
            "image_capabilities": dict(version.image_capabilities_json or {}),
            "config": dict(version.config_json or {}),
            "row_version": int(agent.row_version),
        }

    @staticmethod
    def _grant_view(row):
        return {
            "agent_grant_id": str(row.agent_grant_id),
            "domain_id": str(row.domain_id),
            "agent_id": str(row.agent_id),
            "subject_type": row.subject_type,
            "subject_id": row.subject_id,
            "status": row.status,
            "row_version": int(row.row_version),
        }

    @staticmethod
    def _not_found():
        raise AIOpsAgentError(
            "AIOPS_AGENT_NOT_FOUND", "AIOps Agent 不存在", status_code=404
        )


def _image_capabilities_json(value) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = (
        value
        if isinstance(value, AgentImageCapabilities)
        else AgentImageCapabilities.model_validate(value)
    )
    result: dict[str, Any] = {}
    for mode in ("ocr", "vlm"):
        capability = getattr(parsed, mode)
        if capability is not None:
            result[mode] = {
                "allowed_model_ids": [
                    str(item) for item in capability.allowed_model_ids
                ],
                "default_model_id": str(capability.default_model_id),
            }
    return result
