"""智能工作台 Agent 生命周期和不可变执行规格。"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from assistant_app.entities import AssistantAgentEntity, AssistantAgentVersionEntity
from platform_core.identity import uuid7


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateAgentCommand(_Model):
    domain_id: int = Field(ge=1)
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    models: dict[str, UUID] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"
    actor_id: str = Field(min_length=1, max_length=256)


class UpdateAgentCommand(_Model):
    domain_id: int = Field(ge=1)
    agent_id: UUID
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=None, max_length=100)
    models: dict[str, UUID] | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None
    actor_id: str = Field(min_length=1, max_length=256)


class AgentApplicationError(ValueError):
    """将领域校验转换为稳定的内部 API 错误。"""

    def __init__(self, code: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class AssistantAgentService:
    """只管理智能工作台自己的 Agent 配置和版本，不复制 KC 或问数事实。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def create(self, command: CreateAgentCommand) -> dict[str, Any]:
        self._validate(command.status, command.knowledge_core_id, command.data_model_ids)
        agent_id, version_id = uuid7(), uuid7()
        async with self._uow_factory() as uow:
            assert uow.agents is not None
            agent = AssistantAgentEntity(
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
                await uow.agents.add_version(self._version(
                    version_id=version_id, agent_id=agent_id, version_no=1,
                    knowledge_core_id=command.knowledge_core_id,
                    data_model_ids=command.data_model_ids, models=command.models,
                    instruction=command.instruction, config=command.config,
                    actor_id=command.actor_id,
                ))
                agent.current_version_id = version_id
                await uow.commit()
            except IntegrityError as exc:
                raise AgentApplicationError(
                    "AGENT_NAME_CONFLICT", "当前 Domain 已存在同名 Agent"
                ) from exc
        return await self.get(domain_id=command.domain_id, agent_id=agent_id)

    async def list(self, *, domain_id: int) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.agents is not None
            return [
                self._view(agent, await self._current_version(uow.agents, agent))
                for agent in await uow.agents.list(domain_id=domain_id)
            ]

    async def get(self, *, domain_id: int, agent_id: UUID) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            assert uow.agents is not None
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if agent is None:
                self._not_found()
            return self._view(agent, await self._current_version(uow.agents, agent))

    async def update(self, command: UpdateAgentCommand) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            assert uow.agents is not None
            agent = await uow.agents.get(
                domain_id=command.domain_id, agent_id=command.agent_id, lock=True
            )
            if agent is None:
                self._not_found()
            if int(agent.row_version) != command.expected_row_version:
                raise AgentApplicationError(
                    "STATE_VERSION_CONFLICT", "Agent 配置版本已变化"
                )
            current = await self._current_version(uow.agents, agent)
            changes = command.model_dump(
                exclude={"domain_id", "agent_id", "expected_row_version", "actor_id"},
                exclude_unset=True,
            )
            version_fields = {
                "knowledge_core_id", "data_model_ids", "models", "instruction", "config"
            }
            version_changed = bool(version_fields.intersection(changes))
            knowledge_core_id = changes.get("knowledge_core_id", current.knowledge_core_id)
            data_model_ids = changes.get(
                "data_model_ids", current.data_model_ids_json
            )
            target_status = changes.get("status", agent.status)
            self._validate(target_status, knowledge_core_id, data_model_ids)
            if version_changed:
                version_id = uuid7()
                await uow.agents.add_version(self._version(
                    version_id=version_id, agent_id=agent.agent_id,
                    version_no=await uow.agents.next_version_no(agent_id=agent.agent_id),
                    knowledge_core_id=knowledge_core_id,
                    data_model_ids=data_model_ids,
                    models=changes.get("models", current.models_json),
                    instruction=changes.get("instruction", current.instruction),
                    config=changes.get("config", current.config_json),
                    actor_id=command.actor_id,
                ))
                agent.current_version_id = version_id
            for field in ("display_name", "description", "status"):
                if field in changes:
                    setattr(agent, field, changes[field])
            agent.updated_by = command.actor_id
            agent.row_version = int(agent.row_version) + 1
            try:
                await uow.commit()
            except IntegrityError as exc:
                raise AgentApplicationError(
                    "AGENT_NAME_CONFLICT", "当前 Domain 已存在同名 Agent"
                ) from exc
        return await self.get(domain_id=command.domain_id, agent_id=command.agent_id)

    @staticmethod
    def _validate(status: str, knowledge_core_id: UUID | None, data_model_ids) -> None:
        if len({str(value) for value in data_model_ids}) != len(data_model_ids):
            raise AgentApplicationError(
                "AGENT_DATA_MODEL_DUPLICATED", "问数模型不能重复绑定", status_code=422
            )
        if status == "ACTIVE" and knowledge_core_id is None:
            raise AgentApplicationError(
                "AGENT_KNOWLEDGE_CORE_REQUIRED", "启用 Agent 前必须绑定一个可用 Knowledge Core", status_code=422
            )

    @staticmethod
    def _version(*, version_id, agent_id, version_no, knowledge_core_id, data_model_ids, models, instruction, config, actor_id):
        return AssistantAgentVersionEntity(
            agent_version_id=version_id,
            agent_id=agent_id,
            version_no=version_no,
            knowledge_core_id=knowledge_core_id,
            data_model_ids_json=[str(value) for value in data_model_ids],
            models_json={str(role).strip(): str(UUID(str(model_id))) for role, model_id in models.items()},
            instruction=instruction,
            config_json=dict(config or {}),
            created_by=actor_id,
        )

    @staticmethod
    async def _current_version(repository, agent):
        if agent.current_version_id is None:
            raise AgentApplicationError("AGENT_VERSION_MISSING", "Agent 缺少当前版本")
        version = await repository.current_version(
            agent_id=agent.agent_id, agent_version_id=agent.current_version_id
        )
        if version is None:
            raise AgentApplicationError("AGENT_VERSION_MISSING", "Agent 当前版本不存在")
        return version

    @staticmethod
    def _view(agent, version) -> dict[str, Any]:
        data_model_ids = list(version.data_model_ids_json or [])
        return {
            "agent_id": str(agent.agent_id), "domain_id": int(agent.domain_id),
            "display_name": agent.display_name, "description": agent.description,
            "status": agent.status, "row_version": int(agent.row_version),
            "agent_version_id": str(version.agent_version_id),
            "version_no": int(version.version_no),
            "knowledge_core_id": str(version.knowledge_core_id) if version.knowledge_core_id else None,
            "data_model_ids": data_model_ids, "models": dict(version.models_json or {}),
            "instruction": version.instruction, "config": dict(version.config_json or {}),
            "enabled_capabilities": ["conversation", "document"] + (["data_query"] if data_model_ids else []),
            "created_at": agent.created_at, "updated_at": agent.updated_at,
        }

    @staticmethod
    def _not_found() -> None:
        raise AgentApplicationError("AGENT_NOT_FOUND", "Agent 不存在", status_code=404)
