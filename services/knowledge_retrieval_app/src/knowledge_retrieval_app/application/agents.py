"""知识检索 Agent 生命周期和不可变执行规格。"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from knowledge_retrieval_app.entities import (
    KnowledgeRetrievalAgentEntity,
    KnowledgeRetrievalAgentVersionEntity,
)
from platform_core.identity import uuid7


KnowledgeCapability = Literal["conversation", "document", "data_query"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateAgentCommand(_Model):
    domain_id: int = Field(ge=1)
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[KnowledgeCapability, ...] = Field(
        min_length=1, max_length=3
    )
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
    enabled_capabilities: tuple[KnowledgeCapability, ...] | None = Field(
        default=None, min_length=1, max_length=3
    )
    models: dict[str, UUID] | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None
    actor_id: str = Field(min_length=1, max_length=256)


class AgentApplicationError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class KnowledgeRetrievalAgentService:
    CAPABILITIES = ("conversation", "document", "data_query")
    IMMUTABLE_MODEL_ROLES = frozenset({"memory_embedding"})

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def create(self, command: CreateAgentCommand) -> dict[str, Any]:
        capabilities = self._capabilities(command.enabled_capabilities)
        models = self._models(command.models)
        config = self._config(command.config, capabilities)
        self._validate_activation(
            status=command.status,
            models=models,
            capabilities=capabilities,
            config=config,
        )
        agent_id, version_id = uuid7(), uuid7()
        async with self._uow_factory() as uow:
            agent = KnowledgeRetrievalAgentEntity(
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
                    KnowledgeRetrievalAgentVersionEntity(
                        agent_version_id=version_id,
                        agent_id=agent_id,
                        version_no=1,
                        enabled_capabilities_json=list(capabilities),
                        models_json=models,
                        instruction=command.instruction,
                        config_json=config,
                        created_by=command.actor_id,
                    )
                )
                agent.current_version_id = version_id
                await uow.commit()
            except IntegrityError as exc:
                raise AgentApplicationError(
                    "AGENT_NAME_CONFLICT", "当前 Domain 已存在同名 Agent"
                ) from exc
        return await self.get(domain_id=command.domain_id, agent_id=agent_id)

    async def list(self, *, domain_id: int) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            result = []
            for agent in await uow.agents.list(domain_id=domain_id):
                result.append(self._view(agent, await self._version(uow.agents, agent)))
            return result

    async def get(self, *, domain_id: int, agent_id: UUID) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if agent is None:
                self._not_found()
            return self._view(agent, await self._version(uow.agents, agent))

    async def update(self, command: UpdateAgentCommand) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(
                domain_id=command.domain_id, agent_id=command.agent_id, lock=True
            )
            if agent is None:
                self._not_found()
            if int(agent.row_version) != command.expected_row_version:
                raise AgentApplicationError(
                    "STATE_VERSION_CONFLICT", "Agent 配置版本已变化"
                )
            current = await self._version(uow.agents, agent)
            changes = command.model_dump(
                exclude={"domain_id", "agent_id", "expected_row_version", "actor_id"},
                exclude_unset=True,
            )
            version_changed = bool(
                {
                    "enabled_capabilities",
                    "models",
                    "instruction",
                    "config",
                }.intersection(changes)
            )
            requested_capabilities = changes.get("enabled_capabilities")
            capabilities = self._capabilities(
                requested_capabilities
                if requested_capabilities is not None
                else current.enabled_capabilities_json
            )
            models = self._models(changes.get("models", current.models_json))
            config = self._config(
                changes.get("config", current.config_json or {}),
                capabilities,
            )
            self._validate_activation(
                status=str(changes.get("status", agent.status)),
                models=models,
                capabilities=capabilities,
                config=config,
            )
            for role in self.IMMUTABLE_MODEL_ROLES:
                existing = dict(current.models_json or {}).get(role)
                if existing and models.get(role) != existing:
                    raise AgentApplicationError(
                        "AGENT_MODEL_IMMUTABLE",
                        f"模型角色 {role} 一经设定禁止更换或删除",
                    )
            if version_changed:
                version_id = uuid7()
                await uow.agents.add_version(
                    KnowledgeRetrievalAgentVersionEntity(
                        agent_version_id=version_id,
                        agent_id=agent.agent_id,
                        version_no=await uow.agents.next_version_no(
                            agent_id=agent.agent_id
                        ),
                        enabled_capabilities_json=list(capabilities),
                        models_json=models,
                        instruction=changes.get("instruction", current.instruction),
                        config_json=config,
                        created_by=command.actor_id,
                    )
                )
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

    async def execution_spec(
        self, *, domain_id: int, agent_id: UUID
    ) -> dict[str, Any]:
        row = await self.get(domain_id=domain_id, agent_id=agent_id)
        if row["status"] != "ACTIVE":
            raise AgentApplicationError(
                "AGENT_NOT_ACTIVE", "Agent 未启用，不能创建运行", status_code=422
            )
        return {
            "schema_version": "1.0",
            "owner_app_id": "knowledge_retrieval",
            "domain_id": str(domain_id),
            "consumer_agent_id": row["agent_id"],
            "consumer_agent_version_id": row["agent_version_id"],
            "agent_kind": "KNOWLEDGE_RETRIEVAL",
            "display_name": row["display_name"],
            "enabled_capabilities": row["enabled_capabilities"],
            "models": row["models"],
            "instruction": row["instruction"],
            "resource_context": row["config"],
            "runtime_policy": {},
        }

    async def model_references(self, *, model_id: UUID) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            return [
                {
                    "service": "knowledge-retrieval-app",
                    "domain_id": str(agent.domain_id),
                    "resource_type": "knowledge_retrieval_agent",
                    "resource_id": str(agent.agent_id),
                    "display_name": agent.display_name,
                    "status": agent.status,
                    "binding_role": role,
                }
                for agent, role in await uow.agents.model_references(model_id=model_id)
            ]

    @staticmethod
    def _models(models: dict[str, UUID | str]) -> dict[str, str]:
        return {
            str(role).strip(): str(UUID(str(model_id)))
            for role, model_id in models.items()
        }

    @classmethod
    def _capabilities(
        cls, capabilities: tuple[str, ...] | list[str]
    ) -> tuple[str, ...]:
        """校验知识检索 Agent 能力并按稳定顺序保存。"""
        values = tuple(str(value).strip() for value in capabilities)
        if not values:
            raise AgentApplicationError(
                "AGENT_CAPABILITIES_REQUIRED",
                "知识检索 Agent 至少需要启用一项能力",
                status_code=422,
            )
        if len(set(values)) != len(values):
            raise AgentApplicationError(
                "AGENT_CAPABILITY_DUPLICATED",
                "知识检索 Agent 能力不能重复",
                status_code=422,
            )
        unsupported = sorted(set(values) - set(cls.CAPABILITIES))
        if unsupported:
            raise AgentApplicationError(
                "AGENT_CAPABILITY_UNSUPPORTED",
                f"知识检索 Agent 不支持能力：{unsupported}",
                status_code=422,
            )
        return tuple(value for value in cls.CAPABILITIES if value in values)

    @staticmethod
    def _config(
        config: dict[str, Any], capabilities: tuple[str, ...]
    ) -> dict[str, Any]:
        """由能力组合推导资源模式，避免前端提交相互矛盾的状态。"""
        normalized = dict(config)
        normalized["resource_mode"] = (
            "conversation_only"
            if capabilities == ("conversation",)
            else "managed_resources"
        )
        return normalized

    @staticmethod
    def _validate_activation(
        *,
        status: str,
        models: dict[str, str],
        capabilities: tuple[str, ...],
        config: dict[str, Any],
    ) -> None:
        if status != "ACTIVE":
            return
        if not models:
            raise AgentApplicationError(
                "AGENT_MODELS_REQUIRED",
                "启用 Agent 前必须完成模型配置",
                status_code=422,
            )
        if len(capabilities) > 1 and "router_llm" not in models:
            raise AgentApplicationError(
                "AGENT_ROUTER_MODEL_REQUIRED",
                "启用多能力 Agent 前必须配置 router_llm",
                status_code=422,
            )
        if "data_query" not in capabilities:
            return
        mode = str(config.get("data_query_mode") or "").upper()
        if mode not in {"MCP", "SEMANTIC"}:
            raise AgentApplicationError(
                "AGENT_DATA_QUERY_MODE_REQUIRED",
                "启用业务问数能力前必须选择 MCP 或 SEMANTIC 模式",
                status_code=422,
            )
        if mode == "MCP" and not str(
            config.get("data_profile_name") or ""
        ).strip():
            raise AgentApplicationError(
                "AGENT_DATA_PROFILE_REQUIRED",
                "MCP 问数模式必须配置 data_profile_name",
                status_code=422,
            )

    @staticmethod
    async def _version(repository, agent):
        if agent.current_version_id is None:
            raise AgentApplicationError("AGENT_VERSION_MISSING", "Agent 缺少当前版本")
        version = await repository.current_version(
            agent_id=agent.agent_id,
            agent_version_id=agent.current_version_id,
        )
        if version is None:
            raise AgentApplicationError(
                "AGENT_VERSION_MISSING", "Agent 当前版本不存在"
            )
        return version

    @staticmethod
    def _view(agent, version) -> dict[str, Any]:
        return {
            "agent_id": str(agent.agent_id),
            "domain_id": str(agent.domain_id),
            "display_name": agent.display_name,
            "description": agent.description,
            "status": agent.status,
            "agent_version_id": str(version.agent_version_id),
            "version_no": int(version.version_no),
            "enabled_capabilities": list(version.enabled_capabilities_json or []),
            "models": dict(version.models_json or {}),
            "instruction": version.instruction,
            "config": dict(version.config_json or {}),
            "row_version": int(agent.row_version),
        }

    @staticmethod
    def _not_found() -> None:
        raise AgentApplicationError("AGENT_NOT_FOUND", "Agent 不存在", status_code=404)
