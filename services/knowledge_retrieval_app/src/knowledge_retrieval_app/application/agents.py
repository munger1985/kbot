"""知识检索应用 Agent 生命周期和不可变执行规格。"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from knowledge_retrieval_app.entities import KnowledgeRetrievalAgentEntity, KnowledgeRetrievalAgentVersionEntity
from knowledge_retrieval_app.application.catalog import is_grok_model, load_catalog_model
from platform_core.dictionary import ModelCategory, coerce_model_category
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


class KnowledgeRetrievalAgentService:
    """只管理知识检索应用自己的 Agent 配置和版本，不复制 KC 或问数事实。"""

    _REQUIRED_MODEL_ROLES = frozenset(
        {"context_llm", "composer_llm", "memory_llm", "memory_embedding"}
    )

    def __init__(self, *, uow_factory, catalog_client=None):
        self._uow_factory = uow_factory
        self._catalog = catalog_client

    async def create(self, command: CreateAgentCommand) -> dict[str, Any]:
        await self._validate_x_search_model(command.models)
        self._validate(
            command.status,
            command.knowledge_core_id,
            command.data_model_ids,
            command.models,
        )
        agent_id, version_id = uuid7(), uuid7()
        async with self._uow_factory() as uow:
            assert uow.agents is not None
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

    async def execution_spec(
        self, *, domain_id: int, agent_id: UUID
    ) -> dict[str, Any]:
        """签发供 Agent Runtime 冻结保存的知识检索应用执行规格。"""
        row = await self.get(domain_id=domain_id, agent_id=agent_id)
        if row["status"] != "ACTIVE":
            raise AgentApplicationError(
                "AGENT_NOT_ACTIVE", "Agent 未启用，不能创建会话", status_code=422
            )
        knowledge_core_id = row.get("knowledge_core_id")
        if not knowledge_core_id:
            raise AgentApplicationError(
                "AGENT_KNOWLEDGE_CORE_REQUIRED",
                "Agent 缺少可用 Knowledge Core",
                status_code=422,
            )
        data_model_ids = list(row.get("data_model_ids") or [])
        resource_context = {
            **dict(row.get("config") or {}),
            "resource_mode": "managed_resources",
            "collection_ids": [knowledge_core_id],
            "semantic_model_ids": data_model_ids,
        }
        if data_model_ids:
            resource_context["data_query_mode"] = "SEMANTIC"
        return {
            "schema_version": "1.0",
            "owner_app_id": "knowledge_retrieval",
            "domain_id": domain_id,
            "consumer_agent_id": row["agent_id"],
            "consumer_agent_version_id": row["agent_version_id"],
            "agent_kind": "KNOWLEDGE_RETRIEVAL",
            "display_name": row["display_name"],
            "enabled_capabilities": [
                capability
                for capability in row["enabled_capabilities"]
                if capability in {"conversation", "document", "data_query"}
            ],
            "models": row["models"],
            "instruction": row["instruction"],
            "resource_context": resource_context,
            "runtime_policy": {
                "routing": "document_data_and_conversation",
                "allow_general_conversation": True,
            },
        }

    async def model_references(self, *, model_id: UUID) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.agents is not None
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
            self._validate(
                target_status,
                knowledge_core_id,
                data_model_ids,
                changes.get("models", current.models_json),
            )
            await self._validate_x_search_model(
                changes.get("models", current.models_json)
            )
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

    @classmethod
    def _validate(
        cls, status: str, knowledge_core_id: UUID | None, data_model_ids, models
    ) -> None:
        if len({str(value) for value in data_model_ids}) != len(data_model_ids):
            raise AgentApplicationError(
                "AGENT_DATA_MODEL_DUPLICATED", "问数模型不能重复绑定", status_code=422
            )
        if status == "ACTIVE" and knowledge_core_id is None:
            raise AgentApplicationError(
                "AGENT_KNOWLEDGE_CORE_REQUIRED", "启用 Agent 前必须绑定一个可用 Knowledge Core", status_code=422
            )
        if status != "ACTIVE":
            return
        missing = sorted(cls._REQUIRED_MODEL_ROLES - set(models or {}))
        if missing:
            raise AgentApplicationError(
                "AGENT_MODELS_REQUIRED",
                f"启用 Agent 前必须配置模型角色：{missing}",
                status_code=422,
            )
        if "router_llm" not in set(models or {}):
            raise AgentApplicationError(
                "AGENT_ROUTER_MODEL_REQUIRED",
                "启用问文、问数与对话路由前必须配置 router_llm",
                status_code=422,
            )

    async def _validate_x_search_model(self, models) -> None:
        model_id = (models or {}).get("x_search_llm")
        if model_id is None:
            return
        if self._catalog is None:
            raise AgentApplicationError(
                "MODEL_CATALOG_UNAVAILABLE", "模型目录暂时不可用", status_code=503,
            )
        catalog = await load_catalog_model(self._catalog, UUID(str(model_id)))
        if str(catalog.get("status") or "").upper() != "ACTIVE":
            raise AgentApplicationError(
                "X_SEARCH_MODEL_UNAVAILABLE", "X Search LLM 未启用", status_code=422,
            )
        if coerce_model_category(catalog.get("category")) != ModelCategory.LLM:
            raise AgentApplicationError(
                "X_SEARCH_MODEL_UNAVAILABLE",
                "X Search 只能绑定 LLM 类别的 Grok 模型",
                status_code=422,
            )
        if not is_grok_model(catalog):
            raise AgentApplicationError(
                "X_SEARCH_GROK_REQUIRED", "X Search 只能绑定 Grok 模型", status_code=422,
            )
    @staticmethod
    def _version(*, version_id, agent_id, version_no, knowledge_core_id, data_model_ids, models, instruction, config, actor_id):
        return KnowledgeRetrievalAgentVersionEntity(
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
        models = dict(version.models_json or {})
        capabilities = ["conversation", "document"]
        if data_model_ids:
            capabilities.append("data_query")
        if models.get("x_search_llm"):
            capabilities.append("x_search")
        return {
            "agent_id": str(agent.agent_id), "domain_id": int(agent.domain_id),
            "display_name": agent.display_name, "description": agent.description,
            "status": agent.status, "row_version": int(agent.row_version),
            "agent_version_id": str(version.agent_version_id),
            "version_no": int(version.version_no),
            "knowledge_core_id": str(version.knowledge_core_id) if version.knowledge_core_id else None,
            "data_model_ids": data_model_ids, "models": models,
            "instruction": version.instruction, "config": dict(version.config_json or {}),
            "enabled_capabilities": capabilities,
            "created_at": agent.created_at, "updated_at": agent.updated_at,
        }

    @staticmethod
    def _not_found() -> None:
        raise AgentApplicationError("AGENT_NOT_FOUND", "Agent 不存在", status_code=404)
