"""Root Agent Definition 的配置生命周期。"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from agent_runtime.domain.model_bindings import (
    AGENT_IMMUTABLE_MODEL_ROLES,
    normalize_agent_models,
)
from agent_runtime.entities import AgentDefinitionEntity
from platform_core.identity import uuid7
from platform_core.contracts import AuthContext, PrincipalKind

from .runtime_service import AgentRuntimeConflict


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CreateAgentDefinitionCommand(_Model):
    agent_id: UUID | None = None
    domain_id: int = Field(ge=1)
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[str, ...] = Field(min_length=1)
    models: dict[str, UUID | str]
    do_rerank: bool = False
    data_query_mode: Literal["MCP", "SEMANTIC"] | None = None
    data_profile_name: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="DRAFT", pattern=r"^(DRAFT|ACTIVE)$")
    actor_id: str = Field(min_length=1, max_length=256)


class UpdateAgentDefinitionCommand(_Model):
    domain_id: int = Field(ge=1)
    agent_id: UUID
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[str, ...] | None = None
    models: dict[str, UUID | str] | None = None
    do_rerank: bool | None = None
    data_query_mode: Literal["MCP", "SEMANTIC"] | None = None
    data_profile_name: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: str | None = Field(
        default=None, pattern=r"^(DRAFT|ACTIVE|INACTIVE)$"
    )
    actor_id: str = Field(min_length=1, max_length=256)


class AgentDefinitionView(_Model):
    agent_id: UUID
    domain_id: int
    display_name: str
    description: str | None
    status: str
    enabled_capabilities: tuple[str, ...]
    models: dict[str, UUID]
    do_rerank: bool
    data_query_mode: Literal["MCP", "SEMANTIC"] | None
    data_profile_name: str | None
    instruction: str | None
    config: dict[str, Any]
    row_version: int


class AgentDefinitionService:
    """只维护 Agent Runtime 自有配置，不写 KC Binding。"""

    SUPPORTED_CAPABILITIES = frozenset(
        {"document", "conversation", "data_query", "aiops"}
    )

    async def list_model_references(
        self, *, model_id: UUID,
    ) -> list[dict[str, str]]:
        """返回模型删除检查需要的 Agent 引用。"""
        async with self._uow_factory() as uow:
            return await uow.agents.list_model_references(model_id=model_id)

    def __init__(
        self,
        *,
        uow_factory,
        model_resolver=None,
        data_query_client=None,
        knowledge_core_client=None,
        service_name: str = "kbot-agent-runtime-api",
    ):
        self._uow_factory = uow_factory
        self._model_resolver = model_resolver
        self._data_query_client = data_query_client
        self._knowledge_core_client = knowledge_core_client
        self._service_name = service_name

    async def create(
        self, command: CreateAgentDefinitionCommand
    ) -> AgentDefinitionView:
        capabilities = self._validate_capabilities(
            command.enabled_capabilities
        )
        models = self._validate_models(command.models)
        await self._validate_model_catalog(models)
        agent_id = command.agent_id or uuid7()
        self._validate_runtime_configuration(
            capabilities=capabilities,
            status=command.status,
            router_model=models.get("router_llm"),
            data_query_mode=command.data_query_mode,
            data_profile_name=command.data_profile_name,
        )
        if command.status == "ACTIVE" and command.data_query_mode == "SEMANTIC":
            raise AgentRuntimeConflict(
                "AGENT_SEMANTIC_DRAFT_REQUIRED",
                "SEMANTIC 问数 Agent 必须先创建草稿、完成绑定后再激活",
            )
        if command.status == "ACTIVE" and "document" in capabilities:
            raise AgentRuntimeConflict(
                "AGENT_DOCUMENT_DRAFT_REQUIRED",
                "文档 Agent 必须先创建草稿、完成 Collection 绑定后再激活",
            )
        async with self._uow_factory() as uow:
            row = AgentDefinitionEntity(
                agent_id=agent_id,
                domain_id=command.domain_id,
                display_name=command.display_name,
                description=command.description,
                status=command.status,
                enabled_capabilities_json=list(capabilities),
                models_json=models,
                do_rerank=command.do_rerank,
                data_query_mode=command.data_query_mode,
                data_profile_name=command.data_profile_name,
                instruction=command.instruction,
                config_json=command.config,
                created_by=command.actor_id,
                updated_by=command.actor_id,
            )
            try:
                await uow.agents.add(row)
                await uow.commit()
            except IntegrityError as exc:
                raise AgentRuntimeConflict(
                    "AGENT_CREATE_CONFLICT",
                    "Agent 创建发生数据库约束冲突",
                ) from exc
            return self._view(row)

    async def update(
        self, command: UpdateAgentDefinitionCommand
    ) -> AgentDefinitionView:
        async with self._uow_factory() as uow:
            row = await uow.agents.get_scoped(
                agent_id=command.agent_id,
                domain_id=command.domain_id,
                lock=True,
            )
            if row is None:
                raise AgentRuntimeConflict(
                    "AGENT_NOT_FOUND_OR_DENIED",
                    "Agent 不存在或不属于当前 Domain",
                )
            if int(row.row_version) != command.expected_row_version:
                raise AgentRuntimeConflict(
                    "STATE_VERSION_CONFLICT", "Agent 配置版本已变化"
                )
            values = command.model_dump(
                exclude={
                                        "domain_id",
                    "agent_id",
                    "expected_row_version",
                    "actor_id",
                },
                exclude_unset=True,
            )
            if "enabled_capabilities" in values:
                values["enabled_capabilities_json"] = list(
                    self._validate_capabilities(
                        tuple(values.pop("enabled_capabilities"))
                    )
                )
            effective_capabilities = tuple(
                values.get(
                    "enabled_capabilities_json",
                    row.enabled_capabilities_json or [],
                )
            )
            if values.get("models") is not None:
                values["models_json"] = self._validate_models(
                    values.pop("models")
                )
                await self._validate_model_catalog(values["models_json"])
            else:
                values.pop("models", None)
            effective_models = dict(
                values.get("models_json", row.models_json or {})
            )
            effective_status = str(values.get("status", row.status))
            self._validate_runtime_configuration(
                capabilities=effective_capabilities,
                status=effective_status,
                router_model=effective_models.get("router_llm"),
                data_query_mode=values.get(
                    "data_query_mode",
                    getattr(row, "data_query_mode", None),
                ),
                data_profile_name=values.get(
                    "data_profile_name",
                    getattr(row, "data_profile_name", None),
                ),
            )
            if "models_json" in values:
                current_models = dict(row.models_json or {})
                for role in AGENT_IMMUTABLE_MODEL_ROLES:
                    if current_models.get(role) != effective_models.get(role):
                        raise AgentRuntimeConflict(
                            "AGENT_MODEL_IMMUTABLE",
                            f"模型角色 {role} 一经设定禁止更换或删除",
                        )
            await self._validate_data_query_binding(
                domain_id=command.domain_id,
                actor_id=command.actor_id,
                agent_id=command.agent_id,
                status=effective_status,
                data_query_mode=values.get(
                    "data_query_mode",
                    getattr(row, "data_query_mode", None),
                ),
                required=(
                    effective_status == "ACTIVE"
                    and (
                        row.status != "ACTIVE"
                        or "enabled_capabilities_json" in values
                        or "data_query_mode" in values
                    )
                ),
            )
            await self._validate_knowledge_binding(
                domain_id=command.domain_id,
                actor_id=command.actor_id,
                agent_id=command.agent_id,
                status=effective_status,
                required=(
                    effective_status == "ACTIVE"
                    and "document" in effective_capabilities
                    and (
                        row.status != "ACTIVE"
                        or "enabled_capabilities_json" in values
                    )
                ),
            )
            if "config" in values:
                values["config_json"] = values.pop("config")
            for field, value in values.items():
                setattr(row, field, value)
            row.updated_by = command.actor_id
            row.row_version = int(row.row_version) + 1
            await uow.commit()
            return self._view(row)

    async def get(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
    ) -> AgentDefinitionView:
        async with self._uow_factory() as uow:
            row = await uow.agents.get_scoped(
                agent_id=agent_id,
                domain_id=domain_id,
            )
            if row is None:
                raise AgentRuntimeConflict(
                    "AGENT_NOT_FOUND_OR_DENIED",
                    "Agent 不存在或不属于当前 Domain",
                )
            return self._view(row)

    async def list(
        self, *, domain_id: int
    ) -> list[AgentDefinitionView]:
        async with self._uow_factory() as uow:
            rows = await uow.agents.list_scoped(
                domain_id=domain_id,
            )
            return [self._view(row) for row in rows]

    @classmethod
    def _validate_capabilities(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(item.strip() for item in values))
        unknown = set(normalized) - cls.SUPPORTED_CAPABILITIES
        if not normalized or unknown:
            raise AgentRuntimeConflict(
                "AGENT_CAPABILITY_INVALID",
                f"Agent 能力集合无效：{sorted(unknown)}",
            )
        if "aiops" in normalized and len(normalized) != 1:
            raise AgentRuntimeConflict(
                "AGENT_CAPABILITY_INVALID",
                "AIOps Agent 必须使用独立入口，不能与其他能力混合",
            )
        return normalized

    @staticmethod
    def _validate_runtime_configuration(
        *,
        capabilities: tuple[str, ...],
        status: str,
        router_model: str | None,
        data_query_mode: str | None,
        data_profile_name: str | None,
    ) -> None:
        has_data_query = "data_query" in capabilities
        if not has_data_query and (
            data_query_mode is not None or data_profile_name is not None
        ):
            raise AgentRuntimeConflict(
                "AGENT_DATA_QUERY_CONFIG_INVALID",
                "未启用 data_query 能力时 Mode 和 Profile 必须为空",
            )
        if has_data_query and data_query_mode not in {"MCP", "SEMANTIC"}:
            raise AgentRuntimeConflict(
                "AGENT_DATA_QUERY_MODE_REQUIRED",
                "问数 Agent 必须选择 MCP 或 SEMANTIC 模式",
            )
        if data_query_mode == "MCP" and not str(
            data_profile_name or ""
        ).strip():
            raise AgentRuntimeConflict(
                "AGENT_DATA_PROFILE_REQUIRED",
                "MCP 问数模式必须配置 data_profile_name",
            )
        if data_query_mode == "SEMANTIC" and data_profile_name is not None:
            raise AgentRuntimeConflict(
                "AGENT_DATA_QUERY_CONFIG_INVALID",
                "SEMANTIC 问数模式不能配置 data_profile_name",
            )
        if status != "ACTIVE":
            return
        if len(capabilities) > 1 and not str(
            router_model or ""
        ).strip():
            raise AgentRuntimeConflict(
                "AGENT_ROUTER_MODEL_REQUIRED",
                "多能力 Agent 启用前必须配置 models.router_llm",
            )

    async def _validate_data_query_binding(
        self,
        *,
        domain_id: int,
        actor_id: str,
        agent_id: UUID,
        status: str,
        data_query_mode: str | None,
        required: bool,
    ) -> None:
        if status != "ACTIVE" or not required:
            return
        if self._data_query_client is None:
            raise AgentRuntimeConflict(
                "DATA_QUERY_CATALOG_UNAVAILABLE",
                "Data Query 绑定目录尚未初始化",
            )
        auth_context = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id=self._service_name,
            calling_service=self._service_name,
            request_id=str(uuid7()),
            trace_id=str(uuid7()),
            domain_id=str(domain_id),
            asserted_user_id=actor_id,
        )
        try:
            context = await self._data_query_client.get_planning_context(
                agent_id=agent_id,
                auth_context=auth_context,
            )
        except Exception as exc:
            raise AgentRuntimeConflict(
                "DATA_QUERY_BINDING_VALIDATION_FAILED",
                "无法验证 Agent 的语义模型绑定",
            ) from exc
        models = context.get("models") if isinstance(context, dict) else None
        has_binding = isinstance(models, list) and bool(models)
        if data_query_mode == "SEMANTIC" and not has_binding:
            raise AgentRuntimeConflict(
                "AGENT_SEMANTIC_BINDING_REQUIRED",
                "SEMANTIC 问数 Agent 启用前必须绑定已发布语义模型",
            )
        if data_query_mode != "SEMANTIC" and has_binding:
            raise AgentRuntimeConflict(
                "AGENT_SEMANTIC_BINDING_CONFLICT",
                "非 SEMANTIC 模式不能保留有效语义模型绑定",
            )

    async def _validate_knowledge_binding(
        self,
        *,
        domain_id: int,
        actor_id: str,
        agent_id: UUID,
        status: str,
        required: bool,
    ) -> None:
        if status != "ACTIVE" or not required:
            return
        if self._knowledge_core_client is None:
            raise AgentRuntimeConflict(
                "KNOWLEDGE_CATALOG_UNAVAILABLE",
                "Knowledge Core 绑定目录尚未初始化",
            )
        auth_context = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id=self._service_name,
            calling_service=self._service_name,
            request_id=str(uuid7()),
            trace_id=str(uuid7()),
            domain_id=str(domain_id),
            asserted_user_id=actor_id,
        )
        try:
            response = await self._knowledge_core_client.list_agent_bindings(
                domain_id=domain_id,
                agent_id=agent_id,
                auth_context=auth_context,
            )
        except Exception as exc:
            raise AgentRuntimeConflict(
                "KNOWLEDGE_BINDING_VALIDATION_FAILED",
                "无法验证 Agent 的 Collection 绑定",
            ) from exc
        bindings = (
            response.get("bindings")
            if isinstance(response, dict)
            else None
        )
        if not isinstance(bindings, list) or not any(
            isinstance(item, dict) and item.get("status") == "ACTIVE"
            for item in bindings
        ):
            raise AgentRuntimeConflict(
                "AGENT_COLLECTION_BINDING_REQUIRED",
                "文档 Agent 启用前必须至少绑定一个有效 Collection",
            )

    @staticmethod
    def _validate_models(
        values: dict[str, UUID | str],
    ) -> dict[str, str]:
        try:
            return normalize_agent_models(values)
        except ValueError as exc:
            raise AgentRuntimeConflict(
                "AGENT_MODELS_INVALID", str(exc)
            ) from exc

    async def _validate_model_catalog(
        self, models: dict[str, str]
    ) -> None:
        if self._model_resolver is None:
            raise AgentRuntimeConflict(
                "MODEL_CATALOG_UNAVAILABLE",
                "Agent 模型目录解析器尚未初始化",
            )
        try:
            await self._model_resolver.resolve(models)
        except (LookupError, RuntimeError, ValueError) as exc:
            raise AgentRuntimeConflict(
                "AGENT_MODEL_INVALID", str(exc)
            ) from exc

    @staticmethod
    def _view(row: AgentDefinitionEntity) -> AgentDefinitionView:
        return AgentDefinitionView(
            agent_id=row.agent_id,
            domain_id=int(row.domain_id),
            display_name=row.display_name,
            description=row.description,
            status=row.status,
            enabled_capabilities=tuple(
                row.enabled_capabilities_json or []
            ),
            models=dict(row.models_json or {}),
            do_rerank=bool(row.do_rerank),
            data_query_mode=getattr(row, "data_query_mode", None),
            data_profile_name=row.data_profile_name,
            instruction=row.instruction,
            config=dict(row.config_json or {}),
            row_version=int(row.row_version),
        )
