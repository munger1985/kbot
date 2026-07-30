"""Root Agent Definition 的配置生命周期。"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from agent_runtime.domain.model_bindings import (
    AGENT_IMMUTABLE_MODEL_ROLES,
    normalize_agent_models,
)
from agent_runtime.entities import AgentDefinitionEntity
from platform_core.identity import uuid7

from .runtime_service import AgentRuntimeConflict


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CreateAgentDefinitionCommand(_Model):
    domain_id: int = Field(ge=1)
    agent_key: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[str, ...] = Field(min_length=1)
    models: dict[str, UUID | str]
    do_rerank: bool = False
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
    agent_key: str
    display_name: str
    description: str | None
    status: str
    enabled_capabilities: tuple[str, ...]
    models: dict[str, UUID]
    do_rerank: bool
    data_profile_name: str | None
    instruction: str | None
    config: dict[str, Any]
    row_version: int


class AgentDefinitionService:
    """只维护 Agent Runtime 自有配置，不写 KC Binding。"""

    SUPPORTED_CAPABILITIES = frozenset(
        {"document", "conversation", "mcp_data", "aiops"}
    )

    def __init__(self, *, uow_factory, model_resolver=None):
        self._uow_factory = uow_factory
        self._model_resolver = model_resolver

    async def create(
        self, command: CreateAgentDefinitionCommand
    ) -> AgentDefinitionView:
        capabilities = self._validate_capabilities(
            command.enabled_capabilities
        )
        models = self._validate_models(command.models)
        await self._validate_model_catalog(models)
        self._validate_runtime_configuration(
            capabilities=capabilities,
            status=command.status,
            router_model=models.get("router_llm"),
            data_profile_name=command.data_profile_name,
        )
        async with self._uow_factory() as uow:
            existing = await uow.agents.get_by_key(
                domain_id=command.domain_id,
                agent_key=command.agent_key,
            )
            if existing is not None:
                raise AgentRuntimeConflict(
                    "AGENT_KEY_CONFLICT",
                    "当前 Domain 已存在相同 agent_key",
                )
            row = AgentDefinitionEntity(
                agent_id=uuid7(),
                domain_id=command.domain_id,
                agent_key=command.agent_key,
                display_name=command.display_name,
                description=command.description,
                status=command.status,
                enabled_capabilities_json=list(capabilities),
                models_json=models,
                do_rerank=command.do_rerank,
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
                    "AGENT_KEY_CONFLICT",
                    "当前 Domain 已存在相同 agent_key",
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
            self._validate_runtime_configuration(
                capabilities=effective_capabilities,
                status=str(values.get("status", row.status)),
                router_model=effective_models.get("router_llm"),
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
        data_profile_name: str | None,
    ) -> None:
        if status != "ACTIVE":
            return
        if len(capabilities) > 1 and not str(
            router_model or ""
        ).strip():
            raise AgentRuntimeConflict(
                "AGENT_ROUTER_MODEL_REQUIRED",
                "多能力 Agent 启用前必须配置 models.router_llm",
            )
        if "mcp_data" in capabilities and not str(
            data_profile_name or ""
        ).strip():
            raise AgentRuntimeConflict(
                "AGENT_DATA_PROFILE_REQUIRED",
                "问数 Agent 启用前必须配置 data_profile_name",
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
            agent_key=row.agent_key,
            display_name=row.display_name,
            description=row.description,
            status=row.status,
            enabled_capabilities=tuple(
                row.enabled_capabilities_json or []
            ),
            models=dict(row.models_json or {}),
            do_rerank=bool(row.do_rerank),
            data_profile_name=row.data_profile_name,
            instruction=row.instruction,
            config=dict(row.config_json or {}),
            row_version=int(row.row_version),
        )
