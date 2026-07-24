"""Root Agent Definition 的配置生命周期。"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from agent_runtime.entities import AgentDefinitionEntity
from platform_core.identity import uuid7

from .runtime_service import AgentRuntimeConflict


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CreateAgentDefinitionCommand(_Model):
    app_id: int = Field(ge=1)
    domain_id: int = Field(ge=1)
    agent_key: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[str, ...] = Field(min_length=1)
    router_llm_model_name: str | None = Field(default=None, max_length=128)
    context_llm_model_name: str = Field(min_length=1, max_length=128)
    composer_llm_model_name: str = Field(min_length=1, max_length=128)
    memory_llm_model_name: str = Field(min_length=1, max_length=128)
    query_vlm_model_name: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    chart_llm_model_name: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    memory_embedding_model_name: str = Field(
        min_length=1, max_length=128
    )
    do_rerank: bool = False
    data_profile_name: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="DRAFT", pattern=r"^(DRAFT|ACTIVE)$")
    actor_id: str = Field(min_length=1, max_length=256)


class UpdateAgentDefinitionCommand(_Model):
    app_id: int = Field(ge=1)
    domain_id: int = Field(ge=1)
    agent_id: UUID
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[str, ...] | None = None
    router_llm_model_name: str | None = Field(default=None, max_length=128)
    context_llm_model_name: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    composer_llm_model_name: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    memory_llm_model_name: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    query_vlm_model_name: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    chart_llm_model_name: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    memory_embedding_model_name: str | None = Field(
        default=None, min_length=1, max_length=128
    )
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
    router_llm_model_name: str | None
    context_llm_model_name: str
    composer_llm_model_name: str
    memory_llm_model_name: str
    query_vlm_model_name: str | None
    chart_llm_model_name: str | None
    memory_embedding_model_name: str
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

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def create(
        self, command: CreateAgentDefinitionCommand
    ) -> AgentDefinitionView:
        capabilities = self._validate_capabilities(
            command.enabled_capabilities
        )
        self._validate_runtime_configuration(
            capabilities=capabilities,
            status=command.status,
            router_llm_model_name=command.router_llm_model_name,
            data_profile_name=command.data_profile_name,
        )
        async with self._uow_factory() as uow:
            existing = await uow.agents.get_by_key(
                app_id=command.app_id,
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
                app_id=command.app_id,
                domain_id=command.domain_id,
                agent_key=command.agent_key,
                display_name=command.display_name,
                description=command.description,
                status=command.status,
                enabled_capabilities_json=list(capabilities),
                router_llm_model_name=command.router_llm_model_name,
                context_llm_model_name=command.context_llm_model_name,
                composer_llm_model_name=command.composer_llm_model_name,
                memory_llm_model_name=command.memory_llm_model_name,
                query_vlm_model_name=command.query_vlm_model_name,
                chart_llm_model_name=command.chart_llm_model_name,
                memory_embedding_model_name=(
                    command.memory_embedding_model_name
                ),
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
                app_id=command.app_id,
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
                    "app_id",
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
            self._validate_runtime_configuration(
                capabilities=effective_capabilities,
                status=str(values.get("status", row.status)),
                router_llm_model_name=values.get(
                    "router_llm_model_name",
                    row.router_llm_model_name,
                ),
                data_profile_name=values.get(
                    "data_profile_name",
                    getattr(row, "data_profile_name", None),
                ),
            )
            requested_embedding = values.get(
                "memory_embedding_model_name"
            )
            if (
                requested_embedding is not None
                and requested_embedding
                != row.memory_embedding_model_name
            ):
                raise AgentRuntimeConflict(
                    "MEMORY_EMBEDDING_MODEL_IMMUTABLE",
                    "记忆 Embedding 模型一经设定禁止更换",
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
        app_id: int,
        domain_id: int,
    ) -> AgentDefinitionView:
        async with self._uow_factory() as uow:
            row = await uow.agents.get_scoped(
                agent_id=agent_id,
                app_id=app_id,
                domain_id=domain_id,
            )
            if row is None:
                raise AgentRuntimeConflict(
                    "AGENT_NOT_FOUND_OR_DENIED",
                    "Agent 不存在或不属于当前 Domain",
                )
            return self._view(row)

    async def list(
        self, *, app_id: int, domain_id: int
    ) -> list[AgentDefinitionView]:
        async with self._uow_factory() as uow:
            rows = await uow.agents.list_scoped(
                app_id=app_id, domain_id=domain_id
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
        router_llm_model_name: str | None,
        data_profile_name: str | None,
    ) -> None:
        if status != "ACTIVE":
            return
        if len(capabilities) > 1 and not str(
            router_llm_model_name or ""
        ).strip():
            raise AgentRuntimeConflict(
                "AGENT_ROUTER_MODEL_REQUIRED",
                "多能力 Agent 启用前必须配置 router_llm_model_name",
            )
        if "mcp_data" in capabilities and not str(
            data_profile_name or ""
        ).strip():
            raise AgentRuntimeConflict(
                "AGENT_DATA_PROFILE_REQUIRED",
                "问数 Agent 启用前必须配置 data_profile_name",
            )

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
            router_llm_model_name=row.router_llm_model_name,
            context_llm_model_name=row.context_llm_model_name,
            composer_llm_model_name=row.composer_llm_model_name,
            memory_llm_model_name=row.memory_llm_model_name,
            query_vlm_model_name=row.query_vlm_model_name,
            chart_llm_model_name=row.chart_llm_model_name,
            memory_embedding_model_name=(
                row.memory_embedding_model_name
            ),
            do_rerank=bool(row.do_rerank),
            data_profile_name=row.data_profile_name,
            instruction=row.instruction,
            config=dict(row.config_json or {}),
            row_version=int(row.row_version),
        )
