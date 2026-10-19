"""AIOps 私有 Agent 生命周期。"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError

from aiops_agent.entities import (
    AIOpsAgentEntity,
    AIOpsAgentVersionEntity,
    PolicyEntity,
)
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.tools.capabilities import target_capability_names
from platform_core.contracts.aiops import AIOpsCapability, AgentKnowledgePolicy
from platform_core.identity import uuid7


CONTROLLED_DYNAMIC_PARAMETER_VALUES = {
    "cursor_sharing": ("EXACT", "FORCE"),
    "optimizer_mode": ("ALL_ROWS", "FIRST_ROWS"),
    "statistics_level": ("BASIC", "TYPICAL", "ALL"),
}
CONTROLLED_SYSTEM_PRIVILEGES = (
    "CREATE SESSION",
    "CREATE TABLE",
    "CREATE VIEW",
    "CREATE PROCEDURE",
    "CREATE SEQUENCE",
    "CREATE SYNONYM",
    "CREATE TRIGGER",
    "CREATE TYPE",
)
CONTROLLED_OBJECT_PRIVILEGES = (
    "SELECT",
    "READ",
    "INSERT",
    "UPDATE",
    "DELETE",
    "EXECUTE",
)


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


class AgentModelBindings(_Model):
    """声明 AIOps Agent 各模型调用阶段的独立绑定。"""

    planner_llm: UUID | None = None
    diagnosis_llm: UUID | None = None
    composer_llm: UUID | None = None
    memory_llm: UUID | None = None
    memory_embedding: UUID | None = None


class ControlledDynamicParameterRule(_Model):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    allowed_values: tuple[str, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_values(self):
        normalized = [value.upper() for value in self.allowed_values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("动态参数允许值不能重复")
        if any(
            re.fullmatch(r"[A-Za-z0-9_]{1,128}", value) is None
            for value in self.allowed_values
        ):
            raise ValueError("动态参数允许值格式无效")
        supported = CONTROLLED_DYNAMIC_PARAMETER_VALUES
        if self.name not in supported or not set(normalized) <= set(
            supported[self.name]
        ):
            raise ValueError("动态参数或允许值不在受控动作 Catalog")
        return self


class ControlledActionObjectScopes(_Model):
    schemas: tuple[str, ...] = ()
    exclude_system_objects: bool = True
    dynamic_parameters: tuple[ControlledDynamicParameterRule, ...] = ()
    resource_manager_plans: tuple[str, ...] = ()
    privilege_grantees: tuple[str, ...] = ()
    system_privileges: tuple[str, ...] = ()
    object_privileges: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_scopes(self):
        if len(set(self.schemas)) != len(self.schemas):
            raise ValueError("受控动作 Schema 范围不能重复")
        for schema in self.schemas:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]{0,127}", schema) is None:
                raise ValueError("受控动作 Schema 名称格式无效")
        parameter_names = [item.name.lower() for item in self.dynamic_parameters]
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("动态参数白名单不能重复")
        for label, values in (
            ("Resource Manager Plan", self.resource_manager_plans),
            ("授权用户", self.privilege_grantees),
        ):
            normalized = [value.upper() for value in values]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{label} 白名单不能重复")
            if any(
                re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]{0,127}", value) is None
                for value in values
            ):
                raise ValueError(f"{label} 格式无效")
        supported_privileges = {
            "系统权限": set(CONTROLLED_SYSTEM_PRIVILEGES),
            "对象权限": set(CONTROLLED_OBJECT_PRIVILEGES),
        }
        for label, values in (
            ("系统权限", self.system_privileges),
            ("对象权限", self.object_privileges),
        ):
            normalized = [value.upper() for value in values]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{label} 白名单不能重复")
            if any(
                re.fullmatch(r"[A-Za-z][A-Za-z ]{0,63}", value) is None
                for value in values
            ):
                raise ValueError(f"{label} 格式无效")
            if not set(normalized) <= supported_privileges[label]:
                raise ValueError(f"{label} 不在受控动作 Catalog")
        return self


class ResourceControlledActionExecution(_Model):
    resource_id: UUID
    enabled: bool = False
    allowed_action_ids: tuple[str, ...] = ()
    object_scopes: ControlledActionObjectScopes = Field(
        default_factory=ControlledActionObjectScopes
    )
    max_daily_executions: int | None = Field(default=None, ge=1, le=10000)

    @model_validator(mode="after")
    def validate_action_selection(self):
        if len(set(self.allowed_action_ids)) != len(self.allowed_action_ids):
            raise ValueError("受控动作不能重复")
        if self.enabled != bool(self.allowed_action_ids):
            raise ValueError("启用受控动作时必须明确选择至少一个动作")
        return self


class CreateAIOpsAgentCommand(_Model):
    domain_id: int = Field(ge=1)
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    allowed_business_system_ids: tuple[UUID, ...] = Field(min_length=1, max_length=32)
    capabilities: tuple[AIOpsCapability, ...] = Field(min_length=1, max_length=8)
    knowledge_policy: AgentKnowledgePolicy = Field(
        default_factory=AgentKnowledgePolicy
    )
    controlled_action_execution: tuple[
        ResourceControlledActionExecution, ...
    ] = ()
    auto_alert_enabled: bool = True
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] = "CRITICAL"
    auto_observe_min_target_level: int = Field(default=1, ge=1, le=5)
    alert_cooldown_minutes: int = Field(default=15, ge=0, le=1440)
    models: AgentModelBindings = Field(default_factory=AgentModelBindings)
    image_capabilities: AgentImageCapabilities = Field(
        default_factory=AgentImageCapabilities
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"
    actor_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_resources(self):
        if len(set(self.allowed_business_system_ids)) != len(self.allowed_business_system_ids):
            raise ValueError("allowed_business_system_ids 不能重复")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities 不能重复")
        policy_resources = [item.resource_id for item in self.controlled_action_execution]
        if len(set(policy_resources)) != len(policy_resources):
            raise ValueError("每个资源只能声明一份受控动作策略")
        return self


class UpdateAIOpsAgentCommand(_Model):
    domain_id: int = Field(ge=1)
    agent_id: UUID
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    allowed_business_system_ids: tuple[UUID, ...] | None = Field(
        default=None, min_length=1, max_length=32
    )
    capabilities: tuple[AIOpsCapability, ...] | None = Field(
        default=None, min_length=1, max_length=8
    )
    knowledge_policy: AgentKnowledgePolicy | None = None
    controlled_action_execution: tuple[
        ResourceControlledActionExecution, ...
    ] | None = None
    auto_alert_enabled: bool | None = None
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] | None = None
    auto_observe_min_target_level: int | None = Field(
        default=None, ge=1, le=5
    )
    alert_cooldown_minutes: int | None = Field(default=None, ge=0, le=1440)
    models: AgentModelBindings | None = None
    image_capabilities: AgentImageCapabilities | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None
    actor_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_resources(self):
        if self.allowed_business_system_ids is not None and len(set(self.allowed_business_system_ids)) != len(
            self.allowed_business_system_ids
        ):
            raise ValueError("allowed_business_system_ids 不能重复")
        if self.capabilities is not None and len(set(self.capabilities)) != len(
            self.capabilities
        ):
            raise ValueError("capabilities 不能重复")
        if self.controlled_action_execution is not None:
            policy_resources = [
                item.resource_id for item in self.controlled_action_execution
            ]
            if len(set(policy_resources)) != len(policy_resources):
                raise ValueError("每个资源只能声明一份受控动作策略")
        return self


class ArchiveAIOpsAgentCommand(_Model):
    domain_id: int
    agent_id: UUID
    expected_row_version: int = Field(ge=1)
    actor_id: str = Field(min_length=1, max_length=256)


class AIOpsAgentError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class AIOpsAgentService:
    def __init__(
        self,
        *,
        uow_factory,
        action_registry=None,
        environment: str = "development",
        model_client=None,
    ):
        self._uow_factory = uow_factory
        self._action_registry = action_registry
        self._environment = environment
        self._model_client = model_client

    async def create(self, command: CreateAIOpsAgentCommand) -> dict[str, Any]:
        agent_id, version_id = uuid7(), uuid7()
        async with self._uow_factory() as uow:
            values = command.model_dump()
            await self._validate_resources(
                uow, command.domain_id, command.status, values
            )
            policy = await self._create_policy(
                uow=uow,
                agent_id=agent_id,
                version_no=1,
                display_name=command.display_name,
                values=values,
                actor_id=command.actor_id,
            )
            values["policy_id"] = policy.policy_id
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
                        values=values,
                        actor_id=command.actor_id,
                    )
                )
                await uow.agents.add_version_business_systems(
                    version_id=version_id,
                    business_system_ids=command.allowed_business_system_ids,
                )
                await uow.agents.add_version_resource_policies(
                    version_id=version_id,
                    controlled_action_policies=self._controlled_action_policies(values),
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
            rows = []
            for agent in await uow.agents.list(domain_id=domain_id):
                version = await self._version(uow.agents, agent)
                rows.append(await self._view(uow, agent, version))
            return rows

    async def get(self, *, domain_id: int, agent_id: UUID) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if agent is None or agent.status == "ARCHIVED":
                self._not_found()
            return await self._view(
                uow, agent, await self._version(uow.agents, agent)
            )

    async def archive(self, command: ArchiveAIOpsAgentCommand) -> None:
        """归档 Agent，保留历史运行和会话的审计引用。"""
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(
                domain_id=command.domain_id,
                agent_id=command.agent_id,
                lock=True,
            )
            if agent is None or agent.status == "ARCHIVED":
                self._not_found()
            if int(agent.row_version) != command.expected_row_version:
                raise AIOpsAgentError(
                    "STATE_VERSION_CONFLICT", "Agent 配置版本已变化"
                )
            if await uow.agents.has_schedulable_inspection_plans(
                domain_id=command.domain_id,
                agent_id=command.agent_id,
            ):
                raise AIOpsAgentError(
                    "AIOPS_AGENT_INSPECTION_PLAN_CONFLICT",
                    "Agent 仍被启用或暂停的巡检计划使用，请先停用相关计划",
                )
            current = await self._version(uow.agents, agent)
            policy = await uow.policies.get_scoped(
                policy_id=current.policy_id,
                domain_id=command.domain_id,
            )
            now = datetime.now(UTC)
            if policy is not None and policy.status != "RETIRED":
                policy.status = "RETIRED"
                policy.retired_at = now
                policy.updated_by = command.actor_id
                policy.row_version = int(policy.row_version) + 1
            agent.status = "ARCHIVED"
            agent.updated_by = command.actor_id
            agent.updated_at = now
            agent.row_version = int(agent.row_version) + 1
            await uow.commit()

    async def action_catalog(
        self, *, domain_id: int, resource_id: UUID
    ) -> dict[str, Any]:
        """返回一个业务资源可见动作及当前可执行原因。"""
        async with self._uow_factory() as uow:
            resource = await uow.business_systems.get_resource_scoped(
                resource_id=resource_id,
                domain_id=domain_id,
            )
            database = await uow.business_systems.get_database_profile(resource_id)
            if resource is None or database is None:
                self._not_found()
            target = await uow.targets.get_scoped(
                target_id=database.target_id, domain_id=domain_id
            )
            if target is None:
                self._not_found()
            capabilities = set(target_capability_names(target))
            templates = self._action_registry.compatible(
                db_type=target.db_type,
                db_version=target.version_code or "UNKNOWN",
                capabilities=capabilities,
                entitlements=set(),
                environment=target.environment,
            ) if self._action_registry is not None else ()
            discovered_options = _controlled_action_options(
                dict(target.capabilities_json or {})
            )
            return {
                "resource_id": str(resource.resource_id),
                "catalog_hash": (
                    self._action_registry.catalog_hash
                    if self._action_registry is not None
                    else None
                ),
                "actions": [
                    {
                        "action_id": item.definition.action_template_id,
                        "version": item.definition.version,
                        "variant": item.definition.variant,
                        "action_family": item.definition.action_family,
                        "effect_class": item.definition.effect_class,
                        "execution_mode": item.definition.execution_mode,
                        "executor_kind": item.definition.executor_kind,
                        "risk_level": item.definition.risk_level,
                        "lock_impact": item.definition.lock_impact,
                        "estimated_duration_seconds": (
                            item.definition.estimated_duration_seconds
                        ),
                        "scope_requirements": list(
                            _controlled_scope_requirements(item.definition)
                        ),
                        "status": item.definition.status,
                        "currently_executable": bool(
                            item.definition.status == "ACTIVE"
                            and item.definition.execution_mode
                            == "EXECUTABLE_AFTER_APPROVAL"
                            and target.controlled_change_enabled
                            and target.execution_credential_id is not None
                        ),
                    }
                    for item in templates
                ],
                "scope_options": {
                    **discovered_options,
                    "dynamic_parameters": [
                        {
                            "name": name,
                            "allowed_values": list(values),
                        }
                        for name, values in (
                            CONTROLLED_DYNAMIC_PARAMETER_VALUES.items()
                        )
                    ],
                    "system_privileges": list(
                        CONTROLLED_SYSTEM_PRIVILEGES
                    ),
                    "object_privileges": list(
                        CONTROLLED_OBJECT_PRIVILEGES
                    ),
                },
            }

    async def update(self, command: UpdateAIOpsAgentCommand) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(
                domain_id=command.domain_id,
                agent_id=command.agent_id,
                lock=True,
            )
            if agent is None or agent.status == "ARCHIVED":
                self._not_found()
            if int(agent.row_version) != command.expected_row_version:
                raise AIOpsAgentError(
                    "STATE_VERSION_CONFLICT", "Agent 配置版本已变化"
                )
            current = await self._version(uow.agents, agent)
            current_business_systems = await uow.agents.version_business_system_ids(
                agent_version_id=current.agent_version_id
            )
            current_action_policies = await uow.agents.version_resource_policies(
                agent_version_id=current.agent_version_id
            )
            current_policy = await uow.policies.get_scoped(
                policy_id=current.policy_id, domain_id=command.domain_id
            )
            if current_policy is None:
                raise AIOpsAgentError(
                    "AIOPS_AGENT_POLICY_MISSING", "Agent 执行策略不存在"
                )
            current_rules = dict(current_policy.rules_json or {})
            changes = command.model_dump(
                exclude={"domain_id", "agent_id", "expected_row_version", "actor_id"},
                exclude_unset=True,
            )
            effective_business_system_ids = tuple(
                changes.get("allowed_business_system_ids", tuple(current_business_systems))
            )
            current_controlled_actions = [
                {"resource_id": resource_id, **policy}
                for resource_id, policy in current_action_policies.items()
                if policy
            ]
            effective = {
                "domain_id": command.domain_id,
                "allowed_business_system_ids": effective_business_system_ids,
                "capabilities": changes.get(
                    "capabilities",
                    tuple(current.capabilities_json or ()),
                ),
                "knowledge_policy": changes.get(
                    "knowledge_policy",
                    dict(current.knowledge_policy_json or {}),
                ),
                "controlled_action_execution": changes.get(
                    "controlled_action_execution", current_controlled_actions
                ),
                "auto_alert_enabled": changes.get(
                    "auto_alert_enabled",
                    bool(current_rules.get("auto_alert_enabled", True)),
                ),
                "auto_observe_min_severity": changes.get(
                    "auto_observe_min_severity",
                    current_rules.get("auto_observe_min_severity", "CRITICAL"),
                ),
                "auto_observe_min_target_level": changes.get(
                    "auto_observe_min_target_level",
                    int(current_rules.get("auto_observe_min_target_level", 1)),
                ),
                "alert_cooldown_minutes": changes.get(
                    "alert_cooldown_minutes",
                    int(current_rules.get("alert_cooldown_seconds", 900)) // 60,
                ),
                "models": changes.get("models", current.models_json),
                "image_capabilities": changes.get(
                    "image_capabilities", current.image_capabilities_json
                ),
                "instruction": changes.get("instruction", current.instruction),
                "config": changes.get("config", current.config_json),
            }
            await self._validate_resources(
                uow,
                command.domain_id,
                str(changes.get("status", agent.status)),
                effective,
            )
            version_fields = {
                "allowed_business_system_ids",
                "capabilities",
                "knowledge_policy",
                "controlled_action_execution",
                "auto_alert_enabled",
                "auto_observe_min_severity",
                "auto_observe_min_target_level",
                "alert_cooldown_minutes",
                "models",
                "image_capabilities",
                "instruction",
                "config",
            }
            if version_fields.intersection(changes):
                version_id = uuid7()
                next_version_no = await uow.agents.next_version_no(
                    agent_id=agent.agent_id
                )
                current_policy.status = "RETIRED"
                current_policy.retired_at = datetime.now(UTC)
                current_policy.updated_by = command.actor_id
                current_policy.row_version = int(current_policy.row_version) + 1
                policy = await self._create_policy(
                    uow=uow,
                    agent_id=agent.agent_id,
                    version_no=next_version_no,
                    display_name=changes.get("display_name", agent.display_name),
                    values=effective,
                    actor_id=command.actor_id,
                )
                effective["policy_id"] = policy.policy_id
                await uow.agents.add_version(
                    self._new_version(
                        agent_id=agent.agent_id,
                        version_id=version_id,
                        version_no=next_version_no,
                        values=effective,
                        actor_id=command.actor_id,
                    )
                )
                await uow.agents.add_version_business_systems(
                    version_id=version_id,
                    business_system_ids=effective["allowed_business_system_ids"],
                )
                await uow.agents.add_version_resource_policies(
                    version_id=version_id,
                    controlled_action_policies=self._controlled_action_policies(effective),
                )
                agent.current_version_id = version_id
            for field in (
                "display_name",
                "description",
                "status",
            ):
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
            "enabled_capabilities": row["capabilities"],
            "models": row["models"],
            "instruction": row["instruction"],
            "resource_context": {
                "policy_id": row["policy_id"],
                "allowed_business_system_ids": row["allowed_business_system_ids"],
                "capabilities": row["capabilities"],
                "knowledge_policy": row["knowledge_policy"],
                "controlled_action_execution": row[
                    "controlled_action_execution"
                ],
                "auto_alert_enabled": row["auto_alert_enabled"],
                "auto_observe_min_severity": row["auto_observe_min_severity"],
                "alert_cooldown_minutes": row["alert_cooldown_minutes"],
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

    async def _validate_production_llm_providers(self, values) -> None:
        """生产环境启用 Agent 时，运行时 LLM 必须是本地 DeepSeek。"""
        from aiops_agent.application.diagnosis.local_models import (
            LocalModelBindingError,
            is_production_environment,
            require_production_local_llm,
        )

        if not is_production_environment(self._environment):
            return
        models = dict(values.get("models") or {})
        for role, role_label in (
            ("planner_llm", "规划"),
            ("diagnosis_llm", "诊断"),
            ("composer_llm", "回答组织"),
        ):
            model_id = models.get(role)
            if not model_id:
                continue
            if self._model_client is None:
                raise AIOpsAgentError(
                    "AIOPS_AGENT_MODEL_DIRECTORY_UNAVAILABLE",
                    "生产环境无法校验诊断模型提供方",
                    status_code=503,
                )
            try:
                definition = await self._model_client.get_model(model_id)
            except (LookupError, RuntimeError, TypeError, ValueError) as exc:
                raise AIOpsAgentError(
                    "AIOPS_AGENT_MODEL_DIRECTORY_UNAVAILABLE",
                    f"{role_label}模型目录暂时不可用",
                    status_code=503,
                ) from exc
            try:
                require_production_local_llm(
                    environment=self._environment,
                    provider=(definition or {}).get("provider"),
                    role_label=role_label,
                )
            except LocalModelBindingError as exc:
                raise AIOpsAgentError(
                    exc.code, exc.message, status_code=422
                ) from exc

    async def _validate_resources(
        self, uow, domain_id: int, status: str, values
    ) -> None:
        capabilities = {
            AIOpsCapability(item) for item in values.get("capabilities") or ()
        }
        policies = self._controlled_action_policies(values)
        enabled_policies = {
            resource_id: policy
            for resource_id, policy in policies.items()
            if policy.get("enabled")
        }
        if (
            enabled_policies
            and AIOpsCapability.CONTROLLED_CHANGE not in capabilities
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_CHANGE_CAPABILITY_REQUIRED",
                "配置受控动作前必须启用 CONTROLLED_CHANGE 能力",
                status_code=422,
            )
        if (
            status == "ACTIVE"
            and AIOpsCapability.CONTROLLED_CHANGE in capabilities
            and not enabled_policies
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_CHANGE_POLICY_REQUIRED",
                "启用 CONTROLLED_CHANGE 前必须至少授权一个业务资源",
                status_code=422,
            )
        if (
            status == "ACTIVE"
            and AIOpsCapability.ACTIVE_DIAGNOSIS in capabilities
            and not dict(values.get("models") or {}).get("diagnosis_llm")
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_DIAGNOSIS_MODEL_REQUIRED",
                "启用 Agent 前必须选择诊断模型",
                status_code=422,
            )
        if status == "ACTIVE" and not dict(values.get("models") or {}).get(
            "planner_llm"
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_PLANNER_MODEL_REQUIRED",
                "启用 Agent 前必须选择规划模型",
                status_code=422,
            )
        if status == "ACTIVE" and not dict(values.get("models") or {}).get(
            "composer_llm"
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_COMPOSER_MODEL_REQUIRED",
                "启用 Agent 前必须选择回答组织模型",
                status_code=422,
            )
        if status == "ACTIVE":
            await self._validate_production_llm_providers(values)
        business_system_ids = tuple(
            values.get("allowed_business_system_ids") or ()
        )
        if not business_system_ids:
            raise AIOpsAgentError(
                "AIOPS_AGENT_BUSINESS_SYSTEM_REQUIRED",
                "Agent 至少需要关联一个业务系统",
                status_code=422,
            )
        business_systems = [
            await uow.business_systems.get_scoped(
                business_system_id=business_system_id,
                domain_id=domain_id,
            )
            for business_system_id in business_system_ids
        ]
        if any(system is None for system in business_systems):
            raise AIOpsAgentError(
                "AIOPS_AGENT_RESOURCE_NOT_FOUND", "Agent 引用的业务系统不存在"
            )
        if status == "ACTIVE" and any(
            system.status != "ENABLED" for system in business_systems
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_BUSINESS_SYSTEM_DISABLED",
                "启用 Agent 前，所选业务系统必须已上线",
                status_code=422,
            )
        target_ids = {
            database.target_id
            for business_system_id in business_system_ids
            for link, resource, database, _target in await uow.business_systems.list_system_resource_rows(
                business_system_id
            )
            if link.status == "ACTIVE"
            and resource.lifecycle_status == "ACTIVE"
            and database is not None
        }
        targets = [
            await uow.targets.get_scoped(target_id=target_id, domain_id=domain_id)
            for target_id in sorted(target_ids, key=str)
        ]
        if any(target is None for target in targets):
            raise AIOpsAgentError(
                "AIOPS_AGENT_RESOURCE_NOT_FOUND", "业务系统引用的 Target 不存在"
            )
        target_capabilities = {
            AIOpsCapability.MONITORING_QUERY,
            AIOpsCapability.ALERT_QUERY,
            AIOpsCapability.INSPECTION_QUERY,
            AIOpsCapability.ACTIVE_DIAGNOSIS,
            AIOpsCapability.CONTROLLED_CHANGE,
        }
        if status == "ACTIVE" and capabilities.intersection(
            target_capabilities
        ) and not any(
            target is not None and target.status == "ENABLED" for target in targets
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_NO_ELIGIBLE_TARGET",
                "启用 Agent 前，至少一个业务系统资源必须具备已启用的诊断 Target",
                status_code=422,
            )
        resources_by_id = {
            resource.resource_id: (resource, database, target)
            for business_system_id in business_system_ids
            for link, resource, database, target in await uow.business_systems.list_system_resource_rows(
                business_system_id
            )
            if link.status == "ACTIVE" and resource.lifecycle_status == "ACTIVE"
        }
        for resource_id, action_policy in policies.items():
            resource_context = resources_by_id.get(resource_id)
            if resource_context is None:
                raise AIOpsAgentError(
                    "AIOPS_AGENT_ACTION_RESOURCE_NOT_ALLOWED",
                    "受控动作资源必须属于 Agent 已授权业务系统中的活跃资源",
                    status_code=422,
                )
            if not action_policy.get("enabled"):
                continue
            target = resource_context[2]
            if (
                target is None
                or not target.controlled_change_enabled
                or target.execution_credential_id is None
            ):
                raise AIOpsAgentError(
                    "AIOPS_AGENT_CHANGE_TARGET_REQUIRED",
                    "启用受控动作前，目标必须允许受控变更并配置执行凭据",
                    status_code=422,
                )
            selected = set(action_policy.get("allowed_action_ids") or ())
            compatible = set(self._compatible_action_ids(target))
            if not selected or not selected.issubset(compatible):
                raise AIOpsAgentError(
                    "AIOPS_AGENT_ACTION_NOT_COMPATIBLE",
                    "选择的受控动作与 Target 类型、版本或能力不兼容",
                    status_code=422,
                )
            scopes = dict(action_policy.get("object_scopes") or {})
            required_scopes = self._required_scopes_for_actions(
                target, selected
            )
            missing_scopes = [
                scope
                for scope in required_scopes
                if not scopes.get(scope)
            ]
            if missing_scopes:
                labels = {
                    "schemas": "Schema",
                    "dynamic_parameters": "动态参数及允许值",
                    "resource_manager_plans": "Resource Manager Plan",
                    "privilege_grantees": "允许授权的本地用户",
                    "system_privileges": "系统权限",
                    "object_privileges": "对象权限",
                }
                missing = "、".join(
                    labels.get(item, item) for item in missing_scopes
                )
                raise AIOpsAgentError(
                    "AIOPS_AGENT_ACTION_SCOPE_REQUIRED",
                    f"已选受控动作必须明确选择{missing}",
                    status_code=422,
                )
        return targets

    def _required_scopes_for_actions(
        self, target, action_ids: set[str]
    ) -> tuple[str, ...]:
        """汇总同一动作所有可执行变体的必选范围。"""
        if self._action_registry is None:
            return ()
        compatible = set(self._compatible_action_ids(target))
        requirements: set[str] = set()
        for template in self._action_registry.templates:
            definition = template.definition
            if (
                definition.action_template_id in action_ids
                and definition.action_template_id in compatible
                and definition.db_type == target.db_type
                and definition.execution_mode
                == "EXECUTABLE_AFTER_APPROVAL"
                and definition.status == "ACTIVE"
            ):
                requirements.update(
                    _controlled_scope_requirements(definition)
                )
        return tuple(sorted(requirements))

    def _compatible_action_ids(self, target) -> list[str]:
        if self._action_registry is None:
            return []
        match = re.search(r"\d+", target.version_code or "")
        if match is None:
            return []
        major = int(match.group())
        capabilities = set(target_capability_names(target))
        return sorted(
            {
                template.definition.action_template_id
                for template in self._action_registry.templates
                if template.definition.status == "ACTIVE"
                and template.definition.execution_mode
                == "EXECUTABLE_AFTER_APPROVAL"
                and template.definition.db_type == target.db_type
                and template.definition.supported_version_min <= major
                < template.definition.supported_version_max_exclusive
                and set(template.definition.required_capabilities)
                <= capabilities
                and target.environment
                in template.definition.environment_allowlist
            }
        )

    @staticmethod
    def _controlled_action_policies(values) -> dict[UUID, dict[str, Any]]:
        """把请求中的每个业务资源策略规范化为不可变版本快照。"""
        result: dict[UUID, dict[str, Any]] = {}
        for raw in values.get("controlled_action_execution") or ():
            item = (
                raw.model_dump(mode="python")
                if isinstance(raw, ResourceControlledActionExecution)
                else dict(raw)
            )
            resource_id = UUID(str(item.pop("resource_id")))
            item["allowed_action_ids"] = sorted(
                set(item.get("allowed_action_ids") or ())
            )
            item["object_scopes"] = dict(item.get("object_scopes") or {})
            result[resource_id] = item
        return result

    async def _create_policy(
        self, *, uow, agent_id, version_no, display_name, values, actor_id
    ) -> PolicyEntity:
        now = datetime.now(UTC)
        rules = {
            "schema_version": "ops.policy.v1",
            "readonly_database_enabled": True,
            "auto_alert_enabled": bool(values.get("auto_alert_enabled", True)),
            "auto_observe_min_severity": values.get(
                "auto_observe_min_severity", "CRITICAL"
            ),
            "auto_observe_min_target_level": int(
                values.get("auto_observe_min_target_level", 1)
            ),
            "alert_cooldown_seconds": int(
                values.get("alert_cooldown_minutes", 15)
            )
            * 60,
        }
        policy = PolicyEntity(
            policy_id=uuid7(),
            domain_id=values["domain_id"],
            policy_key=f"agent.{agent_id}",
            version_no=version_no,
            display_name=f"{display_name.strip()} 执行策略",
            rules_json=rules,
            policy_hash=sha256_json(rules),
            status="ACTIVE",
            effective_at=now,
            row_version=1,
            created_by=actor_id,
            updated_by=actor_id,
            created_at=now,
            updated_at=now,
        )
        await uow.policies.add(policy)
        return policy

    @staticmethod
    def _new_version(*, agent_id, version_id, version_no, values, actor_id):
        knowledge_policy = values.get("knowledge_policy") or {}
        if isinstance(knowledge_policy, AgentKnowledgePolicy):
            knowledge_policy = knowledge_policy.model_dump(mode="json")
        return AIOpsAgentVersionEntity(
            agent_version_id=version_id,
            agent_id=agent_id,
            version_no=version_no,
            policy_id=values["policy_id"],
            capabilities_json=[
                str(item) for item in values.get("capabilities") or ()
            ],
            knowledge_policy_json=dict(knowledge_policy),
            models_json={
                key: str(value)
                for key, value in dict(values.get("models") or {}).items()
                if value is not None
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
    async def _view(uow, agent, version):
        business_system_ids = set(await uow.agents.version_business_system_ids(
            agent_version_id=version.agent_version_id
        ))
        candidate_ids = set(await uow.agents.version_target_ids(
            agent_version_id=version.agent_version_id
        ))
        action_policies = await uow.agents.version_resource_policies(
            agent_version_id=version.agent_version_id
        )
        target_candidates = []
        for target_id in sorted(candidate_ids, key=str):
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=int(agent.domain_id),
            )
            if target is None:
                continue
            resource = await uow.business_systems.get_resource_for_target(
                target_id=target_id,
                domain_id=int(agent.domain_id),
            )
            target_candidates.append(
                {
                    "target_id": str(target.target_id),
                    "resource_id": (
                        str(resource.resource_id)
                        if resource is not None
                        else None
                    ),
                    "display_name": target.display_name,
                    "db_type": target.db_type,
                    "status": target.status,
                    "importance_level": int(target.importance_level),
                    "connectivity_status": target.connectivity_status,
                    "readonly_connection_enabled": bool(
                        target.readonly_connection_enabled
                    ),
                    "controlled_change_enabled": bool(
                        target.controlled_change_enabled
                    ),
                }
            )
        business_system_candidates = []
        for business_system_id in sorted(business_system_ids, key=str):
            system = await uow.business_systems.get_scoped(
                business_system_id=business_system_id,
                domain_id=int(agent.domain_id),
            )
            if system is None:
                continue
            business_system_candidates.append(
                {
                    "business_system_id": str(system.business_system_id),
                    "system_code": system.system_code,
                    "display_name": system.display_name,
                    "environment": system.environment,
                    "status": system.status,
                }
            )
        policy = await uow.policies.get_scoped(
            policy_id=version.policy_id, domain_id=int(agent.domain_id)
        )
        rules = dict(policy.rules_json or {}) if policy is not None else {}
        return {
            "agent_id": str(agent.agent_id),
            "domain_id": str(agent.domain_id),
            "display_name": agent.display_name,
            "description": agent.description,
            "status": agent.status,
            "agent_version_id": str(version.agent_version_id),
            "version_no": int(version.version_no),
            "policy_id": str(version.policy_id),
            "allowed_business_system_ids": [
                str(item) for item in sorted(business_system_ids, key=str)
            ],
            "business_system_candidates": business_system_candidates,
            "target_candidates": target_candidates,
            "controlled_action_execution": [
                {"resource_id": str(resource_id), **dict(action_policies[resource_id])}
                for resource_id in sorted(action_policies, key=str)
                if action_policies[resource_id]
            ],
            "auto_alert_enabled": bool(rules.get("auto_alert_enabled", True)),
            "auto_observe_min_severity": rules.get(
                "auto_observe_min_severity", "CRITICAL"
            ),
            "auto_observe_min_target_level": int(
                rules.get("auto_observe_min_target_level", 1)
            ),
            "alert_cooldown_minutes": int(
                rules.get("alert_cooldown_seconds", 900)
            ) // 60,
            "models": dict(version.models_json or {}),
            "capabilities": list(version.capabilities_json or ()),
            "knowledge_policy": dict(version.knowledge_policy_json or {}),
            "instruction": version.instruction,
            "image_capabilities": dict(version.image_capabilities_json or {}),
            "config": dict(version.config_json or {}),
            "row_version": int(agent.row_version),
        }

    @staticmethod
    def _not_found():
        raise AIOpsAgentError(
            "AIOPS_AGENT_NOT_FOUND", "AIOps Agent 不存在", status_code=404
        )


def _controlled_scope_requirements(definition) -> tuple[str, ...]:
    """从动作参数语义推导 Agent–Target 必须显式登记的范围。"""
    parameters = tuple(definition.parameters)
    names = {item.name for item in parameters}
    requirements: set[str] = set()
    object_parameters = [
        item for item in parameters if item.type == "database_object_ref"
    ]
    if any("USER" not in item.object_types for item in object_parameters):
        requirements.add("schemas")
    if {"parameter_name", "parameter_value"} <= names:
        requirements.add("dynamic_parameters")
    if "resource_plan_name" in names:
        requirements.add("resource_manager_plans")
    if "grantee_name" in names:
        requirements.add("privilege_grantees")
        requirements.add(
            "object_privileges" if "object_ref" in names
            else "system_privileges"
        )
    return tuple(sorted(requirements))


def _controlled_action_options(
    capabilities: dict[str, Any],
) -> dict[str, list[str]]:
    """读取 Target 最近一次只读探测固化的动作候选项。"""
    probe = capabilities.get("capability_probe")
    details = probe.get("details") if isinstance(probe, dict) else None
    discovered = (
        details.get("controlled_action_options")
        if isinstance(details, dict)
        else None
    )
    if not isinstance(discovered, dict):
        discovered = capabilities.get("controlled_action_options")
    if not isinstance(discovered, dict):
        discovered = {}
    result = {}
    for key in ("schemas", "resource_manager_plans", "privilege_grantees"):
        values = discovered.get(key, ())
        if not isinstance(values, (list, tuple, set)):
            result[key] = []
            continue
        result[key] = sorted(
            {
                str(value).strip().upper()
                for value in values
                if str(value).strip()
            }
        )
    return result


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
