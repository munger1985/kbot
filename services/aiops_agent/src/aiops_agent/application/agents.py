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
    AIOpsAgentGrantEntity,
    AIOpsAgentVersionEntity,
    PolicyEntity,
)
from aiops_agent.application.configuration.common import sha256_json
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
    diagnostic_source_ids: tuple[UUID, ...] = Field(min_length=1, max_length=16)
    target_id: UUID | None = None
    allow_change_execution: bool = False
    auto_alert_enabled: bool = True
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] = "CRITICAL"
    alert_cooldown_minutes: int = Field(default=15, ge=0, le=1440)
    models: dict[str, UUID] = Field(default_factory=dict)
    image_capabilities: AgentImageCapabilities = Field(
        default_factory=AgentImageCapabilities
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"
    actor_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_resources(self):
        if len(set(self.diagnostic_source_ids)) != len(self.diagnostic_source_ids):
            raise ValueError("diagnostic_source_ids 不能重复")
        if self.allow_change_execution and self.target_id is None:
            raise ValueError("允许执行变更前必须选择数据库直连 Target")
        return self


class UpdateAIOpsAgentCommand(_Model):
    domain_id: int = Field(ge=1)
    agent_id: UUID
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    diagnostic_source_ids: tuple[UUID, ...] | None = Field(
        default=None, min_length=1, max_length=16
    )
    target_id: UUID | None = None
    allow_change_execution: bool | None = None
    auto_alert_enabled: bool | None = None
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] | None = None
    alert_cooldown_minutes: int | None = Field(default=None, ge=0, le=1440)
    models: dict[str, UUID] | None = None
    image_capabilities: AgentImageCapabilities | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None
    actor_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_resources(self):
        if (
            self.diagnostic_source_ids is not None
            and len(set(self.diagnostic_source_ids))
            != len(self.diagnostic_source_ids)
        ):
            raise ValueError("diagnostic_source_ids 不能重复")
        return self


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
    def __init__(self, *, uow_factory, action_registry=None):
        self._uow_factory = uow_factory
        self._action_registry = action_registry

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
                await uow.agents.add_version_sources(
                    version_id=version_id,
                    source_ids=command.diagnostic_source_ids,
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
            if agent is None:
                self._not_found()
            return await self._view(
                uow, agent, await self._version(uow.agents, agent)
            )

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
            current_sources = await uow.agents.version_source_ids(
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
            effective = {
                "domain_id": command.domain_id,
                "diagnostic_source_ids": changes.get(
                    "diagnostic_source_ids", tuple(current_sources)
                ),
                "target_id": changes.get("target_id", current.target_id),
                "allow_change_execution": changes.get(
                    "allow_change_execution",
                    bool(current_rules.get("allow_agent_execution", False)),
                ),
                "auto_alert_enabled": changes.get(
                    "auto_alert_enabled",
                    bool(current_rules.get("auto_alert_enabled", True)),
                ),
                "auto_observe_min_severity": changes.get(
                    "auto_observe_min_severity",
                    current_rules.get("auto_observe_min_severity", "CRITICAL"),
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
                uow, command.domain_id, str(changes.get("status", agent.status)), effective
            )
            version_fields = {
                "diagnostic_source_ids",
                "target_id",
                "allow_change_execution",
                "auto_alert_enabled",
                "auto_observe_min_severity",
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
                await uow.agents.add_version_sources(
                    version_id=version_id,
                    source_ids=effective["diagnostic_source_ids"],
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
                "diagnostic_source_ids": row["diagnostic_source_ids"],
                "target_id": row["target_id"],
                "allow_change_execution": row["allow_change_execution"],
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

    async def _validate_resources(
        self, uow, domain_id: int, status: str, values
    ) -> None:
        if status == "ACTIVE" and not dict(values.get("models") or {}).get(
            "diagnosis_llm"
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_DIAGNOSIS_MODEL_REQUIRED",
                "启用 Agent 前必须选择诊断模型",
                status_code=422,
            )
        source_ids = tuple(values.get("diagnostic_source_ids") or ())
        if not source_ids:
            raise AIOpsAgentError(
                "AIOPS_AGENT_SOURCE_REQUIRED", "Agent 至少需要选择一个监控源",
                status_code=422,
            )
        sources = [
            await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id, domain_id=domain_id
            )
            for source_id in source_ids
        ]
        if any(source is None for source in sources):
            raise AIOpsAgentError(
                "AIOPS_AGENT_RESOURCE_NOT_FOUND", "Agent 引用的监控源不存在"
            )
        if status == "ACTIVE" and any(
            source.status != "ENABLED"
            or source.connectivity_status not in {"CONNECTED", "DEGRADED"}
            for source in sources
        ):
            raise AIOpsAgentError(
                "AIOPS_AGENT_SOURCE_UNAVAILABLE",
                "启用 Agent 前，所选监控源必须已启用且连接健康",
                status_code=422,
            )
        target_id = values.get("target_id")
        target = None
        if target_id is not None:
            target = await uow.targets.get_scoped(
                target_id=target_id, domain_id=domain_id
            )
            if target is None:
                raise AIOpsAgentError(
                    "AIOPS_AGENT_RESOURCE_NOT_FOUND", "Agent 引用的 Target 不存在"
                )
            if status == "ACTIVE" and (
                target.status != "ENABLED"
                or target.connectivity_status not in {"CONNECTED", "DEGRADED"}
            ):
                raise AIOpsAgentError(
                    "AIOPS_AGENT_TARGET_UNAVAILABLE",
                    "启用 Agent 前，数据库直连 Target 必须已启用且连接健康",
                    status_code=422,
                )
            if status == "ACTIVE":
                bindings = await uow.targets.list_source_bindings(
                    target_id=target_id,
                    domain_id=domain_id,
                    active_only=True,
                )
                bound_source_ids = {
                    binding.diagnostic_source_id for binding in bindings
                }
                if not set(source_ids) <= bound_source_ids:
                    raise AIOpsAgentError(
                        "AIOPS_AGENT_SOURCE_NOT_BOUND",
                        "启用 Agent 前，所选监控源必须与数据库直连 Target 建立有效映射",
                        status_code=422,
                    )
        if values.get("allow_change_execution"):
            if target is None:
                raise AIOpsAgentError(
                    "AIOPS_AGENT_TARGET_REQUIRED",
                    "允许执行变更前必须选择数据库直连 Target",
                    status_code=422,
                )
            actions = self._compatible_action_ids(target)
            values["_allowed_action_types"] = actions

    def _compatible_action_ids(self, target) -> list[str]:
        if self._action_registry is None:
            return []
        match = re.search(r"\d+", target.version_code or "")
        if match is None:
            return []
        major = int(match.group())
        capabilities = {
            key
            for key, enabled in dict(target.capabilities_json or {}).items()
            if enabled is True
        }
        features = dict(target.capabilities_json or {}).get("features", [])
        if isinstance(features, list):
            capabilities.update(str(item) for item in features if item)
        return sorted(
            {
                template.definition.action_template_id
                for template in self._action_registry.templates
                if template.definition.status == "ACTIVE"
                and template.definition.execution_capability
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

    async def _create_policy(
        self, *, uow, agent_id, version_no, display_name, values, actor_id
    ) -> PolicyEntity:
        now = datetime.now(UTC)
        rules = {
            "schema_version": "ops.policy.v1",
            "readonly_database_enabled": values.get("target_id") is not None,
            "allow_agent_execution": bool(
                values.get("allow_change_execution", False)
            ),
            "allowed_action_types": list(values.get("_allowed_action_types") or []),
            "auto_alert_enabled": bool(values.get("auto_alert_enabled", True)),
            "auto_observe_min_severity": values.get(
                "auto_observe_min_severity", "CRITICAL"
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
        return AIOpsAgentVersionEntity(
            agent_version_id=version_id,
            agent_id=agent_id,
            version_no=version_no,
            policy_id=values["policy_id"],
            target_id=values.get("target_id"),
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
    async def _view(uow, agent, version):
        source_ids = await uow.agents.version_source_ids(
            agent_version_id=version.agent_version_id
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
            "diagnostic_source_ids": [str(item) for item in source_ids],
            "target_id": str(version.target_id) if version.target_id else None,
            "allow_change_execution": bool(
                rules.get("allow_agent_execution", False)
            ),
            "auto_alert_enabled": bool(rules.get("auto_alert_enabled", True)),
            "auto_observe_min_severity": rules.get(
                "auto_observe_min_severity", "CRITICAL"
            ),
            "alert_cooldown_minutes": int(
                rules.get("alert_cooldown_seconds", 900)
            ) // 60,
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
