"""Action Catalog 的不可变本地契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ActionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionParameter(_ActionContract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    type: Literal[
        "integer",
        "enum",
        "database_object_ref",
        "identifier",
        "boolean",
        "size",
        "duration",
        "timestamp",
        "restricted_string",
    ]
    minimum: int | None = None
    maximum: int | None = None
    enum: tuple[str, ...] = ()
    min_length: int | None = Field(default=None, ge=0, le=4096)
    max_length: int | None = Field(default=None, ge=1, le=4096)
    pattern: str | None = Field(default=None, max_length=512)
    object_types: tuple[str, ...] = ()
    source_fact_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_constraints(self) -> "ActionParameter":
        if self.type in {"integer", "size", "duration"}:
            if self.minimum is None or self.maximum is None:
                raise ValueError("数值 Action 参数必须声明上下限")
            if self.minimum > self.maximum:
                raise ValueError("Action 参数上下限无效")
            if self.enum:
                raise ValueError("数值 Action 参数不能声明枚举")
        elif self.type == "enum":
            if not self.enum:
                raise ValueError("枚举 Action 参数不能为空")
        elif self.enum:
            raise ValueError("非枚举 Action 参数不能声明枚举值")
        if self.type in {"identifier", "restricted_string"}:
            if self.max_length is None or self.pattern is None:
                raise ValueError("受限字符串参数必须声明长度和格式")
        if self.type == "database_object_ref" and not self.object_types:
            raise ValueError("数据库对象引用必须声明允许的对象类型")
        return self


class ActionTemplateDefinition(_ActionContract):
    action_template_id: str = Field(
        pattern=r"^[a-z][a-z0-9_.-]{1,127}$"
    )
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    db_type: Literal["ORACLE", "MYSQL"]
    variant: str = Field(min_length=1, max_length=128)
    supported_version_min: int = Field(ge=0)
    supported_version_max_exclusive: int = Field(gt=0)
    required_capabilities: tuple[str, ...] = ()
    required_entitlements: tuple[str, ...] = ()
    required_privileges: tuple[str, ...] = ()
    action_family: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    effect_class: Literal[
        "SESSION_CONTROL",
        "OBJECT_MAINTENANCE",
        "METADATA_REFRESH",
        "SERVICE_CONTROL",
        "CAPACITY_INCREASE",
        "CONFIGURATION_CHANGE",
        "SECURITY_CHANGE",
        "BACKUP_CREATION",
        "AVAILABILITY_TRANSITION",
        "SOFTWARE_MAINTENANCE",
        "DATA_DELETION",
        "OBJECT_DELETION",
        "RECOVERY_MATERIAL_DELETION",
        "STATE_REPLACEMENT",
        "ARBITRARY_MUTATION",
    ]
    execution_mode: Literal[
        "EXECUTABLE_AFTER_APPROVAL", "MANUAL_ONLY", "UNSUPPORTED"
    ]
    executor_kind: Literal["DATABASE", "EXTERNAL", "NONE"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    environment_allowlist: tuple[str, ...]
    parameters: tuple[ActionParameter, ...]
    command_ref: str | None = Field(
        default=None, pattern=r"^[a-zA-Z0-9_./-]+\.(sql|txt)$"
    )
    command_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    compiler_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    renderer_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    validator_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    verifier_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    renderer_version: Literal["strict-template.v2"]
    precondition_tool_refs: tuple[str, ...]
    verification_tool_refs: tuple[str, ...]
    expected_effects: tuple[str, ...]
    rollback_description: str | None = Field(
        default=None, max_length=2000
    )
    statement_timeout_seconds: int = Field(ge=1, le=300)
    observation_delay_seconds: int = Field(ge=0, le=3600)
    idempotency_class: Literal[
        "IDEMPOTENT", "CHECK_THEN_ACT", "NON_RETRYABLE"
    ]
    concurrency_key: str = Field(min_length=1, max_length=128)
    lock_impact: str = Field(min_length=1, max_length=1000)
    estimated_duration_seconds: int = Field(ge=0, le=86400)
    cancellable: bool
    status: Literal["ACTIVE", "DISABLED", "PLANNED"]

    @model_validator(mode="after")
    def validate_definition(self) -> "ActionTemplateDefinition":
        if self.supported_version_min >= self.supported_version_max_exclusive:
            raise ValueError("Action 数据库版本范围无效")
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Action 参数名称不能重复")
        if self.execution_mode != "UNSUPPORTED" and (
            not self.precondition_tool_refs or not self.verification_tool_refs
        ):
            raise ValueError("Action 必须声明前置检查和执行后验证")
        if (self.command_ref is None) != (self.command_sha256 is None):
            raise ValueError("Action 命令引用和 Hash 必须同时声明")
        if self.execution_mode != "UNSUPPORTED" and self.command_ref is None:
            raise ValueError("可展示的 Action 必须声明命令模板")
        if self.execution_mode == "EXECUTABLE_AFTER_APPROVAL" and (
            self.executor_kind == "NONE" or self.status != "ACTIVE"
        ):
            raise ValueError("可执行 Action 必须启用并声明执行器")
        if (
            self.execution_mode == "MANUAL_ONLY"
            and self.executor_kind != "NONE"
        ):
            raise ValueError("人工 Action 不能声明执行器")
        if self.execution_mode == "UNSUPPORTED" and self.executor_kind != "NONE":
            raise ValueError("不支持的 Action 不能声明执行器")
        return self


@dataclass(frozen=True)
class ResolvedActionTemplate:
    definition: ActionTemplateDefinition
    command_template: str | None
    template_hash: str


class RenderedAction(_ActionContract):
    action_template_id: str
    action_template_version: str
    variant: str
    db_type: Literal["ORACLE", "MYSQL"]
    renderer_version: str
    typed_parameters: dict[str, Any]
    parameters_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    command_text: str
    command_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    template_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    risk_level: str
    action_family: str
    effect_class: str
    execution_mode: str
    executor_kind: str
    precondition_tool_refs: tuple[str, ...]
    verification_tool_refs: tuple[str, ...]
    expected_effects: tuple[str, ...]
    rollback_description: str | None = None
    statement_timeout_seconds: int
    observation_delay_seconds: int
    idempotency_class: str
    concurrency_key: str
    lock_impact: str
    estimated_duration_seconds: int
    cancellable: bool
