"""Action Catalog 的不可变本地契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ActionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionParameter(_ActionContract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    type: Literal["integer", "enum"]
    minimum: int | None = None
    maximum: int | None = None
    enum: tuple[str, ...] = ()
    source_fact_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_constraints(self) -> "ActionParameter":
        if self.type == "integer":
            if self.minimum is None or self.maximum is None:
                raise ValueError("整数 Action 参数必须声明上下限")
            if self.minimum > self.maximum:
                raise ValueError("Action 参数上下限无效")
            if self.enum:
                raise ValueError("整数 Action 参数不能声明枚举")
        elif not self.enum:
            raise ValueError("枚举 Action 参数不能为空")
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
    execution_capability: Literal[
        "ADVISORY_ONLY", "EXECUTABLE_AFTER_APPROVAL"
    ]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    environment_allowlist: tuple[str, ...]
    parameters: tuple[ActionParameter, ...]
    command_ref: str = Field(pattern=r"^[a-zA-Z0-9_./-]+\.sql$")
    command_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    renderer_version: Literal["strict-template.v1"]
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
    status: Literal["ACTIVE", "DISABLED"]

    @model_validator(mode="after")
    def validate_definition(self) -> "ActionTemplateDefinition":
        if self.supported_version_min >= self.supported_version_max_exclusive:
            raise ValueError("Action 数据库版本范围无效")
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Action 参数名称不能重复")
        if not self.precondition_tool_refs or not self.verification_tool_refs:
            raise ValueError("Action 必须声明前置检查和执行后验证")
        return self


@dataclass(frozen=True)
class ResolvedActionTemplate:
    definition: ActionTemplateDefinition
    command_template: str
    template_hash: str


class RenderedAction(_ActionContract):
    action_template_id: str
    action_template_version: str
    variant: str
    db_type: Literal["ORACLE", "MYSQL"]
    renderer_version: str
    typed_parameters: dict[str, int | str]
    parameters_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    command_text: str
    command_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    template_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    risk_level: str
    execution_capability: str
    precondition_tool_refs: tuple[str, ...]
    verification_tool_refs: tuple[str, ...]
    expected_effects: tuple[str, ...]
    rollback_description: str | None = None
    statement_timeout_seconds: int
    observation_delay_seconds: int
    idempotency_class: str
    concurrency_key: str
