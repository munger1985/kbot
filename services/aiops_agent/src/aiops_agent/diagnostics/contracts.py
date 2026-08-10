"""版本化数据库诊断目录的本地契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiagnosticParameter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    type: Literal["string", "integer", "boolean"]
    required: bool = True
    minimum: int | None = None
    maximum: int | None = None
    max_length: int | None = Field(default=None, ge=1, le=4096)
    enum: tuple[str, ...] = ()
    default: Any | None = None

    @model_validator(mode="after")
    def validate_constraints(self) -> "DiagnosticParameter":
        if self.type == "integer" and self.max_length is not None:
            raise ValueError("integer 参数不能设置 max_length")
        if self.type != "integer" and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError("只有 integer 参数允许数值范围")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("参数最小值不能大于最大值")
        if not self.required and self.default is None:
            raise ValueError("可选参数必须具有确定性默认值")
        return self


class DiagnosticOutputColumn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    logical_type: Literal[
        "STRING", "INTEGER", "DECIMAL", "BOOLEAN", "DATETIME"
    ]
    sensitivity: Literal["PUBLIC", "MASKED", "HASHED"] = "PUBLIC"
    nullable: bool = True


class DiagnosticToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_id: str = Field(pattern=r"^db\.[a-z0-9_.-]{1,124}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    db_type: Literal["POSTGRESQL", "ORACLE", "MYSQL"]
    variant: str = Field(min_length=1, max_length=128)
    supported_version_min: int = Field(ge=0)
    supported_version_max_exclusive: int = Field(gt=0)
    required_capabilities: tuple[str, ...] = ()
    required_entitlements: tuple[str, ...] = ()
    required_privileges: tuple[str, ...] = ()
    parameters: tuple[DiagnosticParameter, ...] = ()
    output_columns: tuple[DiagnosticOutputColumn, ...]
    template_ref: str = Field(pattern=r"^[a-zA-Z0-9_./-]+\.sql$")
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timeout_seconds: int = Field(gt=0, le=300)
    max_rows: int = Field(gt=0, le=10000)
    max_bytes: int = Field(gt=0, le=20 * 1024 * 1024)
    cost_level: Literal["LOW", "MEDIUM", "HIGH"]

    @model_validator(mode="after")
    def validate_definition(self) -> "DiagnosticToolDefinition":
        if self.supported_version_min >= self.supported_version_max_exclusive:
            raise ValueError("数据库版本范围无效")
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("参数名称不能重复")
        columns = [item.name for item in self.output_columns]
        if len(columns) != len(set(columns)):
            raise ValueError("输出列名称不能重复")
        return self
