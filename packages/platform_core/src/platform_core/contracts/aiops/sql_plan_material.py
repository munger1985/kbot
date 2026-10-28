"""代表参数的专用保护提交合同；提交材料不构成计划采集授权。"""

from typing import Literal

from pydantic import Field, SecretStr

from .types import AIOpsContract, UUIDv7, UtcDatetime


class SqlPlanMaterialSource(AIOpsContract):
    source_artifact_id: UUIDv7
    step_id: str = Field(min_length=1, max_length=128)
    row_index: int = Field(ge=0, le=255)
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    subject: dict[str, str | bool | None] = Field(max_length=8)


class SqlPlanMaterialSubmit(AIOpsContract):
    submission_id: UUIDv7
    source_artifact_id: UUIDv7
    step_id: str = Field(min_length=1, max_length=128)
    row_index: int = Field(ge=0, le=255)
    protected_document: SecretStr = Field(min_length=2, max_length=65536, repr=False)
    ttl_seconds: int = Field(default=900, ge=60, le=1800, strict=True)


class SqlPlanMaterialView(AIOpsContract):
    material_id: UUIDv7
    ops_run_id: UUIDv7
    source: SqlPlanMaterialSource
    parameter_count: int = Field(ge=0, le=64)
    created_at: UtcDatetime
    expires_at: UtcDatetime
    state: Literal["STORED", "EXPIRED", "REVOKED", "UNAVAILABLE"]
    execution_authorized: Literal[False] = False
    representative_parameters: Literal["USER_DECLARED_UNVERIFIED"] = "USER_DECLARED_UNVERIFIED"
    environment: Literal["USER_DECLARED_UNVERIFIED"] = "USER_DECLARED_UNVERIFIED"
    notice: str = "仅保护保存材料，不触发 EXPLAIN、不重放业务 SQL；仍需独立风险审批及采集能力。"


class SqlPlanMaterialPage(AIOpsContract):
    items: tuple[SqlPlanMaterialView, ...] = Field(max_length=16)
