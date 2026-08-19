"""管理面输入 DTO；禁止任意 SQL，凭据只存在于单次写请求。"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .query_plan import FilterOperator


_KEY_PATTERN = r"^[a-z][a-z0-9._-]{0,127}$"
_OBJECT_PATTERN = r"^[A-Za-z_][A-Za-z0-9_$#-]{0,127}$"
_DATABASE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._$#-]{0,127}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataSourceEndpoint(_Contract):
    host: str = Field(min_length=1, max_length=253, pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
    port: int = Field(ge=1, le=65535)
    database: str = Field(min_length=1, max_length=128, pattern=_DATABASE_PATTERN)
    allowed_schemas: tuple[str, ...] = Field(min_length=1, max_length=32)
    tls_enabled: bool = True

    @model_validator(mode="after")
    def validate_allowed_schemas(self) -> "DataSourceEndpoint":
        if any(not re.fullmatch(_OBJECT_PATTERN, item) for item in self.allowed_schemas):
            raise ValueError("allowed_schemas 包含非法数据库标识符")
        if len(set(self.allowed_schemas)) != len(self.allowed_schemas):
            raise ValueError("allowed_schemas 不能重复")
        return self


class DataSourceCredentials(_Contract):
    """只允许在 BFF 到内部管理服务的单次请求中出现。"""

    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=1024)


class DataSourceConnectionTest(_Contract):
    source_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    endpoint: DataSourceEndpoint
    credentials: DataSourceCredentials


class DataSourceConnectionTestResult(_Contract):
    ok: bool
    database_version: str | None = None
    capabilities: dict[str, object] = Field(default_factory=dict)


class DataSourceCreate(_Contract):
    display_name: str = Field(min_length=1, max_length=256)
    source_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    endpoint: DataSourceEndpoint
    credentials: DataSourceCredentials
    auto_discover_schema: bool = True


class DataSourceUpdate(_Contract):
    display_name: str = Field(min_length=1, max_length=256)
    endpoint: DataSourceEndpoint
    credentials: DataSourceCredentials | None = None
    expected_row_version: int = Field(ge=1)


class DatasetDefinition(_Contract):
    name: str = Field(pattern=_KEY_PATTERN)
    display_name: str = Field(min_length=1, max_length=256)
    physical_schema: str = Field(pattern=_OBJECT_PATTERN)
    physical_object: str = Field(pattern=_OBJECT_PATTERN)
    primary_time_dimension: str | None = Field(default=None, pattern=_KEY_PATTERN)
    scope_column: str | None = Field(default=None, pattern=_OBJECT_PATTERN)


class DimensionDefinition(_Contract):
    name: str = Field(pattern=_KEY_PATTERN)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    dataset: str = Field(pattern=_KEY_PATTERN)
    physical_column: str = Field(pattern=_OBJECT_PATTERN)
    value_type: Literal["STRING", "INTEGER", "DECIMAL", "DATE", "DATETIME", "BOOLEAN"]
    groupable: bool = True
    filterable: bool = True
    sensitivity: Literal["PUBLIC", "INTERNAL", "SENSITIVE"] = "INTERNAL"
    synonyms: tuple[str, ...] = Field(default=(), max_length=32)
    value_normalization: Literal["NONE", "LOWER_TRIM"] = "NONE"
    allowed_filter_operators: tuple[FilterOperator, ...] = (
        "EQ", "NE", "IN", "NOT_IN", "BETWEEN", "GT", "GTE", "LT", "LTE",
        "CONTAINS", "STARTS_WITH", "IS_NULL", "IS_NOT_NULL",
    )
    filter_alias_columns: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_filter_contract(self) -> "DimensionDefinition":
        if not self.allowed_filter_operators:
            raise ValueError("维度必须允许至少一种筛选操作符")
        if len(self.allowed_filter_operators) != len(set(self.allowed_filter_operators)):
            raise ValueError("维度筛选操作符不能重复")
        if len(self.filter_alias_columns) != len(set(self.filter_alias_columns)):
            raise ValueError("维度备用筛选列不能重复")
        return self


class MeasureDefinition(_Contract):
    name: str = Field(pattern=_KEY_PATTERN)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    dataset: str = Field(pattern=_KEY_PATTERN)
    physical_column: str | None = Field(default=None, pattern=_OBJECT_PATTERN)
    aggregation: Literal["COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"]
    value_type: Literal["INTEGER", "DECIMAL"]
    sensitivity: Literal["PUBLIC", "INTERNAL", "SENSITIVE"] = "INTERNAL"

    @model_validator(mode="after")
    def validate_count_column(self) -> "MeasureDefinition":
        if self.aggregation == "COUNT" and self.physical_column is not None:
            raise ValueError("COUNT 指标不允许物理列；请使用 COUNT_DISTINCT 或其他聚合")
        if self.aggregation != "COUNT" and self.physical_column is None:
            raise ValueError("非 COUNT 指标必须指定 physical_column")
        return self


class SemanticModelDefinition(_Contract):
    datasets: tuple[DatasetDefinition, ...] = Field(min_length=1, max_length=64)
    dimensions: tuple[DimensionDefinition, ...] = Field(default=(), max_length=512)
    measures: tuple[MeasureDefinition, ...] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_logical_names(self) -> "SemanticModelDefinition":
        datasets = [item.name for item in self.datasets]
        dimensions = [item.name for item in self.dimensions]
        measures = [item.name for item in self.measures]
        if len(datasets) != len(set(datasets)):
            raise ValueError("Dataset 名称不能重复")
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("Dimension 名称不能重复")
        if len(measures) != len(set(measures)):
            raise ValueError("Measure 名称不能重复")
        dataset_names = set(datasets)
        if any(item.dataset not in dataset_names for item in (*self.dimensions, *self.measures)):
            raise ValueError("Dimension/Measure 必须引用已声明的 Dataset")
        return self


class SemanticModelDraftCreate(_Contract):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    data_source_id: UUID
    schema_snapshot_id: UUID
    definition: SemanticModelDefinition


class QueryBudget(_Contract):
    max_rows: int = Field(default=1000, ge=1, le=10000)
    max_result_bytes: int = Field(default=1_048_576, ge=1024, le=16_777_216)
    statement_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_concurrent_runs: int = Field(default=4, ge=1, le=64)


class SubjectSelector(_Contract):
    actor_ids: tuple[str, ...] = Field(default=(), max_length=1000)
    roles: tuple[str, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def require_subject(self) -> "SubjectSelector":
        if not self.actor_ids and not self.roles:
            raise ValueError("Policy 必须至少指定一个用户或角色")
        return self


class PolicyBindingCreate(_Contract):
    semantic_model_ids: tuple[UUID, ...] = Field(min_length=1, max_length=64)
    subject_selector: SubjectSelector
    budget: QueryBudget = Field(default_factory=QueryBudget)

    @model_validator(mode="after")
    def unique_models(self) -> "PolicyBindingCreate":
        if len(self.semantic_model_ids) != len(set(self.semantic_model_ids)):
            raise ValueError("semantic_model_ids 不能重复")
        return self


class AgentBindingCreate(_Contract):
    consumer_app_id: str = Field(min_length=1, max_length=128)
    agent_id: UUID
    agent_version_id: UUID
    semantic_model_id: UUID
    policy_binding_id: UUID


class AgentBindingMatch(_Contract):
    consumer_app_id: str = Field(min_length=1, max_length=128)
    agent_id: UUID
    agent_version_id: UUID
    semantic_model_ids: tuple[UUID, ...] = Field(default=(), max_length=64)


class AgentBindingMatchResult(_Contract):
    matched: bool


class PublishSemanticModelCommand(_Contract):
    semantic_model_id: UUID
    semantic_model_version_id: UUID
    schema_snapshot_id: UUID
    expected_row_version: int = Field(ge=1)


class SubmitSemanticModelReviewCommand(_Contract):
    expected_row_version: int = Field(ge=1)


class ReturnSemanticModelForRevisionCommand(_Contract):
    review_comment: str = Field(min_length=2, max_length=2000)
    expected_row_version: int = Field(ge=1)

    @field_validator("review_comment")
    @classmethod
    def normalize_review_comment(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("审核意见至少需要 2 个字符")
        return normalized


class RetireSemanticModelVersionCommand(_Contract):
    expected_row_version: int = Field(ge=1)


class DeleteSemanticModelCommand(_Contract):
    expected_row_version: int = Field(ge=1)


class DataSourceView(_Contract):
    data_source_id: UUID
    display_name: str
    source_type: str
    status: str
    current_version: int = Field(ge=1)
    row_version: int = Field(ge=1)


class DataSourceCredentialStatus(_Contract):
    configured: bool
    key_version: str
    updated_at: datetime


class DataSourceDetail(DataSourceView):
    """可安全返回给 BFF 的数据源详情；绝不包含密文或数据库用户名。"""

    endpoint: DataSourceEndpoint
    credential: DataSourceCredentialStatus
    capabilities: dict[str, object] = Field(default_factory=dict)
    error_code: str | None = None
    updated_at: datetime


class DataSourcePage(_Contract):
    items: tuple[DataSourceView, ...]
    next_cursor: UUID | None = None


class DataSourceStatusChange(_Contract):
    status: Literal["DISABLED"]
    expected_row_version: int = Field(ge=1)


class SchemaSnapshotReceipt(_Contract):
    schema_snapshot_id: UUID
    data_source_id: UUID
    status: str
    source_version: int = Field(ge=1)


class SchemaSnapshotObjectView(_Contract):
    schema_snapshot_object_id: UUID
    schema_name: str
    object_name: str
    object_type: str
    selected: bool
    status: str
    attempt_count: int = Field(ge=0)
    metadata_source: str
    column_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None


class SchemaSnapshotSummary(SchemaSnapshotReceipt):
    created_at: datetime
    discovered_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    completed_at: datetime | None = None


class SchemaSnapshotDetail(SchemaSnapshotSummary):
    objects: tuple[SchemaSnapshotObjectView, ...]


class SchemaSnapshotPage(_Contract):
    items: tuple[SchemaSnapshotSummary, ...]


class SchemaObjectSelection(_Contract):
    object_ids: tuple[UUID, ...] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "SchemaObjectSelection":
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("object_ids 不能重复")
        return self


class ManualSchemaDefinition(_Contract):
    ddl: str = Field(min_length=10, max_length=100_000)


class SemanticModelCandidateRequest(_Contract):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    business_context: str | None = Field(default=None, max_length=4000)
    object_ids: tuple[UUID, ...] = Field(default=(), max_length=64)
    ai_model_id: UUID | None = None
    allow_ai_metadata: bool = False


class SemanticModelCandidate(_Contract):
    data_source_id: UUID
    schema_snapshot_id: UUID
    definition: SemanticModelDefinition
    warnings: tuple[str, ...] = ()


class SemanticModelGeneratedDraft(SemanticModelCandidate):
    semantic_model_id: UUID
    semantic_model_version_id: UUID
    version_no: int = Field(ge=1)
    status: str
    row_version: int = Field(ge=1)


class SemanticModelGenerationReceipt(_Contract):
    generation_job_id: UUID
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]


class SemanticModelGenerationView(SemanticModelGenerationReceipt):
    data_source_id: UUID
    schema_snapshot_id: UUID
    semantic_model_id: UUID | None = None
    semantic_model_version_id: UUID | None = None
    error_code: str | None = None


class SemanticModelDraftUpdate(_Contract):
    definition: SemanticModelDefinition
    expected_row_version: int = Field(ge=1)


class SemanticModelValidationRequest(_Contract):
    question: str = Field(min_length=2, max_length=2000)
    ai_model_id: UUID
    idempotency_key: str = Field(min_length=8, max_length=128)
    allow_ai_metadata: bool = False


class SemanticModelValidationReceipt(_Contract):
    data_query_run_id: UUID
    status: str
    query_plan: dict[str, object]


class SemanticModelValidationResult(_Contract):
    data_query_run_id: UUID
    status: str
    error_code: str | None = None
    columns: tuple[dict[str, object], ...] = ()
    preview_rows: tuple[dict[str, object], ...] = ()
    row_count: int | None = Field(default=None, ge=0)
    truncated: bool = False


class SemanticModelDraftView(_Contract):
    semantic_model_id: UUID
    semantic_model_version_id: UUID
    version_no: int = Field(ge=1)
    status: str
    row_version: int = Field(ge=1)


class SemanticModelView(_Contract):
    semantic_model_id: UUID
    display_name: str
    description: str | None = None
    active_version: int | None = Field(default=None, ge=1)
    row_version: int = Field(ge=1)


class SemanticModelVersionView(_Contract):
    semantic_model_version_id: UUID
    semantic_model_id: UUID
    version_no: int = Field(ge=1)
    data_source_id: UUID
    schema_snapshot_id: UUID
    status: str
    row_version: int = Field(ge=1)
    review_comment: str | None = None
    definition: SemanticModelDefinition | None = None


class SemanticModelDetail(SemanticModelView):
    versions: tuple[SemanticModelVersionView, ...]
    updated_at: datetime


class SemanticModelPage(_Contract):
    items: tuple[SemanticModelView, ...]
    next_cursor: UUID | None = None


class SemanticModelSearch(_Contract):
    semantic_model_ids: tuple[UUID, ...] = Field(min_length=1, max_length=5000)
    query: str | None = Field(default=None, max_length=120)
    publication_status: Literal["PUBLISHED", "UNPUBLISHED"] | None = None
    cursor: UUID | None = None
    limit: int = Field(default=20, ge=1, le=100)


class PolicyBindingView(_Contract):
    policy_binding_id: UUID
    status: str
    row_version: int = Field(ge=1)


class PolicyBindingDetail(PolicyBindingView):
    semantic_model_ids: tuple[UUID, ...]
    subject_selector: SubjectSelector
    budget: QueryBudget
    updated_at: datetime


class PolicyBindingPage(_Contract):
    items: tuple[PolicyBindingDetail, ...]
    next_cursor: UUID | None = None


class PolicyBindingStatusChange(_Contract):
    status: Literal["DISABLED"]
    expected_row_version: int = Field(ge=1)


class AgentBindingView(_Contract):
    agent_binding_id: UUID
    consumer_app_id: str
    agent_id: UUID
    agent_version_id: UUID
    semantic_model_id: UUID
    policy_binding_id: UUID
    status: str
    row_version: int = Field(ge=1)


class AgentBindingPage(_Contract):
    items: tuple[AgentBindingView, ...]
    next_cursor: UUID | None = None


class AgentBindingStatusChange(_Contract):
    status: Literal["DISABLED"]
    expected_row_version: int = Field(ge=1)


class VerifiedQueryView(_Contract):
    verified_query_id: UUID
    semantic_model_version_id: UUID
    question_summary: str
    status: str
    verified_by: str | None = None
    verified_at: datetime | None = None
    row_version: int = Field(ge=1)


class PromoteVerifiedQueryCommand(_Contract):
    data_query_run_id: UUID
    assertion: dict[str, object] = Field(default_factory=dict)


class VerifiedQueryPage(_Contract):
    items: tuple[VerifiedQueryView, ...]
    next_cursor: UUID | None = None


class DataQueryAuditView(_Contract):
    audit_id: UUID
    data_query_run_id: UUID | None = None
    actor_id: str | None = None
    trace_id: str
    action: str
    created_at: datetime


class DataQueryAuditPage(_Contract):
    items: tuple[DataQueryAuditView, ...]
    next_cursor: UUID | None = None
