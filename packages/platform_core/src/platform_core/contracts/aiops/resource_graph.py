"""业务系统资源图谱的共享 Wire 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .configuration import SourceBindingCreate, TargetCreate
from .types import AIOpsContract, CursorPage, DatabaseType, JsonObject, PUBLIC_SCHEMA_VERSION, UUIDv7, UtcDatetime


ArchitectureLayerKey = Literal["ACCESS", "APPLICATION", "MIDDLEWARE", "DATA", "INFRASTRUCTURE"]
ResourceIdentityKind = Literal["STABLE", "DISCOVERED", "EPHEMERAL"]
ResourceLifecycleStatus = Literal["DISCOVERED", "ACTIVE", "INACTIVE", "RETIRED"]
SystemResourceStatus = Literal["ACTIVE", "REMOVED"]
RelationOrigin = Literal["MANUAL", "DISCOVERY", "IMPORT"]
RelationConfirmationStatus = Literal["SUGGESTED", "CONFIRMED", "REJECTED"]
RelationVerificationStatus = Literal["UNKNOWN", "MATCHED", "DRIFTED", "UNREACHABLE"]
ResourceCommitMode = Literal["NEW"]
DraftMode = Literal["CREATE", "EDIT"]
DraftStep = Literal["BUSINESS_INFO", "SYSTEM_COMPOSITION", "TOPOLOGY_CONFIRMATION"]
DraftStatus = Literal["ACTIVE", "COMMITTED", "ABANDONED", "EXPIRED"]


class BusinessSystemInfoInput(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    environment: Literal["PROD", "STG", "DEV"]
    importance_level: int = Field(default=3, ge=1, le=5)
    responsibility_group_id: UUIDv7
    description: str | None = Field(default=None, max_length=1000)


class BusinessSystemInfoPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    environment: Literal["PROD", "STG", "DEV"] | None = None
    importance_level: int | None = Field(default=None, ge=1, le=5)
    responsibility_group_id: UUIDv7 | None = None
    description: str | None = Field(default=None, max_length=1000)


class BusinessSystemLifecycleCommand(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    reason: str | None = Field(default=None, max_length=1000)


class ResourceTypeSummary(AIOpsContract):
    resource_type_key: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=128)
    count: int = Field(ge=0)


class ResourceMonitoringInput(AIOpsContract):
    """数据库资源的监控映射；候选引用必须由服务端签发。"""

    source_id: UUIDv7
    candidate_ref: str | None = Field(default=None, min_length=1, max_length=8192)
    host_candidate_ref: str | None = Field(default=None, min_length=1, max_length=8192)
    source_binding: SourceBindingCreate | None = None

    @model_validator(mode="after")
    def validate_mapping_mode(self) -> "ResourceMonitoringInput":
        if (self.candidate_ref is None) == (self.source_binding is None):
            raise ValueError("监控映射必须且只能使用发现候选或稳定定位配置之一")
        if self.source_binding is not None and self.source_binding.source_id != self.source_id:
            raise ValueError("监控映射的 Source ID 不一致")
        if self.candidate_ref is None and self.host_candidate_ref is not None:
            raise ValueError("主机候选必须与数据库候选一起提交")
        return self


class DatabaseResourceCommit(AIOpsContract):
    target: TargetCreate
    monitoring: ResourceMonitoringInput


class BusinessSystemResourceCommit(AIOpsContract):
    """Wizard 草稿中的资源；draft_resource_id 由客户端生成但不向用户展示。"""

    draft_resource_id: UUIDv7
    mode: ResourceCommitMode = "NEW"
    resource_type_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    display_name: str = Field(min_length=1, max_length=256)
    identity_kind: ResourceIdentityKind = "STABLE"
    owner_group_id: UUIDv7
    architecture_layer_key: ArchitectureLayerKey
    responsibility_group_id: UUIDv7 | None = None
    location_name: str | None = Field(default=None, max_length=256)
    display_order: int = Field(default=0, ge=0, le=1_000_000)
    attributes: JsonObject = Field(default_factory=dict)
    database: DatabaseResourceCommit | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "BusinessSystemResourceCommit":
        if self.resource_type_key == "DATABASE" and self.database is None:
            raise ValueError("新数据库资源必须提交数据库连接配置")
        if self.resource_type_key != "DATABASE" and self.database is not None:
            raise ValueError("非数据库资源不能提交数据库连接配置")
        return self


class RelationParticipantCommit(AIOpsContract):
    draft_resource_id: UUIDv7
    participant_role: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    display_order: int = Field(default=0, ge=0, le=1_000_000)


class ResourceRelationCommit(AIOpsContract):
    draft_relation_id: UUIDv7
    relation_type_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    display_name: str | None = Field(default=None, max_length=256)
    responsibility_group_id: UUIDv7 | None = None
    origin: RelationOrigin = "MANUAL"
    confirmation_status: RelationConfirmationStatus = "CONFIRMED"
    attributes: JsonObject = Field(default_factory=dict)
    participants: tuple[RelationParticipantCommit, ...] = Field(min_length=2, max_length=10_000)

    @model_validator(mode="after")
    def validate_participants(self) -> "ResourceRelationCommit":
        resource_ids = [item.draft_resource_id for item in self.participants]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("同一资源不能在一个关系中重复出现")
        return self


class DatabaseResourcePatch(AIOpsContract):
    version_code: str | None = Field(default=None, min_length=1, max_length=64)
    environment: Literal["PROD", "STG", "DEV"] | None = None
    deployment_type: Literal["SINGLE_INSTANCE", "RAC"] | None = None
    rac_node_number: int | None = Field(default=None, ge=1, le=1024)
    db_role: Literal["PRIMARY", "STANDBY", "UNKNOWN"] | None = None


class ManagedResourcePatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    owner_group_id: UUIDv7 | None = None
    location_name: str | None = Field(default=None, max_length=256)
    lifecycle_status: ResourceLifecycleStatus | None = None
    attributes: JsonObject | None = None
    database: DatabaseResourcePatch | None = None


class BusinessSystemResourcePatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    architecture_layer_key: ArchitectureLayerKey | None = None
    responsibility_group_id: UUIDv7 | None = None
    display_order: int | None = Field(default=None, ge=0, le=1_000_000)


class RelationParticipantInput(AIOpsContract):
    system_resource_id: UUIDv7
    participant_role: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    display_order: int = Field(default=0, ge=0, le=1_000_000)


class ResourceRelationInputV2(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    relation_type_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    display_name: str | None = Field(default=None, max_length=256)
    responsibility_group_id: UUIDv7 | None = None
    origin: RelationOrigin = "MANUAL"
    confirmation_status: RelationConfirmationStatus = "CONFIRMED"
    attributes: JsonObject = Field(default_factory=dict)
    participants: tuple[RelationParticipantInput, ...] = Field(min_length=2, max_length=10_000)

    @model_validator(mode="after")
    def validate_participants(self) -> "ResourceRelationInputV2":
        ids = [item.system_resource_id for item in self.participants]
        if len(ids) != len(set(ids)):
            raise ValueError("同一资源不能在一个关系中重复出现")
        return self


class ResourceRelationPatchV2(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, max_length=256)
    responsibility_group_id: UUIDv7 | None = None
    confirmation_status: RelationConfirmationStatus | None = None
    attributes: JsonObject | None = None
    participants: tuple[RelationParticipantInput, ...] | None = Field(default=None, min_length=2, max_length=10_000)

    @model_validator(mode="after")
    def validate_participants(self) -> "ResourceRelationPatchV2":
        if self.participants is not None:
            ids = [item.system_resource_id for item in self.participants]
            if len(ids) != len(set(ids)):
                raise ValueError("同一资源不能在一个关系中重复出现")
        return self


class BusinessSystemResourceGraphCommit(AIOpsContract):
    """原子创建业务系统、资源、内部 Target、监控映射和多参与者关系。"""

    schema_version: str = PUBLIC_SCHEMA_VERSION
    system: BusinessSystemInfoInput
    resources: tuple[BusinessSystemResourceCommit, ...] = Field(min_length=1, max_length=10_000)
    relations: tuple[ResourceRelationCommit, ...] = Field(default=(), max_length=20_000)

    @model_validator(mode="after")
    def validate_references(self) -> "BusinessSystemResourceGraphCommit":
        draft_ids = [item.draft_resource_id for item in self.resources]
        if len(draft_ids) != len(set(draft_ids)):
            raise ValueError("草稿资源引用不能重复")
        relation_ids = [item.draft_relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("草稿关系引用不能重复")
        known = set(draft_ids)
        if any(participant.draft_resource_id not in known for relation in self.relations for participant in relation.participants):
            raise ValueError("资源关系引用了不在当前业务系统草稿中的资源")
        return self


class BusinessSystemDraftCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    mode: DraftMode = "CREATE"
    business_system_id: UUIDv7 | None = None
    current_step: DraftStep = "BUSINESS_INFO"
    payload: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mode(self) -> "BusinessSystemDraftCreate":
        if (self.mode == "EDIT") != (self.business_system_id is not None):
            raise ValueError("编辑草稿必须且只能指定业务系统")
        return self


class BusinessSystemDraftSave(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    revision: int = Field(ge=1)
    current_step: DraftStep
    payload: JsonObject = Field(default_factory=dict)


class BusinessSystemDraftView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    draft_id: UUIDv7
    mode: DraftMode
    business_system_id: UUIDv7 | None = None
    current_step: DraftStep
    revision: int = Field(ge=1)
    status: DraftStatus
    payload: JsonObject = Field(default_factory=dict)
    expires_at: UtcDatetime
    created_at: UtcDatetime
    updated_at: UtcDatetime


class BusinessSystemDraftSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    draft_id: UUIDv7
    mode: DraftMode
    business_system_id: UUIDv7 | None = None
    current_step: DraftStep
    revision: int = Field(ge=1)
    status: DraftStatus
    display_name: str
    environment: str | None = None
    importance_level: int | None = None
    responsibility_group_id: UUIDv7 | None = None
    resource_count: int = Field(default=0, ge=0)
    updated_at: UtcDatetime


class BusinessSystemDraftPage(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[BusinessSystemDraftSummary, ...] = ()
    total: int = Field(ge=0)


class ResourceDiscoveryRequest(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    connector_type: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    credential_id: UUIDv7 | None = None
    request: JsonObject = Field(default_factory=dict)


class ResourceDiscoveryJobView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    discovery_job_id: UUIDv7
    draft_id: UUIDv7
    connector_type: str
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "PARTIAL", "FAILED"]
    result: JsonObject | None = None
    error_code: str | None = None
    started_at: UtcDatetime | None = None
    completed_at: UtcDatetime | None = None
    created_at: UtcDatetime


class DatabaseResourceView(AIOpsContract):
    db_type: DatabaseType
    version_code: str
    environment: str
    deployment_type: Literal["SINGLE_INSTANCE", "RAC"]
    rac_node_number: int | None = Field(default=None, ge=1, le=1024)
    db_role: str
    connectivity_status: str
    monitoring_ready: bool


class ManagedResourceView(AIOpsContract):
    resource_id: UUIDv7
    resource_code: str | None = None
    resource_type_key: str
    display_name: str
    identity_kind: ResourceIdentityKind
    lifecycle_status: ResourceLifecycleStatus
    owner_group_id: UUIDv7
    location_name: str | None = None
    row_version: int = Field(ge=1)
    database: DatabaseResourceView | None = None


class BusinessSystemResourceView(AIOpsContract):
    system_resource_id: UUIDv7
    business_system_id: UUIDv7
    architecture_layer_key: ArchitectureLayerKey
    responsibility_group_id: UUIDv7 | None = None
    effective_responsibility_group_id: UUIDv7
    display_order: int = Field(ge=0)
    status: SystemResourceStatus
    row_version: int = Field(ge=1)
    resource: ManagedResourceView


class RelationParticipantView(AIOpsContract):
    participant_id: UUIDv7
    system_resource_id: UUIDv7
    resource_id: UUIDv7
    participant_role: str
    display_order: int = Field(ge=0)


class ResourceRelationViewV2(AIOpsContract):
    relation_id: UUIDv7
    business_system_id: UUIDv7
    relation_type_key: str
    display_name: str | None = None
    responsibility_group_id: UUIDv7 | None = None
    origin: RelationOrigin
    confirmation_status: RelationConfirmationStatus
    verification_status: RelationVerificationStatus
    attributes: JsonObject = Field(default_factory=dict)
    participants: tuple[RelationParticipantView, ...]
    row_version: int = Field(ge=1)


class ResourceGraphValidationIssue(AIOpsContract):
    severity: Literal["BLOCKING", "WARNING"]
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1000)
    resource_id: UUIDv7 | None = None
    relation_id: UUIDv7 | None = None


class BusinessSystemResourceGraphDetail(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    business_system_id: UUIDv7
    system_code: str
    display_name: str
    environment: str
    importance_level: int = Field(ge=1, le=5)
    responsibility_group_id: UUIDv7
    description: str | None = None
    status: Literal["DRAFT", "ENABLED", "DISABLED"]
    protection_status: Literal["UNCONFIGURED", "PARTIAL", "CONFIGURED"]
    resource_count: int = Field(ge=0)
    resource_type_summary: tuple[ResourceTypeSummary, ...] = ()
    architecture_labels: tuple[str, ...] = ()
    resources: tuple[BusinessSystemResourceView, ...] = ()
    relations: tuple[ResourceRelationViewV2, ...] = ()
    validation_issues: tuple[ResourceGraphValidationIssue, ...] = ()
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class BusinessSystemResourceGraphSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    business_system_id: UUIDv7
    system_code: str
    display_name: str
    environment: str
    importance_level: int = Field(ge=1, le=5)
    responsibility_group_id: UUIDv7
    status: Literal["DRAFT", "ENABLED", "DISABLED"]
    protection_status: Literal["UNCONFIGURED", "PARTIAL", "CONFIGURED"]
    resource_count: int = Field(ge=0)
    resource_type_summary: tuple[ResourceTypeSummary, ...] = ()
    architecture_labels: tuple[str, ...] = ()
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class BusinessSystemResourceGraphPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[BusinessSystemResourceGraphSummary, ...] = ()


class TopologyResourceNode(AIOpsContract):
    system_resource_id: UUIDv7
    resource_id: UUIDv7
    resource_code: str | None = None
    display_name: str
    resource_type_key: str
    architecture_layer_key: ArchitectureLayerKey
    lifecycle_status: ResourceLifecycleStatus


class TopologyRelationNode(AIOpsContract):
    relation_id: UUIDv7
    relation_type_key: str
    display_name: str | None = None
    verification_status: RelationVerificationStatus


class TopologyParticipantEdge(AIOpsContract):
    participant_id: UUIDv7
    relation_id: UUIDv7
    system_resource_id: UUIDv7
    participant_role: str


class TopologySubgraph(AIOpsContract):
    business_system_id: UUIDv7
    resources: tuple[TopologyResourceNode, ...] = ()
    relations: tuple[TopologyRelationNode, ...] = ()
    participants: tuple[TopologyParticipantEdge, ...] = ()
    truncated: bool = False
    next_cursor: str | None = None
