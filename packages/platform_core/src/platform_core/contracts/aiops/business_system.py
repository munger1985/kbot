"""业务系统、可管理组件与资源拓扑的共享 Wire 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .configuration import SourceBindingCreate, TargetCreate
from .types import AIOpsContract, CursorPage, DatabaseType, PUBLIC_SCHEMA_VERSION, UUIDv7, UtcDatetime


BusinessSystemStatus = Literal["DRAFT", "ENABLED", "DISABLED"]
ComponentType = Literal["DATABASE", "APPLICATION", "MIDDLEWARE", "NETWORK"]
ComponentStatus = Literal["ACTIVE", "DISABLED"]
ProtectionStatus = Literal["UNCONFIGURED", "PARTIAL", "CONFIGURED"]
TopologyRole = Literal["INSTANCE", "PRIMARY", "STANDBY", "SOURCE", "REPLICA", "REPORTING"]
ObservedTopologyRole = Literal["UNKNOWN", "PRIMARY", "STANDBY", "SOURCE", "REPLICA", "REPORTING"]
RelationType = Literal[
    "DATA_GUARD",
    "OGG",
    "MYSQL_REPLICATION",
    "POSTGRES_STREAMING",
    "GENERIC_DEPENDENCY",
]
TransportMode = Literal["SYNC", "ASYNC", "FAST_SYNC", "UNKNOWN"]


class BusinessSystemCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    system_code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    display_name: str = Field(min_length=1, max_length=256)
    environment: Literal["PROD", "STG", "DEV"]
    importance_level: int = Field(default=3, ge=1, le=5)
    responsibility_group_id: UUIDv7
    description: str | None = Field(default=None, max_length=1000)


class BusinessSystemPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    system_code: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    environment: Literal["PROD", "STG", "DEV"] | None = None
    importance_level: int | None = Field(default=None, ge=1, le=5)
    responsibility_group_id: UUIDv7
    description: str | None = Field(default=None, max_length=1000)


class ManagedComponentInput(AIOpsContract):
    component_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    display_name: str = Field(min_length=1, max_length=256)
    component_type: ComponentType
    responsibility_group_id: UUIDv7
    description: str | None = Field(default=None, max_length=1000)
    sort_order: int = Field(default=0, ge=0, le=10000)


class BusinessSystemSiteInput(AIOpsContract):
    site_code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    display_name: str = Field(min_length=1, max_length=256)
    site_role: Literal["PRIMARY", "DR", "ACTIVE", "UNKNOWN"] = "UNKNOWN"
    sort_order: int = Field(default=0, ge=0, le=10000)


class ComponentMemberInput(AIOpsContract):
    member_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    component_key: str = Field(min_length=1, max_length=64)
    target_id: UUIDv7
    site_code: str = Field(min_length=1, max_length=64)
    cluster_group_key: str | None = Field(default=None, max_length=64)
    expected_role: TopologyRole
    sort_order: int = Field(default=0, ge=0, le=100000)


class ResourceRelationInput(AIOpsContract):
    relation_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    source_component_key: str = Field(min_length=1, max_length=64)
    target_component_key: str = Field(min_length=1, max_length=64)
    source_member_key: str | None = Field(default=None, min_length=1, max_length=64)
    target_member_key: str | None = Field(default=None, min_length=1, max_length=64)
    relation_type: RelationType
    transport_mode: TransportMode = "UNKNOWN"
    responsibility_group_id: UUIDv7 | None = None
    sort_order: int = Field(default=0, ge=0, le=100000)

    @model_validator(mode="after")
    def validate_endpoints(self) -> "ResourceRelationInput":
        if (self.source_member_key is None) != (self.target_member_key is None):
            raise ValueError("资源关系必须同时使用组件级端点或成员级端点")
        if (
            self.source_component_key == self.target_component_key
            and self.source_member_key is not None
            and self.source_member_key == self.target_member_key
        ):
            raise ValueError("资源关系的起点和终点不能相同")
        return self


def _validate_topology_references(
    *,
    components: tuple[ManagedComponentInput, ...],
    sites: tuple[BusinessSystemSiteInput, ...],
    members: tuple[ComponentMemberInput | "ComponentMemberCommit", ...],
    relations: tuple[ResourceRelationInput, ...],
) -> None:
    component_keys = [item.component_key for item in components]
    site_codes = [item.site_code for item in sites]
    member_keys = [item.member_key for item in members]
    relation_keys = [item.relation_key for item in relations]
    if len(component_keys) != len(set(component_keys)):
        raise ValueError("组件标识不能重复")
    if len(site_codes) != len(set(site_codes)):
        raise ValueError("站点代码不能重复")
    if len(member_keys) != len(set(member_keys)):
        raise ValueError("成员标识不能重复")
    if len(relation_keys) != len(set(relation_keys)):
        raise ValueError("关系标识不能重复")
    known_components = set(component_keys)
    known_sites = set(site_codes)
    known_members = set(member_keys)
    member_component_by_key = {item.member_key: item.component_key for item in members}
    component_type_by_key = {item.component_key: item.component_type for item in components}
    if any(item.component_key not in known_components for item in members):
        raise ValueError("成员引用了不存在的组件")
    if any(item.site_code not in known_sites for item in members):
        raise ValueError("成员引用了不存在的站点")
    if any(component_type_by_key[item.component_key] != "DATABASE" for item in members):
        raise ValueError("一期只有数据库组件可以配置 Target 成员")
    for relation in relations:
        if relation.source_component_key not in known_components or relation.target_component_key not in known_components:
            raise ValueError("资源关系引用了不存在的组件")
        if relation.source_member_key is not None and (
            relation.source_member_key not in known_members
            or relation.target_member_key not in known_members
        ):
            raise ValueError("资源关系引用了不存在的成员")
        if relation.source_member_key is None:
            if relation.source_component_key == relation.target_component_key:
                raise ValueError("组件级资源关系的起点和终点不能相同")
            continue
        if (
            member_component_by_key[relation.source_member_key]
            != relation.source_component_key
            or member_component_by_key[relation.target_member_key]
            != relation.target_component_key
        ):
            raise ValueError("资源关系成员必须属于对应的起点和终点组件")


class BusinessSystemTopologyPut(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    components: tuple[ManagedComponentInput, ...] = Field(min_length=1, max_length=512)
    sites: tuple[BusinessSystemSiteInput, ...] = Field(min_length=1, max_length=512)
    members: tuple[ComponentMemberCommit, ...] = Field(default=(), max_length=10000)
    relations: tuple[ResourceRelationInput, ...] = Field(default=(), max_length=20000)

    @model_validator(mode="after")
    def validate_references(self) -> "BusinessSystemTopologyPut":
        target_ids = [item.target_id for item in self.members if item.target_id is not None]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("同一个 Target 不能重复成为多个组件成员")
        _validate_topology_references(
            components=self.components,
            sites=self.sites,
            members=self.members,
            relations=self.relations,
        )
        return self


class ComponentMonitoringInput(AIOpsContract):
    """Wizard 成员的监控映射；发现型来源只接受有时效的候选引用。"""

    source_id: UUIDv7
    candidate_ref: str | None = Field(default=None, min_length=1, max_length=8192)
    host_candidate_ref: str | None = Field(default=None, min_length=1, max_length=8192)
    source_binding: SourceBindingCreate | None = None

    @model_validator(mode="after")
    def validate_mapping_mode(self) -> "ComponentMonitoringInput":
        if (self.candidate_ref is None) == (self.source_binding is None):
            raise ValueError("监控映射必须且只能使用发现候选或稳定定位配置之一")
        if self.source_binding is not None and self.source_binding.source_id != self.source_id:
            raise ValueError("监控映射的 Source ID 不一致")
        if self.candidate_ref is None and self.host_candidate_ref is not None:
            raise ValueError("主机候选必须与数据库候选一起提交")
        return self


class ComponentTargetPatch(AIOpsContract):
    """业务系统编辑中与拓扑一起原子修改的 Target 身份字段。"""

    display_name: str = Field(min_length=1, max_length=256)
    version_code: str = Field(min_length=1, max_length=64)
    environment: Literal["PROD", "STG", "DEV"]
    db_role: Literal["PRIMARY", "STANDBY", "UNKNOWN"] = "UNKNOWN"


class ComponentMemberCommit(AIOpsContract):
    member_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    component_key: str = Field(min_length=1, max_length=64)
    site_code: str = Field(min_length=1, max_length=64)
    cluster_group_key: str | None = Field(default=None, max_length=64)
    expected_role: TopologyRole
    sort_order: int = Field(default=0, ge=0, le=100000)
    mode: Literal["NEW", "REUSE"]
    target_id: UUIDv7 | None = None
    target: TargetCreate | None = None
    target_patch: ComponentTargetPatch | None = None
    target_row_version: int | None = Field(default=None, ge=1)
    monitoring: ComponentMonitoringInput | None = None

    @model_validator(mode="after")
    def validate_target_mode(self) -> "ComponentMemberCommit":
        if self.mode == "NEW" and (self.target is None or self.target_id is not None):
            raise ValueError("新建成员必须提交 Target 配置且不能提交 Target ID")
        if self.mode == "REUSE" and (self.target_id is None or self.target is not None):
            raise ValueError("复用成员必须提交 Target ID 且不能提交新 Target 配置")
        if self.mode == "NEW" and (self.target_patch is not None or self.target_row_version is not None):
            raise ValueError("新建成员不能提交 Target 修改版本")
        if self.target_patch is not None and self.target_row_version is None:
            raise ValueError("修改已有 Target 必须提交 Target 行版本")
        return self


BusinessSystemTopologyPut.model_rebuild()


class BusinessSystemCommit(AIOpsContract):
    """将业务系统、组件、Target、监控映射与关系作为一个事务提交。"""

    schema_version: str = PUBLIC_SCHEMA_VERSION
    system: BusinessSystemCreate
    components: tuple[ManagedComponentInput, ...] = Field(min_length=1, max_length=512)
    sites: tuple[BusinessSystemSiteInput, ...] = Field(min_length=1, max_length=512)
    members: tuple[ComponentMemberCommit, ...] = Field(default=(), max_length=10000)
    relations: tuple[ResourceRelationInput, ...] = Field(default=(), max_length=20000)

    @model_validator(mode="after")
    def validate_commit(self) -> "BusinessSystemCommit":
        reused_target_ids = [item.target_id for item in self.members if item.target_id is not None]
        if len(reused_target_ids) != len(set(reused_target_ids)):
            raise ValueError("同一个 Target 不能重复成为多个组件成员")
        _validate_topology_references(
            components=self.components,
            sites=self.sites,
            members=self.members,
            relations=self.relations,
        )
        return self


class BusinessSystemSiteView(BusinessSystemSiteInput):
    site_id: UUIDv7


class ManagedComponentView(ManagedComponentInput):
    component_id: UUIDv7
    status: ComponentStatus
    member_count: int = Field(ge=0)
    engine_types: tuple[DatabaseType, ...] = ()
    versions: tuple[str, ...] = ()
    architecture_labels: tuple[str, ...] = ()
    protection_status: ProtectionStatus = "UNCONFIGURED"


class ComponentMemberView(ComponentMemberInput):
    membership_id: UUIDv7
    component_id: UUIDv7
    target_display_name: str
    target_db_type: DatabaseType
    target_version_code: str
    target_environment: str
    target_row_version: int = Field(ge=1)
    observed_role: ObservedTopologyRole = "UNKNOWN"
    connectivity_status: str
    monitoring_ready: bool


class ResourceRelationView(ResourceRelationInput):
    relation_id: UUIDv7


class TopologyValidationIssue(AIOpsContract):
    severity: Literal["BLOCKING", "WARNING"]
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1000)
    component_key: str | None = None
    member_key: str | None = None


class BusinessSystemSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    business_system_id: UUIDv7
    system_code: str
    display_name: str
    environment: str
    importance_level: int = Field(ge=1, le=5)
    responsibility_group_id: UUIDv7 | None = None
    status: BusinessSystemStatus
    protection_status: ProtectionStatus
    component_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    architecture_labels: tuple[str, ...] = ()
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class BusinessSystemDetail(BusinessSystemSummary):
    description: str | None = None
    sites: tuple[BusinessSystemSiteView, ...] = ()
    components: tuple[ManagedComponentView, ...] = ()
    members: tuple[ComponentMemberView, ...] = ()
    relations: tuple[ResourceRelationView, ...] = ()
    validation_issues: tuple[TopologyValidationIssue, ...] = ()
    created_at: UtcDatetime


class BusinessSystemPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[BusinessSystemSummary, ...] = ()
