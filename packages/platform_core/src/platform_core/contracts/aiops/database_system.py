"""数据库系统、拓扑与保护边界的共享 Wire 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .configuration import (
    SourceBindingCreate,
    SupportedDatabaseVersion,
    TargetCreate,
    validate_supported_database_version,
)
from .types import AIOpsContract, CursorPage, DatabaseType, PUBLIC_SCHEMA_VERSION, UUIDv7, UtcDatetime


DeploymentShape = Literal["SINGLE_INSTANCE", "HIGH_AVAILABILITY"]
DeploymentCategory = Literal["SINGLE_INSTANCE", "RAC", "REPLICATION", "COMPOSITE"]
DatabaseSystemCapability = Literal[
    "RAC", "DATA_GUARD", "MYSQL_REPLICATION", "POSTGRES_STREAMING"
]
DatabaseSystemStatus = Literal["DRAFT", "ENABLED", "DISABLED"]
ProtectionStatus = Literal["UNCONFIGURED", "CONFIGURED"]
TopologyRole = Literal[
    "INSTANCE", "PRIMARY", "STANDBY", "SOURCE", "REPLICA", "RAC_INSTANCE"
]
ObservedTopologyRole = Literal[
    "UNKNOWN", "PRIMARY", "STANDBY", "SOURCE", "REPLICA", "RAC_INSTANCE"
]
RelationType = Literal["DATA_GUARD", "MYSQL_REPLICATION", "POSTGRES_STREAMING"]
TransportMode = Literal["SYNC", "ASYNC", "FAST_SYNC", "UNKNOWN"]


_ENGINE_CAPABILITIES: dict[DatabaseType, frozenset[str]] = {
    DatabaseType.ORACLE: frozenset({"RAC", "DATA_GUARD"}),
    DatabaseType.MYSQL: frozenset({"MYSQL_REPLICATION"}),
    DatabaseType.POSTGRESQL: frozenset({"POSTGRES_STREAMING"}),
}


def validate_database_system_architecture(
    *,
    db_type: DatabaseType | str,
    deployment_shape: str,
    deployment_category: str,
    capabilities: tuple[str, ...],
) -> None:
    database_type = DatabaseType(db_type)
    capability_set = set(capabilities)
    if len(capability_set) != len(capabilities):
        raise ValueError("数据库系统能力不能重复")
    if capability_set - _ENGINE_CAPABILITIES[database_type]:
        raise ValueError("所选架构能力与数据库引擎不兼容")
    if deployment_shape == "SINGLE_INSTANCE":
        if deployment_category != "SINGLE_INSTANCE" or capability_set:
            raise ValueError("单实例必须使用 SINGLE_INSTANCE 且不能配置集群能力")
        return
    if deployment_category == "SINGLE_INSTANCE":
        raise ValueError("高可用系统不能使用 SINGLE_INSTANCE 类别")
    expected = {
        (DatabaseType.ORACLE, "RAC"): {"RAC"},
        (DatabaseType.ORACLE, "REPLICATION"): {"DATA_GUARD"},
        (DatabaseType.ORACLE, "COMPOSITE"): {"RAC", "DATA_GUARD"},
        (DatabaseType.MYSQL, "REPLICATION"): {"MYSQL_REPLICATION"},
        (DatabaseType.POSTGRESQL, "REPLICATION"): {"POSTGRES_STREAMING"},
    }.get((database_type, deployment_category))
    if expected is None or capability_set != expected:
        raise ValueError("部署类别、数据库引擎与架构能力不匹配")


class DatabaseSystemCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    db_type: DatabaseType
    version_code: SupportedDatabaseVersion
    deployment_shape: DeploymentShape
    deployment_category: DeploymentCategory
    capabilities: tuple[DatabaseSystemCapability, ...] = Field(default=(), max_length=8)
    environment: Literal["PROD", "STG", "DEV"]
    importance_level: int = Field(default=3, ge=1, le=5)
    responsibility_group_id: UUIDv7 | None = None
    description: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_architecture(self) -> "DatabaseSystemCreate":
        validate_supported_database_version(self.db_type, self.version_code)
        validate_database_system_architecture(
            db_type=self.db_type,
            deployment_shape=self.deployment_shape,
            deployment_category=self.deployment_category,
            capabilities=self.capabilities,
        )
        return self


class DatabaseSystemPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    environment: Literal["PROD", "STG", "DEV"] | None = None
    importance_level: int | None = Field(default=None, ge=1, le=5)
    responsibility_group_id: UUIDv7 | None = None
    description: str | None = Field(default=None, max_length=1000)


class DatabaseSystemSiteInput(AIOpsContract):
    site_code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    display_name: str = Field(min_length=1, max_length=256)
    expected_role: Literal["PRIMARY", "STANDBY", "ACTIVE", "UNKNOWN"] = "UNKNOWN"
    sort_order: int = Field(default=0, ge=0, le=10000)


class DatabaseSystemMemberInput(AIOpsContract):
    member_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    target_id: UUIDv7
    site_code: str = Field(min_length=1, max_length=64)
    group_key: str | None = Field(default=None, max_length=64)
    member_type: Literal["INSTANCE", "RAC_INSTANCE"]
    expected_role: TopologyRole
    sort_order: int = Field(default=0, ge=0, le=10000)


class DatabaseSystemRelationInput(AIOpsContract):
    relation_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    source_member_key: str = Field(min_length=1, max_length=64)
    target_member_key: str = Field(min_length=1, max_length=64)
    relation_type: RelationType
    transport_mode: TransportMode = "UNKNOWN"
    sort_order: int = Field(default=0, ge=0, le=10000)


class DatabaseSystemTopologyPut(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    sites: tuple[DatabaseSystemSiteInput, ...] = Field(min_length=1, max_length=32)
    members: tuple[DatabaseSystemMemberInput, ...] = Field(min_length=1, max_length=128)
    relations: tuple[DatabaseSystemRelationInput, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_references(self) -> "DatabaseSystemTopologyPut":
        site_codes = [item.site_code for item in self.sites]
        member_keys = [item.member_key for item in self.members]
        target_ids = [item.target_id for item in self.members]
        relation_keys = [item.relation_key for item in self.relations]
        if len(site_codes) != len(set(site_codes)):
            raise ValueError("站点代码不能重复")
        if len(member_keys) != len(set(member_keys)):
            raise ValueError("成员标识不能重复")
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("同一个 Target 不能重复成为多个拓扑成员")
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("关系标识不能重复")
        known_sites = set(site_codes)
        known_members = set(member_keys)
        if any(item.site_code not in known_sites for item in self.members):
            raise ValueError("成员引用了不存在的站点")
        for item in self.relations:
            if item.source_member_key not in known_members or item.target_member_key not in known_members:
                raise ValueError("复制关系引用了不存在的成员")
            if item.source_member_key == item.target_member_key:
                raise ValueError("复制关系的起点和终点不能相同")
        return self


class DatabaseSystemMonitoringInput(AIOpsContract):
    """Wizard 成员的监控映射；发现型来源只接受有时效的候选引用。"""
    source_id: UUIDv7
    candidate_ref: str | None = Field(default=None, min_length=1, max_length=8192)
    host_candidate_ref: str | None = Field(default=None, min_length=1, max_length=8192)
    source_binding: SourceBindingCreate | None = None

    @model_validator(mode="after")
    def validate_mapping_mode(self) -> "DatabaseSystemMonitoringInput":
        if (self.candidate_ref is None) == (self.source_binding is None):
            raise ValueError("监控映射必须且只能使用发现候选或稳定定位配置之一")
        if self.source_binding is not None and self.source_binding.source_id != self.source_id:
            raise ValueError("监控映射的 Source ID 不一致")
        if self.candidate_ref is None and self.host_candidate_ref is not None:
            raise ValueError("主机候选必须与数据库候选一起提交")
        return self


class DatabaseSystemMemberCommit(AIOpsContract):
    """原子提交中的成员草稿，可新建 Target 或复用未归属 Target。"""

    member_key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    site_code: str = Field(min_length=1, max_length=64)
    group_key: str | None = Field(default=None, max_length=64)
    member_type: Literal["INSTANCE", "RAC_INSTANCE"]
    expected_role: TopologyRole
    sort_order: int = Field(default=0, ge=0, le=10000)
    mode: Literal["NEW", "REUSE"]
    target_id: UUIDv7 | None = None
    target: TargetCreate | None = None
    monitoring: DatabaseSystemMonitoringInput | None = None

    @model_validator(mode="after")
    def validate_target_mode(self) -> "DatabaseSystemMemberCommit":
        if self.mode == "NEW" and (self.target is None or self.target_id is not None):
            raise ValueError("新建成员必须提交 Target 配置且不能提交 Target ID")
        if self.mode == "REUSE" and (self.target_id is None or self.target is not None):
            raise ValueError("复用成员必须提交 Target ID 且不能提交新 Target 配置")
        return self


class DatabaseSystemCommit(AIOpsContract):
    """将系统、成员 Target、监控映射和拓扑作为一个事务提交。"""

    schema_version: str = PUBLIC_SCHEMA_VERSION
    system: DatabaseSystemCreate
    sites: tuple[DatabaseSystemSiteInput, ...] = Field(min_length=1, max_length=32)
    members: tuple[DatabaseSystemMemberCommit, ...] = Field(min_length=1, max_length=128)
    relations: tuple[DatabaseSystemRelationInput, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_commit(self) -> "DatabaseSystemCommit":
        site_codes = [item.site_code for item in self.sites]
        member_keys = [item.member_key for item in self.members]
        relation_keys = [item.relation_key for item in self.relations]
        reused_target_ids = [item.target_id for item in self.members if item.target_id is not None]
        if len(site_codes) != len(set(site_codes)): raise ValueError("站点代码不能重复")
        if len(member_keys) != len(set(member_keys)): raise ValueError("成员标识不能重复")
        if len(relation_keys) != len(set(relation_keys)): raise ValueError("关系标识不能重复")
        if len(reused_target_ids) != len(set(reused_target_ids)): raise ValueError("同一个 Target 不能重复成为多个拓扑成员")
        known_sites, known_members = set(site_codes), set(member_keys)
        if any(item.site_code not in known_sites for item in self.members): raise ValueError("成员引用了不存在的站点")
        for relation in self.relations:
            if relation.source_member_key not in known_members or relation.target_member_key not in known_members: raise ValueError("复制关系引用了不存在的成员")
            if relation.source_member_key == relation.target_member_key: raise ValueError("复制关系的起点和终点不能相同")
        for item in self.members:
            if item.target is not None and (item.target.db_type != self.system.db_type or item.target.version_code != self.system.version_code):
                raise ValueError(f"成员 {item.member_key} 的数据库引擎或版本与系统不一致")
        return self


class DatabaseSystemSiteView(DatabaseSystemSiteInput):
    site_id: UUIDv7


class DatabaseSystemMemberView(DatabaseSystemMemberInput):
    member_id: UUIDv7
    target_display_name: str
    observed_role: ObservedTopologyRole = "UNKNOWN"
    connectivity_status: str
    monitoring_ready: bool


class DatabaseSystemRelationView(DatabaseSystemRelationInput):
    relation_id: UUIDv7


class TopologyValidationIssue(AIOpsContract):
    severity: Literal["BLOCKING", "WARNING"]
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1000)
    member_key: str | None = None


class DatabaseSystemSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    database_system_id: UUIDv7
    display_name: str
    db_type: DatabaseType
    version_code: str
    deployment_shape: DeploymentShape
    deployment_category: DeploymentCategory
    capabilities: tuple[DatabaseSystemCapability, ...] = ()
    environment: str
    importance_level: int = Field(ge=1, le=5)
    responsibility_group_id: UUIDv7 | None = None
    status: DatabaseSystemStatus
    protection_status: ProtectionStatus
    member_count: int = Field(ge=0)
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class DatabaseSystemDetail(DatabaseSystemSummary):
    description: str | None = None
    sites: tuple[DatabaseSystemSiteView, ...] = ()
    members: tuple[DatabaseSystemMemberView, ...] = ()
    relations: tuple[DatabaseSystemRelationView, ...] = ()
    validation_issues: tuple[TopologyValidationIssue, ...] = ()
    created_at: UtcDatetime


class DatabaseSystemPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[DatabaseSystemSummary, ...] = ()
