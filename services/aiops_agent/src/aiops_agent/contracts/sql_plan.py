"""已有执行计划的安全投影；声明身份和运行时统计不能自行升级来源信任。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer
from platform_core.contracts.aiops.sql_plan_verification import SqlPlanNativeVerification


class SqlPlanNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: int = Field(ge=0, le=127)
    parent_id: int | None = Field(default=None, ge=0, le=127)
    operation: str = Field(min_length=1, max_length=96)
    schema_name: str | None = Field(default=None, max_length=128)
    relation_name: str | None = Field(default=None, max_length=128)
    index_name: str | None = Field(default=None, max_length=128)
    object_name_semantics: Literal["CATALOG_NAME", "ALIAS_OR_NAME", "UNKNOWN"] = "UNKNOWN"
    metrics: dict[str, str] = Field(default_factory=dict, max_length=24)


class SqlPlanDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["SQL_PLAN_DOCUMENT.v1"] = "SQL_PLAN_DOCUMENT.v1"
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    origin: Literal["USER_UPLOAD", "DATABASE_CURSOR", "CONTROLLED_ESTIMATE"] = "USER_UPLOAD"
    native_format: Literal["POSTGRESQL_EXPLAIN_JSON", "MYSQL_EXPLAIN_JSON", "ORACLE_PLAN_ROWS"]
    subject: dict[str, str | bool | None] = Field(default_factory=dict, max_length=8)
    physical_identity: dict[str, str] = Field(default_factory=dict, max_length=8)
    declared_captured_at: str | None = Field(default=None, max_length=64)
    observed_at: str | None = Field(default=None, max_length=64)
    source_sql_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    native_verification: SqlPlanNativeVerification | None = None
    original_parse_context_verified: Literal[False] = False
    parameter_types: tuple[Literal["NULL", "BOOLEAN", "NUMBER", "STRING", "ARRAY", "OBJECT"], ...] = Field(default=(), max_length=64)
    parameter_values_available: Literal[False] = False
    runtime_statistics: Literal["NONE", "PARTIAL", "PRESENT"] = "NONE"
    nodes: tuple[SqlPlanNode, ...] = Field(min_length=1, max_length=128)
    warnings: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_tree(self):
        if self.native_verification is not None and self.origin != "CONTROLLED_ESTIMATE":
            raise ValueError("人工材料不能自报专用Executor原生核验")
        ids = {node.node_id for node in self.nodes}
        if self.native_verification is not None and (any(item.node_id not in ids for item in self.native_verification.resolved_objects)
            or any(index not in ids for index in self.native_verification.unresolved_object_nodes)):
            raise ValueError("原生核验对象不能引用不存在的计划节点")
        if ids != set(range(len(self.nodes))):
            raise ValueError("计划节点必须使用连续且唯一的投影编号")
        for node in self.nodes:
            if node.parent_id is not None and node.parent_id >= node.node_id:
                raise ValueError("计划父节点必须先于子节点，不能循环或悬空")
        expected = {"POSTGRESQL": "POSTGRESQL_EXPLAIN_JSON", "MYSQL": "MYSQL_EXPLAIN_JSON", "ORACLE": "ORACLE_PLAN_ROWS"}
        if self.native_format != expected[self.database_type]:
            raise ValueError("计划原生格式与引擎不一致")
        if self.origin == "DATABASE_CURSOR" and (
            self.database_type != "ORACLE" or self.declared_captured_at is not None
            or self.observed_at is None or self.parameter_types
        ):
            raise ValueError("缓存计划仅能绑定受控 Oracle 观测，不可携带人工声明时间或绑定原值")
        if self.origin == "USER_UPLOAD" and self.observed_at is not None:
            raise ValueError("人工计划不能自行声明数据库来源采集时间")
        if self.origin == "CONTROLLED_ESTIMATE":
            from platform_core.contracts.aiops.sql_plan_execution import SqlPlanEstimateNode
            if (self.observed_at is None or self.declared_captured_at is not None
                or self.source_sql_sha256 is None or self.parameter_types or self.runtime_statistics != "NONE"):
                raise ValueError("受控估算必须绑定原SQL哈希和实际采集时间，不能声明运行时统计或参数")
            for node in self.nodes:
                SqlPlanEstimateNode.model_validate(node.model_dump(exclude={"object_name_semantics"}))
        elif self.source_sql_sha256 is not None:
            raise ValueError("仅受控估算可以携带独立复核的原SQL哈希")
        return self

    @model_serializer(mode="wrap")
    def preserve_historical_document(self, handler):
        value = handler(self)
        if self.native_verification is None:
            value.pop("native_verification", None)
        return value


class SqlPlanAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str
    anchor_ref: str | None = None
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    subject: dict[str, str | int | bool | None] = Field(max_length=16)
    origin: Literal["USER_UPLOAD", "DATABASE_CURSOR", "CONTROLLED_ESTIMATE"] = "USER_UPLOAD"
    identity_match: Literal["DECLARED_MATCH", "SOURCE_MATCH", "MISMATCH", "INCOMPLETE"]
    physical_match: Literal["DECLARED_MATCH", "SOURCE_MATCH", "MISMATCH", "INCOMPLETE"]
    time_relation: Literal["DECLARED_NEAR_OBSERVATION", "SOURCE_NEAR_OBSERVATION", "HISTORICAL_OR_DIFFERENT_TIME", "UNKNOWN"]
    runtime_statistics: Literal["NONE", "PARTIAL", "PRESENT"]
    representative_parameters: Literal["UNVERIFIED"] = "UNVERIFIED"
    production_execution_verified: Literal[False] = False
    node_count: int = Field(ge=1, le=128)
    native_verification: SqlPlanNativeVerification | None = None
    bind_metadata_refs: tuple[str, ...] = Field(default=(), max_length=4)
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_origin(self):
        if self.native_verification is not None and self.origin != "CONTROLLED_ESTIMATE":
            raise ValueError("只有独立回执重放的估算计划可携带原生核验")
        if self.origin == "USER_UPLOAD" and (
            "SOURCE_MATCH" in {self.identity_match, self.physical_match}
            or self.time_relation == "SOURCE_NEAR_OBSERVATION" or self.bind_metadata_refs
        ):
            raise ValueError("人工计划不能升级数据库来源或绑定捕获关系")
        if self.origin == "CONTROLLED_ESTIMATE" and (self.runtime_statistics != "NONE" or self.bind_metadata_refs):
            raise ValueError("估算计划不能升级为运行时计划或绑定捕获证明")
        return self

    @model_serializer(mode="wrap")
    def preserve_historical_assessment(self, handler):
        value = handler(self)
        if self.native_verification is None:
            value.pop("native_verification", None)
        return value
