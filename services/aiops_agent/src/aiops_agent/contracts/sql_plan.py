"""已有执行计划的安全投影；声明身份和运行时统计不能自行升级来源信任。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    origin: Literal["USER_UPLOAD"] = "USER_UPLOAD"
    native_format: Literal["POSTGRESQL_EXPLAIN_JSON", "MYSQL_EXPLAIN_JSON", "ORACLE_PLAN_ROWS"]
    subject: dict[str, str | bool | None] = Field(default_factory=dict, max_length=8)
    physical_identity: dict[str, str] = Field(default_factory=dict, max_length=8)
    declared_captured_at: str | None = Field(default=None, max_length=64)
    parameter_types: tuple[Literal["NULL", "BOOLEAN", "NUMBER", "STRING", "ARRAY", "OBJECT"], ...] = Field(default=(), max_length=64)
    parameter_values_available: Literal[False] = False
    runtime_statistics: Literal["NONE", "PARTIAL", "PRESENT"] = "NONE"
    nodes: tuple[SqlPlanNode, ...] = Field(min_length=1, max_length=128)
    warnings: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_tree(self):
        ids = {node.node_id for node in self.nodes}
        if ids != set(range(len(self.nodes))):
            raise ValueError("计划节点必须使用连续且唯一的投影编号")
        for node in self.nodes:
            if node.parent_id is not None and node.parent_id >= node.node_id:
                raise ValueError("计划父节点必须先于子节点，不能循环或悬空")
        expected = {"POSTGRESQL": "POSTGRESQL_EXPLAIN_JSON", "MYSQL": "MYSQL_EXPLAIN_JSON", "ORACLE": "ORACLE_PLAN_ROWS"}
        if self.native_format != expected[self.database_type]:
            raise ValueError("计划原生格式与引擎不一致")
        return self


class SqlPlanAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str
    anchor_ref: str | None = None
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    subject: dict[str, str | int | bool | None] = Field(max_length=16)
    identity_match: Literal["DECLARED_MATCH", "MISMATCH", "INCOMPLETE"]
    physical_match: Literal["DECLARED_MATCH", "MISMATCH", "INCOMPLETE"]
    time_relation: Literal["DECLARED_NEAR_OBSERVATION", "HISTORICAL_OR_DIFFERENT_TIME", "UNKNOWN"]
    runtime_statistics: Literal["NONE", "PARTIAL", "PRESENT"]
    representative_parameters: Literal["UNVERIFIED"] = "UNVERIFIED"
    production_execution_verified: Literal[False] = False
    node_count: int = Field(ge=1, le=128)
    limitations: tuple[str, ...]
