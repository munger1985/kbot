"""语句语法、计划对象和目录关系分开保存，不把名称关联解释为运行时访问。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer

from aiops_agent.contracts.sql_diagnosis import EvidenceFieldReference


class SqlObjectCatalogCandidate(BaseModel):
    """可见目录中的同名候选，不代表语句解析或运行时访问已确认。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: str = Field(min_length=1, max_length=256)
    relation_name: str = Field(min_length=1, max_length=256)
    catalog_ref: EvidenceFieldReference


class SqlObjectCatalogLookup(BaseModel):
    """供调查选择的版本化目录补证请求；本身不是执行授权。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: Literal["db.sql.object_candidates"] = "db.sql.object_candidates"
    tool_version: Literal["1.0.0"] = "1.0.0"
    input: dict[str, str | int] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_input(self):
        if (set(self.input) != {"table_name", "schema_name", "limit"} or self.input.get("limit") != 40
                or not isinstance(self.input.get("table_name"), str) or not 0 < len(self.input["table_name"]) <= 256
                or not isinstance(self.input.get("schema_name"), str) or len(self.input["schema_name"]) > 256):
            raise ValueError("对象目录补证请求必须保留精确名称及固定枚举预算")
        return self


class SqlObjectReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor: EvidenceFieldReference
    relation_name: str = Field(min_length=1, max_length=256)
    schema_name: str | None = Field(default=None, min_length=1, max_length=256)
    resolution: Literal["EXPLICIT_SCHEMA", "NATIVE_DEFAULT_SCHEMA", "CONTEXT_REQUIRED"]
    context_ref: EvidenceFieldReference | None = None
    candidates: tuple[SqlObjectCatalogCandidate, ...] = Field(default=(), max_length=16)
    lookup: SqlObjectCatalogLookup | None = None
    limitation: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_resolution(self):
        if ((self.resolution == "CONTEXT_REQUIRED") != (self.schema_name is None)
                or (self.resolution == "NATIVE_DEFAULT_SCHEMA") != (self.context_ref is not None)):
            raise ValueError("对象名解析状态、Schema 与原生上下文引用不一致")
        if self.context_ref is not None and (
                self.context_ref.evidence_ref != self.anchor.evidence_ref
                or self.context_ref.row_index != self.anchor.row_index or self.context_ref.field != "schema_name"):
            raise ValueError("默认数据库只能引用同一语句统计行的原生上下文")
        if any(item.relation_name != self.relation_name or self.schema_name is not None
               and item.schema_name != self.schema_name for item in self.candidates):
            raise ValueError("目录候选名称与语句名称不一致")
        if self.lookup is not None and self.lookup.input != {
                "table_name": self.relation_name, "schema_name": self.schema_name or "", "limit": 40}:
            raise ValueError("目录补证请求不得改变原语句名称或上下文")
        return self

    @property
    def evidence_refs(self):
        return (self.anchor, *((self.context_ref,) if self.context_ref is not None else ()),
                *(item.catalog_ref for item in self.candidates))


class SqlObjectRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SQL_TEXT_REFERENCES_OBJECT", "SQL_TEXT_DEFAULT_SCHEMA_CANDIDATE", "PLAN_ROW_REFERENCES_OBJECT", "CATALOG_INDEX_ON",
                  "CATALOG_DESCENDANT_OF", "CATALOG_VIEW_DEPENDS_ON", "CATALOG_SYNONYM_REFERENCES",
                  "CATALOG_NAME_MATCH"]
    anchor: EvidenceFieldReference
    related: EvidenceFieldReference
    schema_name: str = Field(min_length=1, max_length=256)
    relation_name: str = Field(min_length=1, max_length=256)
    limitation: str = Field(min_length=1, max_length=1000)


class SqlPlanObjectReference(BaseModel):
    """计划中解析到的对象；不是依靠语法猜测，也不证明某次生产实际访问。"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    node_id: int = Field(ge=0, le=127)
    source: Literal["CONTROLLED_ESTIMATE_OBJECT", "DATABASE_CURSOR_OBJECT"]
    object_kind: Literal["TABLE", "INDEX"]
    schema_name: str = Field(min_length=1, max_length=128)
    object_name: str = Field(min_length=1, max_length=128)
    native_object_id: str | None = None
    field_ref: EvidenceFieldReference
    subject_anchor_ref: str
    production_execution_verified: Literal[False] = False


class SqlObjectGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["SQL_OBJECT_GRAPH.v1"] = "SQL_OBJECT_GRAPH.v1"
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    subject: dict[str, str | int | bool | None] = Field(min_length=1, max_length=16)
    references: tuple[SqlObjectReference, ...] = Field(default=(), max_length=64)
    relationships: tuple[SqlObjectRelationship, ...] = Field(default=(), max_length=128)
    plan_objects: tuple[SqlPlanObjectReference, ...] = Field(default=(), max_length=128)
    truncated: bool = False
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_engine_context(self):
        if self.database_type != "MYSQL" and any(item.resolution == "NATIVE_DEFAULT_SCHEMA" for item in self.references):
            raise ValueError("只有 MySQL Digest 的原生默认数据库可提供此名称上下文")
        return self

    @model_serializer(mode="wrap")
    def preserve_historical_graph(self, handler):
        value = handler(self)
        if not self.plan_objects:
            value.pop("plan_objects", None)
        return value
