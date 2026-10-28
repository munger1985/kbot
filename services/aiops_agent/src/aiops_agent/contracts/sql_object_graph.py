"""语句语法、计划对象和目录关系分开保存，不把名称关联解释为运行时访问。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.contracts.sql_diagnosis import EvidenceFieldReference


class SqlObjectRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SQL_TEXT_REFERENCES_OBJECT", "PLAN_ROW_REFERENCES_OBJECT", "CATALOG_INDEX_ON",
                  "CATALOG_DESCENDANT_OF", "CATALOG_VIEW_DEPENDS_ON", "CATALOG_SYNONYM_REFERENCES",
                  "CATALOG_NAME_MATCH"]
    anchor: EvidenceFieldReference
    related: EvidenceFieldReference
    schema_name: str = Field(min_length=1, max_length=256)
    relation_name: str = Field(min_length=1, max_length=256)
    limitation: str = Field(min_length=1, max_length=1000)


class SqlObjectGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["SQL_OBJECT_GRAPH.v1"] = "SQL_OBJECT_GRAPH.v1"
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    subject: dict[str, str | int | bool | None] = Field(min_length=1, max_length=16)
    relationships: tuple[SqlObjectRelationship, ...] = Field(default=(), max_length=128)
    truncated: bool = False
    limitations: tuple[str, ...] = ()
