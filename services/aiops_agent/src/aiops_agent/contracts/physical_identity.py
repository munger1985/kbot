"""监控原生身份见证与跨源比较合同；来源自报不等于物理认证。"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from platform_core.contracts.aiops.types import UtcDatetime


class SourceIdentityWitness(BaseModel):
    """只定位身份信息序列，不把同抓取端点的其他指标自动归属到该数据库。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_ref: str = Field(min_length=1, max_length=1024)
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    target_id: UUID
    source_id: UUID
    binding_id: UUID
    source_version: int = Field(ge=1)
    collector_scope: tuple[tuple[str, str], ...] = Field(max_length=16)
    native_identity: tuple[tuple[str, str], ...] = Field(max_length=8)
    sample_times: tuple[UtcDatetime, ...] = Field(max_length=120)
    quality_flags: tuple[str, ...] = Field(default=(), max_length=16)
    protocol_version: Literal["DATABASE_IDENTITY_INFO.v1"] = "DATABASE_IDENTITY_INFO.v1"

    @model_validator(mode="after")
    def validate_scope(self):
        for pairs in (self.collector_scope, self.native_identity):
            if (len(dict(pairs)) != len(pairs) or tuple(sorted(pairs)) != pairs
                or any(not text or len(text) > 256 or any(ord(char) < 32 for char in text)
                       for pair in pairs for text in pair)):
                raise ValueError("身份见证字段必须有界、唯一并按键排序")
        if tuple(sorted(set(self.sample_times))) != self.sample_times:
            raise ValueError("身份样本时点必须去重并按时间排序")
        return self


class PhysicalIdentityNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_ref: str = Field(min_length=1, max_length=512)
    input_ref: str = Field(min_length=1, max_length=1024)
    target_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=128)
    source_version: str = Field(min_length=1, max_length=4096)
    binding_id: str | None = Field(default=None, max_length=128)
    origin: Literal["DIRECT_DATABASE", "MONITORING_REPORT"]
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    native_identity: tuple[tuple[str, str], ...] = Field(max_length=8)
    observed_times: tuple[UtcDatetime, ...] = Field(max_length=120)
    collector_scope: tuple[tuple[str, str], ...] = Field(default=(), max_length=16)
    quality_flags: tuple[str, ...] = Field(default=(), max_length=16)


class PhysicalIdentityEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_node: str = Field(pattern=r"^[a-f0-9]{64}$")
    monitoring_node: str = Field(pattern=r"^[a-f0-9]{64}$")
    relation: Literal["REPORTED_DATABASE_FAMILY_MATCH", "REPORTED_SERVER_MATCH", "CONFLICT", "UNKNOWN"]
    matched_fields: tuple[str, ...] = Field(default=(), max_length=8)
    conflicting_fields: tuple[str, ...] = Field(default=(), max_length=8)
    reasons: tuple[str, ...] = Field(min_length=1, max_length=8)
    nearest_sample_seconds: float | None = Field(default=None, ge=0)
    exact_runtime_instance_verified: Literal[False] = False
    metric_series_assignment_verified: Literal[False] = False
    window_continuity_verified: Literal[False] = False
    independent_corroboration: Literal[False] = False


class PhysicalIdentityGraph(BaseModel):
    """有界来源比较，不为历史连续性、SQL归因或独立互证背书。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["PHYSICAL_IDENTITY_GRAPH.v1"] = "PHYSICAL_IDENTITY_GRAPH.v1"
    nodes: tuple[PhysicalIdentityNode, ...] = Field(max_length=64)
    edges: tuple[PhysicalIdentityEdge, ...] = Field(max_length=128)
    quality_flags: tuple[str, ...] = Field(default=(), max_length=16)
    proximity_seconds: Literal[120] = 120

    @model_validator(mode="after")
    def validate_edges(self):
        nodes = {node.node_id: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError("物理身份图节点不能重复")
        seen = set()
        for edge in self.edges:
            left, right = nodes.get(edge.database_node), nodes.get(edge.monitoring_node)
            key = (edge.database_node, edge.monitoring_node)
            if (left is None or right is None or key in seen or left.origin != "DIRECT_DATABASE"
                or right.origin != "MONITORING_REPORT" or left.target_id != right.target_id):
                raise ValueError("物理身份图的边必须关联同一目标的直连与监控原始观察")
            seen.add(key)
        return self
