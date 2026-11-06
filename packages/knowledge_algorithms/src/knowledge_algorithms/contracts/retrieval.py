"""召回计划、候选读取、关联和证据集合的统一进程内合同。"""

from dataclasses import dataclass
import math

from .content import ContentKind, Provenance
from .errors import require
from .identity import EvidenceKey, Locator, RetrievalSnapshot, SourceSpan, nonempty
from .query import QueryInput, QueryMode
from ..projections.contracts import Representation, RepresentationSpec
from ..references.relations import LinkedManifest, ResolvedRelation
from ..retrieval.fusion import ChannelShard, FusedHit
from ..retrieval.selection import CandidateTargetMatch, CoverageTarget, PayloadCost, SelectionResult


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    channel_key: str
    signal_family: str
    kind: str
    direction: str
    weight: float
    limit: int
    timeout_seconds: float
    required: bool
    representation_spec: RepresentationSpec | None = None

    def __post_init__(self) -> None:
        nonempty(self.channel_key, self.signal_family, self.direction)
        require(self.kind in {"LEXICAL", "DENSE", "MULTIVECTOR", "NEAR_DUPLICATE"}, "召回通道类型无效")
        require(math.isfinite(self.weight) and self.weight > 0 and type(self.limit) is int and self.limit > 0,
                "通道权重或数量无效")
        require(math.isfinite(self.timeout_seconds) and self.timeout_seconds > 0, "通道超时无效")
        require((self.representation_spec is not None) == (self.kind in {"DENSE", "MULTIVECTOR"}), "表示空间与通道类型不符")
        if self.representation_spec is not None:
            require(self.direction in self.representation_spec.directions, "模型未验证当前检索方向")
            require(self.kind == self.representation_spec.kind, "表示输出类型与通道不符")
        require(type(self.required) is bool, "必需通道声明无效")


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    profile_version: str
    channels: tuple[ChannelSpec, ...]
    rrf_constant: float
    candidate_budget: int
    max_primary_count: int
    payload_budget: PayloadCost
    max_cover_states: int
    relation_page_size: int
    max_relation_pages: int
    max_representation_elements: int
    timeout_seconds: float
    max_shards: int = 64

    def __post_init__(self) -> None:
        nonempty(self.profile_version)
        require(all(type(v) is int and v > 0 for v in (
            self.candidate_budget, self.max_primary_count, self.max_cover_states, self.relation_page_size,
            self.max_relation_pages, self.max_representation_elements, self.max_shards)), "召回预算无效")
        require(self.max_primary_count <= self.candidate_budget, "主命中上限不能超过候选预算")
        require(len({c.channel_key for c in self.channels}) == len(self.channels), "通道登记重复")
        require(len({c.signal_family for c in self.channels}) == len(self.channels), "同信号族必须在登记时合为单通道")
        require(math.isfinite(self.rrf_constant) and self.rrf_constant >= 0, "RRF 常数无效")
        require(math.isfinite(self.timeout_seconds) and self.timeout_seconds > 0, "召回超时无效")


@dataclass(frozen=True, slots=True)
class ChannelRequest:
    query: QueryInput
    channel: ChannelSpec
    representation: Representation | None
    input_index: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    key: EvidenceKey
    source_keys: tuple[EvidenceKey, ...]
    kind: ContentKind
    locators: tuple[Locator, ...]
    provenance: Provenance
    allowed_view_ref: str
    content_handle: str
    content_hash: str
    mime_type: str
    byte_count: int
    estimated_cost: PayloadCost
    expires_at: int
    preview_handle: str | None = None
    payload_status: str = "DEFERRED"
    source_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        nonempty(self.allowed_view_ref, self.content_handle, self.content_hash, self.mime_type)
        require(bool(self.source_keys) and bool(self.locators), "证据必须有精确来源和定位")
        require(all(k.owner_binding_ref == self.key.owner_binding_ref for k in self.source_keys), "证据来源不能跨 App")
        require(all(k.document_version_ref == self.key.document_version_ref
                    and k.parse_run_ref == self.key.parse_run_ref and k.projection_ref is None
                    for k in self.source_keys), "证据来源版本、解析代次或规范身份无效")
        require(all(span.node_key in {k.object_key for k in self.source_keys} for span in self.source_spans),
                "片段跨度不属于来源对象")
        require(type(self.byte_count) is int and self.byte_count >= 0, "证据大小无效")
        require(type(self.expires_at) is int and self.expires_at > 0, "受控内容句柄缺少期限")
        require(self.payload_status in {"DEFERRED", "UNAVAILABLE"}, "检索清单不能声称已经材料化或发送")


@dataclass(frozen=True, slots=True)
class EvidenceBatch:
    items: tuple[EvidenceItem, ...]
    missing: tuple[tuple[EvidenceKey, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid_keys: tuple[EvidenceKey, ...]
    excluded: tuple[tuple[EvidenceKey, str], ...]


@dataclass(frozen=True, slots=True)
class RelationRequest:
    source_keys: tuple[EvidenceKey, ...]
    limit: int
    cursor: str | None


@dataclass(frozen=True, slots=True)
class RelationPage:
    relations: tuple[ResolvedRelation, ...]
    exhausted: bool
    continuation: str | None


@dataclass(frozen=True, slots=True)
class ChannelStatus:
    channel_key: str
    required: bool
    status: str
    reason: str | None
    approximate: bool
    exhausted: bool


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage: str
    input_keys: tuple[EvidenceKey, ...]
    output_keys: tuple[EvidenceKey, ...]
    excluded: tuple[tuple[EvidenceKey, str], ...]


@dataclass(frozen=True, slots=True)
class PrimaryHit:
    evidence_key: EvidenceKey
    rank: int
    group_key: str
    fusion_score: float | None


@dataclass(frozen=True, slots=True)
class QueryPlan:
    query_hash: str
    scope_hash: str
    profile_version: str
    mode: QueryMode
    channel_keys: tuple[str, ...]
    targets: tuple[CoverageTarget, ...]
    profile_hash: str
    query_representation_hashes: tuple[tuple[str, int | None, str], ...]


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    request_id: str
    owner_binding_ref: str
    scope_ref: str
    scope_hash: str
    snapshot: RetrievalSnapshot
    query_hash: str
    mode: QueryMode
    status: str
    reason: str | None
    primary_hits: tuple[PrimaryHit, ...]
    evidence: tuple[EvidenceItem, ...]
    groups: tuple[LinkedManifest, ...]
    context_selection: SelectionResult
    matches: tuple[CandidateTargetMatch, ...]
    channel_status: tuple[ChannelStatus, ...]
    trace: tuple[StageRecord, ...]
    plan: QueryPlan
    continuation: str | None
    scope_exhausted: bool
    schema_version: str = "knowledge.evidence_set.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "knowledge.evidence_set.v1", "证据合同版本无效")
        require(self.status in {"COMPLETE", "PARTIAL", "EMPTY", "FAILED"}, "召回执行状态无效")
        require(self.snapshot.owner_binding_ref == self.owner_binding_ref
                and self.snapshot.scope_hash == self.scope_hash, "证据集合的范围或快照不一致")
        keys = {item.key for item in self.evidence}
        require(len(keys) == len(self.evidence) and all(key.owner_binding_ref == self.owner_binding_ref for key in keys),
                "证据集合身份重复或跨 App")
        require(all(hit.evidence_key in keys for hit in self.primary_hits), "主命中缺少对应证据")
        require(set(self.context_selection.selected_evidence_keys) <= keys, "上下文选择超出证据集合")
        require(self.plan.query_hash == self.query_hash and self.plan.scope_hash == self.scope_hash
                and self.plan.mode == self.mode, "冻结计划与证据集合不一致")
        require(all(g.anchor_key in keys and set(g.linked_keys) <= keys for g in self.groups), "图文组引用缺少证据")
