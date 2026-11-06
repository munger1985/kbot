"""Runtime 消费和实际交付的冻结合同；不在算法核调用 Composer。"""

from dataclasses import dataclass

from .errors import ErrorCode, require
from .identity import AppBackendBinding, AuthorizedScope, EvidenceKey, Locator, nonempty
from .query import QueryInput
from .retrieval import EvidenceSet
from ..retrieval.selection import PayloadCost, SelectionResult


@dataclass(frozen=True, slots=True)
class DelegationClaims:
    owner_binding_ref: str
    audience: str
    subject_ref: str
    tenant_ref: str | None
    domain_ref: str
    agent_version_ref: str | None
    run_ref: str | None
    spec_hash: str | None
    scope_hash: str
    operations: tuple[str, ...]
    expires_at: int

    def __post_init__(self) -> None:
        nonempty(self.owner_binding_ref, self.audience, self.subject_ref, self.domain_ref, self.scope_hash,
                 *self.operations)
        require(bool(self.operations) and len(set(self.operations)) == len(self.operations), "委托操作重复或为空")
        require(type(self.expires_at) is int and self.expires_at > 0, "委托期限无效")
        require((self.run_ref is None) == (self.spec_hash is None), "委托 Run/Spec 必须同时存在")

    def validate(self, binding: AppBackendBinding, scope: AuthorizedScope, *, audience: str,
                 operation: str, now: int, run_ref: str | None, spec_hash: str | None) -> None:
        """服务适配器先验证凭据/JWT 与当前主体权限，本方法只核验冻结上下文。"""
        binding.check_owner(self.owner_binding_ref)
        binding.check_owner(scope.owner_binding_ref)
        require(self.scope_hash == scope.scope_hash and self.audience == audience
                and operation in self.operations and self.expires_at > now, "委托范围、受众、操作或期限无效")
        require(self.run_ref == run_ref and self.spec_hash == spec_hash, "委托运行或 Spec 不匹配")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    request_id: str
    scope_ref: str
    scope_hash: str
    query: QueryInput
    profile_ref: str
    payload_budget: PayloadCost
    run_ref: str | None
    spec_hash: str | None
    schema_version: str = "knowledge.retrieval_request.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "knowledge.retrieval_request.v1", "检索请求版本无效")
        nonempty(self.request_id, self.scope_ref, self.scope_hash, self.profile_ref)
        require((self.run_ref is None) == (self.spec_hash is None), "Runtime 请求必须同时绑定 Run 与 Spec")


@dataclass(frozen=True, slots=True)
class CitationLabel:
    run_ref: str
    label: str
    evidence_key: EvidenceKey
    role: str
    locators: tuple[Locator, ...]

    def __post_init__(self) -> None:
        nonempty(self.run_ref, self.label)
        require(self.role in {"PRIMARY", "CONTEXT", "REFERENCED"} and bool(self.locators), "引用角色或定位无效")


@dataclass(frozen=True, slots=True)
class CitationLabelMap:
    run_ref: str
    entries: tuple[CitationLabel, ...]
    schema_version: str = "knowledge.citation_label_map.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "knowledge.citation_label_map.v1", "引用映射版本无效")
        nonempty(self.run_ref)
        require(all(entry.run_ref == self.run_ref for entry in self.entries)
                and len({entry.label for entry in self.entries}) == len(self.entries), "标签重复或跨 Run")


@dataclass(frozen=True, slots=True)
class RetrievalEvidenceArtifact:
    run_ref: str
    turn_ref: str
    evidence_set: EvidenceSet
    labels: CitationLabelMap
    schema_version: str = "knowledge.retrieval_evidence.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "knowledge.retrieval_evidence.v1", "证据 Artifact 版本无效")
        nonempty(self.run_ref, self.turn_ref)
        require(self.labels.run_ref == self.run_ref, "证据 Artifact 标签跨 Run")
        require({entry.evidence_key for entry in self.labels.entries}
                <= {item.key for item in self.evidence_set.evidence}, "标签引用不在冻结证据集合")


@dataclass(frozen=True, slots=True)
class DeliveredEvidence:
    evidence_key: EvidenceKey
    label: str
    content_hash: str
    input_kind: str
    cost: PayloadCost

    def __post_init__(self) -> None:
        nonempty(self.label, self.content_hash)
        require(self.input_kind in {"TEXT", "IMAGE"}, "实际交付模态无效")


@dataclass(frozen=True, slots=True)
class ModelEvidenceDelivery:
    run_ref: str
    model_snapshot_ref: str
    status: str
    prepared: tuple[DeliveredEvidence, ...]
    sent_at: int | None
    schema_version: str = "knowledge.model_evidence_delivery.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "knowledge.model_evidence_delivery.v1", "模型交付版本无效")
        nonempty(self.run_ref, self.model_snapshot_ref)
        require(self.status in {"NOT_APPLICABLE", "PREPARED", "SENT", "FAILED", "UNKNOWN_DELIVERY"}, "交付状态无效")
        require((self.sent_at is not None) == (self.status == "SENT"), "只有真实发送才有 SENT 时间")
        require(self.status != "NOT_APPLICABLE" or not self.prepared, "纯搜索没有模型交付清单")
        require(len({(p.evidence_key, p.label, p.input_kind) for p in self.prepared}) == len(self.prepared),
                "交付清单身份重复")
        require(self.status != "SENT" or bool(self.prepared), "不能把零载荷宣称为已交付证据")
        require(self.sent_at is None or type(self.sent_at) is int and self.sent_at > 0, "交付时间无效")


@dataclass(frozen=True, slots=True)
class GroundedAnswerArtifact:
    run_ref: str
    body: str
    used_labels: tuple[str, ...]
    references: tuple[CitationLabel, ...]
    coverage_status: str
    schema_version: str = "knowledge.grounded_answer.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "knowledge.grounded_answer.v1", "回答 Artifact 版本无效")
        nonempty(self.run_ref)
        require(self.coverage_status in {"COMPLETE", "PARTIAL", "INSUFFICIENT", "UNKNOWN"}, "回答覆盖状态无效")
        require(tuple(entry.label for entry in self.references) == self.used_labels
                and len(set(self.used_labels)) == len(self.used_labels)
                and all(entry.run_ref == self.run_ref for entry in self.references), "回答引用顺序、身份或 Run 不一致")


@dataclass(frozen=True, slots=True)
class ErrorEnvelope:
    code: ErrorCode
    stage: str
    retryable: bool
    schema_version: str = "knowledge.error.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "knowledge.error.v1" and isinstance(self.code, ErrorCode), "错误合同版本或错误码无效")
        nonempty(self.stage)


@dataclass(frozen=True, slots=True)
class AppRetrievalArtifact:
    retrieval_ref: str
    evidence: RetrievalEvidenceArtifact

    def __post_init__(self) -> None:
        nonempty(self.retrieval_ref)


@dataclass(frozen=True, slots=True)
class KnowledgeCompositionArtifact:
    retrieval_ref: str
    evidence: EvidenceSet
    delivery: ModelEvidenceDelivery
    answer: GroundedAnswerArtifact | None
    labels: CitationLabelMap
    excluded: tuple[tuple[EvidenceKey, str], ...]
    context_selection: SelectionResult

    def __post_init__(self) -> None:
        nonempty(self.retrieval_ref)
        keys = {item.key for item in self.evidence.evidence}
        require(self.delivery.run_ref == self.labels.run_ref, "交付与标签跨 Run")
        require({entry.evidence_key for entry in self.labels.entries} <= keys
                and {entry.evidence_key for entry in self.delivery.prepared} <= keys
                and {key for key, _ in self.excluded} <= keys
                and set(self.context_selection.selected_evidence_keys) <= keys,
                "交付、选择或标签身份超出冻结证据")
        by_label = {entry.label: entry.evidence_key for entry in self.labels.entries}
        require(all(by_label.get(entry.label) == entry.evidence_key for entry in self.delivery.prepared),
                "交付标签与引用身份不一致")
        if self.answer is not None:
            require(self.delivery.status == "SENT" and self.answer.run_ref == self.delivery.run_ref,
                    "没有确认交付时不能宣称有来源回答")
            require(set(self.answer.references) <= set(self.labels.entries), "回答引用未交付的身份")
