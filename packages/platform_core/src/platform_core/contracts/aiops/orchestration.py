"""AIOps 统一助手的能力规划、范围解析、委派与回答契约。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .types import AIOpsContract, JsonObject, Sha256Digest, UUIDv7, UtcDatetime


ORCHESTRATION_SCHEMA_VERSION = "aiops.orchestration.v1"
EVIDENCE_SCHEMA_VERSION = "aiops.evidence.v2"
STRUCTURED_ANSWER_SCHEMA_VERSION = "aiops.structured-answer.v2"


class AIOpsCapability(StrEnum):
    RESOURCE_CATALOG_QUERY = "RESOURCE_CATALOG_QUERY"
    KNOWLEDGE_QA = "KNOWLEDGE_QA"
    MONITORING_QUERY = "MONITORING_QUERY"
    ALERT_QUERY = "ALERT_QUERY"
    INSPECTION_QUERY = "INSPECTION_QUERY"
    REPORT_CASE_QUERY = "REPORT_CASE_QUERY"
    ACTIVE_DIAGNOSIS = "ACTIVE_DIAGNOSIS"
    CONTROLLED_CHANGE = "CONTROLLED_CHANGE"


class ScopeType(StrEnum):
    NONE = "NONE"
    SYSTEM = "SYSTEM"
    TOPOLOGY_GROUP = "TOPOLOGY_GROUP"
    RESOURCE_SET = "RESOURCE_SET"
    RESOURCE = "RESOURCE"


class ScopeResolutionStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    RESOLVED = "RESOLVED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    DENIED = "DENIED"


class FreshnessRequirement(StrEnum):
    STORED_ACCEPTABLE = "STORED_ACCEPTABLE"
    LATEST_AVAILABLE = "LATEST_AVAILABLE"
    LIVE_REQUIRED = "LIVE_REQUIRED"


class TurnOutputMode(StrEnum):
    SUMMARY = "SUMMARY"
    DETAIL = "DETAIL"
    TABLE = "TABLE"
    TREND = "TREND"
    REPORT = "REPORT"
    DIAGNOSIS = "DIAGNOSIS"


class DelegationStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_USER = "WAITING_USER"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    DENIED = "DENIED"


class EvidenceSourceKind(StrEnum):
    RESOURCE_CATALOG = "RESOURCE_CATALOG"
    MONITORING = "MONITORING"
    ALERT = "ALERT"
    INSPECTION = "INSPECTION"
    REPORT = "REPORT"
    KNOWLEDGE = "KNOWLEDGE"
    USER_INPUT = "USER_INPUT"
    DIAGNOSIS = "DIAGNOSIS"
    MODEL_INFERENCE = "MODEL_INFERENCE"


class EvidenceTrustLevel(StrEnum):
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    USER_PROVIDED = "USER_PROVIDED"
    MODEL_INFERENCE = "MODEL_INFERENCE"


class EvidenceFreshness(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class StructuredAnswerStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    WAITING_USER = "WAITING_USER"
    DENIED = "DENIED"
    FAILED = "FAILED"


class TimeRangeV1(AIOpsContract):
    """绝对时间范围或相对窗口，两种表达必须二选一。"""

    start_at: UtcDatetime | None = None
    end_at: UtcDatetime | None = None
    relative_window_seconds: int | None = Field(
        default=None, ge=60, le=31_536_000
    )

    @model_validator(mode="after")
    def validate_range(self) -> "TimeRangeV1":
        has_absolute = self.start_at is not None or self.end_at is not None
        if has_absolute and (self.start_at is None or self.end_at is None):
            raise ValueError("绝对时间范围必须同时包含 start_at 和 end_at")
        if has_absolute and self.relative_window_seconds is not None:
            raise ValueError("绝对时间范围与相对窗口不能同时设置")
        if (
            self.start_at is not None
            and self.end_at is not None
            and self.start_at >= self.end_at
        ):
            raise ValueError("start_at 必须早于 end_at")
        return self


class ExecutionBudgetV1(AIOpsContract):
    max_delegations: int = Field(default=8, ge=1, le=32)
    max_parallel_delegations: int = Field(default=4, ge=1, le=16)
    max_live_queries: int = Field(default=12, ge=0, le=100)
    max_diagnosis_runs: int = Field(default=4, ge=0, le=16)
    timeout_seconds: int = Field(default=300, ge=10, le=7200)

    @model_validator(mode="after")
    def validate_parallelism(self) -> "ExecutionBudgetV1":
        if self.max_parallel_delegations > self.max_delegations:
            raise ValueError("并行委派数不能超过委派总数")
        return self


class AgentKnowledgePolicy(AIOpsContract):
    """Agent 版本保存稳定选择器，具体 KC Revision 在每个 Turn 冻结。"""

    enabled: bool = True
    source_kinds: tuple[str, ...] = ("MANUAL", "DIAGNOSIS_CASE")
    knowledge_base_ids: tuple[UUIDv7, ...] = ()
    tag_ids: tuple[UUIDv7, ...] = ()
    max_security_level: int = Field(default=3, ge=1, le=5)
    maximum_references: int = Field(default=8, ge=1, le=20)

    @model_validator(mode="after")
    def validate_selectors(self) -> "AgentKnowledgePolicy":
        allowed = {"MANUAL", "DIAGNOSIS_CASE"}
        if (self.enabled and not self.source_kinds) or not set(self.source_kinds) <= allowed:
            raise ValueError("知识来源只能选择 MANUAL 或 DIAGNOSIS_CASE")
        if len(set(self.source_kinds)) != len(self.source_kinds):
            raise ValueError("知识来源不能重复")
        if len(set(self.tag_ids)) != len(self.tag_ids):
            raise ValueError("知识标签不能重复")
        if self.enabled and not self.knowledge_base_ids:
            raise ValueError("启用知识能力时至少需要绑定一个运维知识库")
        if len(set(self.knowledge_base_ids)) != len(self.knowledge_base_ids):
            raise ValueError("运维知识库不能重复")
        return self


class TrustedContextHint(AIOpsContract):
    """服务端校验过的入口提示，不替代当轮范围解析与授权。"""

    source_kind: str = Field(
        pattern=r"^(BUSINESS_SYSTEM_PAGE|RESOURCE_PAGE|SITUATION|RUN|REPORT|INSPECTION)$"
    )
    business_system_id: UUIDv7 | None = None
    resource_ids: tuple[UUIDv7, ...] = Field(default=(), max_length=64)
    source_resource_id: UUIDv7 | None = None

    @model_validator(mode="after")
    def validate_hint(self) -> "TrustedContextHint":
        if self.business_system_id is None and not self.resource_ids:
            raise ValueError("可信上下文必须包含业务系统或资源提示")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise ValueError("可信上下文中的资源不能重复")
        return self


class TurnPlanStepV1(AIOpsContract):
    step_id: str = Field(pattern=r"^s[1-9][0-9]*$")
    capability: AIOpsCapability
    objective: str = Field(min_length=1, max_length=2000)
    scope_type: ScopeType
    depends_on: tuple[str, ...] = Field(default=(), max_length=16)
    requested_evidence_kinds: tuple[str, ...] = Field(default=(), max_length=16)
    optional: bool = False
    parameters: JsonObject = Field(default_factory=dict)


class TurnPlanV1(AIOpsContract):
    schema_version: str = ORCHESTRATION_SCHEMA_VERSION
    turn_id: UUIDv7
    revision_no: int = Field(ge=1)
    requested_capabilities: tuple[AIOpsCapability, ...] = Field(
        min_length=1, max_length=len(AIOpsCapability)
    )
    requested_facts: tuple[str, ...] = Field(default=(), max_length=64)
    time_range: TimeRangeV1 | None = None
    freshness: FreshnessRequirement = FreshnessRequirement.LATEST_AVAILABLE
    requires_active_diagnosis: bool = False
    output_mode: TurnOutputMode = TurnOutputMode.SUMMARY
    budget: ExecutionBudgetV1 = Field(default_factory=ExecutionBudgetV1)
    steps: tuple[TurnPlanStepV1, ...] = Field(min_length=1, max_length=32)
    policy_snapshot_hash: Sha256Digest

    @model_validator(mode="after")
    def validate_plan(self) -> "TurnPlanV1":
        capabilities = tuple(self.requested_capabilities)
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("requested_capabilities 不能重复")
        step_ids = tuple(step.step_id for step in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("计划步骤 ID 不能重复")
        if len(self.steps) > self.budget.max_delegations:
            raise ValueError("计划步骤数超过执行预算")
        declared = set(step_ids)
        graph: dict[str, tuple[str, ...]] = {}
        for step in self.steps:
            if step.capability not in capabilities:
                raise ValueError("计划步骤使用了未声明的能力")
            if step.step_id in step.depends_on:
                raise ValueError("计划步骤不能依赖自身")
            if not set(step.depends_on) <= declared:
                raise ValueError("计划步骤依赖了未知步骤")
            graph[step.step_id] = step.depends_on
        if set(capabilities) != {step.capability for step in self.steps}:
            raise ValueError("每项请求能力都必须至少有一个计划步骤")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("计划步骤依赖不能形成环")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in graph[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in step_ids:
            visit(step_id)
        diagnoses = AIOpsCapability.ACTIVE_DIAGNOSIS in capabilities
        if self.requires_active_diagnosis != diagnoses:
            raise ValueError("requires_active_diagnosis 与能力计划不一致")
        if diagnoses and self.budget.max_diagnosis_runs < 1:
            raise ValueError("主动诊断必须保留诊断运行预算")
        if (
            self.freshness == FreshnessRequirement.LIVE_REQUIRED
            and self.budget.max_live_queries < 1
        ):
            raise ValueError("实时查询必须保留实时查询预算")
        return self


class BusinessSystemRefV1(AIOpsContract):
    business_system_id: UUIDv7
    display_name: str = Field(min_length=1, max_length=256)
    environment: str | None = Field(default=None, max_length=16)


class ResourceRefV1(AIOpsContract):
    resource_id: UUIDv7
    system_resource_id: UUIDv7
    business_system_id: UUIDv7
    display_name: str = Field(min_length=1, max_length=256)
    resource_type_key: str = Field(min_length=1, max_length=64)
    topology_roles: tuple[str, ...] = Field(default=(), max_length=16)


class TopologyRelationRefV1(AIOpsContract):
    relation_id: UUIDv7
    business_system_id: UUIDv7
    relation_type_key: str = Field(min_length=1, max_length=64)
    display_name: str | None = Field(default=None, max_length=256)


class ScopeResolutionV1(AIOpsContract):
    """用户可见的范围解析；禁止包含内部 Target 身份。"""

    schema_version: str = ORCHESTRATION_SCHEMA_VERSION
    status: ScopeResolutionStatus
    scope_type: ScopeType
    business_systems: tuple[BusinessSystemRefV1, ...] = Field(
        default=(), max_length=32
    )
    resources: tuple[ResourceRefV1, ...] = Field(default=(), max_length=256)
    relations: tuple[TopologyRelationRefV1, ...] = Field(
        default=(), max_length=256
    )
    inherited_from_turn_id: UUIDv7 | None = None
    resolution_basis: tuple[str, ...] = Field(default=(), max_length=16)
    confidence: float | None = Field(default=None, ge=0, le=1)
    authorization_snapshot_hash: Sha256Digest
    resource_graph_snapshot_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "ScopeResolutionV1":
        system_ids = tuple(item.business_system_id for item in self.business_systems)
        resource_ids = tuple(item.resource_id for item in self.resources)
        relation_ids = tuple(item.relation_id for item in self.relations)
        if len(set(system_ids)) != len(system_ids):
            raise ValueError("业务系统范围不能重复")
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("资源范围不能重复")
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("拓扑关系范围不能重复")
        known_systems = set(system_ids)
        if any(item.business_system_id not in known_systems for item in self.resources):
            raise ValueError("资源必须属于已解析的业务系统")
        if any(item.business_system_id not in known_systems for item in self.relations):
            raise ValueError("拓扑关系必须属于已解析的业务系统")
        if self.status == ScopeResolutionStatus.NOT_REQUIRED:
            if self.scope_type != ScopeType.NONE or system_ids or resource_ids or relation_ids:
                raise ValueError("NOT_REQUIRED 范围必须为空")
        elif self.status == ScopeResolutionStatus.DENIED:
            if system_ids or resource_ids or relation_ids:
                raise ValueError("DENIED 范围不能泄露资源引用")
        elif self.status == ScopeResolutionStatus.RESOLVED:
            if self.scope_type == ScopeType.NONE or not system_ids:
                raise ValueError("RESOLVED 范围必须包含业务系统")
            if self.scope_type == ScopeType.RESOURCE and len(resource_ids) != 1:
                raise ValueError("RESOURCE 范围必须且只能包含一个资源")
            if self.scope_type == ScopeType.RESOURCE_SET and not resource_ids:
                raise ValueError("RESOURCE_SET 范围必须包含资源")
            if self.scope_type == ScopeType.TOPOLOGY_GROUP and not relation_ids:
                raise ValueError("TOPOLOGY_GROUP 范围必须包含拓扑关系")
        return self


class ResolvedExecutionTargetV1(AIOpsContract):
    """仅供内部执行器消费的 Resource 到 Target 解析结果。"""

    business_system_id: UUIDv7
    system_resource_id: UUIDv7
    resource_id: UUIDv7
    target_id: UUIDv7
    role: str | None = Field(default=None, max_length=64)
    readiness: str = Field(
        pattern=r"^(READY|NO_BINDING|DISABLED|UNAVAILABLE)$"
    )
    binding_ids: tuple[UUIDv7, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_bindings(self) -> "ResolvedExecutionTargetV1":
        if len(set(self.binding_ids)) != len(self.binding_ids):
            raise ValueError("执行 Target 的 binding_ids 不能重复")
        return self


class DelegationSpecV1(AIOpsContract):
    schema_version: str = ORCHESTRATION_SCHEMA_VERSION
    delegation_id: UUIDv7
    turn_id: UUIDv7
    plan_revision_no: int = Field(ge=1)
    step_id: str = Field(pattern=r"^s[1-9][0-9]*$")
    capability: AIOpsCapability
    objective: str = Field(min_length=1, max_length=2000)
    scope: ScopeResolutionV1
    execution_targets: tuple[ResolvedExecutionTargetV1, ...] = Field(
        default=(), max_length=256
    )
    depends_on: tuple[UUIDv7, ...] = Field(default=(), max_length=16)
    input_artifact_ids: tuple[UUIDv7, ...] = Field(default=(), max_length=64)
    required_evidence_kinds: tuple[str, ...] = Field(default=(), max_length=16)
    idempotency_key: str = Field(min_length=1, max_length=128)
    timeout_seconds: int = Field(ge=10, le=7200)

    @model_validator(mode="after")
    def validate_spec(self) -> "DelegationSpecV1":
        target_ids = tuple(item.target_id for item in self.execution_targets)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("execution_targets 不能重复")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("depends_on 不能重复")
        if len(set(self.input_artifact_ids)) != len(self.input_artifact_ids):
            raise ValueError("input_artifact_ids 不能重复")
        if self.delegation_id in self.depends_on:
            raise ValueError("委派不能依赖自身")
        if (
            self.capability == AIOpsCapability.ACTIVE_DIAGNOSIS
            and not self.execution_targets
        ):
            raise ValueError("主动诊断委派必须包含内部执行 Target")
        return self


class EvidenceSubjectRefV1(AIOpsContract):
    business_system_id: UUIDv7 | None = None
    resource_id: UUIDv7 | None = None
    relation_id: UUIDv7 | None = None

    @model_validator(mode="after")
    def validate_subject(self) -> "EvidenceSubjectRefV1":
        if not any((self.business_system_id, self.resource_id, self.relation_id)):
            raise ValueError("证据主题至少需要一个公开资源引用")
        return self


class EvidenceDescriptorV2(AIOpsContract):
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    evidence_id: UUIDv7
    artifact_id: UUIDv7
    delegation_id: UUIDv7 | None = None
    source_kind: EvidenceSourceKind
    evidence_kind: str = Field(min_length=1, max_length=64)
    subjects: tuple[EvidenceSubjectRefV1, ...] = Field(default=(), max_length=64)
    trust_level: EvidenceTrustLevel
    observed_at: UtcDatetime | None = None
    window_start_at: UtcDatetime | None = None
    window_end_at: UtcDatetime | None = None
    captured_at: UtcDatetime
    freshness: EvidenceFreshness
    confidence: float | None = Field(default=None, ge=0, le=1)
    summary: str = Field(min_length=1, max_length=2000)
    provenance: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence(self) -> "EvidenceDescriptorV2":
        if (self.window_start_at is None) != (self.window_end_at is None):
            raise ValueError("证据窗口必须同时包含开始和结束时间")
        if (
            self.window_start_at is not None
            and self.window_end_at is not None
            and self.window_start_at >= self.window_end_at
        ):
            raise ValueError("证据窗口开始时间必须早于结束时间")
        if (
            self.source_kind == EvidenceSourceKind.MODEL_INFERENCE
            and self.trust_level != EvidenceTrustLevel.MODEL_INFERENCE
        ):
            raise ValueError("模型推断证据必须标记 MODEL_INFERENCE 信任级别")
        if (
            self.trust_level == EvidenceTrustLevel.MODEL_INFERENCE
            and self.source_kind != EvidenceSourceKind.MODEL_INFERENCE
        ):
            raise ValueError("MODEL_INFERENCE 信任级别只能用于模型推断")
        return self


class DelegationResultV1(AIOpsContract):
    schema_version: str = ORCHESTRATION_SCHEMA_VERSION
    delegation_id: UUIDv7
    capability: AIOpsCapability
    status: DelegationStatus
    output_artifact_ids: tuple[UUIDv7, ...] = Field(default=(), max_length=64)
    evidence: tuple[EvidenceDescriptorV2, ...] = Field(default=(), max_length=256)
    gaps: tuple[str, ...] = Field(default=(), max_length=64)
    warnings: tuple[str, ...] = Field(default=(), max_length=64)
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = Field(default=None, max_length=2000)
    started_at: UtcDatetime | None = None
    completed_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "DelegationResultV1":
        if len(set(self.output_artifact_ids)) != len(self.output_artifact_ids):
            raise ValueError("委派输出 Artifact 不能重复")
        if any(item.delegation_id not in {None, self.delegation_id} for item in self.evidence):
            raise ValueError("证据必须属于当前委派")
        if self.started_at and self.completed_at and self.started_at > self.completed_at:
            raise ValueError("委派完成时间不能早于开始时间")
        if self.status in {DelegationStatus.FAILED, DelegationStatus.DENIED}:
            if not self.error_code:
                raise ValueError("失败或拒绝的委派必须包含 error_code")
        elif self.error_code is not None or self.error_message is not None:
            raise ValueError("非失败委派不能包含错误信息")
        return self


class ClarificationCandidateV1(AIOpsContract):
    candidate_id: str = Field(
        min_length=16,
        max_length=256,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    label: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    scope_type: ScopeType


class ClarificationRequestV1(AIOpsContract):
    schema_version: str = ORCHESTRATION_SCHEMA_VERSION
    clarification_id: UUIDv7
    turn_id: UUIDv7
    question: str = Field(min_length=1, max_length=2000)
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")
    candidates: tuple[ClarificationCandidateV1, ...] = Field(
        min_length=2, max_length=20
    )
    allow_free_text: bool = True
    expires_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_candidates(self) -> "ClarificationRequestV1":
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("澄清候选 candidate_id 不能重复")
        return self


class ClarificationResponseV1(AIOpsContract):
    schema_version: str = ORCHESTRATION_SCHEMA_VERSION
    clarification_id: UUIDv7
    idempotency_key: str = Field(min_length=1, max_length=128)
    candidate_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=256,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    free_text: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_response(self) -> "ClarificationResponseV1":
        if (self.candidate_id is None) == (self.free_text is None):
            raise ValueError("澄清响应必须且只能选择候选或填写文字")
        return self


class StructuredAnswerCitationV2(AIOpsContract):
    citation_id: str = Field(pattern=r"^C[1-9][0-9]*$")
    evidence_id: UUIDv7
    label: str = Field(min_length=1, max_length=256)


class StructuredAnswerBlockV2(AIOpsContract):
    block_id: str = Field(pattern=r"^B[1-9][0-9]*$")
    block_type: str = Field(
        pattern=(
            r"^(MARKDOWN|TABLE|CHART|SCOPE_SUMMARY|RESOURCE_TABLE|"
            r"MONITORING_SUMMARY|ALERT_TIMELINE|INSPECTION_SUMMARY|"
            r"REPORT_LIST|CASE_CARDS|DIAGNOSIS_SUMMARY|ACTION_PROPOSAL|"
            r"CLARIFICATION)$"
        )
    )
    schema_version: str = Field(min_length=1, max_length=64)
    payload: JsonObject
    citation_ids: tuple[str, ...] = Field(default=(), max_length=64)


class AIOpsStructuredAnswerV2(AIOpsContract):
    schema_version: str = STRUCTURED_ANSWER_SCHEMA_VERSION
    turn_id: UUIDv7
    status: StructuredAnswerStatus
    scope: ScopeResolutionV1
    capabilities_used: tuple[AIOpsCapability, ...] = Field(
        default=(), max_length=len(AIOpsCapability)
    )
    generated_at: UtcDatetime
    data_as_of: UtcDatetime | None = None
    coverage: str = Field(pattern=r"^(SUPPORTED|PARTIAL|UNSUPPORTED)$")
    gaps: tuple[str, ...] = Field(default=(), max_length=64)
    inference_disclosures: tuple[str, ...] = Field(default=(), max_length=64)
    evidence: tuple[EvidenceDescriptorV2, ...] = Field(default=(), max_length=256)
    citations: tuple[StructuredAnswerCitationV2, ...] = Field(
        default=(), max_length=256
    )
    blocks: tuple[StructuredAnswerBlockV2, ...] = Field(
        min_length=1, max_length=64
    )
    clarification_id: UUIDv7 | None = None

    @model_validator(mode="after")
    def validate_answer(self) -> "AIOpsStructuredAnswerV2":
        if len(set(self.capabilities_used)) != len(self.capabilities_used):
            raise ValueError("capabilities_used 不能重复")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        citation_ids = tuple(item.citation_id for item in self.citations)
        block_ids = tuple(item.block_id for item in self.blocks)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("回答证据不能重复")
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("回答引用 ID 不能重复")
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("回答块 ID 不能重复")
        known_evidence = set(evidence_ids)
        if any(item.evidence_id not in known_evidence for item in self.citations):
            raise ValueError("回答引用必须指向已声明证据")
        known_citations = set(citation_ids)
        if any(not set(block.citation_ids) <= known_citations for block in self.blocks):
            raise ValueError("回答块引用了未知 citation_id")
        waiting = self.status == StructuredAnswerStatus.WAITING_USER
        if waiting != (self.clarification_id is not None):
            raise ValueError("WAITING_USER 回答必须且只能携带 clarification_id")
        return self
