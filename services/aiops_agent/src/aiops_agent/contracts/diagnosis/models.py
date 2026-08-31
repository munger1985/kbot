"""步骤 7 Evidence、假设、根因和报告的严格 Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from platform_core.contracts.aiops.types import UtcDatetime
from aiops_agent.contracts.artifacts.database import DatabaseDiagnosticResult


class _DiagnosisContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DiagnosisScope(_DiagnosisContract):
    schema_version: Literal["DIAGNOSIS_SCOPE.v1"] = "DIAGNOSIS_SCOPE.v1"
    run_id: str
    target_id: str
    agent_id: str
    trigger_type: str
    symptom_codes: tuple[str, ...] = ()
    user_question_summary: str | None = Field(default=None, max_length=2000)
    window_start: UtcDatetime
    window_end: UtcDatetime
    db_type: str
    db_version: str
    environment: str
    target_capabilities: tuple[str, ...] = ()
    allowed_collection_ids: tuple[str, ...] = ()
    policy_snapshot_hash: str
    security_level: int = Field(ge=0, le=9)
    budget_snapshot: dict[str, int]


class EvidenceFact(_DiagnosisContract):
    fact_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_artifact_id: str
    source_json_pointer: str
    source_type: Literal[
        "METRIC_OBSERVATION",
        "EVENT_OBSERVATION",
        "LOG_ENTRY",
        "DATABASE_OBSERVATION",
        "USER_RESULT",
        "KNOWLEDGE_CITATION",
    ]
    source_group_id: str
    trust_level: Literal[
        "SOURCE_VERIFIED",
        "USER_PROVIDED",
        "KNOWLEDGE_CITATION",
    ]
    target_id: str
    observed_subject: str
    metric_or_fact_type: str
    value: Any = None
    unit: str | None = None
    dimensions: dict[str, str] = Field(default_factory=dict)
    window_start: UtcDatetime | None = None
    window_end: UtcDatetime | None = None
    captured_at: UtcDatetime | None = None
    quality_flags: tuple[str, ...] = ()
    security_level: int = Field(default=1, ge=0, le=9)
    fact_summary: str = Field(min_length=1, max_length=2000)


class EvidenceIndex(_DiagnosisContract):
    schema_version: Literal["EVIDENCE_INDEX.v1"] = "EVIDENCE_INDEX.v1"
    target_id: str
    facts: tuple[EvidenceFact, ...] = ()
    gaps: tuple[dict[str, Any], ...] = ()
    fact_count: int = Field(ge=0)
    source_group_count: int = Field(ge=0)
    index_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_counts(self) -> "EvidenceIndex":
        if self.fact_count != len(self.facts):
            raise ValueError("Evidence fact_count 与事实数量不一致")
        groups = {item.source_group_id for item in self.facts}
        if self.source_group_count != len(groups):
            raise ValueError("Evidence source_group_count 不一致")
        if len({item.fact_id for item in self.facts}) != len(self.facts):
            raise ValueError("Evidence fact_id 不能重复")
        return self


class KnowledgeCitationPack(_DiagnosisContract):
    schema_version: Literal["KNOWLEDGE_CITATION_PACK.v1"] = (
        "KNOWLEDGE_CITATION_PACK.v1"
    )
    query: str = Field(min_length=1, max_length=2000)
    citations: tuple[dict[str, Any], ...] = ()
    gap_code: str | None = None


class EvidenceRequestDraft(_DiagnosisContract):
    request_key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    tool_id: str = Field(pattern=r"^db\.[a-z0-9_.-]{1,124}$")
    parameters: dict[str, Any] = Field(default_factory=dict)
    hypothesis_keys: tuple[str, ...] = ()
    diagnostic_question: str = Field(min_length=1, max_length=1000)
    supports_if: str = Field(min_length=1, max_length=1000)
    contradicts_if: str = Field(min_length=1, max_length=1000)
    priority_reason: str = Field(min_length=1, max_length=1000)


class HypothesisDraft(_DiagnosisContract):
    hypothesis_key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    statement: str = Field(min_length=1, max_length=2000)
    mechanism: str = Field(min_length=1, max_length=3000)
    causal_role: Literal["ROOT", "CONTRIBUTOR", "SYMPTOM", "COINCIDENTAL"]
    explained_symptom_codes: tuple[str, ...] = ()
    supporting_fact_refs: tuple[str, ...] = ()
    counter_fact_refs: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()


class DiagnosisRoundDraft(_DiagnosisContract):
    schema_version: Literal["DIAGNOSIS_ROUND_DRAFT.v1"] = (
        "DIAGNOSIS_ROUND_DRAFT.v1"
    )
    round_no: int = Field(ge=1, le=10)
    hypotheses: tuple[HypothesisDraft, ...] = ()
    evidence_requests: tuple[EvidenceRequestDraft, ...] = ()
    stop_recommendation: Literal["CONTINUE", "FINALIZE", "INCONCLUSIVE"]
    stop_reason: str = Field(min_length=1, max_length=2000)
    model_gap_code: str | None = None
    invocation_receipt: ModelInvocationReceipt | None = None


class ValidatedEvidenceRequest(_DiagnosisContract):
    request_key: str
    tool_id: str
    tool_version: str
    variant: str
    parameters: dict[str, Any]
    request_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    hypothesis_keys: tuple[str, ...]


class RejectedEvidenceRequest(_DiagnosisContract):
    request_key: str
    tool_id: str
    reason_code: str


class ValidatedEvidencePlan(_DiagnosisContract):
    schema_version: Literal["VALIDATED_EVIDENCE_PLAN.v1"] = (
        "VALIDATED_EVIDENCE_PLAN.v1"
    )
    round_no: int = Field(ge=1, le=10)
    accepted: tuple[ValidatedEvidenceRequest, ...] = ()
    rejected: tuple[RejectedEvidenceRequest, ...] = ()
    remaining_tool_budget: int = Field(ge=0)
    decision: Literal["COLLECT", "FINALIZE", "STOP_INCONCLUSIVE"]


class DiagnosisEvidenceCollection(_DiagnosisContract):
    schema_version: Literal["DIAGNOSIS_EVIDENCE_COLLECTION.v1"] = (
        "DIAGNOSIS_EVIDENCE_COLLECTION.v1"
    )
    round_no: int = Field(ge=1, le=10)
    results: tuple[DatabaseDiagnosticResult, ...] = ()


class TestResult(_DiagnosisContract):
    request_key: str
    outcome: Literal["SUPPORTS", "CONTRADICTS", "NEUTRAL", "UNAVAILABLE"]
    strength: Literal["DIRECT", "CORRELATED", "CONTEXTUAL"]
    fact_refs: tuple[str, ...] = ()


class HypothesisAssessment(_DiagnosisContract):
    hypothesis_key: str
    status: Literal["SUPPORTED", "WEAKENED", "REJECTED", "UNTESTED"]
    causal_role: Literal["ROOT", "CONTRIBUTOR", "SYMPTOM", "COINCIDENTAL"]
    supporting_fact_refs: tuple[str, ...] = ()
    counter_fact_refs: tuple[str, ...] = ()
    test_results: tuple[TestResult, ...] = ()
    remaining_gaps: tuple[str, ...] = ()


class DiagnosisRoundAssessment(_DiagnosisContract):
    schema_version: Literal["DIAGNOSIS_ROUND_ASSESSMENT.v1"] = (
        "DIAGNOSIS_ROUND_ASSESSMENT.v1"
    )
    round_no: int = Field(ge=1, le=10)
    hypothesis_assessments: tuple[HypothesisAssessment, ...] = ()
    suggested_root_cause_level: Literal[
        "CONFIRMED", "PROBABLE", "POSSIBLE", "INCONCLUSIVE"
    ]
    recommended_next_step: Literal[
        "CONTINUE", "FINALIZE", "STOP_INCONCLUSIVE"
    ]
    rationale_summary: str = Field(min_length=1, max_length=3000)
    model_gap_code: str | None = None
    invocation_receipt: ModelInvocationReceipt | None = None


class ModelInvocationReceipt(_DiagnosisContract):
    schema_version: Literal["MODEL_INVOCATION_RECEIPT.v1"] = (
        "MODEL_INVOCATION_RECEIPT.v1"
    )
    purpose: str
    schema_id: str
    model_technical_name: str
    model_revision: str
    prompt_id: str
    prompt_version: str
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_version_id: str | None = None
    prompt_source: str | None = None
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_request_id: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    duration_ms: int = Field(ge=0)
    finish_reason: str | None = None
    repair_count: int = Field(default=0, ge=0, le=1)


class RootCauseAssessment(_DiagnosisContract):
    schema_version: Literal["ROOT_CAUSE_ASSESSMENT.v1"] = (
        "ROOT_CAUSE_ASSESSMENT.v1"
    )
    target_id: str
    suggested_level: Literal[
        "CONFIRMED", "PROBABLE", "POSSIBLE", "INCONCLUSIVE"
    ]
    eligible_ceiling: Literal[
        "CONFIRMED", "PROBABLE", "POSSIBLE", "INCONCLUSIVE"
    ]
    effective_level: Literal[
        "CONFIRMED", "PROBABLE", "POSSIBLE", "INCONCLUSIVE"
    ]
    primary_hypothesis_key: str | None = None
    contributing_hypothesis_keys: tuple[str, ...] = ()
    supporting_fact_refs: tuple[str, ...] = ()
    counter_fact_refs: tuple[str, ...] = ()
    unresolved_gaps: tuple[str, ...] = ()
    downgrade_reasons: tuple[str, ...] = ()


class GroundingVerification(_DiagnosisContract):
    schema_version: Literal["GROUNDING_VERIFICATION.v1"] = (
        "GROUNDING_VERIFICATION.v1"
    )
    status: Literal["PASS", "REVISE", "BLOCK"]
    invalid_fact_refs: tuple[str, ...] = ()
    ungrounded_claims: tuple[str, ...] = ()
    ignored_counter_evidence: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    model_gap_code: str | None = None
    invocation_receipt: ModelInvocationReceipt | None = None


class SolutionDraft(_DiagnosisContract):
    schema_version: Literal["SOLUTION_DRAFT.v1"] = "SOLUTION_DRAFT.v1"
    immediate_mitigations: tuple[str, ...] = ()
    long_term_remediations: tuple[str, ...] = ()
    candidate_action_template_refs: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    verification_plan: tuple[str, ...] = ()
    knowledge_citations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class DirectQuestionAnswer(_DiagnosisContract):
    """针对已验证证据生成的可追溯直接答案。"""

    answer_kind: Literal["EVIDENCE_FACT"]
    status: Literal["ANSWERED", "PARTIAL"]
    question_summary: str = Field(min_length=1, max_length=2000)
    answer_text: str = Field(min_length=1, max_length=3000)
    fact_refs: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()


class DiagnosisReportDraft(_DiagnosisContract):
    schema_version: Literal["DIAGNOSIS_REPORT_DRAFT.v1"] = (
        "DIAGNOSIS_REPORT_DRAFT.v1"
    )
    target_id: str
    status: Literal["READY", "PARTIAL", "DEGRADED"]
    output_kind: Literal["SIMPLE_CONCLUSION", "DIAGNOSIS_REPORT"]
    recommendation_level: Literal["NONE", "BRIEF", "FULL"]
    report_decision_reasons: tuple[str, ...] = ()
    issue_detected: bool = False
    root_cause: RootCauseAssessment
    facts: tuple[EvidenceFact, ...] = ()
    hypotheses: tuple[HypothesisAssessment, ...] = ()
    hypothesis_details: tuple[HypothesisDraft, ...] = ()
    diagnosis_rationale: str | None = Field(default=None, max_length=3000)
    rejected_evidence_requests: tuple[RejectedEvidenceRequest, ...] = ()
    direct_answer: DirectQuestionAnswer | None = None
    solution: SolutionDraft
    gaps: tuple[str, ...] = ()
    verification: GroundingVerification
    model_receipt_hashes: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)
