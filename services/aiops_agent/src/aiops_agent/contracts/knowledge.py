"""运维知识提炼、发布与检索合同。"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeScopeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=512)


class KnowledgeBaseWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    models: dict[str, UUID]
    expected_row_version: int | None = Field(default=None, ge=1)


class ManualUploadMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=256)
    publisher: str | None = Field(default=None, max_length=256)
    document_version: str | None = Field(default=None, max_length=128)
    published_date: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=2000)
    security_level: int = Field(default=1, ge=1, le=5)


KnowledgeTagCategory = Literal["RESOURCE_CATEGORY", "TECHNOLOGY_STACK", "VENDOR"]


class KnowledgeTagScopeSnapshot(BaseModel):
    """审批读事务关闭后仍可安全使用的标签范围快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    tag_id: UUID
    category: KnowledgeTagCategory
    display_name: str = Field(min_length=1, max_length=128)


class KnowledgeTagWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: KnowledgeTagCategory
    display_name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1000)
    sort_order: int = Field(default=0, ge=0, le=1_000_000)
    expected_row_version: int | None = Field(default=None, ge=1)


class KnowledgeTagStatusCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)


class KnowledgeTagEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_version_id: str
    file_name: str
    quote: str = Field(min_length=1, max_length=1000)
    locator: dict[str, Any] = Field(default_factory=dict)


class KnowledgeTagSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag_id: str
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[KnowledgeTagEvidence, ...] = Field(default=(), max_length=8)


class KnowledgeTagAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["OPS_KNOWLEDGE_TAG_ANALYSIS.v1"] = "OPS_KNOWLEDGE_TAG_ANALYSIS.v1"
    suggestions: tuple[KnowledgeTagSuggestion, ...] = Field(default=(), max_length=64)


class ManualProcedureStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ordinal: int = Field(ge=1)
    description: str
    command_text: str | None = None
    source_evidence_ids: tuple[str, ...] = ()


class ManualProcedureCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["OPS_PROCEDURE_CARD.v1"] = "OPS_PROCEDURE_CARD.v1"
    procedure_key: str
    title: str
    intent: str
    scope: dict[str, Any] = Field(default_factory=dict)
    preconditions: tuple[dict[str, Any], ...] = ()
    stop_conditions: tuple[dict[str, Any], ...] = ()
    steps: tuple[ManualProcedureStep, ...] = ()
    validations: tuple[dict[str, Any], ...] = ()
    rollback: tuple[dict[str, Any], ...] = ()
    risks: tuple[dict[str, Any], ...] = ()
    source_locator: dict[str, Any] = Field(default_factory=dict)


class ManualProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["OPS_MANUAL_PROFILE.v1"] = "OPS_MANUAL_PROFILE.v1"
    title: str
    source: dict[str, Any]
    scope: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    topics: tuple[str, ...] = ()
    procedures: tuple[ManualProcedureCard, ...] = ()
    warnings: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    source_evidence_ids: tuple[str, ...] = ()


class CaseRedaction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    status: Literal["REVIEW_REQUIRED", "REVIEWED"] = "REVIEW_REQUIRED"
    replacement_counts: dict[str, int] = Field(default_factory=dict)
    semantic_provenance: dict[str, Any] = Field(default_factory=dict)


class DiagnosisCaseProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["OPS_DIAGNOSIS_CASE.v1"] = "OPS_DIAGNOSIS_CASE.v1"
    case_kind: Literal["DIAGNOSTIC_REFERENCE", "VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION", "NEGATIVE_EXPERIENCE"]
    problem_signature: dict[str, Any]
    environment: dict[str, Any]
    root_cause: dict[str, Any]
    actions: tuple[dict[str, Any], ...] = ()
    verification: dict[str, Any]
    applicability: tuple[str, ...] = ()
    contraindications: tuple[str, ...] = ()
    source: dict[str, Any]
    redaction: CaseRedaction | None = None

    @model_validator(mode="after")
    def validate_verified_effect(self):
        if self.case_kind in {"VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION"}:
            measured = self.verification
            if not (
                measured.get("controlled_verification_ref")
                and measured.get("comparability") == "COMPARABLE"
                and measured.get("guardrails_passed") is True
                and measured.get("measured_effect") == "IMPROVED"
                and self.actions and all(item.get("evidence_refs") for item in self.actions)
            ):
                raise ValueError("已验证案例必须关联真实执行、同口径改善和护栏证据")
            if self.case_kind == "VERIFIED_RESOLUTION" and measured.get("symptom_recovered") is not True:
                raise ValueError("已验证解决案例必须有目标症状恢复证据")
        if self.case_kind == "NEGATIVE_EXPERIENCE" and not (
            self.actions and all(item.get("evidence_refs") for item in self.actions)
            and self.verification.get("result") in {"NOT_ACHIEVED", "ADVERSE", "DEGRADED", "UNCHANGED"}
        ):
            raise ValueError("失败经验必须有可追溯的尝试动作及失败或无效结果")
        return self


class DiagnosisCaseDraft(BaseModel):
    """案例人工修订合同；来源与证据由服务端从基线版本继承。"""

    model_config = ConfigDict(extra="forbid")
    expected_asset_row_version: int = Field(ge=1)
    base_version_id: UUID
    display_name: str = Field(min_length=1, max_length=256)
    profile: DiagnosisCaseProfile


class DiagnosisClosureCommand(BaseModel):
    """由诊断操作者确认的结构化收口；证据只能引用当前会话已有 Artifact。"""

    model_config = ConfigDict(extra="forbid")
    included_turn_ids: tuple[UUID, ...] = Field(min_length=1, max_length=200)
    display_name: str = Field(min_length=1, max_length=256)
    symptoms: tuple[str, ...] = Field(min_length=1, max_length=64)
    root_cause_summary: str = Field(min_length=1, max_length=4000)
    actions: tuple[str, ...] = Field(default=(), max_length=16)
    action_evidence_ids: tuple[UUID, ...] = Field(default=(), max_length=64)
    verification_result: Literal[
        "NOT_VERIFIED", "RESOLVED", "IMPROVED", "NOT_ACHIEVED",
        "ADVERSE", "INCONCLUSIVE",
    ] = "NOT_VERIFIED"
    before_evidence_ids: tuple[UUID, ...] = Field(default=(), max_length=64)
    after_evidence_ids: tuple[UUID, ...] = Field(default=(), max_length=64)
    applicability: tuple[str, ...] = Field(default=(), max_length=32)
    contraindications: tuple[str, ...] = Field(default=(), max_length=32)


class DiagnosisClosure(BaseModel):
    """报告与案例共同读取的不可变诊断事实源。"""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["DIAGNOSIS_CLOSURE.v1"] = "DIAGNOSIS_CLOSURE.v1"
    conversation_id: UUID
    source_turn_ids: tuple[UUID, ...]
    target_id: UUID
    closed_by: str = Field(min_length=1, max_length=256)
    closed_at: datetime
    profile: DiagnosisCaseProfile


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    purpose: Literal[
        "DIAGNOSE", "EXPLAIN", "PLAN", "CHANGE", "VERIFY", "CATALOG"
    ] = "DIAGNOSE"
    problem_class: str | None = Field(default=None, max_length=128)
    components: tuple[str, ...] = Field(default=(), max_length=16)
    error_codes: tuple[str, ...] = Field(default=(), max_length=16)
    signal_names: tuple[str, ...] = Field(default=(), max_length=16)
    database_type: str | None = Field(default=None, max_length=32)
    database_major_version: str | None = Field(default=None, max_length=32)
    topology: str | None = Field(default=None, max_length=64)
    tag_ids: tuple[str, ...] = Field(default=(), max_length=64)
    knowledge_base_ids: tuple[UUID, ...] = Field(default=(), max_length=32)
    source_kinds: tuple[Literal["MANUAL", "DIAGNOSIS_CASE"], ...] = (
        "MANUAL", "DIAGNOSIS_CASE"
    )
    max_results: int = Field(default=8, ge=1, le=20)
    max_security_level: int = Field(default=3, ge=1, le=5)


class KnowledgeReviewCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)
    comment: str | None = Field(default=None, max_length=2000)
