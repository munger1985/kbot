"""问题驱动诊断的严格判断与不可变调查记录，不赋予执行权限。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.contracts.sql_diagnosis import EvidenceFieldReference, SqlRecommendation
from aiops_agent.contracts.diagnosis_state import ClaimRevision, InvestigationGoal
from aiops_agent.contracts.measurements import EvidenceProvenance, DerivedMeasurement
from aiops_agent.contracts.calculation import CalculationEvaluation
from aiops_agent.contracts.series_calculation import SeriesComputation
from aiops_agent.contracts.evidence import SourcePaginationObservation
from aiops_agent.contracts.implementation import ImplementationRunbook
from aiops_agent.contracts.sql_object_graph import SqlObjectGraph
from aiops_agent.contracts.sql_plan import SqlPlanDocument, SqlPlanAssessment
from platform_core.contracts.aiops.executor import ObjectAccessObservation, DiagnosticPaginationObservation


class InvestigationParent(BaseModel):
    """服务端从已重新授权的不可变上一轮读取，不能由模型填写。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    revision: int = Field(ge=1)
    target_id: str
    run_id: str
    artifact_id: str
    content_hash: str = Field(min_length=1)
    claims: tuple["InvestigationClaim", ...] = Field(max_length=16)
    historical_claims: tuple["InvestigationHistoricalClaim", ...] = Field(default=(), max_length=128)


class InvestigationHistoricalClaim(BaseModel):
    """跨多轮仍有效的判断以原 Artifact 和行标识定位，不混入本轮观测。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    content_hash: str
    run_id: str
    claim: "InvestigationClaim"


class InvestigationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^c[0-9]+$")
    kind: Literal["FACT", "CAUSE", "RISK", "PREDICTION", "EXCLUDED", "INTERPRETATION"]
    subject: str = Field(min_length=1, max_length=1000)
    statement: str = Field(min_length=1, max_length=3000)
    level: Literal["OBSERVED", "SUPPORTED", "HYPOTHESIS", "UNKNOWN"]
    applicable_time: str = Field(min_length=1, max_length=1000)
    evidence_fields: tuple[EvidenceFieldReference, ...] = Field(default=(), max_length=24)
    counterevidence_fields: tuple[EvidenceFieldReference, ...] = Field(default=(), max_length=24)
    reasoning: str = Field(min_length=1, max_length=3000)
    alternatives: tuple[str, ...] = Field(default=(), max_length=8)
    limitations: tuple[str, ...] = Field(default=(), max_length=12)
    revision_state: Literal["ACTIVE", "REFUTED", "SUPERSEDED"] = "ACTIVE"

    @model_validator(mode="after")
    def validate_strength(self):
        if self.level in {"OBSERVED", "SUPPORTED"} and not self.evidence_fields:
            raise ValueError("已观测或有支持的主张必须定位原始行与字段")
        if self.kind != "FACT" and self.level == "OBSERVED":
            raise ValueError("只有直接事实可以标记为已观测，原因和预测不是物理观测")
        if self.kind != "FACT" and not self.limitations:
            raise ValueError("解释、原因、风险、预测及排除必须说明适用限制")
        if self.counterevidence_fields and self.level == "OBSERVED":
            raise ValueError("存在反证的主张不能直接作为已确认事实")
        return self


class DiagnosisAnswerDraft(BaseModel):
    """所有普通诊断先提交逐项判断，再生成用户正文。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DBA_DIAGNOSTIC_ASSESSMENT.v1"] = "DBA_DIAGNOSTIC_ASSESSMENT.v1"
    claims: tuple[InvestigationClaim, ...] = Field(min_length=1, max_length=16)
    recommendations: tuple[SqlRecommendation, ...] = Field(default=(), max_length=12)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=12)
    goal_satisfied: bool
    goal_assessments: tuple[InvestigationGoal, ...] = Field(default=(), max_length=16)
    continues_investigation: bool = False
    claim_revisions: tuple[ClaimRevision, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_links(self):
        ids = {item.claim_id for item in self.claims}
        if len(ids) != len(self.claims):
            raise ValueError("主张 ID 不能重复")
        action_ids = {item.recommendation_id for item in self.recommendations}
        if len(action_ids) != len(self.recommendations):
            raise ValueError("建议 ID 不能重复")
        active_ids = {item.claim_id for item in self.claims if item.revision_state == "ACTIVE"}
        if self.claim_revisions and not self.continues_investigation:
            raise ValueError("修订必须显式声明继续上一调查")
        if any(item.current_claim_id not in active_ids for item in self.claim_revisions):
            raise ValueError("修订必须引用本轮有效判断")
        prior_ids = [(item.prior_artifact_id, item.prior_claim_id) for item in self.claim_revisions]
        if len(prior_ids) != len(set(prior_ids)):
            raise ValueError("同一旧判断不能在一次修订中被多次处置")
        goal_ids = [item.goal_id for item in self.goal_assessments]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("子目标 ID 不能重复")
        for goal in self.goal_assessments:
            if not set(goal.claim_refs) <= active_ids:
                raise ValueError("子目标只能引用本轮有效判断")
            if goal.status == "SATISFIED" and not any(
                item.claim_id in goal.claim_refs and item.level in {"OBSERVED", "SUPPORTED"}
                for item in self.claims
            ):
                raise ValueError("已完成子目标必须有有效事实或支持判断")
            if goal.status in {"BLOCKED", "NOT_APPLICABLE"} and not goal.explanation:
                raise ValueError("阻塞或不适用的子目标必须解释原因")
        if self.goal_assessments and self.goal_satisfied != all(
            item.status in {"SATISFIED", "NOT_APPLICABLE"} for item in self.goal_assessments
        ):
            raise ValueError("整体完成状态必须与全部子目标一致")
        for action in self.recommendations:
            if not set(action.conclusion_ids) <= active_ids:
                raise ValueError("建议只能引用当前有效主张")
            if action.intent in {"MITIGATE", "REPAIR"} and not action.requires_approval:
                raise ValueError("处置建议不能绕过独立审批")
        if self.goal_satisfied and not any(
            item.level in {"OBSERVED", "SUPPORTED"} and item.revision_state == "ACTIVE"
            for item in self.claims
        ):
            raise ValueError("仅有假设或未知时不能声明目标已回答")
        return self


class InvestigationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str
    artifact_id: str
    tool_id: str
    source_id: str
    captured_at: str
    measurement_semantics: str
    trust_level: Literal["SOURCE_VERIFIED", "USER_PROVIDED", "MODEL_INFERENCE"]
    row_count: int = Field(ge=0)
    truncated: bool
    followup_run_id: str | None = None
    provenance: EvidenceProvenance | None = None
    sql_plan: SqlPlanDocument | None = None
    object_access: tuple[ObjectAccessObservation, ...] = Field(default=(), max_length=1)
    pagination: DiagnosticPaginationObservation | None = None
    source_pagination: SourcePaginationObservation | None = None
    derivations: tuple[DerivedMeasurement, ...] = ()
    calculation_graphs: tuple[CalculationEvaluation, ...] = Field(default=(), max_length=128)
    series_computations: tuple[SeriesComputation, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def validate_series_budget(self):
        if self.source_pagination is not None and (
            self.pagination is not None or sum(self.source_pagination.returned_page_rows) != self.row_count
            or self.truncated != (self.source_pagination.stop_reason != "EXHAUSTED")
        ):
            raise ValueError("调查来源分页与原始观察不一致")
        if self.pagination is not None and (
            sum(self.pagination.returned_page_rows) != self.row_count
            or self.truncated != (self.pagination.stop_reason != "EXHAUSTED")
        ):
            raise ValueError("调查分页范围与原始观察不一致")
        if sum(len(item.samples) for item in self.series_computations) > 8192:
            raise ValueError("调查监控计算总输入超出预算")
        indexes = [item.row_index for item in self.series_computations]
        if len(indexes) != len(set(indexes)):
            raise ValueError("同一摘要行不能绑定多份监控计算")
        return self


class InvestigationAccessGap(BaseModel):
    """保留访问受限的原工具引用，不把历史拒绝转换成新的现场数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str = Field(min_length=1, max_length=512)
    source_id: str
    step_id: str
    code: str
    object_access: tuple[ObjectAccessObservation, ...] = Field(min_length=1, max_length=1)


class InvestigationRecord(BaseModel):
    """冻结于 Turn 结果 Artifact；正文、报告与案例共享相同修订。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["DBA_INVESTIGATION_RECORD.v1"] = "DBA_INVESTIGATION_RECORD.v1"
    record_id: str
    revision: int = Field(default=1, ge=1)
    previous_revision: str | None = None
    target_id: str
    run_id: str
    task_id: str
    original_request: str
    task_kind: str
    generated_at: str
    observations: tuple[InvestigationObservation, ...]
    claims: tuple[InvestigationClaim, ...]
    recommendations: tuple[SqlRecommendation, ...]
    goal_assessments: tuple[InvestigationGoal, ...]
    unresolved_questions: tuple[str, ...] = ()
    stop_reason: Literal["GOAL_ANSWERED", "EVIDENCE_EXHAUSTED", "BUDGET_EXHAUSTED", "PARTIAL_ANSWER", "SAFETY_STOP", "NEEDS_AUTHORIZATION"]
    execution_refs: tuple[str, ...] = ()
    verification_refs: tuple[str, ...] = ()
    parent: InvestigationParent | None = None
    claim_revisions: tuple[ClaimRevision, ...] = ()
    runbook: ImplementationRunbook | None = None
    sql_object_graph: SqlObjectGraph | None = None
    sql_subject_graphs: tuple[SqlObjectGraph, ...] = Field(default=(), max_length=20)
    sql_plan_assessments: tuple[SqlPlanAssessment, ...] = Field(default=(), max_length=80)
    access_gaps: tuple[InvestigationAccessGap, ...] = Field(default=(), max_length=128)

    def validate_access_gaps(self, gaps):
        expected = tuple(InvestigationAccessGap(
            evidence_ref=gap.evidence_ref, source_id=gap.source_id, step_id=gap.step_id,
            code=gap.code, object_access=gap.object_access,
        ) for gap in gaps if gap.object_access)
        if self.access_gaps != expected:
            raise ValueError("调查访问缺口与原工具引用或历史观察不一致")

    @property
    def sql_graphs(self) -> tuple[SqlObjectGraph, ...]:
        return (self.sql_object_graph,) if self.sql_object_graph is not None else self.sql_subject_graphs

    @model_validator(mode="after")
    def validate_record(self):
        draft = DiagnosisAnswerDraft(
            claims=self.claims, recommendations=self.recommendations,
            unresolved_questions=self.unresolved_questions,
            goal_satisfied=self.stop_reason == "GOAL_ANSWERED",
            goal_assessments=self.goal_assessments,
            continues_investigation=self.parent is not None,
            claim_revisions=self.claim_revisions,
        )
        refs = {item.evidence_ref for item in self.observations}
        if len(refs) != len(self.observations):
            raise ValueError("调查观测引用不能重复")
        for claim in draft.claims:
            if any(item.evidence_ref not in refs for item in (*claim.evidence_fields, *claim.counterevidence_fields)):
                raise ValueError("主张必须引用本记录冻结的观测")
        if self.sql_object_graph is not None and self.sql_subject_graphs:
            raise ValueError("单主体根图与多主体图不能同时存在")
        from aiops_agent.playbooks import canonical_hash
        subjects = [canonical_hash(graph.subject) for graph in self.sql_graphs]
        if len(subjects) != len(set(subjects)):
            raise ValueError("同一完整 SQL 主体不能重复冻结对象图")
        if len({graph.database_type for graph in self.sql_graphs}) > 1:
            raise ValueError("同一 Target 的对象图不能混合数据库引擎")
        for graph in self.sql_graphs:
            for edge in graph.relationships:
                if edge.anchor.evidence_ref not in refs or edge.related.evidence_ref not in refs:
                    raise ValueError("对象关系必须引用本记录冻结的观测")
        for plan in self.sql_plan_assessments:
            if plan.evidence_ref not in refs or (plan.anchor_ref is not None and plan.anchor_ref not in refs) or not any(
                graph.database_type == plan.database_type and graph.subject == plan.subject for graph in self.sql_graphs
            ):
                raise ValueError("计划核验必须绑定本记录的材料与 SQL 主体")
        claim_ids = {item.claim_id for item in self.claims}
        if any(not set(goal.claim_refs) <= claim_ids for goal in self.goal_assessments):
            raise ValueError("目标完成状态引用了不存在的主张")
        if (self.revision == 1) != (self.previous_revision is None):
            raise ValueError("调查修订必须关联前一不可变版本")
        if self.parent is not None:
            if self.target_id != self.parent.target_id or self.record_id != self.parent.record_id:
                raise ValueError("调查修订不能改变目标或调查身份")
            if self.revision != self.parent.revision + 1 or self.previous_revision != self.parent.artifact_id:
                raise ValueError("调查修订必须连续关联服务端冻结的上一版本")
            previous_claims = {(self.parent.artifact_id, item.claim_id): item for item in self.parent.claims if item.revision_state == "ACTIVE"}
            previous_claims.update({
                (item.artifact_id, item.claim.claim_id): item.claim for item in self.parent.historical_claims
                if item.claim.revision_state == "ACTIVE"
            })
            current_claims = {item.claim_id: item for item in self.claims}
            changed_keys = set()
            for change in self.claim_revisions:
                key = (change.prior_artifact_id or self.parent.artifact_id, change.prior_claim_id)
                if key in changed_keys:
                    raise ValueError("同一来源判断不能重复修订")
                changed_keys.add(key)
                old, new = previous_claims.get(key), current_claims[change.current_claim_id]
                if old is None or old.subject != new.subject or old.applicable_time != new.applicable_time:
                    raise ValueError("修订不能引用缺失、已撤回、不同对象或不同时间阶段的判断")
                if change.relation == "REFUTES" and not new.evidence_fields:
                    raise ValueError("驳回历史判断必须有本轮反证引用")
        elif self.revision != 1 or self.claim_revisions:
            raise ValueError("缺少授权来源不能生成跨轮修订")
        return self

    def validate_observations(self, evidence):
        expected = {item.evidence_ref: item for item in self.observations}
        actual = {item.evidence_ref: item for item in evidence}
        if len(actual) != len(evidence) or set(expected) != set(actual):
            raise ValueError("调查记录与结果证据集合不一致")
        for ref, observation in expected.items():
            fact = actual[ref]
            for key in InvestigationObservation.model_fields:
                if str(getattr(observation, key)) != str(getattr(fact, key)):
                    raise ValueError("调查记录与原观测的身份、时间、口径或信任标记不一致")
        from aiops_agent.domain.evidence.sql_object_graph import replay_sql_object_graph
        for graph in self.sql_graphs:
            replay_sql_object_graph(graph, evidence)
        from aiops_agent.domain.evidence.sql_plan_matching import replay_sql_plan_assessments
        replay_sql_plan_assessments(self.sql_plan_assessments, self.sql_graphs, evidence)
