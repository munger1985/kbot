"""步骤 7 诊断证据、模型角色和确定性 Gate Handler。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.contracts.diagnosis import (
    DiagnosisReportDraft,
    DiagnosisEvidenceCollection,
    DiagnosisRoundAssessment,
    DiagnosisRoundDraft,
    DiagnosisScope,
    EvidenceIndex,
    GroundingVerification,
    HypothesisAssessment,
    KnowledgeCitationPack,
    RootCauseAssessment,
    SolutionDraft,
    ValidatedEvidencePlan,
)
from aiops_agent.domain.diagnosis import (
    EvidenceRequestBudget,
    assess_root_cause,
    normalize_evidence_artifacts,
    validate_evidence_requests,
)
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.orchestration.diagnosis import DiagnosisPromptRegistry
from platform_core.security import create_service_auth_context

from .handlers import TaskExecutionContext


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        UTC
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _artifact(context: TaskExecutionContext, schema: str) -> dict[str, Any]:
    return next(
        item["payload"]
        for item in reversed(context.input_artifacts)
        if item["schema_version"] == schema
    )


def _artifacts(
    context: TaskExecutionContext, schema: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        item["payload"]
        for item in context.input_artifacts
        if item["schema_version"] == schema
    )


def _artifact_from_task(
    context: TaskExecutionContext, schema: str, task_key: str
) -> dict[str, Any]:
    return next(
        item["payload"]
        for item in context.input_artifacts
        if item["schema_version"] == schema
        and item.get("provenance", {}).get("task_key") == task_key
    )


def _round_no(task_key: str) -> int:
    match = re.search(r"diagnosis:r(\d+):", task_key)
    if not match:
        raise ValueError("诊断轮次 Task Key 无效")
    return int(match.group(1))


class DiagnosisScopeHandler:
    async def execute(self, context: TaskExecutionContext) -> DiagnosisScope:
        plan = context.plan_snapshot
        target = plan["target"]
        diagnosis = plan["diagnosis"]
        window = diagnosis["window"]
        return DiagnosisScope(
            run_id=context.run_id,
            target_id=context.target_id,
            agent_id=context.agent_id,
            trigger_type=context.trigger_type,
            symptom_codes=tuple(diagnosis.get("symptom_codes", ())),
            user_question_summary=diagnosis.get("question_summary"),
            window_start=_parse_time(window["start"]),
            window_end=_parse_time(window["end"]),
            db_type=target["db_type"],
            db_version=target.get("version_code") or "UNKNOWN",
            environment=target["environment"],
            target_capabilities=tuple(
                diagnosis.get("target_capabilities", ())
            ),
            allowed_collection_ids=tuple(
                diagnosis.get("allowed_collection_ids", ())
            ),
            policy_snapshot_hash=diagnosis["policy_snapshot_hash"],
            security_level=int(target["security_level"]),
            budget_snapshot=dict(diagnosis["budget"]),
        )


class BuildEvidenceIndexHandler:
    async def execute(self, context: TaskExecutionContext) -> EvidenceIndex:
        source_schemas = {
            "OBSERVATION_SET.v1",
            "DATABASE_DIAGNOSTIC_RESULT.v1",
            "DIAGNOSIS_EVIDENCE_COLLECTION.v1",
            "EVIDENCE_INDEX.v1",
            "KNOWLEDGE_CITATION_PACK.v1",
        }
        sources = tuple(
            item
            for item in context.input_artifacts
            if item["schema_version"] in source_schemas
        )
        return normalize_evidence_artifacts(
            sources,
            target_id=context.target_id,
            max_facts=int(
                context.plan_snapshot["diagnosis"]["budget"][
                    "max_evidence_facts"
                ]
            ),
        )


class KnowledgeCitationHandler:
    def __init__(self, *, knowledge_client, caller_service: str):
        self._client = knowledge_client
        self._caller = caller_service

    async def execute(
        self, context: TaskExecutionContext
    ) -> KnowledgeCitationPack:
        diagnosis = context.plan_snapshot["diagnosis"]
        collection_ids = tuple(
            UUID(item) for item in diagnosis["allowed_collection_ids"]
        )
        query = diagnosis["question_summary"] or "数据库故障诊断"
        if not collection_ids:
            return KnowledgeCitationPack(
                query=query,
                gap_code="KNOWLEDGE_SCOPE_EMPTY",
            )
        auth_context = create_service_auth_context(
            caller_service=self._caller,
            request_id=context.task_id,
            trace_id=context.trace_id,
        )
        domain_id = int(context.plan_snapshot["target"]["domain_id"])
        try:
            discovery = await self._client.discover(
                query=query,
                collection_ids=collection_ids,
                domain_id=domain_id,
                agent_id=context.agent_id,
                auth_context=auth_context,
                max_security_level=min(
                    3,
                    int(
                        context.plan_snapshot["target"][
                            "security_level"
                        ]
                    ),
                ),
                per_collection_limit=8,
            )
            candidates = discovery.get("candidates", [])
            if not candidates:
                return KnowledgeCitationPack(query=query)
            evidence = await self._client.retrieve_evidence(
                query=query,
                candidates=candidates[:16],
                domain_id=domain_id,
                agent_id=context.agent_id,
                auth_context=auth_context,
                max_security_level=min(
                    3,
                    int(
                        context.plan_snapshot["target"][
                            "security_level"
                        ]
                    ),
                ),
                max_evidence=8,
                context_limit=3,
            )
            return KnowledgeCitationPack(
                query=query,
                citations=tuple(evidence.get("citations", ())),
            )
        except Exception:
            return KnowledgeCitationPack(
                query=query,
                gap_code="KNOWLEDGE_CORE_UNAVAILABLE",
            )


class DiagnosisRoundDraftHandler:
    def __init__(self, *, model_client, prompts: DiagnosisPromptRegistry):
        self._model = model_client
        self._prompts = prompts

    async def execute(
        self, context: TaskExecutionContext
    ) -> DiagnosisRoundDraft:
        round_no = _round_no(context.task_key)
        evidence = EvidenceIndex.model_validate(
            _artifact_from_task(
                context,
                "EVIDENCE_INDEX.v1",
                f"diagnosis:evidence:r{round_no - 1}",
            )
        )
        prior_assessments = _artifacts(
            context, "DIAGNOSIS_ROUND_ASSESSMENT.v1"
        )
        if prior_assessments:
            prior = DiagnosisRoundAssessment.model_validate(
                max(
                    prior_assessments,
                    key=lambda item: int(item["round_no"]),
                )
            )
            if prior.recommended_next_step != "CONTINUE":
                return DiagnosisRoundDraft(
                    round_no=round_no,
                    stop_recommendation="FINALIZE",
                    stop_reason="上一轮已满足终止条件",
                )
        diagnosis = context.plan_snapshot["diagnosis"]
        if not diagnosis["model"]["enabled"]:
            return self._fallback(round_no, "MODEL_DISABLED")
        prompt = self._prompts.resolve("round_draft")
        tool_cards = tuple(
            {
                "tool_id": item["tool_id"],
                "parameters": {
                    parameter["name"]: {
                        key: value
                        for key, value in parameter.items()
                        if key not in {"default"}
                    }
                    for parameter in item.get(
                        "parameter_definitions", []
                    )
                },
                "cost_level": item["cost_level"],
            }
            for item in context.plan_snapshot[
                "database_diagnostics"
            ].get("tools", [])
        )
        input_payload = {
            "round_no": round_no,
            "scope": _artifact(context, "DIAGNOSIS_SCOPE.v1"),
            "facts": [
                {
                    "fact_id": fact.fact_id,
                    "fact_type": fact.metric_or_fact_type,
                    "summary": fact.fact_summary,
                    "quality_flags": fact.quality_flags,
                    "source_group_id": fact.source_group_id,
                }
                for fact in evidence.facts
            ],
            "gaps": evidence.gaps,
            "tool_cards": tool_cards,
            "prior_assessment": (
                max(
                    prior_assessments,
                    key=lambda item: int(item["round_no"]),
                )
                if prior_assessments
                else None
            ),
        }
        try:
            result = await self._model.generate_structured(
                purpose="diagnosis.round_draft",
                output_model=DiagnosisRoundDraft,
                model_snapshot=diagnosis["model"],
                prompt_ref={**prompt.ref(), "content": prompt.content},
                input_payload=input_payload,
                max_output_tokens=diagnosis["budget"][
                    "max_output_tokens_per_call"
                ],
                deadline=(
                    _parse_time(context.deadline_at)
                    if context.deadline_at
                    else None
                ),
                idempotency_key=f"{context.task_id}:{context.attempt}",
            )
            draft = DiagnosisRoundDraft.model_validate(
                result.output.model_dump()
            )
            if draft.round_no != round_no:
                raise ValueError("模型返回的诊断轮次不匹配")
            self._validate_fact_refs(draft, evidence)
            return draft.model_copy(
                update={"invocation_receipt": result.receipt}
            )
        except (AIOpsModelError, ValueError):
            return self._fallback(round_no, "MODEL_OUTPUT_INVALID")

    @staticmethod
    def _fallback(round_no: int, code: str) -> DiagnosisRoundDraft:
        return DiagnosisRoundDraft(
            round_no=round_no,
            stop_recommendation="INCONCLUSIVE",
            stop_reason="结构化诊断模型本次不可用",
            model_gap_code=code,
        )

    @staticmethod
    def _validate_fact_refs(
        draft: DiagnosisRoundDraft, evidence: EvidenceIndex
    ) -> None:
        valid = {item.fact_id for item in evidence.facts}
        for hypothesis in draft.hypotheses:
            refs = set(hypothesis.supporting_fact_refs) | set(
                hypothesis.counter_fact_refs
            )
            if not refs <= valid:
                raise ValueError("模型引用了不存在的 FactRef")


class EvidenceRequestValidatorHandler:
    def __init__(self, *, registry: DiagnosticRegistry):
        self._registry = registry

    async def execute(
        self, context: TaskExecutionContext
    ) -> ValidatedEvidencePlan:
        draft = DiagnosisRoundDraft.model_validate(
            _artifact(context, "DIAGNOSIS_ROUND_DRAFT.v1")
        )
        budget = context.plan_snapshot["diagnosis"]["budget"]
        previous_plans = [
            ValidatedEvidencePlan.model_validate(item)
            for item in _artifacts(
                context, "VALIDATED_EVIDENCE_PLAN.v1"
            )
            if int(item["round_no"]) < draft.round_no
        ]
        used = sum(len(item.accepted) for item in previous_plans)
        prior_fingerprints = frozenset(
            request.request_fingerprint
            for item in previous_plans
            for request in item.accepted
        )
        return validate_evidence_requests(
            draft,
            database_snapshot=context.plan_snapshot[
                "database_diagnostics"
            ],
            registry=self._registry,
            budget=EvidenceRequestBudget(
                remaining_tool_calls=max(
                    0, int(budget["max_tool_calls"]) - used
                ),
                prior_fingerprints=prior_fingerprints,
            ),
        )


class DiagnosisEvidenceCollectHandler:
    """按已验证计划调用目录工具，不接受模型提供的 SQL 或版本。"""

    def __init__(self, *, database_handler):
        self._database_handler = database_handler

    async def execute(
        self, context: TaskExecutionContext
    ) -> DiagnosisEvidenceCollection:
        plan = ValidatedEvidencePlan.model_validate(
            _artifact(context, "VALIDATED_EVIDENCE_PLAN.v1")
        )
        results = []
        for request in plan.accepted:
            plan_snapshot = dict(context.plan_snapshot)
            database = dict(plan_snapshot["database_diagnostics"])
            tools = []
            for frozen in database["tools"]:
                copied = dict(frozen)
                if copied["tool_id"] == request.tool_id:
                    copied["parameters"] = dict(request.parameters)
                tools.append(copied)
            database["tools"] = tools
            plan_snapshot["database_diagnostics"] = database
            child_context = replace(
                context,
                task_key=f"diagnostic:{request.tool_id}",
                plan_snapshot=plan_snapshot,
            )
            results.append(
                await self._database_handler.execute(child_context)
            )
        return DiagnosisEvidenceCollection(
            round_no=plan.round_no,
            results=tuple(results),
        )


class DiagnosisRoundAssessmentHandler:
    def __init__(self, *, model_client, prompts: DiagnosisPromptRegistry):
        self._model = model_client
        self._prompts = prompts

    async def execute(
        self, context: TaskExecutionContext
    ) -> DiagnosisRoundAssessment:
        round_no = _round_no(context.task_key)
        evidence = EvidenceIndex.model_validate(
            _artifact_from_task(
                context,
                "EVIDENCE_INDEX.v1",
                f"diagnosis:evidence:r{round_no}",
            )
        )
        draft = DiagnosisRoundDraft.model_validate(
            _artifact(context, "DIAGNOSIS_ROUND_DRAFT.v1")
        )
        plan = ValidatedEvidencePlan.model_validate(
            _artifact(context, "VALIDATED_EVIDENCE_PLAN.v1")
        )
        prior_assessments = _artifacts(
            context, "DIAGNOSIS_ROUND_ASSESSMENT.v1"
        )
        evidence_versions = _artifacts(context, "EVIDENCE_INDEX.v1")
        if (
            prior_assessments
            and len(evidence_versions) >= 2
            and len(
                {item["index_hash"] for item in evidence_versions}
            )
            == 1
        ):
            return DiagnosisRoundAssessment.model_validate(
                max(
                    prior_assessments,
                    key=lambda item: int(item["round_no"]),
                )
            ).model_copy(
                update={
                    "round_no": round_no,
                    "recommended_next_step": "STOP_INCONCLUSIVE",
                    "rationale_summary": (
                        "本轮没有产生新的有效证据，已按无进展策略终止"
                    ),
                }
            )
        if prior_assessments and not draft.hypotheses:
            return DiagnosisRoundAssessment.model_validate(
                max(
                    prior_assessments,
                    key=lambda item: int(item["round_no"]),
                )
            ).model_copy(
                update={
                    "round_no": round_no,
                    "recommended_next_step": "FINALIZE",
                }
            )
        diagnosis = context.plan_snapshot["diagnosis"]
        if not diagnosis["model"]["enabled"] or not draft.hypotheses:
            return self._fallback(
                draft, draft.model_gap_code, round_no=round_no
            )
        prompt = self._prompts.resolve("round_assess")
        input_payload = {
            "round_no": round_no,
            "facts": [
                {
                    "fact_id": item.fact_id,
                    "summary": item.fact_summary,
                    "source_group_id": item.source_group_id,
                    "quality_flags": item.quality_flags,
                }
                for item in evidence.facts
            ],
            "hypotheses": [
                item.model_dump(mode="json") for item in draft.hypotheses
            ],
            "evidence_plan": plan.model_dump(mode="json"),
            "prior_assessment": (
                max(
                    prior_assessments,
                    key=lambda item: int(item["round_no"]),
                )
                if prior_assessments
                else None
            ),
        }
        try:
            result = await self._model.generate_structured(
                purpose="diagnosis.round_assess",
                output_model=DiagnosisRoundAssessment,
                model_snapshot=diagnosis["model"],
                prompt_ref={**prompt.ref(), "content": prompt.content},
                input_payload=input_payload,
                max_output_tokens=diagnosis["budget"][
                    "max_output_tokens_per_call"
                ],
                deadline=(
                    _parse_time(context.deadline_at)
                    if context.deadline_at
                    else None
                ),
                idempotency_key=f"{context.task_id}:{context.attempt}",
            )
            assessment = DiagnosisRoundAssessment.model_validate(
                result.output.model_dump()
            )
            if assessment.round_no != round_no:
                raise ValueError("模型返回的评估轮次不匹配")
            self._validate_refs(assessment, evidence, draft)
            if (
                round_no >= int(diagnosis["budget"]["max_rounds"])
                and assessment.recommended_next_step == "CONTINUE"
            ):
                assessment = assessment.model_copy(
                    update={
                        "recommended_next_step": "STOP_INCONCLUSIVE"
                    }
                )
            return assessment.model_copy(
                update={"invocation_receipt": result.receipt}
            )
        except (AIOpsModelError, ValueError):
            return self._fallback(
                draft, "MODEL_OUTPUT_INVALID", round_no=round_no
            )

    @staticmethod
    def _fallback(
        draft: DiagnosisRoundDraft,
        code: str | None,
        *,
        round_no: int,
    ) -> DiagnosisRoundAssessment:
        return DiagnosisRoundAssessment(
            round_no=round_no,
            hypothesis_assessments=tuple(
                HypothesisAssessment(
                    hypothesis_key=item.hypothesis_key,
                    status="UNTESTED",
                    causal_role=item.causal_role,
                    supporting_fact_refs=item.supporting_fact_refs,
                    counter_fact_refs=item.counter_fact_refs,
                    remaining_gaps=item.unresolved_questions,
                )
                for item in draft.hypotheses
            ),
            suggested_root_cause_level="INCONCLUSIVE",
            recommended_next_step="STOP_INCONCLUSIVE",
            rationale_summary="模型不可用，无法完成语义证据评估",
            model_gap_code=code or "MODEL_DISABLED",
        )

    @staticmethod
    def _validate_refs(
        assessment: DiagnosisRoundAssessment,
        evidence: EvidenceIndex,
        draft: DiagnosisRoundDraft,
    ) -> None:
        valid_facts = {item.fact_id for item in evidence.facts}
        valid_hypotheses = {
            item.hypothesis_key for item in draft.hypotheses
        }
        for item in assessment.hypothesis_assessments:
            if item.hypothesis_key not in valid_hypotheses:
                raise ValueError("模型评估了不存在的假设")
            refs = set(item.supporting_fact_refs) | set(
                item.counter_fact_refs
            )
            refs.update(
                ref for result in item.test_results for ref in result.fact_refs
            )
            if not refs <= valid_facts:
                raise ValueError("模型评估引用了不存在的 FactRef")


class RootCauseAssessmentHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> RootCauseAssessment:
        return assess_root_cause(
            target_id=context.target_id,
            evidence=EvidenceIndex.model_validate(
                _artifact(context, "EVIDENCE_INDEX.v1")
            ),
            assessment=DiagnosisRoundAssessment.model_validate(
                _artifact(
                    context, "DIAGNOSIS_ROUND_ASSESSMENT.v1"
                )
            ),
        )


class GroundingVerificationHandler:
    def __init__(self, *, model_client, prompts: DiagnosisPromptRegistry):
        self._model = model_client
        self._prompts = prompts

    async def execute(
        self, context: TaskExecutionContext
    ) -> GroundingVerification:
        evidence = EvidenceIndex.model_validate(
            _artifact(context, "EVIDENCE_INDEX.v1")
        )
        root = RootCauseAssessment.model_validate(
            _artifact(context, "ROOT_CAUSE_ASSESSMENT.v1")
        )
        valid = {item.fact_id for item in evidence.facts}
        invalid = tuple(
            ref
            for ref in (
                *root.supporting_fact_refs,
                *root.counter_fact_refs,
            )
            if ref not in valid
        )
        issues = []
        if invalid:
            issues.append("根因结论包含不存在的 FactRef")
        if (
            root.effective_level in {"CONFIRMED", "PROBABLE"}
            and not root.supporting_fact_refs
        ):
            issues.append("高等级根因没有当前状态事实")
        deterministic = GroundingVerification(
            status="BLOCK" if issues else "PASS",
            invalid_fact_refs=invalid,
            issues=tuple(issues),
        )
        diagnosis = context.plan_snapshot["diagnosis"]
        if issues or not diagnosis["model"]["enabled"]:
            return deterministic
        prompt = self._prompts.resolve("grounding_verify")
        input_payload = {
            "root_cause": root.model_dump(mode="json"),
            "facts": [
                {
                    "fact_id": item.fact_id,
                    "summary": item.fact_summary,
                    "quality_flags": item.quality_flags,
                }
                for item in evidence.facts
            ],
        }
        try:
            result = await self._model.generate_structured(
                purpose="diagnosis.grounding_verify",
                output_model=GroundingVerification,
                model_snapshot=diagnosis["model"],
                prompt_ref={**prompt.ref(), "content": prompt.content},
                input_payload=input_payload,
                max_output_tokens=min(
                    2048,
                    diagnosis["budget"]["max_output_tokens_per_call"],
                ),
                deadline=(
                    _parse_time(context.deadline_at)
                    if context.deadline_at
                    else None
                ),
                idempotency_key=f"{context.task_id}:{context.attempt}",
            )
            verification = GroundingVerification.model_validate(
                result.output.model_dump()
            )
            if any(
                ref not in valid for ref in verification.invalid_fact_refs
            ):
                raise ValueError("Verifier 返回未知 FactRef")
            return verification.model_copy(
                update={"invocation_receipt": result.receipt}
            )
        except (AIOpsModelError, ValueError):
            return GroundingVerification(
                status="REVISE",
                issues=("语义引用检查模型本次不可用",),
                model_gap_code="MODEL_OUTPUT_INVALID",
            )


class SolutionDraftHandler:
    async def execute(self, context: TaskExecutionContext) -> SolutionDraft:
        root = RootCauseAssessment.model_validate(
            _artifact(context, "ROOT_CAUSE_ASSESSMENT.v1")
        )
        if root.effective_level in {"CONFIRMED", "PROBABLE"}:
            return SolutionDraft(
                immediate_mitigations=(
                    "依据已验证根因选择低风险缓解措施，并在执行前复核目标范围",
                ),
                long_term_remediations=(
                    "将根因机制转化为容量、配置或流程层面的长期治理项",
                ),
                verification_plan=(
                    "按相同指标和数据库诊断工具复测并生成前后对比",
                ),
                limitations=root.unresolved_gaps,
            )
        return SolutionDraft(
            immediate_mitigations=(
                "保持只读观测并补充能够区分主要假设的证据",
            ),
            verification_plan=(
                "补齐证据后重新执行诊断，当前不建议自动变更",
            ),
            limitations=(
                *root.unresolved_gaps,
                *root.downgrade_reasons,
            ),
        )


class DiagnosisReportHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> DiagnosisReportDraft:
        evidence = EvidenceIndex.model_validate(
            _artifact(context, "EVIDENCE_INDEX.v1")
        )
        assessment = DiagnosisRoundAssessment.model_validate(
            _artifact(context, "DIAGNOSIS_ROUND_ASSESSMENT.v1")
        )
        root = RootCauseAssessment.model_validate(
            _artifact(context, "ROOT_CAUSE_ASSESSMENT.v1")
        )
        verification = GroundingVerification.model_validate(
            _artifact(context, "GROUNDING_VERIFICATION.v1")
        )
        solution = SolutionDraft.model_validate(
            _artifact(context, "SOLUTION_DRAFT.v1")
        )
        model_gaps = tuple(
            code
            for code in (
                assessment.model_gap_code,
                verification.model_gap_code,
            )
            if code is not None
        )
        receipts = tuple(
            item.model_dump(mode="json")
            for item in (
                assessment.invocation_receipt,
                verification.invocation_receipt,
            )
            if item is not None
        )
        receipt_hashes = tuple(_hash(item) for item in receipts)
        status = (
            "DEGRADED"
            if model_gaps
            else "PARTIAL"
            if evidence.gaps or root.effective_level == "INCONCLUSIVE"
            else "READY"
        )
        return DiagnosisReportDraft(
            target_id=context.target_id,
            status=status,
            root_cause=root,
            facts=evidence.facts,
            hypotheses=assessment.hypothesis_assessments,
            solution=solution,
            gaps=(
                *model_gaps,
                *(str(item.get("code", "EVIDENCE_GAP")) for item in evidence.gaps),
            ),
            verification=verification,
            model_receipt_hashes=receipt_hashes,
            provenance={
                "diagnosis_kernel": "step7.v1",
                "evidence_index_hash": evidence.index_hash,
                "deterministic_grade_policy": True,
                "mutation_enabled": False,
            },
        )
