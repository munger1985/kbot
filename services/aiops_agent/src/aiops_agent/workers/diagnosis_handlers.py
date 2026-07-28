"""步骤 7 诊断证据、模型角色和确定性 Gate Handler。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from loguru import logger

from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.contracts.diagnosis import (
    DiagnosisReportDraft,
    DiagnosisEvidenceCollection,
    DiagnosisRoundAssessment,
    DiagnosisRoundDraft,
    DiagnosisScope,
    DirectQuestionAnswer,
    EvidenceIndex,
    GroundingVerification,
    HypothesisAssessment,
    KnowledgeCitationPack,
    RootCauseAssessment,
    SolutionDraft,
    ValidatedEvidencePlan,
)
from aiops_agent.contracts.hitl import (
    HitlOutcome,
    InputSuspension,
    ManualSqlRequest,
)
from aiops_agent.domain.diagnosis import (
    EvidenceRequestBudget,
    assess_root_cause,
    normalize_evidence_artifacts,
    validate_evidence_requests,
)
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.orchestration.hitl import ManualSqlBuilder
from aiops_agent.orchestration.diagnosis import DiagnosisPromptRegistry
from platform_core.security import create_service_auth_context
from platform_core.identity import uuid7

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
            "HITL_OUTCOME.v1",
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
            # Collection 是可选的经验增强源；空范围表示仅依赖当前状态证据和模型。
            return KnowledgeCitationPack(query=query)
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
                "returns": [
                    {
                        "name": column["name"],
                        "type": column["logical_type"],
                    }
                    for column in item.get("output_columns", [])
                ],
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
            request_payload = input_payload
            for repair_count in range(2):
                result = await self._model.generate_structured(
                    purpose=(
                        "diagnosis.round_draft"
                        if repair_count == 0
                        else "diagnosis.round_draft.repair"
                    ),
                    output_model=DiagnosisRoundDraft,
                    model_snapshot=diagnosis["model"],
                    prompt_ref={
                        **prompt.ref(),
                        "content": prompt.content,
                    },
                    input_payload=request_payload,
                    max_output_tokens=diagnosis["budget"][
                        "max_output_tokens_per_call"
                    ],
                    deadline=(
                        _parse_time(context.deadline_at)
                        if context.deadline_at
                        else None
                    ),
                    idempotency_key=(
                        f"{context.task_id}:{context.attempt}:"
                        f"{repair_count}"
                    ),
                )
                draft = DiagnosisRoundDraft.model_validate(
                    result.output.model_dump()
                )
                try:
                    if draft.round_no != round_no:
                        raise ValueError("模型返回的诊断轮次不匹配")
                    self._validate_fact_refs(draft, evidence)
                    self._validate_tool_refs(draft, tool_cards)
                except ValueError as exc:
                    if repair_count > 0:
                        raise
                    logger.warning(
                        "诊断假设输出需要修复：task_id={} round_no={} "
                        "error={}",
                        context.task_id,
                        round_no,
                        str(exc),
                    )
                    request_payload = {
                        **input_payload,
                        "validation_feedback": {
                            "error": str(exc),
                            "instruction": (
                                "重新生成完整对象；tool_id 只能逐字选择 "
                                "allowed_tool_ids 中的值"
                            ),
                            "allowed_tool_ids": [
                                item["tool_id"] for item in tool_cards
                            ],
                        },
                    }
                    continue
                receipt = result.receipt.model_copy(
                    update={"repair_count": repair_count}
                )
                return draft.model_copy(
                    update={"invocation_receipt": receipt}
                )
            raise ValueError("诊断假设修复后仍不满足约束")
        except AIOpsModelError as exc:
            logger.warning(
                "诊断假设生成降级：task_id={} round_no={} code={} error={}",
                context.task_id,
                round_no,
                exc.code,
                str(exc),
            )
            return self._fallback(round_no, exc.code)
        except ValueError as exc:
            logger.warning(
                "诊断假设业务校验失败：task_id={} round_no={} error={}",
                context.task_id,
                round_no,
                str(exc),
            )
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

    @staticmethod
    def _validate_tool_refs(
        draft: DiagnosisRoundDraft,
        tool_cards: tuple[dict[str, Any], ...],
    ) -> None:
        allowed = {item["tool_id"] for item in tool_cards}
        invalid = sorted(
            {
                request.tool_id
                for request in draft.evidence_requests
                if request.tool_id not in allowed
            }
        )
        if invalid:
            raise ValueError(
                "模型请求了未登记工具："
                f"{', '.join(invalid)}；允许工具："
                f"{', '.join(sorted(allowed)) or '无'}"
            )


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
        evidence_task_key = (
            "diagnosis:evidence:final"
            if context.task_key.endswith(":assess-manual")
            else f"diagnosis:evidence:r{round_no}"
        )
        evidence = EvidenceIndex.model_validate(
            _artifact_from_task(
                context,
                "EVIDENCE_INDEX.v1",
                evidence_task_key,
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
        except AIOpsModelError as exc:
            logger.warning(
                "诊断轮次评估降级：task_id={} round_no={} code={} error={}",
                context.task_id,
                round_no,
                exc.code,
                str(exc),
            )
            return self._fallback(
                draft, exc.code, round_no=round_no
            )
        except ValueError as exc:
            logger.warning(
                "诊断轮次评估业务校验失败：task_id={} round_no={} error={}",
                context.task_id,
                round_no,
                str(exc),
            )
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


class InteractiveDiagnosisHandler:
    """只在聊天且自动诊断无法取证时请求用户手工执行目录 SQL。"""

    _CONNECTIVITY_GAPS = {
        "TARGET_UNREACHABLE",
        "TIMEOUT",
        "SECRET_UNAVAILABLE",
        "SECRET_NOT_CONFIGURED",
        "DATABASE_ACCESS_DISABLED",
        "ENDPOINT_NOT_CONFIGURED",
        "DIAGNOSTIC_ACCESS_DENIED",
        "DIAGNOSTIC_POLICY_DENIED",
        "DIAGNOSTIC_SECRET_MISSING",
        "TARGET_ENDPOINT_MISSING",
    }

    def __init__(self, *, registry: DiagnosticRegistry):
        self._builder = ManualSqlBuilder(registry)

    async def execute(
        self, context: TaskExecutionContext
    ) -> HitlOutcome | InputSuspension:
        assessment = DiagnosisRoundAssessment.model_validate(
            _artifact(context, "DIAGNOSIS_ROUND_ASSESSMENT.v1")
        )
        evidence = EvidenceIndex.model_validate(
            _artifact(context, "EVIDENCE_INDEX.v1")
        )
        direct_answer = DiagnosisReportHandler._direct_answer(
            question=context.plan_snapshot.get("diagnosis", {}).get(
                "question_summary"
            ),
            evidence=evidence,
        )
        if (
            context.trigger_type != "CHAT"
            or assessment.recommended_next_step != "STOP_INCONCLUSIVE"
            or (
                direct_answer is not None
                and direct_answer.status == "ANSWERED"
            )
            or not self._requires_manual_input(evidence)
        ):
            return HitlOutcome(status="NOT_REQUIRED")

        database = context.plan_snapshot["database_diagnostics"]
        frozen_tools = tuple(database.get("tools", ()))
        plans = tuple(
            ValidatedEvidencePlan.model_validate(item)
            for item in _artifacts(
                context, "VALIDATED_EVIDENCE_PLAN.v1"
            )
        )
        selected = self._select_tools(frozen_tools, plans)
        if not selected:
            return HitlOutcome(
                status="NOT_REQUIRED",
                gap_code="MANUAL_DIAGNOSTIC_REQUEST_EMPTY",
            )
        hypotheses = tuple(
            item.hypothesis_key
            for item in assessment.hypothesis_assessments
            if item.status in {"SUPPORTED", "UNTESTED"}
        )
        queries = tuple(
            self._builder.from_catalog(
                tool_snapshot=item,
                db_type=database["db_type"],
                parameters=dict(item.get("parameters", {})),
                query_id=item["tool_id"],
                purpose=f"补充 {item['tool_id']} 的数据库现场证据",
                diagnostic_question=(
                    f"目标数据库的 {item['tool_id']} 当前状态是什么？"
                ),
                supports_if="返回结果与待验证假设一致",
                contradicts_if="返回结果明确排除待验证假设",
            )
            for item in selected
        )
        now = datetime.now(UTC)
        expires_at = min(
            now + timedelta(hours=2),
            _parse_time(context.deadline_at)
            if context.deadline_at
            else now + timedelta(hours=2),
        )
        hitl_id = uuid7()
        target = context.plan_snapshot["target"]
        request = ManualSqlRequest(
            hitl_id=str(hitl_id),
            run_id=context.run_id,
            round_no=int(assessment.round_no),
            target_id=context.target_id,
            target_display_name=target["target_key"],
            db_type=database["db_type"],
            db_version=database["configured_version"],
            expected_instance_identity=self._expected_identity(
                evidence, database["configured_version"]
            ),
            evidence_gap_refs=tuple(
                str(item.get("code", "UNKNOWN")) for item in evidence.gaps
            ),
            hypothesis_keys=hypotheses,
            queries=queries,
            instructions=(
                "请使用目标数据库的只读账号逐条执行 SQL。",
                "不要修改 SQL，也不要提交来自其他数据库实例的结果。",
                "请直接粘贴数据库客户端的完整原始输出，不需要选择或转换格式。",
            ),
            expires_at=expires_at,
        )
        return InputSuspension(
            hitl_id=str(hitl_id),
            request_type="MANUAL_DIAGNOSTIC_SQL",
            assignee_user_id=context.actor_id,
            prompt_text="自动诊断无法取得足够数据库证据，请手工执行只读 SQL。",
            response_schema={
                "schema_version": "HITL_RESPONSE.v1",
                "input": "RAW_DATABASE_OUTPUT",
                "auto_detect": True,
                "query_ids": [item.query_id for item in queries],
            },
            request_artifact_type="MANUAL_SQL_REQUEST",
            request_schema_version="MANUAL_SQL_REQUEST.v1",
            request_payload=request.model_dump(mode="json"),
            expires_at=expires_at,
            idempotency_key=f"{context.task_id}:manual-diagnostic",
        )

    @classmethod
    def _requires_manual_input(cls, evidence: EvidenceIndex) -> bool:
        return any(
            str(item.get("code", "")) in cls._CONNECTIVITY_GAPS
            for item in evidence.gaps
        )

    @staticmethod
    def _select_tools(
        frozen_tools: tuple[dict[str, Any], ...],
        plans: tuple[ValidatedEvidencePlan, ...],
    ) -> tuple[dict[str, Any], ...]:
        by_id = {item["tool_id"]: item for item in frozen_tools}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for plan in reversed(plans):
            for request in plan.accepted:
                if request.tool_id in seen or request.tool_id not in by_id:
                    continue
                frozen = dict(by_id[request.tool_id])
                frozen["parameters"] = dict(request.parameters)
                selected.append(frozen)
                seen.add(request.tool_id)
                if len(selected) >= 3:
                    return tuple(selected)
        return tuple(selected)

    @staticmethod
    def _expected_identity(
        evidence: EvidenceIndex, configured_version: str
    ) -> dict[str, str]:
        identity = next(
            (
                item.value
                for item in evidence.facts
                if item.metric_or_fact_type == "db.instance.identity"
                and isinstance(item.value, dict)
            ),
            {},
        )
        result = {"configured_version": configured_version}
        for key in ("product", "instance_name"):
            if identity.get(key):
                result[key] = str(identity[key])
        return result


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
        except AIOpsModelError as exc:
            logger.warning(
                "诊断引用检查降级：task_id={} code={} error={}",
                context.task_id,
                exc.code,
                str(exc),
            )
            return GroundingVerification(
                status="REVISE",
                issues=("语义引用检查模型本次不可用",),
                model_gap_code=exc.code,
            )
        except ValueError as exc:
            logger.warning(
                "诊断引用检查业务校验失败：task_id={} error={}",
                context.task_id,
                str(exc),
            )
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
    @staticmethod
    def _output_decision(
        *,
        question: str,
        trigger_type: str,
        root_level: str,
        has_direct_answer: bool,
        has_recommendations: bool,
        status: str,
    ) -> tuple[str, str, tuple[str, ...], bool]:
        """按问题、触发方式和诊断结果决定输出深度。"""
        explicit_report = any(
            keyword in question.lower()
            for keyword in (
                "报告",
                "report",
                "根因分析",
                "故障分析",
                "性能分析",
                "巡检",
            )
        )
        issue_detected = (
            not has_direct_answer
            and root_level
            in {"CONFIRMED", "PROBABLE", "POSSIBLE"}
        )
        automatic_trigger = trigger_type in {"ALERT", "SCHEDULE"}
        output_kind = (
            "DIAGNOSIS_REPORT"
            if explicit_report or issue_detected or automatic_trigger
            else "SIMPLE_CONCLUSION"
        )
        decision_reasons = tuple(
            reason
            for condition, reason in (
                (explicit_report, "USER_REQUESTED_REPORT"),
                (issue_detected, "ISSUE_DETECTED"),
                (automatic_trigger, "AUTOMATIC_TRIGGER"),
                (
                    not explicit_report
                    and not issue_detected
                    and not automatic_trigger,
                    "INFORMATIONAL_QUERY",
                ),
            )
            if condition
        )
        recommendation_level = (
            "FULL"
            if issue_detected
            else "BRIEF"
            if has_recommendations or status != "READY"
            else "NONE"
        )
        return (
            output_kind,
            recommendation_level,
            decision_reasons,
            issue_detected,
        )

    _STORAGE_SUBJECTS = ("表空间", "存储空间", "磁盘空间", "storage")
    _REMAINING_INTENTS = (
        "还有多少",
        "还剩",
        "剩余",
        "可用",
        "余量",
        "remaining",
    )
    _UTILIZATION_INTENTS = (
        "使用率",
        "占用率",
        "用了多少",
        "utilization",
        "used percent",
    )

    @classmethod
    def _direct_answer(
        cls,
        *,
        question: str | None,
        evidence: EvidenceIndex,
    ) -> DirectQuestionAnswer | None:
        """优先回答可由可信监控事实直接计算的问题。"""
        normalized = (question or "").strip().lower()
        asks_storage = any(
            item in normalized for item in cls._STORAGE_SUBJECTS
        )
        asks_remaining = any(
            item in normalized for item in cls._REMAINING_INTENTS
        )
        asks_utilization = any(
            item in normalized for item in cls._UTILIZATION_INTENTS
        )
        if (
            not normalized
            or not asks_storage
            or not (asks_remaining or asks_utilization)
        ):
            return None
        series_facts = tuple(
            item
            for item in evidence.facts
            if item.source_type == "MONITOR_METRIC"
            and item.metric_or_fact_type
            in {
                "db.storage.utilization.series.last",
                "db.storage.free_bytes.series.last",
                "db.storage.max_bytes.series.last",
            }
            and isinstance(item.value, (int, float))
            and bool(item.dimensions.get("tablespace"))
        )
        asks_bytes = any(
            item in normalized
            for item in ("gb", "tb", "字节", "容量", "多大")
        )
        limitation = (
            "当前监控事实只有百分比口径，不能据此计算剩余 GB/TB。"
        )
        if series_facts:
            by_tablespace: dict[str, dict[str, EvidenceFact]] = {}
            for item in series_facts:
                tablespace = str(item.dimensions["tablespace"])
                by_tablespace.setdefault(tablespace, {})[
                    item.metric_or_fact_type
                ] = item
            rows = []
            refs = []
            has_free_bytes = False

            def _used_percent(item):
                fact = item[1].get(
                    "db.storage.utilization.series.last"
                )
                return float(fact.value) if fact is not None else -1.0

            ordered_tablespaces = sorted(
                by_tablespace.items(),
                key=_used_percent,
                reverse=True,
            )
            for tablespace, metrics in ordered_tablespaces:
                utilization = metrics.get(
                    "db.storage.utilization.series.last"
                )
                free_bytes = metrics.get(
                    "db.storage.free_bytes.series.last"
                )
                max_bytes = metrics.get(
                    "db.storage.max_bytes.series.last"
                )
                parts = []
                if free_bytes is not None and asks_remaining:
                    has_free_bytes = True
                    parts.append(
                        f"可用 {float(free_bytes.value) / (1024 ** 3):.2f} GiB"
                    )
                    refs.append(free_bytes.fact_id)
                if max_bytes is not None:
                    if asks_remaining:
                        parts.append(
                            f"最大 "
                            f"{float(max_bytes.value) / (1024 ** 3):.2f} GiB"
                        )
                        refs.append(max_bytes.fact_id)
                if utilization is not None:
                    used = max(
                        0.0, min(100.0, float(utilization.value))
                    )
                    if asks_utilization:
                        parts.append(f"使用率 {used:.2f}%")
                    if asks_remaining:
                        parts.append(f"剩余 {100.0 - used:.2f}%")
                    refs.append(utilization.fact_id)
                if parts:
                    rows.append(f"{tablespace}：{'，'.join(parts)}")
            if rows:
                bytes_gap = (
                    "当前监控未提供表空间可用字节数，只能回答剩余百分比。"
                )
                status = (
                    "ANSWERED"
                    if has_free_bytes or not asks_bytes
                    else "PARTIAL"
                )
                limitations = (
                    (bytes_gap,)
                    if asks_bytes and not has_free_bytes
                    else ()
                )
                return DirectQuestionAnswer(
                    answer_kind="MONITOR_FACT",
                    status=status,
                    question_summary=question or normalized,
                    answer_text=(
                        (
                            "当前监控窗口内各表空间情况如下："
                            if asks_utilization and asks_remaining
                            else "当前监控窗口内各表空间使用率如下："
                            if asks_utilization
                            else "当前监控窗口内各表空间余量如下："
                        )
                        + "；".join(rows)
                        + "。"
                        + (
                            f" {bytes_gap}"
                            if asks_bytes and not has_free_bytes
                            else ""
                        )
                    ),
                    fact_refs=tuple(dict.fromkeys(refs)),
                    limitations=limitations,
                )
        candidates = {
            item.metric_or_fact_type: item
            for item in evidence.facts
            if item.source_type == "MONITOR_METRIC"
            and item.metric_or_fact_type
            in {
                "db.storage.utilization.last",
                "db.storage.utilization.max",
                "db.storage.utilization.avg",
            }
            and isinstance(item.value, (int, float))
        }
        selected = next(
            (
                candidates[key]
                for key in (
                    "db.storage.utilization.last",
                    "db.storage.utilization.max",
                    "db.storage.utilization.avg",
                )
                if key in candidates
            ),
            None,
        )
        if selected is None:
            return None
        used_percent = max(0.0, min(100.0, float(selected.value)))
        remaining_percent = 100.0 - used_percent
        aggregate_limitation = (
            "当前监控查询只保留了聚合值，Prometheus 结果中没有 "
            "tablespace 标签，无法列出具体表空间名称。"
        )
        return DirectQuestionAnswer(
            answer_kind="MONITOR_FACT",
            status="PARTIAL",
            question_summary=question or normalized,
            answer_text=(
                f"当前监控窗口内，最高表空间使用率聚合值为 "
                f"{used_percent:.2f}%"
                + (
                    f"，对应剩余约 {remaining_percent:.2f}%"
                    if asks_remaining
                    else ""
                )
                + "。"
                + f" {aggregate_limitation}"
                + (f" {limitation}" if asks_bytes else "")
            ),
            fact_refs=(selected.fact_id,),
            limitations=(
                aggregate_limitation,
                *((limitation,) if asks_bytes else ()),
            ),
        )

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
        round_drafts = _artifacts(
            context, "DIAGNOSIS_ROUND_DRAFT.v1"
        )
        evidence_plans = tuple(
            ValidatedEvidencePlan.model_validate(item)
            for item in _artifacts(
                context, "VALIDATED_EVIDENCE_PLAN.v1"
            )
        )
        drafts_with_hypotheses = tuple(
            item for item in round_drafts if item.get("hypotheses")
        )
        latest_draft = (
            DiagnosisRoundDraft.model_validate(
                max(
                    drafts_with_hypotheses or round_drafts,
                    key=lambda item: int(item["round_no"]),
                )
            )
            if round_drafts
            else None
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
        direct_answer = self._direct_answer(
            question=context.plan_snapshot["diagnosis"].get(
                "question_summary"
            ),
            evidence=evidence,
        )
        status = (
            (
                "READY"
                if direct_answer.status == "ANSWERED"
                else "PARTIAL"
            )
            if direct_answer is not None
            else "DEGRADED"
            if model_gaps
            else "PARTIAL"
            if evidence.gaps or root.effective_level == "INCONCLUSIVE"
            else "READY"
        )
        question = str(
            context.plan_snapshot["diagnosis"].get("question_summary") or ""
        )
        has_recommendations = bool(
            solution.immediate_mitigations
            or solution.long_term_remediations
            or solution.candidate_action_template_refs
        )
        (
            output_kind,
            recommendation_level,
            decision_reasons,
            issue_detected,
        ) = self._output_decision(
            question=question,
            trigger_type=context.trigger_type,
            root_level=root.effective_level,
            has_direct_answer=direct_answer is not None,
            has_recommendations=has_recommendations,
            status=status,
        )
        return DiagnosisReportDraft(
            target_id=context.target_id,
            status=status,
            output_kind=output_kind,
            recommendation_level=recommendation_level,
            report_decision_reasons=decision_reasons,
            issue_detected=issue_detected,
            root_cause=root,
            facts=evidence.facts,
            hypotheses=(
                ()
                if direct_answer is not None
                else assessment.hypothesis_assessments
            ),
            hypothesis_details=(
                ()
                if direct_answer is not None
                else latest_draft.hypotheses if latest_draft else ()
            ),
            diagnosis_rationale=(
                None
                if direct_answer is not None
                else assessment.rationale_summary
            ),
            rejected_evidence_requests=tuple(
                ()
                if direct_answer is not None
                else (
                    request
                    for plan in evidence_plans
                    for request in plan.rejected
                )
            ),
            direct_answer=direct_answer,
            solution=(
                SolutionDraft(limitations=direct_answer.limitations)
                if direct_answer is not None
                else solution
            ),
            gaps=(
                direct_answer.limitations
                if direct_answer is not None
                else (
                    *model_gaps,
                    *(
                        str(item.get("code", "EVIDENCE_GAP"))
                        for item in evidence.gaps
                    ),
                )
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
