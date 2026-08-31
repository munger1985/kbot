"""Turn 证据充分性判断与自然回答组合 Handler。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from typing import Any

from loguru import logger

from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.contracts.change import ProposalOutcome
from aiops_agent.contracts.evidence import LogEvidenceSet, ObservationSet
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import (
    AIOpsTurnResult,
    DbaAnswerDraft,
    DbaAnswerProgress,
    DbaSufficiencyAssessment,
    TurnAnswerBlock,
    TurnEvidenceFact,
    TurnEvidenceGap,
)
from platform_core.contracts.aiops import (
    AnswerBlockType,
    InvestigationAssessment,
    MeasurementSemantics,
    SufficiencyStatus,
)

from .handlers import TaskExecutionContext


def _fact_trust_level(value: object) -> str:
    normalized = str(value or "MODEL_INFERENCE")
    return (
        normalized
        if normalized
        in {"SOURCE_VERIFIED", "USER_PROVIDED", "MODEL_INFERENCE"}
        else "MODEL_INFERENCE"
    )


class DbaEvidenceAssessmentHandler:
    """归一真实Evidence，并让模型评估假设、证据需求和下一步。"""

    _PROMPT = """
你是一名资深Oracle DBA调查评估者。根据Task Frame、调查计划和本轮真实Evidence，逐项更新
假设与剩余未知项，判断证据是否足以回答，并选择ANSWER、REPLAN、ASK_USER或STOP_UNSAFE。
不得虚构证据。系统仍有授权Tool可以补证时优先REPLAN；只有系统无法取得关键证据时才
ASK_USER。部分结论可以成立时应明确边界，不要因为单项缺失否定全部已验证事实。
sufficiency_status 只能从 ANSWERABLE、PARTIAL、NEEDS_CLARIFICATION、NEEDS_EVIDENCE、
CAPABILITY_UNAVAILABLE、UNSAFE 中选择，不得创建组合状态或同义状态。
""".strip()

    def __init__(self, *, model_client=None) -> None:
        self._model = model_client

    async def execute(
        self, context: TaskExecutionContext
    ) -> DbaSufficiencyAssessment:
        facts: list[TurnEvidenceFact] = []
        gaps: list[TurnEvidenceGap] = []
        reasons: list[str] = []
        database_gap_found = False
        monitoring_gap_found = False
        user_input_is_evidence = any(
            artifact.get("schema_version") == "aiops.input-envelope.v1"
            and any(
                bool(item.get("contains_user_evidence"))
                for item in dict(artifact.get("payload") or {}).get(
                    "materials", ()
                )
                if isinstance(item, dict)
            )
            for artifact in context.input_artifacts
        )
        for artifact in context.input_artifacts:
            schema_version = artifact.get("schema_version")
            if schema_version == "SOURCE_RUN_EVIDENCE.v1":
                payload = dict(artifact.get("payload") or {})
                artifact_id = str(artifact["artifact_id"])
                source_payload = payload.get("payload")
                facts.append(
                    TurnEvidenceFact(
                        evidence_ref=f"artifact:{artifact_id}#source-run",
                        artifact_id=artifact_id,
                        source_id="source.run-evidence",
                        step_id="inherit",
                        tool_id="source.run.final-artifact",
                        trust_level=_fact_trust_level(
                            payload.get("source_trust_level")
                        ),
                        measurement_semantics=(
                            MeasurementSemantics.NOT_APPLICABLE
                        ),
                        presentation_kind="MARKDOWN",
                        captured_at=str(
                            payload.get("captured_at")
                            or datetime.now().isoformat()
                        ),
                        columns=(
                            {"name": "source_schema", "logical_type": "STRING"},
                            {"name": "result", "logical_type": "JSON"},
                        ),
                        rows=((payload.get("source_schema_version"), source_payload),),
                        row_count=1,
                    )
                )
                continue
            if schema_version == "USER_PROVIDED_INPUT.v1":
                payload = dict(artifact.get("payload") or {})
                text = str(payload.get("text", "")).strip()
                if text and (
                    bool(payload.get("contains_evidence"))
                    or user_input_is_evidence
                ):
                    artifact_id = str(artifact["artifact_id"])
                    facts.append(
                        TurnEvidenceFact(
                            evidence_ref=f"artifact:{artifact_id}#user-input",
                            artifact_id=artifact_id,
                            source_id="user.provided-evidence",
                            step_id="input",
                            tool_id="user.input",
                            trust_level="USER_PROVIDED",
                            measurement_semantics=(
                                MeasurementSemantics.NOT_APPLICABLE
                            ),
                            presentation_kind="MARKDOWN",
                            captured_at=str(
                                payload.get("received_at")
                                or datetime.now().isoformat()
                            ),
                            columns=(
                                {"name": "content", "logical_type": "STRING"},
                            ),
                            rows=((text,),),
                            row_count=1,
                        )
                    )
                continue
            if schema_version == "OBSERVATION_SET.v1":
                result = ObservationSet.model_validate(artifact["payload"])
                fact = self._monitoring_fact(
                    artifact_id=str(artifact["artifact_id"]),
                    result=result,
                )
                if fact is not None:
                    facts.append(fact)
                for gap in result.gaps:
                    monitoring_gap_found = True
                    gaps.append(
                        TurnEvidenceGap(
                            source_id="monitoring.overview",
                            step_id=gap.metric_code or gap.binding_id,
                            code=gap.code,
                            detail=gap.detail,
                            retryable=gap.retryable,
                        )
                    )
                continue
            if schema_version == "LOG_EVIDENCE_SET.v1":
                result = LogEvidenceSet.model_validate(artifact["payload"])
                artifact_id = str(artifact["artifact_id"])
                if result.entries:
                    facts.append(
                        TurnEvidenceFact(
                            evidence_ref=f"artifact:{artifact_id}#loki",
                            artifact_id=artifact_id,
                            source_id="oracle.alert-log",
                            step_id="loki",
                            tool_id="loki.query_range",
                            measurement_semantics=(
                                MeasurementSemantics.HISTORICAL_SAMPLES
                            ),
                            presentation_kind="TABLE",
                            captured_at=result.collected_at.isoformat(),
                            columns=(
                                {"name": "observed_at", "logical_type": "DATETIME"},
                                {"name": "line", "logical_type": "STRING"},
                                {"name": "labels", "logical_type": "JSON"},
                            ),
                            rows=tuple(
                                (
                                    entry.observed_at.isoformat(),
                                    entry.line,
                                    dict(entry.labels),
                                )
                                for entry in result.entries
                            ),
                            row_count=len(result.entries),
                            truncated=result.truncated,
                        )
                    )
                for gap in result.gaps:
                    monitoring_gap_found = True
                    gaps.append(
                        TurnEvidenceGap(
                            source_id="oracle.alert-log",
                            step_id="loki",
                            code=gap.code,
                            detail=gap.detail,
                            retryable=gap.retryable,
                        )
                    )
                continue
            if schema_version != "DBA_TOOL_RESULT.v1":
                continue
            result = DbaToolResult.model_validate(artifact["payload"])
            artifact_id = str(artifact["artifact_id"])
            for outcome in result.tool_outcomes:
                if outcome.observation is not None:
                    observation = outcome.observation
                    if (
                        outcome.tool_id != "db.instance.identity"
                        or result.source_type == "TOOL"
                    ):
                        facts.append(
                            TurnEvidenceFact(
                                evidence_ref=(
                                    f"artifact:{artifact_id}#"
                                    f"{outcome.step_id}"
                                ),
                                artifact_id=artifact_id,
                                source_id=result.source_id,
                                step_id=outcome.step_id,
                                tool_id=outcome.tool_id,
                                measurement_semantics=(
                                    result.measurement_semantics
                                ),
                                presentation_kind=result.presentation_kind,
                                captured_at=observation.captured_at.isoformat(),
                                columns=tuple(
                                    column.model_dump(mode="json")
                                    for column in observation.columns
                                ),
                                rows=observation.rows,
                                row_count=observation.row_count,
                                truncated=observation.truncated,
                                warnings=observation.warnings,
                            )
                        )
                if outcome.gap is not None:
                    database_gap_found = True
                    gaps.append(
                        TurnEvidenceGap(
                            source_id=result.source_id,
                            step_id=outcome.step_id,
                            code=outcome.gap.code,
                            detail=outcome.gap.detail,
                            retryable=outcome.gap.retryable,
                        )
                    )

        for gap in dict(
            context.plan_snapshot.get("monitoring", {})
        ).get("initial_gaps", ()):
            monitoring_gap_found = True
            gaps.append(
                TurnEvidenceGap(
                    source_id="monitoring.overview",
                    step_id=str(gap.get("binding_id", "source")),
                    code=str(gap.get("code", "MONITORING_UNAVAILABLE")),
                    detail=str(gap.get("detail", "监控源不可用")),
                )
            )

        answer_context = dict(context.plan_snapshot.get("answer_context", {}))
        task_frame = dict(answer_context.get("task_frame", {}))
        requested_window = bool(task_frame.get("time_scope"))
        cumulative_only = bool(facts) and all(
            fact.measurement_semantics
            == MeasurementSemantics.CUMULATIVE_SINCE_LOAD
            for fact in facts
        )
        if requested_window and cumulative_only:
            reasons.append(
                "请求的是时间窗口数据，但当前证据只有实例启动后的累计口径"
            )
        if database_gap_found:
            reasons.append("部分受控取证步骤未能返回可验证结果")
        if monitoring_gap_found:
            reasons.append("部分监控指标查询失败、无采样或监控源不可用")

        objectives = {str(item) for item in task_frame.get("objectives", ())}
        can_answer_from_expertise = bool(objectives) and objectives <= {
            "UNDERSTAND", "EXPLAIN", "PLAN"
        }
        if not facts and can_answer_from_expertise and not gaps:
            status = SufficiencyStatus.ANSWERABLE
            reasons.append("该问题可以依据 DBA 专业知识回答，无需伪造外部证据")
        elif not facts:
            status = SufficiencyStatus.NEEDS_EVIDENCE
            reasons.append("当前没有取得能够回答问题的主题证据")
        elif reasons:
            status = SufficiencyStatus.PARTIAL
        else:
            status = SufficiencyStatus.ANSWERABLE
        deterministic = DbaSufficiencyAssessment(
            status=status,
            evidence=tuple(facts),
            gaps=tuple(gaps),
            reasons=tuple(reasons),
        )
        if self._model is None:
            return deterministic
        model_snapshot = dict(answer_context.get("model") or {})
        if not model_snapshot:
            return deterministic
        prompt_ref = {
            "prompt_id": "aiops.investigation-assessor",
            "prompt_version": "1",
            "prompt_sha256": hashlib.sha256(
                self._PROMPT.encode("utf-8")
            ).hexdigest(),
            "content": self._PROMPT,
        }
        try:
            result = await self._model.generate_structured(
                purpose="aiops.investigation-assessment",
                output_model=InvestigationAssessment,
                model_snapshot=model_snapshot,
                prompt_ref=prompt_ref,
                input_payload={
                    "task_frame": task_frame,
                    "investigation_plan": dict(
                        answer_context.get("investigation_plan") or {}
                    ),
                    "deterministic_sufficiency": deterministic.model_dump(
                        mode="json"
                    ),
                },
                deadline=(
                    datetime.fromisoformat(context.deadline_at)
                    if context.deadline_at
                    else None
                ),
                idempotency_key=(
                    f"turn:{context.run_id}:assessment:{context.attempt}"
                ),
            )
        except AIOpsModelError as exc:
            logger.warning(
                "调查证据模型评估降级为确定性结果：task_id={} code={}",
                context.task_id,
                exc.code,
            )
            return deterministic
        investigation = InvestigationAssessment.model_validate(result.output)
        assessed_status = SufficiencyStatus(investigation.sufficiency_status)
        return deterministic.model_copy(
            update={
                "status": assessed_status,
                "reasons": tuple(
                    dict.fromkeys(
                        (*deterministic.reasons, investigation.reason)
                    )
                ),
                "investigation": investigation,
            }
        )

    @staticmethod
    def _monitoring_fact(
        *,
        artifact_id: str,
        result: ObservationSet,
    ) -> TurnEvidenceFact | None:
        """把同一监控源的多指标时间序列压缩为一个可折叠事实。"""
        rows: list[tuple[Any, ...]] = []
        warnings: list[str] = []
        for observation in result.observations:
            warnings.extend(observation.warnings)
            for series in observation.series:
                points = [
                    point
                    for point in series.points
                    if point.quality == "GOOD" and point.value is not None
                ]
                if not points:
                    continue
                numeric_values = [
                    float(point.value)
                    for point in points
                    if isinstance(point.value, (int, float))
                    and not isinstance(point.value, bool)
                ]
                dimensions = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(series.dimensions.items())
                )
                rows.append(
                    (
                        observation.metric_code,
                        dimensions or "-",
                        points[-1].value,
                        (
                            round(sum(numeric_values) / len(numeric_values), 4)
                            if numeric_values
                            else None
                        ),
                        round(max(numeric_values), 4)
                        if numeric_values
                        else None,
                        observation.unit,
                        observation.window_start.isoformat(),
                        observation.window_end.isoformat(),
                        round(observation.coverage_ratio, 4),
                    )
                )
        if not rows:
            return None
        columns = (
            {"name": "metric_code", "logical_type": "STRING"},
            {"name": "dimensions", "logical_type": "STRING"},
            {"name": "latest", "logical_type": "DECIMAL"},
            {"name": "average", "logical_type": "DECIMAL"},
            {"name": "maximum", "logical_type": "DECIMAL"},
            {"name": "unit", "logical_type": "STRING"},
            {"name": "window_start", "logical_type": "DATETIME"},
            {"name": "window_end", "logical_type": "DATETIME"},
            {"name": "coverage_ratio", "logical_type": "DECIMAL"},
        )
        return TurnEvidenceFact(
            evidence_ref=f"artifact:{artifact_id}#prometheus",
            artifact_id=artifact_id,
            source_id="monitoring.overview",
            step_id="prometheus",
            tool_id="metric.query_range",
            measurement_semantics=MeasurementSemantics.HISTORICAL_SAMPLES,
            presentation_kind="TABLE",
            captured_at=result.collected_at.isoformat(),
            columns=columns,
            rows=tuple(rows),
            row_count=len(rows),
            truncated=any(item.truncated for item in result.observations),
            warnings=tuple(dict.fromkeys(warnings)),
        )


class DbaAnswerComposeHandler:
    """模型写自然语言，表格和图表由服务端从已验证证据生成。"""

    def __init__(self, *, model_client, prompts) -> None:
        self._model = model_client
        self._prompts = prompts

    async def execute(self, context: TaskExecutionContext) -> AIOpsTurnResult:
        assessment = self._assessment(context.input_artifacts)
        if assessment.status in {
            SufficiencyStatus.NEEDS_CLARIFICATION,
            SufficiencyStatus.NEEDS_EVIDENCE,
            SufficiencyStatus.CAPABILITY_UNAVAILABLE,
            SufficiencyStatus.UNSAFE,
        }:
            return self._waiting_result(assessment, context)

        answer_context = dict(context.plan_snapshot.get("answer_context", {}))
        prompt = self._prompts.resolve("answer_compose", "1")
        result = await self._model.generate_structured(
            purpose="aiops.dba-answer-compose",
            output_model=DbaAnswerDraft,
            model_snapshot=dict(answer_context["model"]),
            prompt_ref={**prompt.ref(), "content": prompt.content},
            input_payload={
                "question": str(answer_context.get("question", "")),
                "input_envelope": dict(
                    answer_context.get("input_envelope", {})
                ),
                "task_frame": dict(answer_context.get("task_frame", {})),
                "sufficiency": assessment.model_dump(mode="json"),
            },
            deadline=self._deadline(context.deadline_at),
            idempotency_key=f"turn:{context.run_id}:answer:{context.attempt}",
        )
        draft = DbaAnswerDraft.model_validate(result.output)
        allowed_refs = {item.evidence_ref for item in assessment.evidence}
        if not set(draft.evidence_refs) <= allowed_refs:
            raise ValueError("回答引用了本轮批准证据之外的内容")

        blocks: list[TurnAnswerBlock] = [
            TurnAnswerBlock(
                block_type=AnswerBlockType.MARKDOWN,
                schema_version="AIOPS_MARKDOWN_BLOCK.v1",
                payload={"markdown": draft.markdown},
                evidence_refs=draft.evidence_refs,
            )
        ]
        blocks.extend(self._data_blocks(assessment.evidence))
        proposal_block = self._proposal_block(context.input_artifacts)
        if proposal_block is not None:
            blocks.append(proposal_block)
        evidence_request = self._evidence_request_block(
            assessment, context
        )
        if evidence_request is not None:
            blocks.append(evidence_request)
        status = (
            "COMPLETED"
            if assessment.status == SufficiencyStatus.ANSWERABLE
            else "PARTIAL"
        )
        return AIOpsTurnResult(
            status=status,
            sufficiency_status=assessment.status,
            blocks=tuple(blocks),
            model_receipt=result.receipt.model_dump(mode="json"),
        )

    async def execute_stream(self, context: TaskExecutionContext):
        """使用模型原生SSE生成正文，校验后逐块投递用户可见增量。"""
        assessment = self._assessment(context.input_artifacts)
        if assessment.status in {
            SufficiencyStatus.NEEDS_CLARIFICATION,
            SufficiencyStatus.NEEDS_EVIDENCE,
            SufficiencyStatus.CAPABILITY_UNAVAILABLE,
            SufficiencyStatus.UNSAFE,
        }:
            result = self._waiting_result(assessment, context).model_copy(
                update={"answer_streamed": True}
            )
            markdown = str(result.blocks[0].payload.get("markdown", ""))
            for index, delta in enumerate(self._answer_deltas(markdown), start=1):
                yield DbaAnswerProgress(
                    event_type="answer.delta",
                    event_key=f"answer-delta:{index}",
                    payload={"chunk_index": index, "delta": delta},
                )
            yield result
            return

        answer_context = dict(context.plan_snapshot.get("answer_context", {}))
        prompt = self._prompts.resolve("answer_stream", "1")
        labels = {
            f"E{index}": fact.evidence_ref
            for index, fact in enumerate(assessment.evidence, start=1)
        }
        evidence_payload = []
        for label, fact in zip(
            labels,
            assessment.evidence,
            strict=True,
        ):
            evidence_payload.append(
                {"label": label, **fact.model_dump(mode="json")}
            )
        yield DbaAnswerProgress(
            event_type="thinking.delta",
            event_key="answer-thinking:compose",
            payload={
                "delta": "正在基于本轮已验证证据组织回答",
                "public_summary": "正在组织带引用的诊断回答",
            },
        )
        started = time.monotonic()
        answer = ""
        used_labels: tuple[str, ...] = ()
        validation_error = ""
        last_input_payload: dict[str, Any] = {}
        for attempt in range(1, 3):
            parts: list[str] = []
            last_input_payload = {
                "question": str(answer_context.get("question", "")),
                "input_envelope": dict(
                    answer_context.get("input_envelope", {})
                ),
                "task_frame": dict(answer_context.get("task_frame", {})),
                "sufficiency_status": str(assessment.status),
                "limitations": list(assessment.reasons),
                "evidence_gaps": [
                    item.model_dump(mode="json")
                    for item in assessment.gaps
                ],
                "evidence": evidence_payload,
                "allowed_citation_labels": list(labels),
                "previous_invalid_answer": answer or None,
                "validation_error": validation_error or None,
            }
            async for chunk in self._model.stream_text(
                purpose="aiops.dba-answer-stream",
                model_snapshot=dict(answer_context["model"]),
                prompt_ref={**prompt.ref(), "content": prompt.content},
                input_payload=last_input_payload,
                deadline=self._deadline(context.deadline_at),
                idempotency_key=(
                    f"turn:{context.run_id}:answer-stream:{attempt}"
                ),
            ):
                parts.append(chunk)
            answer = "".join(parts).strip()
            try:
                used_labels = self._validate_streamed_answer(answer, labels)
                break
            except ValueError as exc:
                validation_error = str(exc)
                if attempt == 1:
                    yield DbaAnswerProgress(
                        event_type="thinking.delta",
                        event_key="answer-thinking:retry",
                        payload={
                            "delta": "回答引用未通过校验，正在重新生成",
                            "public_summary": "正在修正证据引用",
                        },
                    )
        else:
            raise ValueError("模型连续两次未生成可验证的诊断回答")

        markdown = self._strip_citation_labels(answer)
        evidence_refs = tuple(labels[label] for label in used_labels)
        for index, delta in enumerate(self._answer_deltas(markdown), start=1):
            yield DbaAnswerProgress(
                event_type="answer.delta",
                event_key=f"answer-delta:{index}",
                payload={"chunk_index": index, "delta": delta},
            )
        blocks: list[TurnAnswerBlock] = [
            TurnAnswerBlock(
                block_type=AnswerBlockType.MARKDOWN,
                schema_version="AIOPS_MARKDOWN_BLOCK.v1",
                payload={"markdown": markdown},
                evidence_refs=evidence_refs,
            )
        ]
        blocks.extend(self._data_blocks(assessment.evidence))
        proposal_block = self._proposal_block(context.input_artifacts)
        if proposal_block is not None:
            blocks.append(proposal_block)
        evidence_request = self._evidence_request_block(
            assessment, context
        )
        if evidence_request is not None:
            blocks.append(evidence_request)
        yield AIOpsTurnResult(
            status=(
                "COMPLETED"
                if assessment.status == SufficiencyStatus.ANSWERABLE
                else "PARTIAL"
            ),
            sufficiency_status=assessment.status,
            blocks=tuple(blocks),
            answer_streamed=True,
            model_receipt={
                "purpose": "aiops.dba-answer-stream",
                "model_technical_name": answer_context["model"][
                    "technical_name"
                ],
                "model_revision": answer_context["model"]["revision"],
                **prompt.ref(),
                "input_sha256": self._hash(last_input_payload),
                "output_sha256": self._hash(answer),
                "duration_ms": int((time.monotonic() - started) * 1000),
            },
        )

    @staticmethod
    def _assessment(
        artifacts: tuple[dict[str, Any], ...]
    ) -> DbaSufficiencyAssessment:
        matches = [
            item["payload"]
            for item in artifacts
            if item.get("schema_version") == "DBA_SUFFICIENCY.v1"
        ]
        if len(matches) != 1:
            raise ValueError("回答任务必须且只能接收一个充分性 Artifact")
        return DbaSufficiencyAssessment.model_validate(matches[0])

    @staticmethod
    def _proposal_block(
        artifacts: tuple[dict[str, Any], ...]
    ) -> TurnAnswerBlock | None:
        payload = next(
            (
                item["payload"]
                for item in artifacts
                if item.get("schema_version") == "PROPOSAL_OUTCOME.v1"
            ),
            None,
        )
        if payload is None:
            return None
        outcome = ProposalOutcome.model_validate(payload)
        if outcome.status != "CREATED" or outcome.proposal is None:
            return None
        proposal = outcome.proposal
        return TurnAnswerBlock(
            block_type=AnswerBlockType.PROPOSAL_SUMMARY,
            schema_version="AIOPS_PROPOSAL_SUMMARY_BLOCK.v1",
            payload={
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.proposal_hash,
                "row_version": 1,
                "status": "PENDING_APPROVAL",
                "action_template_id": proposal.action_template_id,
                "risk_level": proposal.risk_level,
                "rationale": proposal.rationale,
                "impact": proposal.impact,
                "parameters": proposal.canonical_parameters,
                "expires_at": proposal.expires_at.isoformat(),
            },
            evidence_refs=proposal.evidence_refs,
        )

    @staticmethod
    def _waiting_result(
        assessment: DbaSufficiencyAssessment,
        context: TaskExecutionContext,
    ) -> AIOpsTurnResult:
        if assessment.clarification_question:
            message = assessment.clarification_question
        else:
            detail = "；".join(assessment.reasons) or "当前证据不足"
            message = (
                f"我暂时还不能可靠回答这个问题：{detail}。"
                "系统已经使用该 Agent 绑定的 Target 只读凭据自动取证，"
                "但没有取得足够结果。"
            )
        blocks = [
            TurnAnswerBlock(
                block_type=AnswerBlockType.MARKDOWN,
                schema_version="AIOPS_MARKDOWN_BLOCK.v1",
                payload={"markdown": message},
            )
        ]
        evidence_request = DbaAnswerComposeHandler._evidence_request_block(
            assessment, context
        )
        if evidence_request is not None:
            blocks.append(evidence_request)
        return AIOpsTurnResult(
            status="WAITING_USER",
            sufficiency_status=assessment.status,
            blocks=tuple(blocks),
        )

    @staticmethod
    def _evidence_request_block(
        assessment: DbaSufficiencyAssessment,
        context: TaskExecutionContext,
    ) -> TurnAnswerBlock | None:
        execution = dict(
            context.plan_snapshot.get("investigation_execution", {})
        )
        invocations = dict(execution.get("invocations", {}))
        tools_by_step: dict[tuple[str, str], dict[str, Any]] = {}
        for invocation in invocations.values():
            source_id = str(invocation.get("playbook_id", ""))
            for tool in invocation.get("tools", ()):
                tools_by_step[(source_id, str(tool.get("step_id", "")))] = tool
        requests: list[tuple[TurnEvidenceGap, dict[str, Any]]] = []
        monitoring_gaps = [
            gap
            for gap in assessment.gaps
            if gap.source_id == "monitoring.overview"
        ]
        seen_tools: set[str] = set()
        for gap in assessment.gaps:
            tool = tools_by_step.get((gap.source_id, gap.step_id))
            tool_id = str((tool or {}).get("tool_id", ""))
            if tool is None or not tool_id or tool_id in seen_tools:
                continue
            seen_tools.add(tool_id)
            requests.append((gap, tool))
        if not requests and not monitoring_gaps:
            return None
        lines = [
            "\n### 要继续完成这次诊断",
            "",
        ]
        if requests:
            for gap, tool in requests:
                tool_id = str(tool["tool_id"])
                lines.append(
                    f"- `{tool_id}`：{gap.detail}（`{gap.code}`）"
                )
            lines.extend(
                [
                    "",
                    "请优先修正 Target 只读凭据、对象权限或数据库版本兼容问题。",
                    "如果暂时不能调整，可以执行下面的只读 SQL，",
                    "再把结果以文字或截图粘贴到对话中：",
                ]
            )
            for _, tool in requests:
                sql = DbaAnswerComposeHandler._manual_sql(tool)
                if not sql:
                    continue
                privileges = tuple(tool.get("required_privileges", ()))
                lines.extend(
                    [
                        "",
                        f"#### {tool['tool_id']}",
                        (
                            f"所需对象权限：`{', '.join(privileges)}`"
                            if privileges
                            else "无需额外对象权限"
                        ),
                        "```sql",
                        sql,
                        "```",
                    ]
                )
        if monitoring_gaps:
            queries = DbaAnswerComposeHandler._monitoring_queries(
                context
            )
            lines.extend(
                [
                    "",
                    "#### Prometheus 未返回的指标",
                ]
            )
            for gap in monitoring_gaps:
                lines.append(
                    f"- `{gap.step_id}`：{gap.detail}（`{gap.code}`）"
                )
                query = queries.get(gap.step_id)
                if query:
                    lines.extend(["```promql", query, "```"])
            lines.extend(
                [
                    "",
                    "请在 Prometheus 中执行上述 PromQL，并检查 Target 映射、Exporter 自定义指标和记录规则。",
                    "修复后直接回复“监控已补齐”，我会在下一轮重新自动取证；",
                    "如果暂时无法修改监控，请把查询结果粘贴到对话中继续分析。",
                ]
            )
        return TurnAnswerBlock(
            block_type=AnswerBlockType.EVIDENCE_REQUEST,
            schema_version="AIOPS_EVIDENCE_REQUEST_BLOCK.v1",
            payload={"markdown": "\n".join(lines)},
        )

    @staticmethod
    def _monitoring_queries(
        context: TaskExecutionContext,
    ) -> dict[str, str]:
        """生成与本轮冻结配置一致的可重放 PromQL。"""
        result: dict[str, str] = {}
        monitoring = dict(context.plan_snapshot.get("monitoring", {}))
        for binding in monitoring.get("bindings", ()):
            source = dict(binding.get("source", {}))
            if source.get("source_type") != "PROMETHEUS":
                continue
            overrides = dict(
                binding.get("mapping_overrides") or {}
            ).get("prometheus_queries") or {}
            source_key = str(binding.get("source_locator_key", ""))
            locator = dict(binding.get("source_locator") or {})
            host_key = str(locator.get("host_target_key") or source_key)
            escaped_source = source_key.replace("\\", "\\\\").replace(
                '"', '\\"'
            )
            escaped_host = host_key.replace("\\", "\\\\").replace(
                '"', '\\"'
            )
            for metric in binding.get("metrics", ()):
                metric_code = str(metric.get("metric_code", ""))
                provider = dict(metric.get("providers", {})).get(
                    "PROMETHEUS"
                ) or {}
                template = overrides.get(metric_code) or provider.get(
                    "query_template"
                )
                if not metric_code or not isinstance(template, str):
                    continue
                result.setdefault(
                    metric_code,
                    template.replace(
                        "${external_target}", escaped_source
                    ).replace("${host_target}", escaped_host),
                )
        return result

    @staticmethod
    def _manual_sql(tool: dict[str, Any]) -> str:
        """把冻结且已校验的参数写入仅供人工补证的目录 SQL。"""
        sql = str(tool.get("manual_sql", "")).strip()
        for name, value in dict(tool.get("parameters", {})).items():
            if value is None:
                literal = "NULL"
            elif isinstance(value, bool):
                literal = "1" if value else "0"
            elif isinstance(value, (int, float)):
                literal = str(value)
            else:
                literal = "'" + str(value).replace("'", "''") + "'"
            sql = re.sub(rf":{re.escape(str(name))}\b", literal, sql)
        return sql

    @staticmethod
    def _data_blocks(
        evidence: tuple[TurnEvidenceFact, ...],
    ) -> tuple[TurnAnswerBlock, ...]:
        blocks: list[TurnAnswerBlock] = []
        for fact in evidence:
            columns = list(fact.columns)
            if fact.presentation_kind in {"TABLE", "TABLE_AND_CHART"}:
                blocks.append(
                    TurnAnswerBlock(
                        block_type=AnswerBlockType.TABLE,
                        schema_version="AIOPS_TABLE_BLOCK.v1",
                        payload={
                            "title": fact.tool_id,
                            "columns": columns,
                            "rows": [list(row) for row in fact.rows],
                            "row_count": fact.row_count,
                            "truncated": fact.truncated,
                            "captured_at": fact.captured_at,
                            "measurement_semantics": (
                                fact.measurement_semantics
                            ),
                        },
                        evidence_refs=(fact.evidence_ref,),
                    )
                )
            chart = DbaAnswerComposeHandler._chart_payload(fact)
            if (
                chart is not None
                and fact.presentation_kind in {"CHART", "TABLE_AND_CHART"}
            ):
                blocks.append(
                    TurnAnswerBlock(
                        block_type=AnswerBlockType.CHART,
                        schema_version="AIOPS_CHART_BLOCK.v1",
                        payload=chart,
                        evidence_refs=(fact.evidence_ref,),
                    )
                )
        return tuple(blocks)

    @staticmethod
    def _chart_payload(fact: TurnEvidenceFact) -> dict[str, Any] | None:
        dimension = next(
            (
                index
                for index, column in enumerate(fact.columns)
                if column.get("logical_type") in {"STRING", "DATETIME"}
            ),
            None,
        )
        metric = next(
            (
                index
                for index, column in enumerate(fact.columns)
                if column.get("logical_type") in {"INTEGER", "DECIMAL"}
            ),
            None,
        )
        if dimension is None or metric is None or not fact.rows:
            return None
        return {
            "title": fact.tool_id,
            "chart_type": "BAR",
            "category": fact.columns[dimension]["name"],
            "metric": fact.columns[metric]["name"],
            "categories": [row[dimension] for row in fact.rows[:50]],
            "series": [row[metric] for row in fact.rows[:50]],
            "captured_at": fact.captured_at,
            "measurement_semantics": fact.measurement_semantics,
        }

    @staticmethod
    def _deadline(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    @staticmethod
    def _validate_streamed_answer(
        answer: str, labels: dict[str, str]
    ) -> tuple[str, ...]:
        if not answer:
            raise ValueError("模型返回的回答为空")
        used = tuple(dict.fromkeys(re.findall(r"\[(E\d+)\]", answer)))
        unknown = set(used) - labels.keys()
        if unknown:
            raise ValueError(f"回答使用了未知证据引用：{sorted(unknown)}")
        if labels and not used:
            raise ValueError("有验证证据的回答必须实际引用证据")
        return used

    @staticmethod
    def _strip_citation_labels(answer: str) -> str:
        return re.sub(r"\s*\[E\d+\]", "", answer).strip()

    @staticmethod
    def _answer_deltas(answer: str) -> tuple[str, ...]:
        return tuple(
            answer[index:index + 120]
            for index in range(0, len(answer), 120)
        ) or ("",)

    @staticmethod
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
