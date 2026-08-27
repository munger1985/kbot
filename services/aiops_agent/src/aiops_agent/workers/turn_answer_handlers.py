"""Turn 证据充分性判断与自然回答组合 Handler。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
import time
from typing import Any

from aiops_agent.contracts.skill_execution import DbaSkillResult
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
    MeasurementSemantics,
    SufficiencyStatus,
)

from .handlers import TaskExecutionContext


class DbaEvidenceAssessmentHandler:
    """只按本轮 Skill Artifact 判断证据，不读取历史 Turn。"""

    async def execute(
        self, context: TaskExecutionContext
    ) -> DbaSufficiencyAssessment:
        facts: list[TurnEvidenceFact] = []
        gaps: list[TurnEvidenceGap] = []
        reasons: list[str] = []
        for artifact in context.input_artifacts:
            if artifact.get("schema_version") != "DBA_SKILL_RESULT.v1":
                continue
            result = DbaSkillResult.model_validate(artifact["payload"])
            artifact_id = str(artifact["artifact_id"])
            for outcome in result.tool_outcomes:
                if outcome.observation is not None:
                    observation = outcome.observation
                    if outcome.tool_id != "db.instance.identity":
                        facts.append(
                            TurnEvidenceFact(
                                evidence_ref=(
                                    f"artifact:{artifact_id}#"
                                    f"{outcome.step_id}"
                                ),
                                artifact_id=artifact_id,
                                skill_id=result.skill_id,
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
                    gaps.append(
                        TurnEvidenceGap(
                            skill_id=result.skill_id,
                            step_id=outcome.step_id,
                            code=outcome.gap.code,
                            detail=outcome.gap.detail,
                            retryable=outcome.gap.retryable,
                        )
                    )

        answer_context = dict(context.plan_snapshot.get("answer_context", {}))
        intent = dict(answer_context.get("intent", {}))
        clarification = intent.get("clarification_question")
        if clarification:
            return DbaSufficiencyAssessment(
                status=SufficiencyStatus.NEEDS_CLARIFICATION,
                evidence=tuple(facts),
                gaps=tuple(gaps),
                reasons=("当前问题存在影响诊断范围的必要歧义",),
                clarification_question=str(clarification),
            )

        time_window = dict(intent.get("time_window") or {})
        requested_window = time_window.get("mode") in {"RECENT", "ABSOLUTE"}
        cumulative_only = bool(facts) and all(
            fact.measurement_semantics
            == MeasurementSemantics.CUMULATIVE_SINCE_LOAD
            for fact in facts
        )
        if requested_window and cumulative_only:
            reasons.append(
                "请求的是时间窗口数据，但当前证据只有实例启动后的累计口径"
            )
        if gaps:
            reasons.append("部分受控取证步骤未能返回可验证结果")

        if not facts:
            status = SufficiencyStatus.NEEDS_EVIDENCE
            reasons.append("当前没有取得能够回答问题的主题证据")
        elif reasons:
            status = SufficiencyStatus.PARTIAL
        else:
            status = SufficiencyStatus.ANSWERABLE
        return DbaSufficiencyAssessment(
            status=status,
            evidence=tuple(facts),
            gaps=tuple(gaps),
            reasons=tuple(reasons),
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
                "intent": dict(answer_context.get("intent", {})),
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
                "intent": dict(answer_context.get("intent", {})),
                "sufficiency_status": str(assessment.status),
                "limitations": list(assessment.reasons),
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
        execution = dict(context.plan_snapshot.get("skill_execution", {}))
        invocations = dict(execution.get("invocations", {}))
        tools_by_step: dict[tuple[str, str], dict[str, Any]] = {}
        for invocation in invocations.values():
            skill_id = str(invocation.get("skill_id", ""))
            for tool in invocation.get("tools", ()):
                tools_by_step[(skill_id, str(tool.get("step_id", "")))] = tool
        requests: list[tuple[TurnEvidenceGap, dict[str, Any]]] = []
        seen_tools: set[str] = set()
        for gap in assessment.gaps:
            tool = tools_by_step.get((gap.skill_id, gap.step_id))
            tool_id = str((tool or {}).get("tool_id", ""))
            if tool is None or not tool_id or tool_id in seen_tools:
                continue
            seen_tools.add(tool_id)
            requests.append((gap, tool))
        if not requests:
            return None
        lines = [
            "\n### 自动取证未完成的项目",
            "",
        ]
        for gap, tool in requests:
            tool_id = str(tool["tool_id"])
            lines.append(f"- `{tool_id}`：{gap.detail}（`{gap.code}`）")
        lines.extend(
            [
                "",
                "请优先修正 Target 只读凭据或数据库对象权限后重新提问。",
                "如果暂时不能调整权限，也可以在目标数据库中执行下面的只读 SQL，",
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
        return TurnAnswerBlock(
            block_type=AnswerBlockType.EVIDENCE_REQUEST,
            schema_version="AIOPS_EVIDENCE_REQUEST_BLOCK.v1",
            payload={"markdown": "\n".join(lines)},
        )

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
