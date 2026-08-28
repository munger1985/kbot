"""专业 DBA Turn 证据充分性与自然回答测试。"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.turn_answer import (
    DbaAnswerDraft,
    DbaSufficiencyAssessment,
    TurnEvidenceGap,
)
from aiops_agent.application.runtime import AIOpsRuntimeService
from aiops_agent.adapters.model_serving import AIOpsStructuredModelClient
from aiops_agent.orchestration.diagnosis import DiagnosisPromptRegistry
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.task_worker import AIOpsTaskWorker
from aiops_agent.workers.turn_answer_handlers import (
    DbaAnswerComposeHandler,
    DbaEvidenceAssessmentHandler,
)
from platform_core.contracts.aiops import (
    AppendOpsTaskProgressCommand,
    AnswerBlockType,
    SufficiencyStatus,
    InvestigationAssessment,
)
from platform_core.identity import uuid7


class _AnswerModel:
    def __init__(self, *, evidence_refs: tuple[str, ...]) -> None:
        self.evidence_refs = evidence_refs

    async def generate_structured(self, **kwargs) -> StructuredModelResult:
        digest = "a" * 64
        return StructuredModelResult(
            output=DbaAnswerDraft(
                markdown=(
                    "当前累计耗时最高的是 SQL_ID `abc123`。"
                    "这组数据是实例启动后的累计值，不代表最近十五分钟增量。"
                ),
                evidence_refs=self.evidence_refs,
            ),
            receipt=ModelInvocationReceipt(
                purpose=kwargs["purpose"],
                schema_id="DBA_ANSWER_DRAFT.v1",
                model_technical_name="test-model",
                model_revision="1",
                prompt_id=kwargs["prompt_ref"]["prompt_id"],
                prompt_version=kwargs["prompt_ref"]["prompt_version"],
                prompt_sha256=kwargs["prompt_ref"]["prompt_sha256"],
                input_sha256=digest,
                output_sha256=digest,
                duration_ms=1,
            ),
        )


class _StreamAnswerModel:
    def __init__(self, answers: tuple[str, ...]) -> None:
        self.answers = list(answers)
        self.calls = 0

    async def stream_text(self, **_):
        answer = self.answers[self.calls]
        self.calls += 1
        midpoint = max(1, len(answer) // 2)
        yield answer[:midpoint]
        yield answer[midpoint:]


class _AssessmentModel:
    async def generate_structured(self, **kwargs) -> StructuredModelResult:
        digest = "a" * 64
        return StructuredModelResult(
            output=InvestigationAssessment(
                round_no=1,
                sufficiency_status="PARTIAL",
                verified_facts=("已取得Top SQL累计统计",),
                remaining_unknowns=("最近十五分钟增量",),
                hypothesis_updates={"h1": "SUPPORTED"},
                evidence_gaps=("缺少时间窗口增量",),
                next_action="ANSWER",
                progress_made=True,
                reason="现有证据足以给出有边界的回答",
            ),
            receipt=ModelInvocationReceipt(
                purpose=kwargs["purpose"],
                schema_id="aiops.investigation-assessment.v1",
                model_technical_name="test-model",
                model_revision="1",
                prompt_id=kwargs["prompt_ref"]["prompt_id"],
                prompt_version=kwargs["prompt_ref"]["prompt_version"],
                prompt_sha256=kwargs["prompt_ref"]["prompt_sha256"],
                input_sha256=digest,
                output_sha256=digest,
                duration_ms=1,
            ),
        )


class _ProgressService:
    def __init__(self) -> None:
        self.commands = []

    async def append_task_progress(self, command):
        self.commands.append(command)


class _ProjectionTurns:
    def __init__(self, *, invocation) -> None:
        self.invocation = invocation
        self.evidence = []
        self.messages = []
        self.blocks = []
        self.citations = []
        self.events = []

    async def get_playbook_invocation_by_task(self, **_):
        return self.invocation

    async def list_tool_invocations(self, **_):
        return []

    async def get_evidence_by_artifact(self, *, turn_id, artifact_id):
        return next(
            (
                row
                for row in self.evidence
                if row.turn_id == turn_id and row.artifact_id == artifact_id
            ),
            None,
        )

    async def add_evidence(self, row):
        self.evidence.append(row)
        return row

    async def add_event(self, row):
        self.events.append(row)
        return row

    async def get_message_by_artifact(self, *, turn_id, artifact_id):
        return next(
            (
                row
                for row in self.messages
                if row.turn_id == turn_id and row.artifact_id == artifact_id
            ),
            None,
        )

    async def add_message(self, row):
        self.messages.append(row)
        return row

    async def list_evidence(self, *, turn_id):
        return [row for row in self.evidence if row.turn_id == turn_id]

    async def add_answer_block(self, row):
        self.blocks.append(row)
        return row

    async def add_answer_citation(self, row):
        self.citations.append(row)
        return row


class _ProjectionConversations:
    def __init__(self, conversation) -> None:
        self.conversation = conversation

    async def get_conversation(self, **_):
        return self.conversation


class _ProgressRuns:
    def __init__(self, *, run, task, now) -> None:
        self.run = run
        self.task = task
        self.now = now
        self.events = []
        self.prior_event = None

    async def database_now(self):
        return self.now

    async def get_task(self, **_):
        return self.task

    async def get_run(self, **_):
        return self.run

    async def get_event_by_key(self, **_):
        return self.prior_event

    async def append_event(self, **kwargs):
        event = SimpleNamespace(
            sequence_no=len(self.events) + 1,
            **kwargs,
        )
        self.events.append(event)
        return event


class _ProgressTurns:
    def __init__(self, *, turn, run_id) -> None:
        self.turn = turn
        self.run_id = run_id
        self.events = []

    async def get_run_link_by_ops_run_id(self, *, ops_run_id):
        return (
            SimpleNamespace(turn_id=self.turn.turn_id)
            if ops_run_id == self.run_id
            else None
        )

    async def get_turn(self, **_):
        return self.turn

    async def add_event(self, row):
        self.events.append(row)
        return row


class _ProgressUow:
    def __init__(self, *, runs, turns) -> None:
        self.runs = runs
        self.turns = turns
        self.commit_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def commit(self):
        self.commit_count += 1


def _context(*, artifacts=(), recent: bool = False) -> TaskExecutionContext:
    task_frame = {
        "objectives": ["ASSESS"],
        "problem_statement": "分析最近十五分钟的 Top SQL",
        "success_criteria": ["识别主要 SQL 负载"],
        "time_scope": "最近十五分钟" if recent else None,
    }
    return TaskExecutionContext(
        run_id=str(uuid7()),
        task_id=str(uuid7()),
        task_key="evidence:assess",
        target_id=str(uuid7()),
        agent_id=str(uuid7()),
        trigger_type="API",
        trace_id="trace-turn-answer",
        attempt=1,
        deadline_at=None,
        plan_snapshot={
            "answer_context": {
                "question": "分析最近十五分钟的 Top SQL",
                "task_frame": task_frame,
                "model": {"technical_name": "test-model", "revision": "1"},
            }
        },
        policy_snapshot={},
        input_artifacts=artifacts,
    )


def _skill_artifact(*, semantics: str, row_count: int = 1) -> dict:
    artifact_id = str(uuid7())
    rows = [["abc123", 120.5]] if row_count else []
    return {
        "artifact_id": artifact_id,
        "schema_version": "DBA_SKILL_RESULT.v1",
        "payload": {
            "schema_version": "DBA_SKILL_RESULT.v1",
            "skill_id": "oracle.sql.top_current",
            "skill_version": "1.0.0",
            "manifest_hash": "b" * 64,
            "output_schema": "oracle.sql.top_current.output.v1",
            "measurement_semantics": semantics,
            "presentation_kind": "TABLE_AND_CHART",
            "status": "SUCCEEDED",
            "tool_outcomes": [
                {
                    "step_id": "top_sql",
                    "tool_id": "db.sql.top_current",
                    "tool_version": "1.0.0",
                    "status": "SUCCEEDED",
                    "observation": {
                        "schema_version": "DATABASE_OBSERVATION.v1",
                        "executor_request_id": str(uuid7()),
                        "target_id": str(uuid7()),
                        "tool_id": "db.sql.top_current",
                        "tool_version": "1.0.0",
                        "variant": "oracle-19-current",
                        "template_sha256": "c" * 64,
                        "db_type": "ORACLE",
                        "db_version": "19c",
                        "capability_snapshot_hash": "d" * 64,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "duration_ms": 8,
                        "columns": [
                            {
                                "name": "SQL_ID",
                                "logical_type": "STRING",
                                "sensitivity": "PUBLIC",
                            },
                            {
                                "name": "ELAPSED_SECONDS",
                                "logical_type": "DECIMAL",
                                "sensitivity": "PUBLIC",
                            },
                        ],
                        "rows": rows,
                        "row_count": row_count,
                        "truncated": False,
                        "result_sha256": "e" * 64,
                        "parameters_sha256": "f" * 64,
                    },
                }
            ],
        },
    }


def _monitoring_artifact() -> dict:
    now = datetime.now(UTC)
    start = now - timedelta(minutes=15)
    artifact_id = str(uuid7())
    target_id = str(uuid7())
    binding_id = str(uuid7())
    source_id = str(uuid7())
    return {
        "artifact_id": artifact_id,
        "schema_version": "OBSERVATION_SET.v1",
        "payload": {
            "schema_version": "OBSERVATION_SET.v1",
            "target_id": target_id,
            "binding_id": binding_id,
            "source_id": source_id,
            "collected_at": now.isoformat(),
            "observations": [
                {
                    "metric_code": "host.cpu.utilization",
                    "semantic_version": "1.0.0",
                    "unit": "percent",
                    "value_kind": "GAUGE",
                    "window_start": start.isoformat(),
                    "window_end": now.isoformat(),
                    "requested_step_seconds": 60,
                    "effective_step_seconds": 60,
                    "source_id": source_id,
                    "source_type": "PROMETHEUS",
                    "source_version": 1,
                    "target_id": target_id,
                    "binding_id": binding_id,
                    "external_target_fingerprint": "a" * 64,
                    "series": [
                        {
                            "dimensions": {"target_key": "oracle-dev-190"},
                            "points": [
                                {
                                    "observed_at": start.isoformat(),
                                    "value": 12.5,
                                },
                                {
                                    "observed_at": now.isoformat(),
                                    "value": 18.5,
                                },
                            ],
                        }
                    ],
                    "expected_points": 2,
                    "actual_points": 2,
                    "coverage_ratio": 1,
                }
            ],
        },
    }


class DbaTurnAnswerTest(unittest.TestCase):
    def test_runtime_projects_direct_tool_without_playbook_parent(self) -> None:
        service = object.__new__(AIOpsRuntimeService)
        turn = SimpleNamespace(
            turn_id=uuid7(),
            conversation_id=uuid7(),
            domain_id=7,
            created_by="dba@example.com",
            event_cursor=0,
            status="COLLECTING",
        )
        turns = _ProjectionTurns(invocation=None)
        uow = SimpleNamespace(turns=turns)
        skill_input = _skill_artifact(semantics="CURRENT_ACTIVITY")
        artifact = SimpleNamespace(
            artifact_id=uuid7(), payload_json=skill_input["payload"]
        )

        asyncio.run(
            service._project_skill_result(
                uow=uow,
                turn=turn,
                task=SimpleNamespace(ops_task_id=uuid7(), attempt_count=1),
                artifact=artifact,
                payload=artifact.payload_json,
                now=datetime.now(UTC),
            )
        )

        self.assertEqual(1, len(turns.evidence))
        self.assertNotIn(
            "playbook.completed", {event.event_type for event in turns.events}
        )

    def test_assessment_model_updates_hypotheses_and_next_action(self) -> None:
        result = asyncio.run(
            DbaEvidenceAssessmentHandler(
                model_client=_AssessmentModel()
            ).execute(
                _context(
                    artifacts=(
                        _skill_artifact(
                            semantics="CUMULATIVE_SINCE_LOAD"
                        ),
                    ),
                    recent=True,
                )
            )
        )

        self.assertEqual(SufficiencyStatus.PARTIAL, result.status)
        self.assertIsNotNone(result.investigation)
        self.assertEqual("ANSWER", result.investigation.next_action)
        self.assertEqual(
            "SUPPORTED", result.investigation.hypothesis_updates["h1"]
        )

    def test_chat_monitor_health_uses_run_domain(self) -> None:
        source_id = uuid7()
        binding_id = uuid7()
        run = SimpleNamespace(
            domain_id=7,
            target_id=uuid7(),
            plan_snapshot_json={
                "monitoring": {
                    "bindings": [
                        {
                            "binding_id": str(binding_id),
                            "binding_version": 1,
                            "source": {
                                "source_id": str(source_id),
                                "config_version": 1,
                            },
                        }
                    ]
                }
            },
        )
        source = SimpleNamespace(
            diagnostic_source_id=source_id,
            domain_id=7,
            connectivity_status="CONNECTED",
            connectivity_version=1,
        )
        diagnostic_sources = SimpleNamespace(
            get_scoped=AsyncMock(return_value=source),
            reduce_connectivity=AsyncMock(),
        )
        targets = SimpleNamespace(
            get_source_binding_scoped=AsyncMock(return_value=None)
        )
        uow = SimpleNamespace(
            diagnostic_sources=diagnostic_sources,
            targets=targets,
        )

        asyncio.run(
            AIOpsRuntimeService._reduce_observation_health(
                uow=uow,
                run=run,
                payload={
                    "binding_id": str(binding_id),
                    "observations": [],
                    "gaps": [],
                },
                now=datetime.now(UTC),
            )
        )

        diagnostic_sources.get_scoped.assert_awaited_once_with(
            diagnostic_source_id=source_id,
            domain_id=7,
        )

    def test_prometheus_observation_is_aggregated_as_one_fact(self) -> None:
        result = asyncio.run(
            DbaEvidenceAssessmentHandler().execute(
                _context(artifacts=(_monitoring_artifact(),))
            )
        )

        self.assertEqual(SufficiencyStatus.ANSWERABLE, result.status)
        self.assertEqual(1, len(result.evidence))
        fact = result.evidence[0]
        self.assertEqual("monitoring.overview", fact.skill_id)
        self.assertEqual("HISTORICAL_SAMPLES", fact.measurement_semantics)
        self.assertEqual("host.cpu.utilization", fact.rows[0][0])
        self.assertEqual(15.5, fact.rows[0][3])

    def test_waiting_user_includes_exact_readonly_sql_and_gap_reason(self) -> None:
        assessment = DbaSufficiencyAssessment(
            status=SufficiencyStatus.NEEDS_EVIDENCE,
            gaps=(
                TurnEvidenceGap(
                    skill_id="oracle.sql.top_current",
                    step_id="top_sql",
                    code="PRIVILEGE_MISSING",
                    detail="Target 只读凭据缺少对象查询权限",
                ),
            ),
            reasons=("当前没有取得能够回答问题的主题证据",),
        )
        context = _context(
            artifacts=(
                {
                    "artifact_id": str(uuid7()),
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            )
        )
        context = replace(
            context,
            plan_snapshot={
                **context.plan_snapshot,
                "skill_execution": {
                    "invocations": {
                        "skill:1:oracle.sql.top_current": {
                            "skill_id": "oracle.sql.top_current",
                            "tools": [
                                {
                                    "step_id": "top_sql",
                                    "tool_id": "db.sql.top_current",
                                    "manual_sql": "SELECT * FROM v$sqlstats WHERE ROWNUM <= :limit",
                                    "parameters": {"limit": 10},
                                    "required_privileges": ["V_$SQLSTATS"],
                                }
                            ],
                        }
                    }
                },
            },
        )
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=()),
            prompts=DiagnosisPromptRegistry.load(),
        )

        result = asyncio.run(handler.execute(context))

        self.assertEqual("WAITING_USER", result.status)
        self.assertEqual(
            AnswerBlockType.EVIDENCE_REQUEST,
            result.blocks[1].block_type,
        )
        markdown = result.blocks[1].payload["markdown"]
        self.assertIn("PRIVILEGE_MISSING", markdown)
        self.assertIn("V_$SQLSTATS", markdown)
        self.assertIn("ROWNUM <= 10", markdown)
        self.assertNotIn(":limit", markdown)

    def test_recent_request_with_cumulative_evidence_is_partial(self) -> None:
        context = _context(
            artifacts=(
                _skill_artifact(semantics="CUMULATIVE_SINCE_LOAD"),
            ),
            recent=True,
        )

        result = asyncio.run(DbaEvidenceAssessmentHandler().execute(context))

        self.assertEqual(SufficiencyStatus.PARTIAL, result.status)
        self.assertEqual(1, len(result.evidence))
        self.assertIn("累计口径", result.reasons[0])

    def test_partial_answer_requests_missing_prometheus_evidence(self) -> None:
        base = asyncio.run(
            DbaEvidenceAssessmentHandler().execute(
                _context(
                    artifacts=(
                        _skill_artifact(semantics="CURRENT_ACTIVITY"),
                    )
                )
            )
        )
        assessment = base.model_copy(
            update={
                "status": SufficiencyStatus.PARTIAL,
                "gaps": (
                    TurnEvidenceGap(
                        skill_id="monitoring.overview",
                        step_id="host.cpu.utilization",
                        code="SOURCE_NO_DATA",
                        detail="Prometheus 未返回采样",
                    ),
                ),
                "reasons": ("主机 CPU 监控未返回采样",),
            }
        )
        context = _context(
            artifacts=(
                {
                    "artifact_id": str(uuid7()),
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            )
        )
        context = replace(
            context,
            plan_snapshot={
                **context.plan_snapshot,
                "monitoring": {
                    "bindings": [
                        {
                            "source_locator_key": "oracle-dev-190",
                            "source_locator": {
                                "host_target_key": "dev-db-host-190"
                            },
                            "source": {"source_type": "PROMETHEUS"},
                            "mapping_overrides": {},
                            "metrics": [
                                {
                                    "metric_code": "host.cpu.utilization",
                                    "providers": {
                                        "PROMETHEUS": {
                                            "query_template": (
                                                "node_cpu_seconds_total{"
                                                'target_key="${host_target}"}'
                                            )
                                        }
                                    },
                                }
                            ],
                        }
                    ]
                },
            },
        )
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(
                evidence_refs=(assessment.evidence[0].evidence_ref,)
            ),
            prompts=DiagnosisPromptRegistry.load(),
        )

        result = asyncio.run(handler.execute(context))

        self.assertEqual("PARTIAL", result.status)
        self.assertEqual(
            AnswerBlockType.EVIDENCE_REQUEST,
            result.blocks[-1].block_type,
        )
        markdown = result.blocks[-1].payload["markdown"]
        self.assertIn("dev-db-host-190", markdown)
        self.assertIn("监控已补齐", markdown)

    def test_empty_current_result_is_still_answerable_fact(self) -> None:
        context = _context(
            artifacts=(
                _skill_artifact(
                    semantics="CURRENT_ACTIVITY",
                    row_count=0,
                ),
            )
        )

        result = asyncio.run(DbaEvidenceAssessmentHandler().execute(context))

        self.assertEqual(SufficiencyStatus.ANSWERABLE, result.status)
        self.assertEqual(0, result.evidence[0].row_count)

    def test_answer_uses_narrative_plus_server_generated_data_blocks(self) -> None:
        assessment = asyncio.run(
            DbaEvidenceAssessmentHandler().execute(
                _context(
                    artifacts=(
                        _skill_artifact(semantics="CURRENT_ACTIVITY"),
                    )
                )
            )
        )
        evidence_ref = assessment.evidence[0].evidence_ref
        context = _context(
            artifacts=(
                {
                    "artifact_id": str(uuid7()),
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            )
        )
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=(evidence_ref,)),
            prompts=DiagnosisPromptRegistry.load(),
        )

        result = asyncio.run(handler.execute(context))

        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(
            [AnswerBlockType.MARKDOWN, AnswerBlockType.TABLE, AnswerBlockType.CHART],
            [item.block_type for item in result.blocks],
        )
        self.assertNotIn(evidence_ref, result.blocks[0].payload["markdown"])
        self.assertEqual((evidence_ref,), result.blocks[0].evidence_refs)

    def test_answer_rejects_reference_outside_current_turn(self) -> None:
        assessment = asyncio.run(
            DbaEvidenceAssessmentHandler().execute(
                _context(
                    artifacts=(
                        _skill_artifact(semantics="CURRENT_ACTIVITY"),
                    )
                )
            )
        )
        context = _context(
            artifacts=(
                {
                    "artifact_id": str(uuid7()),
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            )
        )
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=("artifact:other#fact",)),
            prompts=DiagnosisPromptRegistry.load(),
        )

        with self.assertRaisesRegex(ValueError, "批准证据之外"):
            asyncio.run(handler.execute(context))

    def test_streamed_answer_is_validated_then_emits_progress(self) -> None:
        assessment = asyncio.run(
            DbaEvidenceAssessmentHandler().execute(
                _context(
                    artifacts=(
                        _skill_artifact(semantics="CURRENT_ACTIVITY"),
                    )
                )
            )
        )
        context = _context(
            artifacts=(
                {
                    "artifact_id": str(uuid7()),
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            )
        )
        model = _StreamAnswerModel(
            ("当前累计耗时最高的是 SQL_ID abc123。[E1]",)
        )
        handler = DbaAnswerComposeHandler(
            model_client=model,
            prompts=DiagnosisPromptRegistry.load(),
        )

        async def collect():
            return [item async for item in handler.execute_stream(context)]

        items = asyncio.run(collect())
        final = items[-1]

        self.assertEqual("thinking.delta", items[0].event_type)
        self.assertTrue(
            any(item.event_type == "answer.delta" for item in items[:-1])
        )
        self.assertTrue(final.answer_streamed)
        self.assertNotIn("[E1]", final.blocks[0].payload["markdown"])
        self.assertEqual(
            (assessment.evidence[0].evidence_ref,),
            final.blocks[0].evidence_refs,
        )

    def test_streamed_answer_retries_unknown_reference(self) -> None:
        assessment = asyncio.run(
            DbaEvidenceAssessmentHandler().execute(
                _context(
                    artifacts=(
                        _skill_artifact(semantics="CURRENT_ACTIVITY"),
                    )
                )
            )
        )
        context = _context(
            artifacts=(
                {
                    "artifact_id": str(uuid7()),
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            )
        )
        model = _StreamAnswerModel(
            ("错误引用。[E9]", "已按证据修正。[E1]")
        )
        handler = DbaAnswerComposeHandler(
            model_client=model,
            prompts=DiagnosisPromptRegistry.load(),
        )

        async def collect():
            return [item async for item in handler.execute_stream(context)]

        items = asyncio.run(collect())

        self.assertEqual(2, model.calls)
        self.assertTrue(
            any(item.event_key == "answer-thinking:retry" for item in items[:-1])
        )
        self.assertEqual("已按证据修正。", items[-1].blocks[0].payload["markdown"])

    def test_task_worker_commits_stream_progress_before_final_artifact(self) -> None:
        service = _ProgressService()
        worker = AIOpsTaskWorker(
            runtime_service=service,
            handler_registry=object(),
            worker_id="worker-1",
            lease_seconds=30,
            heartbeat_seconds=10,
            poll_interval_seconds=1,
        )

        class _Handler:
            async def execute_stream(self, context):
                del context
                from aiops_agent.contracts.turn_answer import (
                    AIOpsTurnResult,
                    DbaAnswerProgress,
                    TurnAnswerBlock,
                )

                yield DbaAnswerProgress(
                    event_type="answer.delta",
                    event_key="answer-delta:1",
                    payload={"delta": "回答"},
                )
                yield AIOpsTurnResult(
                    status="COMPLETED",
                    sufficiency_status="ANSWERABLE",
                    blocks=(
                        TurnAnswerBlock(
                            block_type="MARKDOWN",
                            schema_version="AIOPS_MARKDOWN_BLOCK.v1",
                            payload={"markdown": "回答"},
                        ),
                    ),
                    answer_streamed=True,
                )

        lease = SimpleNamespace(
            task_id=uuid7(),
            lease_token=uuid7(),
            trace_id="trace-stream",
            attempt=1,
        )
        result = asyncio.run(
            worker._invoke_handler(
                manifest=SimpleNamespace(implementation=_Handler()),
                context=object(),
                current={"lease": lease},
            )
        )

        self.assertEqual(1, len(service.commands))
        self.assertEqual("answer.delta", service.commands[0].event_type)
        self.assertTrue(result.answer_streamed)

    def test_model_sse_decoder_accepts_openai_delta(self) -> None:
        line = (
            'data: {"choices":[{"delta":{"content":"诊断结果"}}]}\n'
        )

        self.assertEqual(
            "诊断结果",
            AIOpsStructuredModelClient._decode_stream_line(line),
        )
        self.assertIsNone(
            AIOpsStructuredModelClient._decode_stream_line("data: [DONE]\n")
        )

    def test_runtime_progress_is_committed_to_run_and_turn_streams(self) -> None:
        now = datetime.now(UTC)
        run_id = uuid7()
        task_id = uuid7()
        lease_token = uuid7()
        run = SimpleNamespace(
            ops_run_id=run_id,
            status="RUNNING",
            workflow_kind="CHAT_TURN",
            domain_id=7,
            cancel_requested_at=None,
            deadline_at=now + timedelta(minutes=5),
            row_version=1,
        )
        task = SimpleNamespace(
            ops_task_id=task_id,
            ops_run_id=run_id,
            status="RUNNING",
            lease_owner="worker-1",
            lease_token=lease_token,
            lease_until=now + timedelta(minutes=1),
            row_version=1,
        )
        turn = SimpleNamespace(turn_id=uuid7(), event_cursor=4)
        runs = _ProgressRuns(run=run, task=task, now=now)
        turns = _ProgressTurns(turn=turn, run_id=run_id)
        uow = _ProgressUow(runs=runs, turns=turns)
        service = object.__new__(AIOpsRuntimeService)
        service._uow_factory = lambda: uow

        command = AppendOpsTaskProgressCommand(
            task_id=task_id,
            worker_id="worker-1",
            lease_token=lease_token,
            trace_id="trace-progress",
            event_type="answer.delta",
            event_key="answer-delta:1",
            payload={"chunk_index": 1, "delta": "诊断"},
        )
        receipt = asyncio.run(service.append_task_progress(command))

        self.assertEqual(1, uow.commit_count)
        self.assertEqual("answer.delta", runs.events[0].event_type)
        self.assertEqual("answer.delta", turns.events[0].event_type)
        self.assertEqual(5, turn.event_cursor)
        self.assertEqual(task_id, receipt.task_id)

        runs.prior_event = runs.events[0]
        repeated = asyncio.run(service.append_task_progress(command))
        self.assertEqual(receipt.event_cursor, repeated.event_cursor)
        with self.assertRaises(Exception) as raised:
            asyncio.run(
                service.append_task_progress(
                    command.model_copy(
                        update={
                            "payload": {
                                "chunk_index": 1,
                                "delta": "不同内容",
                            }
                        }
                    )
                )
            )
        self.assertIn("幂等键对应的内容不一致", str(raised.exception))

    def test_runtime_projects_skill_evidence_answer_and_citation(self) -> None:
        service = object.__new__(AIOpsRuntimeService)
        turn = SimpleNamespace(
            turn_id=uuid7(),
            conversation_id=uuid7(),
            domain_id=7,
            created_by="dba@example.com",
            event_cursor=0,
            status="COLLECTING",
            sufficiency_status=None,
            completed_at=None,
        )
        conversation = SimpleNamespace(
            last_message_no=1,
            updated_by=None,
            updated_at=None,
        )
        invocation = SimpleNamespace(
            turn_id=turn.turn_id,
            playbook_invocation_id=uuid7(),
            status="PLANNED",
            output_artifact_id=None,
            attempt_count=0,
            completed_at=None,
        )
        turns = _ProjectionTurns(invocation=invocation)
        uow = SimpleNamespace(
            turns=turns,
            conversations=_ProjectionConversations(conversation),
        )
        skill_input = _skill_artifact(semantics="CURRENT_ACTIVITY")
        skill_artifact = SimpleNamespace(
            artifact_id=uuid7(),
            payload_json=skill_input["payload"],
        )
        task = SimpleNamespace(ops_task_id=uuid7(), attempt_count=1)
        now = datetime.now(UTC)

        asyncio.run(
            service._project_skill_result(
                uow=uow,
                turn=turn,
                task=task,
                artifact=skill_artifact,
                payload=skill_artifact.payload_json,
                now=now,
            )
        )
        monitoring_input = _monitoring_artifact()
        monitoring_artifact = SimpleNamespace(
            artifact_id=uuid7(),
            payload_json=monitoring_input["payload"],
        )
        asyncio.run(
            service._project_monitoring_result(
                uow=uow,
                turn=turn,
                artifact=monitoring_artifact,
                payload=monitoring_artifact.payload_json,
            )
        )
        evidence_ref = f"artifact:{skill_artifact.artifact_id}#top_sql"
        monitoring_ref = (
            f"artifact:{monitoring_artifact.artifact_id}#prometheus"
        )
        answer_artifact = SimpleNamespace(
            artifact_id=uuid7(),
            payload_json={},
        )
        answer_payload = {
            "schema_version": "AIOPS_TURN_RESULT.v1",
            "status": "COMPLETED",
            "sufficiency_status": "ANSWERABLE",
            "blocks": [
                {
                    "block_type": "MARKDOWN",
                    "schema_version": "AIOPS_MARKDOWN_BLOCK.v1",
                    "payload": {"markdown": "当前没有阻塞会话。"},
                    "evidence_refs": [evidence_ref, monitoring_ref],
                }
            ],
        }

        asyncio.run(
            service._project_turn_answer(
                uow=uow,
                turn=turn,
                artifact=answer_artifact,
                payload=answer_payload,
                now=now,
            )
        )

        self.assertEqual("SUCCEEDED", invocation.status)
        self.assertEqual(skill_artifact.artifact_id, invocation.output_artifact_id)
        self.assertEqual(2, len(turns.evidence))
        self.assertEqual(
            "HISTORICAL_SAMPLES",
            turns.evidence[1].measurement_semantics,
        )
        self.assertEqual(1, len(turns.messages))
        self.assertEqual(1, len(turns.blocks))
        self.assertEqual(2, len(turns.citations))
        self.assertEqual("COMPLETED", turn.status)
        self.assertEqual("当前没有阻塞会话。", turns.messages[0].payload_json["text"])


if __name__ == "__main__":
    unittest.main()
