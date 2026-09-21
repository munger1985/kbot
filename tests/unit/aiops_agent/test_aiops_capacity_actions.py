"""容量方案只能观察 / AUTOEXTEND / RESIZE，永不 ADD DATAFILE。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from aiops_agent.actions import ActionRegistry
from aiops_agent.application.diagnosis import (
    CAPACITY_ADD_ACTION_IDS,
    allows_capacity_action,
    compile_findings,
    decide_capacity_actions,
)
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.turn_answer import (
    DbaAnswerDraft,
    DbaSufficiencyAssessment,
    DiagnosisAnswerDraft,
    TurnEvidenceFact,
)
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.workers.change_handlers import ChatActionPlanHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.contracts.aiops import (
    AnswerBlockType,
    MeasurementSemantics,
    SufficiencyStatus,
)
from platform_core.identity import uuid7


TEST_PROMPT_SNAPSHOT = {"test": {"prompt_version_id": str(uuid7())}}
_STORAGE_COLUMNS = (
    "file_name",
    "current_size_mb",
    "current_max_size_mb",
    "autoextensible",
    "current_next_mb",
    "requested_size_mb",
    "requested_next_mb",
    "requested_max_size_mb",
    "status",
    "online_status",
)
_TABLESPACE_COLUMNS = (
    "tablespace_name",
    "allocated_mb",
    "used_mb",
    "free_mb",
    "used_percent",
    "maximum_mb",
    "maximum_headroom_mb",
    "file_count",
)


class _Prompt:
    content = "测试诊断 Prompt"

    @staticmethod
    def ref() -> dict[str, str]:
        return {
            "prompt_id": "aiops_agent.test",
            "prompt_version": "1.0.0",
            "prompt_sha256": "d" * 64,
            "prompt_version_id": str(uuid7()),
            "prompt_source": "DATABASE",
        }


class _TestPrompts:
    async def resolve(self, *_args, **_kwargs):
        return _Prompt()


class _AnswerModel:
    def __init__(self, *, evidence_refs: tuple[str, ...] = ()) -> None:
        self.evidence_refs = evidence_refs
        self.calls = []

    async def generate_structured(self, **kwargs) -> StructuredModelResult:
        self.calls.append(kwargs)
        output_model = kwargs.get("output_model") or DbaAnswerDraft
        if output_model is DiagnosisAnswerDraft:
            output = DiagnosisAnswerDraft(
                analysis_markdown="表空间 USERS 使用率偏高。",
                solution_markdown="按容量方案处理，不加文件。",
                evidence_refs=self.evidence_refs,
            )
        else:
            output = DbaAnswerDraft(
                markdown="解释类回答",
                evidence_refs=self.evidence_refs,
            )
        digest = "a" * 64
        return StructuredModelResult(
            output=output,
            receipt=ModelInvocationReceipt(
                purpose=kwargs["purpose"],
                schema_id="DBA_DIAGNOSIS_ANSWER_DRAFT.v1",
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


def _fact(
    *,
    tool_id: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
    evidence_ref: str | None = None,
) -> TurnEvidenceFact:
    return TurnEvidenceFact(
        evidence_ref=evidence_ref or f"artifact:{tool_id}#fact",
        artifact_id=str(uuid7()),
        source_id=tool_id,
        step_id="a1",
        tool_id=tool_id,
        trust_level="SOURCE_VERIFIED",
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind="TABLE",
        captured_at=datetime.now(UTC).isoformat(),
        columns=tuple({"name": name} for name in columns),
        rows=rows,
        row_count=len(rows),
    )


def _tablespace_fact(
    row: tuple[object, ...] = (
        "USERS",
        1000.0,
        900.0,
        100.0,
        90.0,
        2000.0,
        1100.0,
        2,
    ),
) -> TurnEvidenceFact:
    return _fact(
        tool_id="db.storage.capacity",
        columns=_TABLESPACE_COLUMNS,
        rows=(row,),
        evidence_ref="artifact:capacity#fact",
    )


def _action_state_fact(
    *,
    tool_id: str = "db.storage.datafile.action_state",
    row: tuple[object, ...] = (
        "+DATA/DB/data01.dbf",
        1024,
        1024,
        "NO",
        0,
        2048,
        128,
        4096,
        "AVAILABLE",
        "ONLINE",
    ),
) -> TurnEvidenceFact:
    return _fact(
        tool_id=tool_id,
        columns=_STORAGE_COLUMNS,
        rows=(row,),
        evidence_ref="artifact:storage#fact",
    )


def _answer_context(
    *,
    artifacts,
    target_facts=(),
) -> TaskExecutionContext:
    return TaskExecutionContext(
        run_id=str(uuid7()),
        task_id=str(uuid7()),
        task_key="evidence:assess",
        target_id=str(uuid7()),
        agent_id=str(uuid7()),
        trigger_type="API",
        trace_id="trace-capacity",
        attempt=1,
        deadline_at=None,
        plan_snapshot={
            "answer_context": {
                "question": "USERS 表空间快满了怎么办？",
                "workflow_kind": "",
                "task_frame": {
                    "objectives": ["ASSESS"],
                    "problem_statement": "分析表空间容量",
                    "success_criteria": ["给出容量方案"],
                    "action_intent": "NONE",
                },
                "model": {"technical_name": "test-model", "revision": "1"},
                "prompts": TEST_PROMPT_SNAPSHOT,
            },
            "target_facts": list(target_facts),
        },
        policy_snapshot={},
        input_artifacts=artifacts,
    )


def _sufficiency_artifact(evidence: tuple[TurnEvidenceFact, ...]) -> dict:
    assessment = DbaSufficiencyAssessment(
        status=SufficiencyStatus.ANSWERABLE,
        evidence=evidence,
        reasons=("已取得本轮只读证据",),
    )
    return {
        "artifact_id": str(uuid7()),
        "schema_version": "DBA_SUFFICIENCY.v1",
        "payload": assessment.model_dump(mode="json"),
    }


def _storage_plan_context(
    *,
    evidence: tuple[TurnEvidenceFact, ...],
    target_facts=(),
    allowed_action_ids: tuple[str, ...] = (
        "db.storage.datafile.autoextend",
        "db.storage.tempfile.autoextend",
        "db.storage.datafile.resize",
        "db.storage.tempfile.resize",
        "db.storage.datafile.add",
        "db.storage.tempfile.add",
    ),
) -> TaskExecutionContext:
    assessment = DbaSufficiencyAssessment(
        status=SufficiencyStatus.ANSWERABLE,
        evidence=evidence,
    )
    return TaskExecutionContext(
        run_id="run-capacity",
        task_id="action-capacity",
        task_key="change:action-plan",
        target_id="target-1",
        agent_id="agent-1",
        trigger_type="CHAT",
        trace_id="trace-capacity-action",
        attempt=1,
        deadline_at=None,
        plan_snapshot={
            "capability_snapshot": {
                "target_capabilities": ["dba_catalog_views"]
            },
            "answer_context": {
                "task_frame": {
                    "action_intent": "EXECUTE",
                    "requires_change": True,
                }
            },
            "target_facts": list(target_facts),
            "change_context": {
                "target": {
                    "db_type": "ORACLE",
                    "version_code": "19.0.0",
                    "environment": "PROD",
                    "status": "ENABLED",
                    "connectivity_status": "CONNECTED",
                    "execution_secret_configured": True,
                },
                "policy": {"rules": {}},
                "controlled_action_execution": {
                    "enabled": True,
                    "allowed_action_ids": list(allowed_action_ids),
                    "object_scopes": {
                        "schemas": [],
                        "exclude_system_objects": True,
                    },
                },
            },
        },
        policy_snapshot={},
        input_artifacts=(
            {
                "schema_version": "DBA_SUFFICIENCY.v1",
                "payload": assessment.model_dump(mode="json"),
            },
        ),
    )


class CapacityDecisionTest(unittest.TestCase):
    def test_facts_and_autoextend_headroom_allow_autoextend_only(self) -> None:
        compilation = compile_findings((_tablespace_fact(),), target_id="tgt-1")
        plan = decide_capacity_actions(
            findings=compilation.findings,
            target_facts=({"fact_type": "ASM_DISKGROUP", "status": "ACTIVE"},),
        )
        self.assertEqual(1, len(plan.decisions))
        self.assertEqual("AUTOEXTEND", plan.decisions[0].kind)
        self.assertEqual(1000.0, plan.decisions[0].autoextend_headroom_mb)
        self.assertEqual(
            (
                "db.storage.datafile.autoextend",
                "db.storage.tempfile.autoextend",
            ),
            plan.decisions[0].allowed_action_ids,
        )
        self.assertTrue(
            allows_capacity_action(plan, "db.storage.datafile.autoextend")
        )
        self.assertFalse(
            allows_capacity_action(plan, "db.storage.datafile.resize")
        )
        self.assertFalse(
            allows_capacity_action(plan, "db.storage.datafile.add")
        )
        self.assertEqual(
            sorted(CAPACITY_ADD_ACTION_IDS),
            plan.as_dict()["blocked_action_ids"],
        )

    def test_facts_without_autoextend_headroom_allow_resize_only(self) -> None:
        compilation = compile_findings(
            (
                _tablespace_fact(
                    ("USERS", 1000.0, 900.0, 100.0, 90.0, 1000.0, 100.0, 2)
                ),
            ),
            target_id="tgt-1",
        )
        plan = decide_capacity_actions(
            findings=compilation.findings,
            target_facts=({"fact_type": "DATAFILE_PATH", "status": "ACTIVE"},),
        )
        self.assertEqual("RESIZE", plan.decisions[0].kind)
        self.assertEqual(0.0, plan.decisions[0].autoextend_headroom_mb)
        self.assertEqual(
            (
                "db.storage.datafile.resize",
                "db.storage.tempfile.resize",
            ),
            plan.decisions[0].allowed_action_ids,
        )
        self.assertTrue(
            allows_capacity_action(plan, "db.storage.datafile.resize")
        )
        self.assertFalse(
            allows_capacity_action(plan, "db.storage.datafile.autoextend")
        )
        self.assertFalse(allows_capacity_action(plan, "db.storage.tempfile.add"))

    def test_missing_maximum_uses_headroom_minus_free(self) -> None:
        fact = _fact(
            tool_id="db.storage.capacity",
            columns=(
                "tablespace_name",
                "used_percent",
                "free_mb",
                "maximum_headroom_mb",
                "file_count",
            ),
            rows=(("USERS", 90.0, 100.0, 1100.0, 2),),
        )
        compilation = compile_findings((fact,), target_id="tgt-1")
        plan = decide_capacity_actions(
            findings=compilation.findings,
            target_facts=({"fact_type": "ASM_DISKGROUP", "status": "ACTIVE"},),
        )
        self.assertEqual("AUTOEXTEND", plan.decisions[0].kind)
        self.assertEqual(1000.0, plan.decisions[0].autoextend_headroom_mb)

    def test_missing_or_retired_facts_observe_without_grow_or_add(self) -> None:
        compilation = compile_findings((_tablespace_fact(),), target_id="tgt-1")
        for facts in (
            (),
            ({"fact_type": "ASM_DISKGROUP", "status": "RETIRED"},),
            ({"fact_type": "TABLESPACE_PLACEMENT", "status": "ACTIVE"},),
        ):
            plan = decide_capacity_actions(
                findings=compilation.findings,
                target_facts=facts,
            )
            self.assertEqual("OBSERVE", plan.decisions[0].kind)
            self.assertEqual((), plan.decisions[0].allowed_action_ids)
            self.assertEqual((), plan.allowed_action_ids)
            self.assertFalse(
                allows_capacity_action(plan, "db.storage.datafile.autoextend")
            )
            self.assertFalse(
                allows_capacity_action(plan, "db.storage.datafile.resize")
            )
            self.assertFalse(
                allows_capacity_action(plan, "db.storage.datafile.add")
            )


class CapacityAnswerInputTest(unittest.TestCase):
    def test_diagnosis_model_input_includes_capacity_plan(self) -> None:
        fact = _tablespace_fact()
        model = _AnswerModel(evidence_refs=(fact.evidence_ref,))
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=model,
                prompts=_TestPrompts(),
            ).execute(
                _answer_context(
                    artifacts=(_sufficiency_artifact((fact,)),),
                    target_facts=(),
                )
            )
        )
        payload = model.calls[0]["input_payload"]
        self.assertEqual("OBSERVE", payload["capacity_plan"]["decisions"][0]["kind"])
        self.assertEqual([], payload["capacity_plan"]["allowed_action_ids"])
        self.assertEqual(
            ["db.storage.datafile.add", "db.storage.tempfile.add"],
            payload["capacity_plan"]["blocked_action_ids"],
        )
        self.assertIn(AnswerBlockType.FACT_CONFIRMATION, [item.block_type for item in result.blocks])

    def test_missing_facts_keep_confirmation_and_never_add_file(self) -> None:
        fact = _tablespace_fact()
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
                prompts=_TestPrompts(),
            ).execute(
                _answer_context(
                    artifacts=(_sufficiency_artifact((fact,)),),
                    target_facts=(),
                )
            )
        )
        self.assertEqual(
            AnswerBlockType.FACT_CONFIRMATION,
            result.blocks[3].block_type,
        )
        self.assertNotIn("command_preview", result.blocks[3].payload)
        self.assertNotIn("proposal_id", result.blocks[3].payload)


class CapacityActionPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ActionRegistry.load()

    def test_action_state_without_facts_does_not_compile_storage(self) -> None:
        plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(
                _storage_plan_context(
                    evidence=(_tablespace_fact(), _action_state_fact()),
                    target_facts=(),
                )
            )
        )
        self.assertEqual("NO_ACTION", plan.decision)
        self.assertEqual((), plan.actions)
        self.assertIn(
            "VERIFIED_ACTION_PARAMETERS_UNAVAILABLE",
            plan.decision_reasons,
        )

    def test_facts_and_tablespace_finding_compile_autoextend(self) -> None:
        plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(
                _storage_plan_context(
                    evidence=(_tablespace_fact(), _action_state_fact()),
                    target_facts=(
                        {"fact_type": "ASM_DISKGROUP", "status": "ACTIVE"},
                    ),
                )
            )
        )
        self.assertEqual("AGENT_EXECUTE", plan.decision)
        self.assertEqual(
            ["db.storage.datafile.autoextend"],
            [item.action_template_id for item in plan.actions],
        )
        self.assertNotIn(
            "db.storage.datafile.add",
            [item.action_template_id for item in plan.actions],
        )

    def test_facts_without_headroom_compile_resize(self) -> None:
        plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(
                _storage_plan_context(
                    evidence=(
                        _tablespace_fact(
                            (
                                "USERS",
                                1000.0,
                                900.0,
                                100.0,
                                90.0,
                                1000.0,
                                100.0,
                                2,
                            )
                        ),
                        _action_state_fact(),
                    ),
                    target_facts=(
                        {"fact_type": "DATAFILE_PATH", "status": "ACTIVE"},
                    ),
                )
            )
        )
        self.assertEqual("AGENT_EXECUTE", plan.decision)
        self.assertEqual(
            ["db.storage.datafile.resize"],
            [item.action_template_id for item in plan.actions],
        )

    def test_never_includes_add_templates(self) -> None:
        plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(
                _storage_plan_context(
                    evidence=(_tablespace_fact(), _action_state_fact()),
                    target_facts=(
                        {"fact_type": "ASM_DISKGROUP", "status": "ACTIVE"},
                    ),
                    allowed_action_ids=(
                        "db.storage.datafile.add",
                        "db.storage.tempfile.add",
                        "db.storage.datafile.autoextend",
                    ),
                )
            )
        )
        action_ids = [item.action_template_id for item in plan.actions]
        self.assertEqual(["db.storage.datafile.autoextend"], action_ids)
        self.assertTrue(
            CAPACITY_ADD_ACTION_IDS.isdisjoint(action_ids)
        )


class InspectionCapacityRecommendationTest(unittest.TestCase):
    def test_recommendations_never_add_datafile_or_unconstrained_grow(self) -> None:
        columns = tuple(
            {"name": name}
            for name in (
                "metric_code",
                "dimensions",
                "forecast_horizon_days",
                "forecast_utilization_percent",
                "estimated_days_to_limit",
                "forecast_confidence",
                "history_elapsed_days",
            )
        )

        def source(percent: float, remaining: float):
            return type(
                "Source",
                (),
                {
                    "evidence": (
                        type(
                            "Evidence",
                            (),
                            {
                                "tool_id": "metric.query_range",
                                "columns": columns,
                                "rows": (
                                    (
                                        "db.storage.used_bytes",
                                        "tablespace=USERS",
                                        30.0,
                                        percent,
                                        remaining,
                                        "MEDIUM",
                                        30.0,
                                    ),
                                ),
                            },
                        )(),
                    )
                },
            )()

        texts = []
        for percent, remaining in ((96.5, 24.0), (86.0, 80.0), (70.0, 120.0)):
            texts.extend(
                AIOpsRuntimeService._inspection_capacity_recommendations(
                    source(percent, remaining)
                )
            )
        texts.extend(
            AIOpsRuntimeService._inspection_capacity_recommendations(
                type("Source", (), {"evidence": ()})()
            )
        )
        joined = "\n".join(texts)
        self.assertIn("AUTOEXTEND 或 RESIZE", joined)
        self.assertIn("只需观察", joined)
        self.assertNotIn("ADD DATAFILE", joined)
        self.assertNotIn("扩容", joined)


if __name__ == "__main__":
    unittest.main()
