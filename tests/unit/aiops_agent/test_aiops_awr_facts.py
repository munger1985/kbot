"""AWR Fact Tool 只消费已确认快照行，不把 HTML 当事实源。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from aiops_agent.application.diagnosis import summarize_awr_facts
from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.turn_answer import (
    DbaAnswerDraft,
    DbaSufficiencyAssessment,
    DiagnosisAnswerDraft,
    TurnEvidenceFact,
)
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.contracts.aiops import (
    AnswerBlockType,
    MeasurementSemantics,
    SufficiencyStatus,
)
from platform_core.identity import uuid7


TEST_PROMPT_SNAPSHOT = {"test": {"prompt_version_id": str(uuid7())}}
_LOAD_COLUMNS = (
    "snapshot_id",
    "instance_number",
    "elapsed_seconds",
    "metric_name",
    "total_value",
    "per_second",
    "unit",
)
_WAIT_COLUMNS = (
    "snapshot_id",
    "instance_number",
    "wait_class",
    "event_name",
    "total_waits",
    "time_waited_seconds",
    "avg_wait_ms",
    "wait_rank",
)
_SQL_COLUMNS = (
    "snapshot_id",
    "instance_number",
    "sql_id",
    "plan_hash_value",
    "executions",
    "elapsed_seconds",
    "cpu_seconds",
    "buffer_gets",
    "disk_reads",
    "sql_rank",
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
                analysis_markdown="Load Profile 上升，等待以 db file sequential read 为主。",
                solution_markdown="先核对已确认快照上的 Fact，再打开原生 HTML。",
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
    step_id: str = "a1",
    evidence_ref: str | None = None,
) -> TurnEvidenceFact:
    return TurnEvidenceFact(
        evidence_ref=evidence_ref or f"artifact:{tool_id}#fact",
        artifact_id=str(uuid7()),
        source_id=tool_id,
        step_id=step_id,
        tool_id=tool_id,
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind="TABLE",
        captured_at=datetime.now(UTC).isoformat(),
        columns=tuple({"name": name} for name in columns),
        rows=rows,
        row_count=len(rows),
    )


def _html_fact(*, tool_id: str = "db.oracle.awr.report", step_id: str = "a9") -> TurnEvidenceFact:
    return _fact(
        tool_id=tool_id,
        columns=("output",),
        rows=(("<html>awr</html>",),),
        step_id=step_id,
        evidence_ref=f"artifact:{tool_id}#html",
    )


def _load_row(
    snapshot_id: int,
    metric_name: str,
    *,
    total_value: float,
    per_second: float,
    elapsed_seconds: float = 3600.0,
) -> tuple[object, ...]:
    return (
        snapshot_id,
        1,
        elapsed_seconds,
        metric_name,
        total_value,
        per_second,
        "COUNT",
    )


def _wait_row(
    snapshot_id: int,
    event_name: str,
    *,
    total_waits: float,
    time_waited_seconds: float,
    wait_rank: int = 1,
) -> tuple[object, ...]:
    return (
        snapshot_id,
        1,
        "User I/O",
        event_name,
        total_waits,
        time_waited_seconds,
        round(time_waited_seconds * 1000 / total_waits, 3) if total_waits else None,
        wait_rank,
    )


def _sql_row(
    snapshot_id: int,
    sql_id: str,
    *,
    elapsed_seconds: float,
    sql_rank: int = 1,
) -> tuple[object, ...]:
    return (
        snapshot_id,
        1,
        sql_id,
        123456,
        10.0,
        elapsed_seconds,
        elapsed_seconds / 2,
        1000.0,
        20.0,
        sql_rank,
    )


def _answer_context(artifacts) -> TaskExecutionContext:
    return TaskExecutionContext(
        run_id=str(uuid7()),
        task_id=str(uuid7()),
        task_key="evidence:assess",
        target_id=str(uuid7()),
        agent_id=str(uuid7()),
        trigger_type="API",
        trace_id="trace-awr-facts",
        attempt=1,
        deadline_at=None,
        plan_snapshot={
            "answer_context": {
                "question": "对比昨天和今天的 AWR",
                "workflow_kind": "",
                "task_frame": {
                    "objectives": ["ASSESS"],
                    "problem_statement": "分析已确认快照上的负载变化",
                    "success_criteria": ["给出 Fact 结论并保留 HTML 链接"],
                    "action_intent": "NONE",
                },
                "model": {"technical_name": "test-model", "revision": "1"},
                "prompts": TEST_PROMPT_SNAPSHOT,
            },
            "target_facts": [],
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


class AwrFactSummaryTest(unittest.TestCase):
    def test_html_evidence_is_not_used_as_facts(self) -> None:
        plan = summarize_awr_facts(
            (
                _html_fact(),
                _html_fact(tool_id="db.oracle.awr.diff_report", step_id="a10"),
                _html_fact(tool_id="db.oracle.ash.report", step_id="a11"),
            )
        )

        self.assertFalse(plan.used_html_as_facts)
        self.assertEqual((), plan.windows)
        self.assertEqual((), plan.comparison)
        self.assertEqual((), plan.trend)
        self.assertEqual(
            (
                "db.oracle.awr.report",
                "db.oracle.awr.diff_report",
                "db.oracle.ash.report",
            ),
            plan.html_report_tool_ids,
        )

    def test_single_window_aggregates_load_wait_and_sql(self) -> None:
        evidence = (
            _fact(
                tool_id="db.oracle.awr.load_profile",
                columns=_LOAD_COLUMNS,
                rows=(
                    _load_row(11, "execute count", total_value=1000.0, per_second=0.278),
                    _load_row(12, "execute count", total_value=2000.0, per_second=0.556),
                    _load_row(11, "redo size", total_value=3600.0, per_second=1.0),
                ),
                evidence_ref="artifact:load#w1",
            ),
            _fact(
                tool_id="db.oracle.awr.top_wait",
                columns=_WAIT_COLUMNS,
                rows=(
                    _wait_row(
                        11,
                        "db file sequential read",
                        total_waits=100.0,
                        time_waited_seconds=10.0,
                    ),
                    _wait_row(
                        12,
                        "db file sequential read",
                        total_waits=50.0,
                        time_waited_seconds=5.0,
                    ),
                    _wait_row(
                        12,
                        "log file sync",
                        total_waits=20.0,
                        time_waited_seconds=1.0,
                        wait_rank=2,
                    ),
                ),
                evidence_ref="artifact:wait#w1",
            ),
            _fact(
                tool_id="db.oracle.awr.top_sql",
                columns=_SQL_COLUMNS,
                rows=(
                    _sql_row(11, "sqlAAAA", elapsed_seconds=8.0),
                    _sql_row(12, "sqlAAAA", elapsed_seconds=4.0),
                    _sql_row(12, "sqlBBBB", elapsed_seconds=1.0, sql_rank=2),
                ),
                evidence_ref="artifact:sql#w1",
            ),
            _html_fact(),
        )

        plan = summarize_awr_facts(evidence)
        self.assertFalse(plan.used_html_as_facts)
        self.assertEqual(("db.oracle.awr.report",), plan.html_report_tool_ids)
        self.assertEqual(1, len(plan.windows))
        window = plan.windows[0]
        self.assertEqual(11, window.begin_snapshot_id)
        self.assertEqual(12, window.end_snapshot_id)
        self.assertEqual(2, window.interval_count)
        self.assertEqual(
            (
                "artifact:load#w1",
                "artifact:wait#w1",
                "artifact:sql#w1",
            ),
            window.evidence_refs,
        )
        load = {item["metric_name"]: item for item in window.load_profile}
        self.assertEqual(3000.0, load["execute count"]["total_value"])
        self.assertEqual(0.417, load["execute count"]["per_second"])
        self.assertEqual(7200.0, load["execute count"]["elapsed_seconds"])
        waits = {item["event_name"]: item for item in window.top_wait}
        self.assertEqual(1, waits["db file sequential read"]["wait_rank"])
        self.assertEqual(15.0, waits["db file sequential read"]["time_waited_seconds"])
        self.assertEqual(100.0, waits["db file sequential read"]["avg_wait_ms"])
        sqls = {item["sql_id"]: item for item in window.top_sql}
        self.assertEqual(1, sqls["sqlAAAA"]["sql_rank"])
        self.assertEqual(12.0, sqls["sqlAAAA"]["elapsed_seconds"])
        self.assertEqual((), plan.comparison)

    def test_two_windows_emit_comparison_directions(self) -> None:
        baseline = (
            _fact(
                tool_id="db.oracle.awr.load_profile",
                columns=_LOAD_COLUMNS,
                rows=(
                    _load_row(10, "execute count", total_value=3600.0, per_second=1.0),
                ),
                evidence_ref="artifact:load#baseline",
            ),
            _fact(
                tool_id="db.oracle.awr.top_wait",
                columns=_WAIT_COLUMNS,
                rows=(
                    _wait_row(
                        10,
                        "db file sequential read",
                        total_waits=100.0,
                        time_waited_seconds=10.0,
                    ),
                ),
                evidence_ref="artifact:wait#baseline",
            ),
            _fact(
                tool_id="db.oracle.awr.top_sql",
                columns=_SQL_COLUMNS,
                rows=(_sql_row(10, "sqlAAAA", elapsed_seconds=8.0),),
                evidence_ref="artifact:sql#baseline",
            ),
        )
        after = (
            _fact(
                tool_id="db.oracle.awr.load_profile",
                columns=_LOAD_COLUMNS,
                rows=(
                    _load_row(20, "execute count", total_value=7200.0, per_second=2.0),
                ),
                evidence_ref="artifact:load#after",
            ),
            _fact(
                tool_id="db.oracle.awr.top_wait",
                columns=_WAIT_COLUMNS,
                rows=(
                    _wait_row(
                        20,
                        "db file sequential read",
                        total_waits=100.0,
                        time_waited_seconds=9.5,
                    ),
                ),
                evidence_ref="artifact:wait#after",
            ),
            _fact(
                tool_id="db.oracle.awr.top_sql",
                columns=_SQL_COLUMNS,
                rows=(_sql_row(20, "sqlAAAA", elapsed_seconds=4.0),),
                evidence_ref="artifact:sql#after",
            ),
        )

        plan = summarize_awr_facts(baseline + after)
        self.assertEqual(2, len(plan.windows))
        by_kind = {(item["kind"], item["name"]): item for item in plan.comparison}
        self.assertEqual("UP", by_kind[("LOAD_PROFILE", "execute count")]["direction"])
        self.assertEqual(100.0, by_kind[("LOAD_PROFILE", "execute count")]["change_pct"])
        self.assertEqual("FLAT", by_kind[("TOP_WAIT", "db file sequential read")]["direction"])
        self.assertEqual("DOWN", by_kind[("TOP_SQL", "sqlAAAA")]["direction"])
        self.assertEqual(-50.0, by_kind[("TOP_SQL", "sqlAAAA")]["change_pct"])

    def test_snapshot_sequence_emits_trend(self) -> None:
        plan = summarize_awr_facts(
            (
                _fact(
                    tool_id="db.oracle.awr.load_profile",
                    columns=_LOAD_COLUMNS,
                    rows=(
                        _load_row(11, "execute count", total_value=3600.0, per_second=1.0),
                        _load_row(12, "execute count", total_value=5400.0, per_second=1.5),
                        _load_row(13, "execute count", total_value=7200.0, per_second=2.0),
                    ),
                    evidence_ref="artifact:load#trend",
                ),
            )
        )

        self.assertEqual(1, len(plan.trend))
        trend = plan.trend[0]
        self.assertEqual("LOAD_PROFILE", trend["kind"])
        self.assertEqual("execute count", trend["name"])
        self.assertEqual(11, trend["begin_snapshot_id"])
        self.assertEqual(13, trend["end_snapshot_id"])
        self.assertEqual(3, trend["interval_count"])
        self.assertEqual(1.0, trend["first"])
        self.assertEqual(2.0, trend["last"])
        self.assertEqual("UP", trend["direction"])
        self.assertEqual(100.0, trend["change_pct"])

    def test_rows_without_snapshot_id_are_skipped(self) -> None:
        plan = summarize_awr_facts(
            (
                _fact(
                    tool_id="db.oracle.awr.load_profile",
                    columns=("metric_name", "total_value", "per_second"),
                    rows=(("execute count", 1000.0, 1.0),),
                    evidence_ref="artifact:load#invalid",
                ),
                _fact(
                    tool_id="db.session.active",
                    columns=("sid",),
                    rows=((88,),),
                    evidence_ref="artifact:session#ignored",
                ),
            )
        )
        self.assertEqual((), plan.windows)
        self.assertEqual((), plan.trend)
        self.assertFalse(plan.used_html_as_facts)
        self.assertEqual((), plan.html_report_tool_ids)


class AwrFactComposeTest(unittest.TestCase):
    def test_compose_payload_uses_facts_and_keeps_html_links(self) -> None:
        load = _fact(
            tool_id="db.oracle.awr.load_profile",
            columns=_LOAD_COLUMNS,
            rows=(_load_row(11, "execute count", total_value=3600.0, per_second=1.0),),
            step_id="a2",
            evidence_ref="artifact:load#fact",
        )
        html = _html_fact(step_id="a1")
        model = _AnswerModel(evidence_refs=(load.evidence_ref,))
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=model,
                prompts=_TestPrompts(),
            ).execute(
                _answer_context((_sufficiency_artifact((load, html)),))
            )
        )

        payload = model.calls[0]["input_payload"]
        self.assertIn("awr_facts", payload)
        self.assertFalse(payload["awr_facts"]["used_html_as_facts"])
        self.assertEqual(1, len(payload["awr_facts"]["windows"]))
        self.assertEqual(
            ["db.oracle.awr.report"],
            payload["awr_facts"]["html_report_tool_ids"],
        )
        html_evidence = next(
            item
            for item in payload["sufficiency"]["evidence"]
            if item["tool_id"] == "db.oracle.awr.report"
        )
        self.assertEqual([], html_evidence["rows"])
        self.assertEqual([], html_evidence["columns"])
        fact_evidence = next(
            item
            for item in payload["sufficiency"]["evidence"]
            if item["tool_id"] == "db.oracle.awr.load_profile"
        )
        self.assertGreater(len(fact_evidence["rows"]), 0)

        html_block = next(
            item
            for item in result.blocks
            if item.block_type == AnswerBlockType.HTML_REPORT_LINKS
        )
        self.assertEqual(
            ["db.oracle.awr.report"],
            [item["tool_id"] for item in html_block.payload["reports"]],
        )
        self.assertEqual("a1", html_block.payload["reports"][0]["action_id"])
        self.assertNotIn(
            "db.oracle.awr.load_profile",
            [item["tool_id"] for item in html_block.payload["reports"]],
        )


if __name__ == "__main__":
    unittest.main()
