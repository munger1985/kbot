"""发现结果绑定回延期目录工具，并在不经模型的情况下继续取证。"""

from __future__ import annotations

import unittest
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiops_agent.application.investigation.discovery import (
    catalog_direct_actions,
)
from aiops_agent.application.investigation.discovery_binding import (
    PRODUCT_TIMEZONE,
    bind_discovery_parameters,
    bound_continuation_actions,
    decide_discovery_continuation,
    parse_time_windows,
)
from aiops_agent.application.investigation.service import (
    TurnPlanningService,
    _ReplanInputSnapshot,
)
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import ToolExecutionSnapshotBuilder
from platform_core.contracts.aiops.investigation import (
    InvestigationAction,
    InvestigationPlan,
    InvestigationPlanningOutput,
)
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.identity import uuid7


SNAPSHOT_ROWS = (
    (100, 1, "2026-09-14T00:00:00+08:00", "2026-09-14T01:00:00+08:00"),
    (101, 1, "2026-09-14T01:00:00+08:00", "2026-09-14T02:00:00+08:00"),
    (124, 1, "2026-09-15T00:00:00+08:00", "2026-09-15T01:00:00+08:00"),
    (125, 1, "2026-09-15T01:00:00+08:00", "2026-09-15T02:00:00+08:00"),
)


def _action(**fields) -> InvestigationAction:
    payload = {
        "action_id": "a1",
        "question": "生成 AWR 报告",
        "tool_id": "db.oracle.awr.report",
        "input": {},
        "expected_evidence_kind": "AWR_REPORT",
        "measurement_semantics": "SNAPSHOT_DELTA",
        "deferred": False,
    }
    payload.update(fields)
    return InvestigationAction.model_validate(payload)


def _plan(*actions: InvestigationAction) -> InvestigationPlan:
    return InvestigationPlan(revision_no=1, actions=actions)


def _snapshot_result() -> dict:
    return {
        "schema_version": "DBA_TOOL_RESULT.v1",
        "tool_outcomes": [
            {
                "tool_id": "db.oracle.awr.snapshots",
                "status": "SUCCEEDED",
                "observation": {
                    "columns": [
                        {"name": "snapshot_id"},
                        {"name": "instance_number"},
                        {"name": "begin_time"},
                        {"name": "end_time"},
                    ],
                    "rows": [list(row) for row in SNAPSHOT_ROWS],
                },
            }
        ],
    }


def _envelope() -> dict:
    return {
        "materials": [
            {
                "item_no": 1,
                "material_kind": "QUESTION",
                "summary": "对比昨天和今天 1 点到 2 点的 AWR",
                "confidence": 1,
            }
        ],
        "explicit_question": "对比昨天和今天 1 点到 2 点的 AWR",
    }


def _task_frame() -> dict:
    return {
        "objectives": ["DIAGNOSE"],
        "problem_statement": "对比两个整点窗口的 AWR",
        "success_criteria": ["生成 AWR 对比报告"],
    }


class DiscoveryBindingTest(unittest.TestCase):
    def test_datetime_snapshot_ids_bind_to_nearest_end_time(self) -> None:
        plan = _plan(
            _action(
                action_id="a1",
                tool_id="db.oracle.awr.snapshots",
                question="列出可用 AWR 快照",
                expected_evidence_kind="AWR_SNAPSHOTS",
                input={},
            ),
            _action(
                action_id="a2",
                deferred=True,
                depends_on=("a1",),
                input={
                    "begin_snapshot_id": "2026-09-14T01:00:00+08:00",
                    "end_snapshot_id": "2026-09-14T02:00:00+08:00",
                },
            ),
        )

        bound = bind_discovery_parameters(
            plan=plan,
            tool_results=(_snapshot_result(),),
        )

        self.assertEqual("BOUND", bound.status)
        report = bound.plan.actions[1]
        self.assertFalse(report.deferred)
        self.assertEqual(
            {"begin_snapshot_id": 100, "end_snapshot_id": 101},
            report.input,
        )

    def test_two_bound_reports_synthesize_diff_report(self) -> None:
        plan = _plan(
            _action(
                action_id="a1",
                tool_id="db.oracle.awr.snapshots",
                question="列出可用 AWR 快照",
                expected_evidence_kind="AWR_SNAPSHOTS",
                input={},
            ),
            _action(
                action_id="a2",
                deferred=True,
                depends_on=("a1",),
                question="生成昨天 1 点到 2 点的 AWR",
                input={
                    "begin_snapshot_id": "2026-09-14T01:00:00+08:00",
                    "end_snapshot_id": "2026-09-14T02:00:00+08:00",
                },
            ),
            _action(
                action_id="a3",
                deferred=True,
                depends_on=("a1",),
                question="生成今天 1 点到 2 点的 AWR",
                input={
                    "begin_snapshot_id": "2026-09-15T01:00:00+08:00",
                    "end_snapshot_id": "2026-09-15T02:00:00+08:00",
                },
            ),
            _action(
                action_id="a4",
                tool_id="db.oracle.awr.diff_report",
                deferred=True,
                depends_on=("a2", "a3"),
                question="对比两段 AWR",
                expected_evidence_kind="AWR_DIFF_REPORT",
                input={},
            ),
        )

        bound = bind_discovery_parameters(
            plan=plan,
            tool_results=(_snapshot_result(),),
        )

        self.assertEqual("BOUND", bound.status)
        self.assertEqual((), bound.unbound_action_ids)
        self.assertEqual(
            {"begin_snapshot_id": 100, "end_snapshot_id": 101},
            bound.plan.actions[1].input,
        )
        self.assertEqual(
            {"begin_snapshot_id": 124, "end_snapshot_id": 125},
            bound.plan.actions[2].input,
        )
        self.assertEqual(
            {
                "baseline_begin_snapshot_id": 100,
                "baseline_end_snapshot_id": 101,
                "after_begin_snapshot_id": 124,
                "after_end_snapshot_id": 125,
            },
            bound.plan.actions[3].input,
        )
        self.assertFalse(any(action.deferred for action in bound.plan.actions))

    def test_bound_discovery_continues_without_rerunning_snapshots(self) -> None:
        prior = _plan(
            _action(
                action_id="a1",
                tool_id="db.oracle.awr.snapshots",
                question="列出可用 AWR 快照",
                expected_evidence_kind="AWR_SNAPSHOTS",
                input={},
            ),
            _action(
                action_id="a2",
                deferred=True,
                depends_on=("a1",),
                input={
                    "begin_snapshot_id": "2026-09-14T01:00:00+08:00",
                    "end_snapshot_id": "2026-09-14T02:00:00+08:00",
                },
            ),
            _action(
                action_id="a3",
                deferred=True,
                depends_on=("a1",),
                question="生成今天 1 点到 2 点的 AWR",
                input={
                    "begin_snapshot_id": "2026-09-15T01:00:00+08:00",
                    "end_snapshot_id": "2026-09-15T02:00:00+08:00",
                },
            ),
            _action(
                action_id="a4",
                tool_id="db.oracle.awr.diff_report",
                deferred=True,
                depends_on=("a2", "a3"),
                question="对比两段 AWR",
                expected_evidence_kind="AWR_DIFF_REPORT",
                input={},
            ),
        )

        decision = decide_discovery_continuation(
            plan=prior,
            tool_results=(_snapshot_result(),),
        )

        self.assertEqual("CONTINUE", decision.action)
        self.assertEqual(
            [
                "db.oracle.awr.report",
                "db.oracle.awr.report",
                "db.oracle.awr.diff_report",
            ],
            [action.tool_id for action in decision.continuation_actions],
        )
        self.assertFalse(
            any(action.deferred for action in decision.continuation_actions)
        )
        self.assertEqual(
            ("a2", "a3"),
            decision.continuation_actions[2].depends_on,
        )
        self.assertEqual((), decision.continuation_actions[0].depends_on)
        self.assertNotIn(
            "db.oracle.awr.snapshots",
            [action.tool_id for action in decision.continuation_actions],
        )

    def test_prepare_skips_deferred_illegal_snapshot_ids(self) -> None:
        investigation = InvestigationPlanningOutput.model_validate(
            {
                "input_envelope": _envelope(),
                "task_frame": _task_frame(),
                "plan": _plan(
                    _action(
                        action_id="a1",
                        tool_id="db.oracle.awr.snapshots",
                        question="列出可用 AWR 快照",
                        expected_evidence_kind="AWR_SNAPSHOTS",
                        input={},
                    ),
                    _action(
                        action_id="a2",
                        deferred=True,
                        depends_on=("a1",),
                        input={
                            "begin_snapshot_id": "2026-09-14T01:00:00+08:00",
                            "end_snapshot_id": "2026-09-14T02:00:00+08:00",
                        },
                    ),
                ).model_dump(mode="json"),
            }
        )
        service = object.__new__(TurnPlanningService)
        service._tool_snapshot_builder = ToolExecutionSnapshotBuilder(
            playbook_registry=PlaybookRegistry.load(),
            diagnostic_registry=DiagnosticRegistry.load(),
        )
        context = SimpleNamespace(
            resolved_uploads=(),
            capabilities=DbaCapabilitySnapshot(
                agent_id=str(uuid7()),
                agent_version_id=str(uuid7()),
                target_id=str(uuid7()),
                database_type="ORACLE",
                database_version="19c",
                target_enabled=True,
                target_reachable=True,
                target_capabilities=("DB_READONLY",),
            ),
        )

        prepared, *_rest = service._prepare_query_inputs(
            investigation=investigation,
            context=context,
        )

        self.assertEqual(
            ["db.oracle.awr.snapshots"],
            [
                action.tool_id
                for action in catalog_direct_actions(prepared.plan.actions)
            ],
        )
        self.assertTrue(prepared.plan.actions[1].deferred)
        self.assertEqual(
            {
                "begin_snapshot_id": "2026-09-14T01:00:00+08:00",
                "end_snapshot_id": "2026-09-14T02:00:00+08:00",
            },
            prepared.plan.actions[1].input,
        )

    def test_wall_clock_binds_utc_tagged_oracle_timestamps(self) -> None:
        plan = _plan(
            _action(
                action_id="a1",
                tool_id="db.oracle.awr.snapshots",
                question="列出可用 AWR 快照",
                expected_evidence_kind="AWR_SNAPSHOTS",
                input={},
            ),
            _action(
                action_id="a2",
                deferred=True,
                depends_on=("a1",),
                input={
                    "begin_snapshot_id": "2026-09-14T01:00:00+08:00",
                    "end_snapshot_id": "2026-09-14T02:00:00+08:00",
                },
            ),
        )
        result = {
            "schema_version": "DBA_TOOL_RESULT.v1",
            "tool_outcomes": [
                {
                    "tool_id": "db.oracle.awr.snapshots",
                    "status": "SUCCEEDED",
                    "observation": {
                        "columns": [
                            {"name": "snapshot_id"},
                            {"name": "instance_number"},
                            {"name": "begin_time"},
                            {"name": "end_time"},
                        ],
                        "rows": [
                            [
                                Decimal("100"),
                                Decimal("1"),
                                "2026-09-14T00:00:00+00:00",
                                "2026-09-14T01:00:00+00:00",
                            ],
                            [
                                Decimal("101"),
                                Decimal("1"),
                                "2026-09-14T01:00:00+00:00",
                                "2026-09-14T02:00:00+00:00",
                            ],
                        ],
                    },
                }
            ],
        }

        bound = bind_discovery_parameters(plan=plan, tool_results=(result,))

        self.assertEqual("BOUND", bound.status)
        self.assertEqual(
            {"begin_snapshot_id": 100, "end_snapshot_id": 101},
            bound.plan.actions[1].input,
        )

    def test_empty_input_binds_iso_datetimes_from_question(self) -> None:
        plan = _plan(
            _action(
                action_id="a1",
                tool_id="db.oracle.awr.snapshots",
                question="列出可用 AWR 快照",
                expected_evidence_kind="AWR_SNAPSHOTS",
                input={},
            ),
            _action(
                action_id="a2",
                deferred=True,
                depends_on=("a1",),
                question=(
                    "生成 2026-09-14T01:00:00+08:00 到 "
                    "2026-09-14T02:00:00+08:00 的 AWR"
                ),
                input={},
            ),
        )

        bound = bind_discovery_parameters(
            plan=plan,
            tool_results=(_snapshot_result(),),
        )

        self.assertEqual("BOUND", bound.status)
        self.assertEqual(
            {"begin_snapshot_id": 100, "end_snapshot_id": 101},
            bound.plan.actions[1].input,
        )

    def test_empty_input_binds_iso_datetimes_from_turn_question(self) -> None:
        plan = _plan(
            _action(
                action_id="a1",
                tool_id="db.oracle.awr.snapshots",
                question="列出可用 AWR 快照",
                expected_evidence_kind="AWR_SNAPSHOTS",
                input={},
            ),
            _action(
                action_id="a2",
                deferred=True,
                depends_on=("a1",),
                question="生成 AWR 报告",
                input={},
            ),
        )

        bound = bind_discovery_parameters(
            plan=plan,
            tool_results=(_snapshot_result(),),
            question=(
                "请生成 2026-09-14T01:00:00+08:00 到 "
                "2026-09-14T02:00:00+08:00 的 AWR"
            ),
        )

        self.assertEqual("BOUND", bound.status)
        self.assertEqual(
            {"begin_snapshot_id": 100, "end_snapshot_id": 101},
            bound.plan.actions[1].input,
        )

    def test_parse_relative_chinese_windows(self) -> None:
        now = datetime(2026, 9, 17, 9, 59, 8, tzinfo=PRODUCT_TIMEZONE)
        single = parse_time_windows(
            "请生成数据库在昨天2:00-3:00的awr报告",
            now=now,
        )
        self.assertEqual(
            (
                (
                    datetime(2026, 9, 16, 2, 0, tzinfo=PRODUCT_TIMEZONE),
                    datetime(2026, 9, 16, 3, 0, tzinfo=PRODUCT_TIMEZONE),
                ),
            ),
            single,
        )
        dual = parse_time_windows(
            "对比昨天和今天 1 点到 2 点的 AWR",
            now=now,
        )
        self.assertEqual(
            (
                (
                    datetime(2026, 9, 16, 1, 0, tzinfo=PRODUCT_TIMEZONE),
                    datetime(2026, 9, 16, 2, 0, tzinfo=PRODUCT_TIMEZONE),
                ),
                (
                    datetime(2026, 9, 17, 1, 0, tzinfo=PRODUCT_TIMEZONE),
                    datetime(2026, 9, 17, 2, 0, tzinfo=PRODUCT_TIMEZONE),
                ),
            ),
            dual,
        )
        self.assertEqual((), parse_time_windows("列出可用快照", now=now))

    def test_relative_chinese_window_binds_from_question(self) -> None:
        now = datetime(2026, 9, 17, 9, 59, 8, tzinfo=PRODUCT_TIMEZONE)
        result = {
            "schema_version": "DBA_TOOL_RESULT.v1",
            "tool_outcomes": [
                {
                    "tool_id": "db.oracle.awr.snapshots",
                    "status": "SUCCEEDED",
                    "observation": {
                        "columns": [
                            {"name": "snapshot_id"},
                            {"name": "instance_number"},
                            {"name": "begin_time"},
                            {"name": "end_time"},
                        ],
                        "rows": [
                            [
                                209,
                                1,
                                "2026-09-16T01:00:00+08:00",
                                "2026-09-16T02:00:00+08:00",
                            ],
                            [
                                210,
                                1,
                                "2026-09-16T02:00:00+08:00",
                                "2026-09-16T03:00:00+08:00",
                            ],
                        ],
                    },
                }
            ],
        }
        plan = _plan(
            _action(
                action_id="a1",
                tool_id="db.oracle.awr.snapshots",
                question="列出可用 AWR 快照",
                expected_evidence_kind="AWR_SNAPSHOTS",
                input={},
            ),
            _action(
                action_id="a2",
                deferred=True,
                depends_on=("a1",),
                question="生成 AWR 报告",
                input={},
            ),
        )
        with patch(
            "aiops_agent.application.investigation.discovery_binding._product_now",
            return_value=now,
        ):
            bound = bind_discovery_parameters(
                plan=plan,
                tool_results=(result,),
                question="请生成数据库在昨天2:00-3:00的awr报告",
            )
        self.assertEqual("BOUND", bound.status)
        self.assertEqual(
            {"begin_snapshot_id": 209, "end_snapshot_id": 210},
            bound.plan.actions[1].input,
        )


class _LoaderUow:
    def __init__(self) -> None:
        self.turn = SimpleNamespace(
            turn_id=uuid7(),
            domain_id=7,
            current_plan_artifact_id=uuid7(),
            task_frame_artifact_id=uuid7(),
        )
        self.run = SimpleNamespace(
            ops_run_id=uuid7(),
            plan_snapshot_json={
                "answer_context": {"input_envelope": _envelope()}
            },
        )
        self.plan_artifact = SimpleNamespace(
            artifact_id=self.turn.current_plan_artifact_id,
            artifact_key="turn-investigation-plan:1",
            schema_version="DBA_INVESTIGATION_PLAN.v1",
            payload_json=_plan(
                _action(deferred=True, input={"begin_snapshot_id": "x"})
            ).model_dump(mode="json"),
        )
        self.task_frame_artifact = SimpleNamespace(
            artifact_id=self.turn.task_frame_artifact_id,
            artifact_key="turn-task-frame:1",
            schema_version="DBA_TASK_FRAME.v1",
            payload_json=_task_frame(),
        )
        self.assessment_artifact = SimpleNamespace(
            artifact_id=uuid7(),
            artifact_key="evidence:assess",
            schema_version="DBA_SUFFICIENCY.v1",
            payload_json={"schema_version": "DBA_SUFFICIENCY.v1"},
        )
        self.tool_artifact = SimpleNamespace(
            artifact_id=uuid7(),
            artifact_key="diagnostic:a1",
            schema_version="DBA_TOOL_RESULT.v1",
            payload_json=_snapshot_result(),
            ops_run_id=self.run.ops_run_id,
        )
        self.turns = SimpleNamespace(get_turn=self._get_turn)
        self.runs = SimpleNamespace(
            get_run=self._get_run,
            get_artifact=self._get_artifact,
            list_artifacts=self._list_artifacts,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def _get_turn(self, *, domain_id, turn_id, lock=False):
        del lock
        if domain_id == 7 and turn_id == self.turn.turn_id:
            return self.turn
        return None

    async def _get_run(self, *, ops_run_id, lock=False):
        del lock
        return self.run if ops_run_id == self.run.ops_run_id else None

    async def _get_artifact(self, *, artifact_id):
        for item in (
            self.plan_artifact,
            self.task_frame_artifact,
            self.assessment_artifact,
            self.tool_artifact,
        ):
            if item.artifact_id == artifact_id:
                return item
        return None

    async def _list_artifacts(self, *, ops_run_id):
        if ops_run_id != self.run.ops_run_id:
            return []
        return [
            self.plan_artifact,
            self.task_frame_artifact,
            self.assessment_artifact,
            self.tool_artifact,
        ]


class ReplanInputLoaderTest(unittest.IsolatedAsyncioTestCase):
    async def test_load_replan_inputs_collects_tool_results_and_envelope(
        self,
    ) -> None:
        uow = _LoaderUow()
        service = object.__new__(TurnPlanningService)
        service._uow_factory = lambda: uow
        context = SimpleNamespace(
            domain_id=7,
            turn_id=uow.turn.turn_id,
            ops_run_id=uow.run.ops_run_id,
        )

        loaded = await service._load_replan_inputs(
            context=context,
            assessment_artifact_id=uow.assessment_artifact.artifact_id,
        )

        self.assertEqual(_envelope(), loaded.input_envelope)
        self.assertEqual(1, len(loaded.tool_results))
        self.assertEqual(
            "DBA_TOOL_RESULT.v1",
            loaded.tool_results[0]["schema_version"],
        )
        self.assertTrue(loaded.prior_plan["actions"][0]["deferred"])


class BoundContinuationHelperTest(unittest.TestCase):
    def test_continuation_strips_missing_discovery_dependencies(self) -> None:
        prior = _plan(
            _action(action_id="a1", tool_id="db.oracle.awr.snapshots"),
            _action(action_id="a2", deferred=True, depends_on=("a1",)),
        )
        bound = _plan(
            prior.actions[0],
            prior.actions[1].model_copy(
                update={
                    "deferred": False,
                    "input": {
                        "begin_snapshot_id": 100,
                        "end_snapshot_id": 101,
                    },
                }
            ),
        )
        continued = bound_continuation_actions(
            prior_plan=prior,
            bound_plan=bound,
        )
        self.assertEqual(("a2",), tuple(item.action_id for item in continued))
        self.assertEqual((), continued[0].depends_on)


def _awr_capabilities() -> DbaCapabilitySnapshot:
    return DbaCapabilitySnapshot(
        agent_id=str(uuid7()),
        agent_version_id=str(uuid7()),
        target_id=str(uuid7()),
        database_type="ORACLE",
        database_version="19c",
        target_enabled=True,
        target_reachable=True,
        target_capabilities=("DB_READONLY",),
        privileges=("DBA_HIST_SNAPSHOT", "V_$DATABASE"),
    )


def _continuation_plan() -> InvestigationPlan:
    return _plan(
        _action(
            action_id="a1",
            tool_id="db.oracle.awr.snapshots",
            question="列出可用 AWR 快照",
            expected_evidence_kind="AWR_SNAPSHOTS",
            input={},
        ),
        _action(
            action_id="a2",
            deferred=True,
            depends_on=("a1",),
            question="生成昨天 1 点到 2 点的 AWR",
            input={
                "begin_snapshot_id": "2026-09-14T01:00:00+08:00",
                "end_snapshot_id": "2026-09-14T02:00:00+08:00",
            },
        ),
        _action(
            action_id="a3",
            deferred=True,
            depends_on=("a1",),
            question="生成今天 1 点到 2 点的 AWR",
            input={
                "begin_snapshot_id": "2026-09-15T01:00:00+08:00",
                "end_snapshot_id": "2026-09-15T02:00:00+08:00",
            },
        ),
        _action(
            action_id="a4",
            tool_id="db.oracle.awr.diff_report",
            deferred=True,
            depends_on=("a2", "a3"),
            question="对比两段 AWR",
            expected_evidence_kind="AWR_DIFF_REPORT",
            input={},
        ),
    )


class ExecuteReplanContinuationTest(unittest.IsolatedAsyncioTestCase):
    def _service(self, *, prior: InvestigationPlan, tool_results: tuple[dict, ...]):
        captured: dict = {}
        service = object.__new__(TurnPlanningService)
        context = SimpleNamespace(
            workflow_kind="CHAT_TURN",
            agent_id=uuid7(),
            domain_id=7,
            trace_id="trace-awr",
            turn_id=uuid7(),
            question="对比昨天和今天 1 点到 2 点的 AWR",
            capabilities=_awr_capabilities(),
            resolved_uploads=(),
        )
        service._prepare = AsyncMock(return_value=context)
        service._load_replan_inputs = AsyncMock(
            return_value=_ReplanInputSnapshot(
                prior_plan=prior.model_dump(mode="json"),
                task_frame=_task_frame(),
                assessment={"schema_version": "DBA_SUFFICIENCY.v1"},
                prior_artifacts=(),
                tool_results=tool_results,
                input_envelope=_envelope(),
            )
        )
        service._agent_catalog = SimpleNamespace(
            resolve_planner_model=AsyncMock(return_value={"technical_name": "planner"}),
            resolve_diagnosis_model=AsyncMock(return_value={"technical_name": "diag"}),
        )
        registry = PlaybookRegistry.load()
        service._playbook_registry = registry
        service._tool_snapshot_builder = ToolExecutionSnapshotBuilder(
            playbook_registry=registry,
            diagnostic_registry=DiagnosticRegistry.load(),
        )
        service._investigation_reasoner = SimpleNamespace(replan=AsyncMock())
        service._prepare_query_inputs = (
            TurnPlanningService._prepare_query_inputs.__get__(
                service, TurnPlanningService
            )
        )
        service._investigation_from_continuation = (
            TurnPlanningService._investigation_from_continuation.__get__(
                service, TurnPlanningService
            )
        )

        async def _persist(**kwargs):
            captured.update(kwargs)
            return {"status": "COLLECTING", "tools": [
                action.tool_id for action in kwargs["investigation"].plan.actions
            ]}

        service._compile_and_persist_replan = _persist
        service.fall_back_from_replan = AsyncMock(
            return_value={"status": "ANSWERING"}
        )
        return service, captured

    async def test_bound_discovery_skips_model_and_persists_reports(self) -> None:
        service, captured = self._service(
            prior=_continuation_plan(),
            tool_results=(_snapshot_result(),),
        )

        result = await TurnPlanningService.execute_replan(
            service,
            {
                "domain_id": 7,
                "turn_id": "turn",
                "ops_run_id": "run",
                "assessment_artifact_id": str(uuid7()),
                "revision_no": 2,
            },
        )

        self.assertEqual("COLLECTING", result["status"])
        service._investigation_reasoner.replan.assert_not_called()
        service.fall_back_from_replan.assert_not_called()
        self.assertIsNone(captured["planning_receipt"])
        self.assertEqual(
            [
                "db.oracle.awr.report",
                "db.oracle.awr.report",
                "db.oracle.awr.diff_report",
            ],
            [action.tool_id for action in captured["investigation"].plan.actions],
        )
        self.assertEqual(
            {
                "baseline_begin_snapshot_id": 100,
                "baseline_end_snapshot_id": 101,
                "after_begin_snapshot_id": 124,
                "after_end_snapshot_id": 125,
            },
            captured["investigation"].plan.actions[2].input,
        )

    async def test_unbound_deferred_skips_model_replan(self) -> None:
        service, _captured = self._service(
            prior=_continuation_plan(),
            tool_results=(),
        )

        result = await TurnPlanningService.execute_replan(
            service,
            {
                "domain_id": 7,
                "turn_id": "turn",
                "ops_run_id": "run",
                "assessment_artifact_id": str(uuid7()),
                "revision_no": 2,
            },
        )

        self.assertEqual("ANSWERING", result["status"])
        service._investigation_reasoner.replan.assert_not_called()
        service.fall_back_from_replan.assert_awaited()


if __name__ == "__main__":
    unittest.main()
