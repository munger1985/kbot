"""步骤 9B Advisory 人工结果与只读效果验证测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.orchestration import (
    build_advisory_verification_blueprint,
)
from aiops_agent.workers.change_handlers import ActionVerificationHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
from platform_core.identity import uuid7


class AdvisoryVerificationHandlerTest(unittest.TestCase):
    def test_absent_target_is_verified(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._context(active_rows=(), blocking_rows=())
            )
        )
        self.assertEqual(result.status, "VERIFIED")
        self.assertFalse(result.target_still_present)
        self.assertFalse(result.blocking_still_present)

    def test_remaining_blocker_is_not_achieved(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._context(
                    active_rows=((1, 42, 9),),
                    blocking_rows=((1, 100, 1, 42),),
                )
            )
        )
        self.assertEqual(result.status, "NOT_ACHIEVED")
        self.assertTrue(result.target_still_present)
        self.assertTrue(result.blocking_still_present)

    def test_gap_never_becomes_success(self) -> None:
        context = self._context(active_rows=(), blocking_rows=())
        artifacts = tuple(
            item
            for item in context.input_artifacts
            if item["payload"].get("tool_id")
            != "db.session.blocking_chain"
        )
        result = asyncio.run(
            ActionVerificationHandler().execute(
                context.__class__(
                    **{**context.__dict__, "input_artifacts": artifacts}
                )
            )
        )
        self.assertEqual(result.status, "INCONCLUSIVE")
        self.assertIn(
            "VERIFICATION_EVIDENCE_MISSING", result.gap_codes
        )

    def test_partition_index_usable_is_verified(self) -> None:
        target_id = str(uuid7())
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "action_template_id": "db.index.partition.rebuild",
            "canonical_parameters": {
                "index_ref": {
                    "schema": "APP",
                    "object_type": "INDEX",
                    "object_name": "IX_ORDERS",
                    "partition": "P_202609",
                },
                "partition_name": "P_202609",
                "online": True,
            },
            "verification_tool_refs": ["db.index.partition.health"],
            "source_result_status": "SUCCEEDED",
        }
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-partition-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id="db.index.partition.health",
                    columns=(
                        "owner",
                        "index_name",
                        "partition_name",
                        "status",
                    ),
                    rows=(("APP", "IX_ORDERS", "P_202609", "USABLE"),),
                ),
            ),
        )

        result = asyncio.run(ActionVerificationHandler().execute(context))

        self.assertEqual("VERIFIED", result.status)
        self.assertTrue(result.effect_achieved)

    def test_coalesced_index_valid_is_verified(self) -> None:
        target_id = str(uuid7())
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "action_template_id": "db.index.coalesce",
            "canonical_parameters": {
                "index_ref": {
                    "schema": "APP",
                    "object_type": "INDEX",
                    "object_name": "IX_ORDERS",
                }
            },
            "verification_tool_refs": ["db.index.coalesce_candidate"],
            "source_result_status": "SUCCEEDED",
        }
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-coalesce-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id="db.index.coalesce_candidate",
                    columns=("owner", "index_name", "status"),
                    rows=(("APP", "IX_ORDERS", "VALID"),),
                ),
            ),
        )

        result = asyncio.run(ActionVerificationHandler().execute(context))

        self.assertEqual("VERIFIED", result.status)
        self.assertTrue(result.effect_achieved)

    def test_cancelled_sql_absence_is_verified_without_session_disconnect(self):
        target_id = str(uuid7())
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "action_template_id": "db.session.cancel_sql",
            "canonical_parameters": {
                "session_id": 42,
                "serial_number": 9,
                "instance_id": 1,
                "sql_id": "0abc123def456",
            },
            "verification_tool_refs": ["db.session.current_sql"],
            "source_result_status": "SUCCEEDED",
        }
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-cancel-sql-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id="db.session.current_sql",
                    columns=(
                        "instance_id",
                        "session_id",
                        "serial_number",
                        "sql_id",
                    ),
                    rows=((1, 42, 9, None),),
                ),
            ),
        )

        result = asyncio.run(ActionVerificationHandler().execute(context))

        self.assertEqual("VERIFIED", result.status)
        self.assertTrue(result.effect_achieved)

    def test_compiled_object_valid_is_verified(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._object_compile_context(
                    rows=(("APP", "PROC_A", "PROCEDURE", "VALID"),)
                )
            )
        )

        self.assertEqual("VERIFIED", result.status)
        self.assertTrue(result.effect_achieved)
        self.assertFalse(result.adverse_effect)

    def test_compiled_object_disappearance_is_adverse(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._object_compile_context(rows=())
            )
        )

        self.assertEqual("ADVERSE", result.status)
        self.assertFalse(result.effect_achieved)
        self.assertTrue(result.adverse_effect)

    def test_gathered_table_statistics_is_verified(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._statistics_context(
                    rows=(
                        (
                            "APP",
                            "ORDERS",
                            "NO",
                            "N",
                            "2026-09-02T03:00:00Z",
                            "NO",
                            None,
                        ),
                    )
                )
            )
        )

        self.assertEqual("VERIFIED", result.status)
        self.assertTrue(result.effect_achieved)

    def test_stale_table_statistics_is_not_achieved(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._statistics_context(
                    rows=(
                        (
                            "APP",
                            "ORDERS",
                            "NO",
                            "N",
                            "2026-08-01T00:00:00Z",
                            "YES",
                            None,
                        ),
                    )
                )
            )
        )

        self.assertEqual("NOT_ACHIEVED", result.status)
        self.assertFalse(result.effect_achieved)

    def test_statistics_lock_state_changes_are_verified(self) -> None:
        cases = (
            ("db.statistics.lock", "ALL"),
            ("db.statistics.unlock", None),
        )
        for action_id, locked in cases:
            with self.subTest(action_id=action_id):
                result = asyncio.run(
                    ActionVerificationHandler().execute(
                        self._statistics_context(
                            action_id=action_id,
                            rows=(
                                (
                                    "APP",
                                    "ORDERS",
                                    "NO",
                                    "N",
                                    "2026-09-02T03:00:00Z",
                                    "NO",
                                    locked,
                                ),
                            ),
                        )
                    )
                )
                self.assertEqual("VERIFIED", result.status)
                self.assertTrue(result.effect_achieved)

    def test_scheduler_job_running_is_verified(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._scheduler_context(
                    rows=(
                        (
                            "APP",
                            "NIGHTLY_JOB",
                            "TRUE",
                            "RUNNING",
                            "2026-09-02T03:00:00Z",
                            None,
                            7,
                            1,
                        ),
                    )
                )
            )
        )

        self.assertEqual("VERIFIED", result.status)
        self.assertTrue(result.effect_achieved)

    def test_scheduler_job_new_failure_is_not_achieved(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._scheduler_context(
                    rows=(
                        (
                            "APP",
                            "NIGHTLY_JOB",
                            "TRUE",
                            "SCHEDULED",
                            "2026-09-02T03:00:00Z",
                            "+00 00:00:01",
                            8,
                            2,
                        ),
                    )
                )
            )
        )

        self.assertEqual("NOT_ACHIEVED", result.status)
        self.assertFalse(result.effect_achieved)

    def test_scheduler_state_changes_are_verified(self) -> None:
        cases = (
            ("db.scheduler.job.enable", "TRUE", "SCHEDULED"),
            ("db.scheduler.job.disable", "FALSE", "DISABLED"),
            ("db.scheduler.job.stop", "TRUE", "SCHEDULED"),
        )
        for action_id, enabled, state in cases:
            with self.subTest(action_id=action_id):
                result = asyncio.run(
                    ActionVerificationHandler().execute(
                        self._scheduler_context(
                            action_id=action_id,
                            rows=(
                                (
                                    "APP",
                                    "NIGHTLY_JOB",
                                    enabled,
                                    state,
                                    None,
                                    None,
                                    7,
                                    1,
                                ),
                            ),
                        )
                    )
                )
                self.assertEqual("VERIFIED", result.status)
                self.assertTrue(result.effect_achieved)

    def test_user_lock_state_changes_are_verified(self) -> None:
        cases = (
            ("db.user.lock", "LOCKED"),
            ("db.user.unlock", "OPEN"),
            ("db.user.password.expire", "EXPIRED"),
        )
        for action_id, status in cases:
            with self.subTest(action_id=action_id):
                result = asyncio.run(
                    ActionVerificationHandler().execute(
                        self._user_context(
                            action_id=action_id,
                            rows=((
                                "APPUSER",
                                status,
                                None,
                                None,
                                "DEFAULT",
                                "PASSWORD",
                                "N",
                                "NO",
                            ),),
                        )
                    )
                )
                self.assertEqual("VERIFIED", result.status)
                self.assertTrue(result.effect_achieved)

    def test_user_disappearance_after_state_change_is_adverse(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._user_context(action_id="db.user.lock", rows=())
            )
        )
        self.assertEqual("ADVERSE", result.status)
        self.assertTrue(result.adverse_effect)

    def test_storage_parameter_resource_and_privilege_effects_are_verified(self):
        cases = (
            (
                "db.storage.datafile.resize",
                {"file_name": "+DATA/DB/data01.dbf", "new_size_mb": 2048},
                "db.storage.datafile.action_state",
                (
                    "file_name",
                    "current_size_mb",
                    "current_max_size_mb",
                    "autoextensible",
                    "current_next_mb",
                    "status",
                    "online_status",
                ),
                (("+DATA/DB/data01.dbf", 2048, 2048, "NO", 0, "AVAILABLE", "ONLINE"),),
            ),
            (
                "db.parameter.set",
                {"parameter_name": "cursor_sharing", "parameter_value": "FORCE"},
                "db.parameter.dynamic_state",
                ("parameter_name", "current_value"),
                (("cursor_sharing", "FORCE"),),
            ),
            (
                "db.resource_manager.plan.switch",
                {"resource_plan_name": "APP_PLAN"},
                "db.resource_manager.plan_state",
                ("resource_plan_name", "current_plan_name"),
                (("APP_PLAN", "APP_PLAN"),),
            ),
            (
                "db.user.privilege.grant",
                {"grantee_name": "APPUSER", "privilege": "CREATE SESSION"},
                "db.user.system_privilege_state",
                ("grantee_name", "privilege", "is_granted"),
                (("APPUSER", "CREATE SESSION", "YES"),),
            ),
        )
        for action_id, parameters, tool_id, columns, rows in cases:
            with self.subTest(action_id=action_id):
                result = asyncio.run(
                    ActionVerificationHandler().execute(
                        self._single_action_context(
                            action_id=action_id,
                            parameters=parameters,
                            tool_id=tool_id,
                            columns=columns,
                            rows=rows,
                        )
                    )
                )
                self.assertEqual("VERIFIED", result.status)
                self.assertTrue(result.effect_achieved)

    def test_blueprint_has_no_proposal_or_execute_task(self) -> None:
        blueprint = build_advisory_verification_blueprint(
            (
                "db.instance.identity",
                "db.session.active",
                "db.session.blocking_chain",
            )
        )
        self.assertEqual(blueprint.final_task_key, "verify")
        self.assertNotIn(
            "PROPOSE", {item.task_type for item in blueprint.tasks}
        )
        self.assertNotIn(
            "EXECUTE", {item.task_type for item in blueprint.tasks}
        )

    def _context(
        self,
        *,
        active_rows: tuple,
        blocking_rows: tuple,
    ) -> TaskExecutionContext:
        proposal_id = str(uuid7())
        source_run_id = str(uuid7())
        result_artifact_id = str(uuid7())
        target_id = str(uuid7())
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": proposal_id,
            "source_run_id": source_run_id,
            "result_artifact_id": result_artifact_id,
            "action_template_id": "db.session.terminate",
            "canonical_parameters": {
                "session_id": 42,
                "serial_number": 9,
                "instance_id": 1,
            },
            "verification_tool_refs": [
                "db.session.active",
                "db.session.blocking_chain",
            ],
            "source_result_status": "EXECUTED",
        }
        return TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": (
                        "ADVISORY_VERIFICATION_SCOPE.v1"
                    ),
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id="db.session.active",
                    columns=(
                        "instance_id",
                        "session_id",
                        "serial_number",
                    ),
                    rows=active_rows,
                ),
                self._diagnostic(
                    target_id=target_id,
                    tool_id="db.session.blocking_chain",
                    columns=(
                        "waiting_instance_id",
                        "waiting_session_id",
                        "blocking_instance_id",
                        "blocking_session_id",
                    ),
                    rows=blocking_rows,
                ),
            ),
        )

    def _single_action_context(
        self,
        *,
        action_id: str,
        parameters: dict,
        tool_id: str,
        columns: tuple[str, ...],
        rows: tuple,
    ) -> TaskExecutionContext:
        target_id = str(uuid7())
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "action_template_id": action_id,
            "canonical_parameters": parameters,
            "verification_tool_refs": [tool_id],
            "source_result_status": "SUCCEEDED",
        }
        return TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-controlled-action-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id=tool_id,
                    columns=columns,
                    rows=rows,
                ),
            ),
        )

    def _object_compile_context(self, *, rows: tuple) -> TaskExecutionContext:
        target_id = str(uuid7())
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "action_template_id": "db.object.compile",
            "canonical_parameters": {
                "object_type": "PROCEDURE",
                "object_ref": {
                    "schema": "APP",
                    "object_type": "PROCEDURE",
                    "object_name": "PROC_A",
                },
            },
            "verification_tool_refs": ["db.object.status"],
            "source_result_status": "SUCCEEDED",
        }
        return TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-object-compile-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id="db.object.status",
                    columns=(
                        "owner",
                        "object_name",
                        "object_type",
                        "status",
                    ),
                    rows=rows,
                ),
            ),
        )

    def _statistics_context(
        self,
        *,
        rows: tuple,
        action_id: str = "db.statistics.gather",
    ) -> TaskExecutionContext:
        target_id = str(uuid7())
        tool_id = {
            "db.statistics.gather": "db.table.statistics",
            "db.statistics.lock": "db.table.statistics.lock_candidate",
            "db.statistics.unlock": "db.table.statistics.unlock_candidate",
        }[action_id]
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "action_template_id": action_id,
            "canonical_parameters": {
                "table_ref": {
                    "schema": "APP",
                    "object_type": "TABLE",
                    "object_name": "ORDERS",
                }
            },
            "verification_tool_refs": [tool_id],
            "source_result_status": "SUCCEEDED",
        }
        return TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-statistics-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id=tool_id,
                    columns=(
                        "owner",
                        "table_name",
                        "partitioned",
                        "temporary",
                        "last_analyzed",
                        "stale_stats",
                        "stattype_locked",
                    ),
                    rows=rows,
                ),
            ),
        )

    def _scheduler_context(
        self,
        *,
        rows: tuple,
        action_id: str = "db.scheduler.job.run",
    ) -> TaskExecutionContext:
        target_id = str(uuid7())
        tool_id = {
            "db.scheduler.job.run": "db.scheduler.job.status",
            "db.scheduler.job.enable": "db.scheduler.job.enable_candidate",
            "db.scheduler.job.disable": "db.scheduler.job.disable_candidate",
            "db.scheduler.job.stop": "db.scheduler.job.stop_candidate",
        }[action_id]
        parameters = {
            "job_ref": {
                "schema": "APP",
                "object_type": "SCHEDULER_JOB",
                "object_name": "NIGHTLY_JOB",
            }
        }
        if action_id == "db.scheduler.job.run":
            parameters.update(
                {"previous_run_count": 7, "previous_failure_count": 1}
            )
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "action_template_id": action_id,
            "canonical_parameters": parameters,
            "verification_tool_refs": [tool_id],
            "source_result_status": "SUCCEEDED",
        }
        return TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-scheduler-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id=tool_id,
                    columns=(
                        "owner",
                        "job_name",
                        "enabled",
                        "state",
                        "last_start_date",
                        "last_run_duration",
                        "run_count",
                        "failure_count",
                    ),
                    rows=rows,
                ),
            ),
        )

    def _user_context(self, *, action_id: str, rows: tuple) -> TaskExecutionContext:
        target_id = str(uuid7())
        tool_id = {
            "db.user.lock": "db.user.lock_candidate",
            "db.user.unlock": "db.user.unlock_candidate",
            "db.user.password.expire": "db.user.password_expire_candidate",
        }[action_id]
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "action_template_id": action_id,
            "canonical_parameters": {
                "user_ref": {
                    "schema": "APPUSER",
                    "object_type": "USER",
                    "object_name": "APPUSER",
                }
            },
            "verification_tool_refs": [tool_id],
            "source_result_status": "SUCCEEDED",
        }
        return TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-user-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id=tool_id,
                    columns=(
                        "username",
                        "account_status",
                        "lock_date",
                        "expiry_date",
                        "profile",
                        "authentication_type",
                        "oracle_maintained",
                        "common",
                    ),
                    rows=rows,
                ),
            ),
        )

    @staticmethod
    def _diagnostic(
        *,
        target_id: str,
        tool_id: str,
        columns: tuple[str, ...],
        rows: tuple,
    ) -> dict:
        return {
            "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
            "payload": {
                "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
                "target_id": target_id,
                "tool_id": tool_id,
                "status": "SUCCEEDED",
                "observation": {
                    "schema_version": "DATABASE_OBSERVATION.v1",
                    "executor_request_id": str(uuid7()),
                    "target_id": target_id,
                    "tool_id": tool_id,
                    "tool_version": "1.0.0",
                    "variant": "oracle_19_plus_gv",
                    "template_sha256": "1" * 64,
                    "db_type": "ORACLE",
                    "db_version": "19.0.0",
                    "capability_snapshot_hash": "2" * 64,
                    "captured_at": "2026-07-24T10:00:00Z",
                    "duration_ms": 5,
                    "columns": [
                        {
                            "name": name,
                            "logical_type": "INTEGER",
                            "sensitivity": "PUBLIC",
                        }
                        for name in columns
                    ],
                    "rows": rows,
                    "row_count": len(rows),
                    "truncated": False,
                    "result_sha256": (
                        "3" * 64
                        if tool_id == "db.session.active"
                        else "4" * 64
                    ),
                    "parameters_sha256": "5" * 64,
                },
            },
        }


class AdvisoryVerificationOutboxTest(unittest.TestCase):
    def test_executed_result_creates_idempotent_verify_run(self) -> None:
        runtime = AsyncMock()
        fallback = AsyncMock()
        sink = AIOpsDomainOutboxSink(
            runtime_service=runtime,
            fallback=fallback,
        )
        proposal_id = str(uuid7())
        payload = {
            "proposal_id": proposal_id,
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
                        "domain_id": 200,
            "actor_id": "portal:user-1",
            "agent_id": str(uuid7()),
            "target_id": str(uuid7()),
            "action_template_id": "db.session.terminate",
            "canonical_parameters": {"session_id": 42},
            "verification_tool_refs": [
                "db.session.active",
                "db.session.blocking_chain",
            ],
            "source_result_status": "EXECUTED",
            "trace_id": "trace-1",
        }
        asyncio.run(
            sink.publish("OPS_ADVISORY_RESULT_RECORDED", payload)
        )
        command = runtime.create_run.await_args.args[0]
        self.assertEqual(
            command.idempotency_key,
            f"proposal:{proposal_id}:manual-result:verify",
        )
        self.assertEqual(
            command.blueprint_id, "change.advisory-verify"
        )
        fallback.publish.assert_not_awaited()

    def test_execution_result_creates_execution_scoped_verify_run(
        self,
    ) -> None:
        runtime = AsyncMock()
        sink = AIOpsDomainOutboxSink(
            runtime_service=runtime,
            fallback=AsyncMock(),
        )
        execution_id = str(uuid7())
        payload = {
            "execution_id": execution_id,
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
                        "domain_id": 200,
            "actor_id": "portal:user-1",
            "agent_id": str(uuid7()),
            "target_id": str(uuid7()),
            "trace_id": "trace-execution",
        }
        asyncio.run(
            sink.publish("OPS_EXECUTION_VERIFY_REQUESTED", payload)
        )
        command = runtime.create_run.await_args.args[0]
        self.assertEqual(
            command.idempotency_key,
            f"execution:{execution_id}:verify",
        )
        self.assertEqual(
            command.client_metadata["trigger"], "execution_result"
        )


class ProposalExpiryTest(unittest.TestCase):
    def test_reconciler_expires_orphaned_hitl_without_reopening_run(
        self,
    ) -> None:
        now = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
        run = SimpleNamespace(
            ops_run_id=uuid7(),
            status="EXPIRED",
        )
        task = SimpleNamespace(
            ops_task_id=uuid7(),
            ops_run_id=run.ops_run_id,
            status="EXPIRED",
        )
        hitl = SimpleNamespace(
            hitl_id=uuid7(),
            ops_task_id=task.ops_task_id,
            status="PENDING",
            expires_at=now - timedelta(seconds=1),
            responded_by=None,
            responded_at=None,
            response_json=None,
            response_hash=None,
        )
        uow = SimpleNamespace(
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now),
                lock_due_run=AsyncMock(return_value=None),
                get_task=AsyncMock(return_value=task),
                get_run=AsyncMock(return_value=run),
                append_event=AsyncMock(),
            ),
            changes=SimpleNamespace(
                find_expired_proposal=AsyncMock(return_value=None),
                find_expired_hitl=AsyncMock(return_value=hitl),
                get_hitl=AsyncMock(return_value=hitl),
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        service = AIOpsRuntimeService(
            uow_factory=lambda: context,
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        worked = asyncio.run(
            service.reconcile_once(trace_id="trace-orphan-hitl")
        )
        self.assertTrue(worked)
        self.assertEqual(hitl.status, "EXPIRED")
        self.assertEqual(run.status, "EXPIRED")
        self.assertEqual(task.status, "EXPIRED")
        self.assertEqual(
            hitl.response_json["reason"],
            "PARENT_STATE_NOT_WAITING_INPUT",
        )
        uow.commit.assert_awaited_once()

    def test_reconciler_expires_advisory_proposal(self) -> None:
        now = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
        proposal = SimpleNamespace(
            proposal_id=uuid7(),
            ops_run_id=uuid7(),
            ops_task_id=uuid7(),
            status="ADVISORY_READY",
            expires_at=now - timedelta(seconds=1),
            updated_at=now - timedelta(minutes=1),
        )
        run = SimpleNamespace(ops_run_id=proposal.ops_run_id)
        uow = SimpleNamespace(
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now),
                lock_due_run=AsyncMock(return_value=None),
                get_run=AsyncMock(return_value=run),
                append_event=AsyncMock(),
            ),
            changes=SimpleNamespace(
                find_expired_proposal=AsyncMock(
                    return_value=proposal
                ),
                get_proposal=AsyncMock(return_value=proposal),
                get_pending_hitl=AsyncMock(return_value=None),
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        factory = lambda: context
        service = AIOpsRuntimeService(
            uow_factory=factory,
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        worked = asyncio.run(
            service.reconcile_once(trace_id="trace-expiry")
        )
        self.assertTrue(worked)
        self.assertEqual(proposal.status, "EXPIRED")
        uow.commit.assert_awaited_once()
        event = uow.runs.append_event.await_args.kwargs
        self.assertEqual(event["event_type"], "proposal.expired")

    def test_reconciler_turns_stale_running_into_unknown(self) -> None:
        now = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
        execution = SimpleNamespace(
            execution_id=uuid7(),
            proposal_id=uuid7(),
            ops_run_id=uuid7(),
            ops_task_id=uuid7(),
            target_id=uuid7(),
            status="RUNNING",
            status_version=3,
            deadline_at=now - timedelta(seconds=1),
            action_template_id="db.session.terminate",
            executor_request_id=str(uuid7()),
            executor_instance_id="executor-test",
            grant_jti_hash="a" * 64,
            proposal_hash="b" * 64,
            command_hash="c" * 64,
            result_artifact_id=None,
            result_hash=None,
            completed_at=None,
            error_code=None,
            error_message=None,
            updated_at=now,
        )
        artifact_id = uuid7()
        run = SimpleNamespace(
            ops_run_id=execution.ops_run_id,
            actor_id="portal:user-1",
            agent_id=uuid7(),
            target_id=execution.target_id,
            plan_snapshot_json={
                "target": {
                                        "domain_id": 200,
                    "security_level": 3,
                }
            },
        )
        proposal = SimpleNamespace(
            proposal_id=execution.proposal_id,
            ops_task_id=execution.ops_task_id,
        )

        async def add_artifact(entity):
            entity.artifact_id = artifact_id
            return entity

        uow = SimpleNamespace(
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now),
                lock_due_run=AsyncMock(return_value=None),
                get_run=AsyncMock(return_value=run),
                add_artifact=AsyncMock(side_effect=add_artifact),
                append_event=AsyncMock(),
            ),
            changes=SimpleNamespace(
                find_expired_proposal=AsyncMock(return_value=None),
                find_expired_hitl=AsyncMock(return_value=None),
                find_due_execution=AsyncMock(return_value=execution),
                get_proposal=AsyncMock(return_value=proposal),
                get_execution=AsyncMock(return_value=execution),
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        service = AIOpsRuntimeService(
            uow_factory=lambda: context,
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        worked = asyncio.run(
            service.reconcile_once(trace_id="trace-reconcile")
        )
        self.assertTrue(worked)
        self.assertEqual(execution.status, "UNKNOWN")
        self.assertEqual(execution.status_version, 4)
        self.assertEqual(execution.result_artifact_id, artifact_id)
        uow.outbox.add.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
