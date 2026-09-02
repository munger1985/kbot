"""Oracle 动态只读 SQL 的 AST 安全策略测试。"""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiops_agent.application.investigation import prepare_dynamic_queries
from aiops_agent.application.investigation.discovery import available_tools
from aiops_agent.application.investigation.reasoner import (
    InvestigationPlanValidationError,
)
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.diagnostics.dynamic_query import (
    DynamicQueryPolicySnapshot,
    DynamicQueryRejected,
    OracleDynamicQueryPolicy,
)
from aiops_agent.diagnostics.grants import (
    DiagnosticGrantCodec,
    DiagnosticGrantError,
    canonical_sha256,
)
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.executor import DynamicDiagnosticExecutorService
from aiops_agent.executor.drivers import DriverQueryResult
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.tools import (
    InvestigationTaskCompiler,
    ToolExecutionSnapshotBuilder,
)
from aiops_agent.workers.database_handlers import (
    DynamicQueryInvocationHandler,
)
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.errors import RetryableTaskError
from platform_core.contracts.aiops import InvestigationPlanningOutput
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticLimits,
    DynamicDiagnosticExecutionGrant,
    DynamicReadDiagnosticRequest,
    OracleDynamicQueryPolicyGrant,
    ReadDiagnosticResult,
)
from platform_core.contracts.aiops.playbooks import (
    DbaCapabilitySnapshot,
    DbaPlaybookPlan,
)
from platform_core.identity import uuid7


class OracleDynamicQueryPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = OracleDynamicQueryPolicy(
            DynamicQueryPolicySnapshot(max_rows=50)
        )

    def test_select_is_normalized_bounded_and_hashed(self) -> None:
        result = self.policy.validate(
            """
            WITH active_sessions AS (
                SELECT inst_id, sid, serial#, status
                  FROM gv$session
                 WHERE status = :status
            )
            SELECT inst_id, sid, serial# AS serial_number
              FROM active_sessions
             ORDER BY inst_id, sid
            """,
            {"status": "ACTIVE"},
        )
        self.assertIn("FETCH FIRST 50 ROWS ONLY", result.normalized_sql)
        self.assertEqual(result.referenced_objects, ("gv$session",))
        self.assertEqual(
            result.projected_columns,
            ("inst_id", "sid", "serial_number"),
        )
        self.assertEqual(result.bind_names, ("status",))
        self.assertEqual(len(result.query_sha256), 64)
        self.assertEqual(len(result.policy_sha256), 64)

    def test_sys_catalog_object_can_be_explicitly_allowed(self) -> None:
        policy = OracleDynamicQueryPolicy(
            DynamicQueryPolicySnapshot(
                allowed_objects=("SYS.X$KCBWH",),
                allow_catalog_object_families=False,
            )
        )
        result = policy.validate(
            "SELECT indx, why0 FROM sys.x$kcbwh",
        )
        self.assertEqual(result.referenced_objects, ("sys.x$kcbwh",))

    def test_existing_lower_limit_is_preserved(self) -> None:
        result = self.policy.validate(
            "SELECT sid FROM v$session FETCH FIRST 10 ROWS ONLY"
        )
        self.assertIn("FETCH FIRST 10 ROWS ONLY", result.normalized_sql)
        self.assertEqual(result.max_rows, 10)

        self._assert_rejected(
            "SELECT sid FROM v$session FETCH FIRST 10 ROWS WITH TIES",
            "DYNAMIC_SQL_LIMIT_INVALID",
        )

    def test_boolean_connectors_and_join_conditions_are_allowed(self) -> None:
        result = self.policy.validate(
            """
            SELECT
                s.inst_id AS instance_id,
                s.sid AS session_id
            FROM gv$session s
            JOIN gv$process p
              ON p.inst_id = s.inst_id
             AND p.addr = s.paddr
            WHERE s.type = :session_type
              AND (s.status = :active_status OR s.username IS NOT NULL)
            """,
            {
                "session_type": "USER",
                "active_status": "ACTIVE",
            },
        )

        self.assertEqual(
            result.referenced_objects,
            ("gv$process", "gv$session"),
        )
        self.assertEqual(
            result.bind_names,
            ("active_status", "session_type"),
        )
        self.assertIn(" AND ", result.normalized_sql)
        self.assertIn(" OR ", result.normalized_sql)

    def test_dml_and_multiple_statements_are_rejected(self) -> None:
        self._assert_rejected(
            "DELETE FROM v$session",
            "DYNAMIC_SQL_NOT_SELECT",
        )
        self._assert_rejected(
            "SELECT sid FROM v$session; SELECT 1 AS value FROM dual",
            "DYNAMIC_SQL_MULTIPLE_STATEMENTS",
        )

    def test_lock_database_link_and_application_table_are_rejected(
        self,
    ) -> None:
        self._assert_rejected(
            "SELECT sid FROM v$session FOR UPDATE",
            "DYNAMIC_SQL_LOCK_FORBIDDEN",
        )
        self._assert_rejected(
            "SELECT sid FROM v$session@remote",
            "DYNAMIC_SQL_DATABASE_LINK_FORBIDDEN",
        )
        self._assert_rejected(
            "SELECT customer_name FROM app.customers",
            "DYNAMIC_SQL_SCHEMA_FORBIDDEN",
        )
        self._assert_rejected(
            "SELECT secret_value FROM secrets",
            "DYNAMIC_SQL_OBJECT_FORBIDDEN",
        )

    def test_package_and_unknown_function_are_rejected(self) -> None:
        self._assert_rejected(
            "SELECT dbms_lock.sleep(1) AS result FROM dual",
            "DYNAMIC_SQL_PACKAGE_CALL_FORBIDDEN",
        )
        self._assert_rejected(
            "SELECT custom_function(sid) AS result FROM v$session",
            "DYNAMIC_SQL_FUNCTION_FORBIDDEN",
        )

    def test_projection_alias_and_bind_parameters_are_strict(self) -> None:
        self._assert_rejected(
            "SELECT sid + 1 FROM v$session",
            "DYNAMIC_SQL_COLUMN_ALIAS_REQUIRED",
        )
        self._assert_rejected(
            "SELECT sid FROM v$session WHERE status = :status",
            "DYNAMIC_SQL_PARAMETERS_MISMATCH",
        )
        self._assert_rejected(
            "SELECT sid FROM v$session",
            "DYNAMIC_SQL_PARAMETERS_MISMATCH",
            {"unused": 1},
        )

    def test_dictionary_columns_are_automatic_and_public(
        self,
    ) -> None:
        star = self.policy.validate("SELECT * FROM v$session")
        self.assertEqual(star.execution_decision, "AUTO_EXECUTE")
        self.assertEqual(star.projected_columns, ("*",))
        self.assertEqual(star.column_sensitivities, ("PUBLIC",))
        self.assertEqual((), star.approval_reason_codes)
        sql_text = self.policy.validate(
            "SELECT sql_text AS sample FROM v$sqlstats",
        )
        self.assertEqual(sql_text.execution_decision, "AUTO_EXECUTE")
        self.assertEqual(sql_text.column_sensitivities, ("PUBLIC",))
        self.assertEqual((), sql_text.approval_reason_codes)

    def test_count_star_is_an_automatic_aggregate(self) -> None:
        result = self.policy.validate(
            "SELECT COUNT(*) AS object_count FROM all_objects"
        )
        self.assertEqual(result.execution_decision, "AUTO_EXECUTE")
        self.assertEqual(result.projected_columns, ("object_count",))

    def test_oracle_relative_time_with_sysdate_is_allowed(self) -> None:
        result = self.policy.validate(
            "SELECT owner, object_name, created FROM dba_objects "
            "WHERE created >= SYSDATE - :days "
            "ORDER BY created DESC, owner, object_name",
            {"days": 7},
        )

        self.assertIn("created >= SYSDATE - :days", result.normalized_sql)
        self.assertIn("FETCH FIRST 50 ROWS ONLY", result.normalized_sql)
        self.assertEqual({"days": 7}, result.parameters)

    def test_oracle_surface_function_names_survive_ast_normalization(
        self,
    ) -> None:
        result = self.policy.validate(
            "SELECT TO_DATE('2026-09-01', 'YYYY-MM-DD') AS sample_date, "
            "SUBSTR(instance_name, 1, 8) AS instance_prefix "
            "FROM v$instance"
        )

        self.assertIn(
            "TO_DATE('2026-09-01', 'YYYY-MM-DD')",
            result.normalized_sql,
        )
        self.assertIn("SUBSTR(instance_name, 1, 8)", result.normalized_sql)

    def test_unknown_function_remains_forbidden_after_name_mapping(self) -> None:
        self._assert_rejected(
            "SELECT str_to_magic(instance_name) AS value FROM v$instance",
            "DYNAMIC_SQL_FUNCTION_FORBIDDEN",
        )

    def _assert_rejected(
        self,
        sql: str,
        code: str,
        parameters: dict | None = None,
    ) -> None:
        with self.assertRaises(DynamicQueryRejected) as raised:
            self.policy.validate(sql, parameters)
        self.assertEqual(raised.exception.code, code)


class FakeDynamicDriver:
    db_type = "ORACLE"

    async def execute_dynamic(self, **kwargs):
        return DriverQueryResult(
            columns=("sid", "wait_seconds"),
            rows=((10, 1.25), (11, None)),
            truncated=False,
            db_version="23.26.1.0.0",
        )


class StaticDynamicDriver:
    db_type = "ORACLE"

    def __init__(self, *, columns, rows, database_types=()) -> None:
        self.columns = columns
        self.rows = rows
        self.database_types = database_types

    async def execute_dynamic(self, **kwargs):
        return DriverQueryResult(
            columns=self.columns,
            rows=self.rows,
            truncated=False,
            db_version="23.26.1.0.0",
            database_types=self.database_types,
        )


class DynamicDiagnosticExecutorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.codec = DiagnosticGrantCodec(
            secret="d" * 32,
            issuer="kbot-aiops-worker",
            audience="kbot-aiops-db-executor",
        )
        self.snapshot = DynamicQueryPolicySnapshot(max_rows=25)
        self.validated = OracleDynamicQueryPolicy(self.snapshot).validate(
            "SELECT sid, wait_time_micro / 1000000 AS wait_seconds "
            "FROM v$session WHERE status = :status",
            {"status": "ACTIVE"},
        )
        now = datetime.now(UTC).replace(microsecond=0)
        self.grant = DynamicDiagnosticExecutionGrant(
            issuer="kbot-aiops-worker",
            audience="kbot-aiops-db-executor",
            grant_id=uuid7(),
            issued_at=now,
            expires_at=now + timedelta(seconds=30),
            run_id=uuid7(),
            task_id=uuid7(),
            lease_token_hash="a" * 64,
            target_id=uuid7(),
            domain_id=100,
            target_row_version=2,
            connection_profile=DiagnosticConnectionProfile(
                host="db.internal",
                port=1521,
                service="PDB01",
                tls_enabled=False,
            ),
            diagnostic_credential_id=uuid7(),
            query_sha256=self.validated.query_sha256,
            policy_sha256=self.validated.policy_sha256,
            policy_snapshot=OracleDynamicQueryPolicyGrant.model_validate(
                self.snapshot.model_dump(mode="json")
            ),
            parameters_sha256=canonical_sha256(
                self.validated.parameters
            ),
            projected_columns=self.validated.projected_columns,
            capability_snapshot_hash="b" * 64,
            limits=DiagnosticLimits(
                statement_timeout_seconds=8,
                max_result_rows=25,
                max_result_bytes=65536,
            ),
            trace_id="trace-dynamic-1",
        )

    def _request(self, *, sql: str | None = None):
        return DynamicReadDiagnosticRequest(
            executor_request_id=uuid7(),
            grant=self.codec.issue_dynamic(self.grant),
            sql=sql or self.validated.normalized_sql,
            parameters=self.validated.parameters,
            idempotency_key="dynamic-request-1",
        )

    def _service(self, control_plane, *, driver=None):
        return DynamicDiagnosticExecutorService(
            grant_codec=self.codec,
            control_plane=control_plane,
            oracle_driver=driver or FakeDynamicDriver(),
            hard_limits=DiagnosticLimits(
                statement_timeout_seconds=30,
                max_result_rows=100,
                max_result_bytes=1048576,
            ),
            concurrency=1,
        )

    async def test_executor_revalidates_and_returns_typed_observation(
        self,
    ) -> None:
        control_plane = AsyncMock()
        control_plane.issue_credential.return_value = SimpleNamespace(
            username="private-user",
            password="hidden",
        )
        result = await self._service(control_plane).execute(self._request())

        self.assertEqual("SUCCEEDED", result.status)
        assert result.observation is not None
        self.assertEqual(
            ("INTEGER", "DECIMAL"),
            tuple(item.logical_type for item in result.observation.columns),
        )
        self.assertEqual(
            "oracle-dynamic-readonly.v1",
            result.observation.provenance["executor_policy"],
        )
        self.assertNotIn("hidden", result.model_dump_json())

    async def test_sql_tampering_is_rejected_before_credential(self) -> None:
        control_plane = AsyncMock()
        request = self._request(
            sql=self.validated.normalized_sql.replace(
                "v$session", "v$instance"
            )
        )

        with self.assertRaises(DiagnosticGrantError) as raised:
            await self._service(control_plane).execute(request)

        self.assertEqual(
            "DYNAMIC_QUERY_BINDING_MISMATCH", raised.exception.code
        )
        control_plane.issue_credential.assert_not_awaited()

    async def test_automatic_wildcard_result_is_bounded_and_public(self) -> None:
        validated = OracleDynamicQueryPolicy(self.snapshot).validate(
            "SELECT * FROM v$session"
        )
        grant = self.grant.model_copy(
            update={
                "query_sha256": validated.query_sha256,
                "policy_sha256": validated.policy_sha256,
                "parameters_sha256": canonical_sha256({}),
                "projected_columns": validated.projected_columns,
            }
        )
        request = DynamicReadDiagnosticRequest(
            executor_request_id=uuid7(),
            grant=self.codec.issue_dynamic(grant),
            sql=validated.normalized_sql,
            parameters={},
            idempotency_key="dynamic-wildcard-request-1",
        )
        control_plane = AsyncMock()
        control_plane.issue_credential.return_value = SimpleNamespace(
            username="private-user",
            password="hidden",
        )

        result = await self._service(control_plane).execute(request)

        self.assertEqual(result.status, "SUCCEEDED")
        assert result.observation is not None
        self.assertTrue(
            all(
                column.sensitivity == "PUBLIC"
                for column in result.observation.columns
            )
        )

    async def test_output_columns_mismatch_has_specific_gap(self) -> None:
        control_plane = AsyncMock()
        control_plane.issue_credential.return_value = SimpleNamespace(
            username="private-user",
            password="hidden",
        )
        driver = StaticDynamicDriver(
            columns=("sid", "unexpected_column"),
            rows=((10, 1),),
        )

        result = await self._service(
            control_plane, driver=driver
        ).execute(self._request())

        self.assertEqual("GAP", result.status)
        self.assertEqual("OUTPUT_COLUMNS_MISMATCH", result.error_code)

    async def test_binary_or_lob_result_has_specific_gap(self) -> None:
        control_plane = AsyncMock()
        control_plane.issue_credential.return_value = SimpleNamespace(
            username="private-user",
            password="hidden",
        )
        driver = StaticDynamicDriver(
            columns=("sid", "wait_seconds"),
            rows=((10, b"binary"),),
            database_types=("DB_TYPE_NUMBER", "DB_TYPE_BLOB"),
        )

        request = self._request()
        with patch(
            "aiops_agent.executor.dynamic_service.logger.warning"
        ) as warning:
            result = await self._service(
                control_plane, driver=driver
            ).execute(request)

        self.assertEqual("GAP", result.status)
        self.assertEqual("OUTPUT_VALUE_TYPE_UNSUPPORTED", result.error_code)
        logged = warning.call_args.args
        self.assertIn(request.executor_request_id, logged)
        self.assertIn(self.grant.run_id, logged)
        self.assertIn(self.grant.task_id, logged)
        self.assertIn(self.grant.trace_id, logged)
        self.assertIn(self.grant.query_sha256, logged)
        self.assertIn("wait_seconds", logged)
        self.assertIn("DB_TYPE_BLOB", logged)
        self.assertIn("bytes", logged)

    async def test_mixed_column_types_have_specific_gap(self) -> None:
        control_plane = AsyncMock()
        control_plane.issue_credential.return_value = SimpleNamespace(
            username="private-user",
            password="hidden",
        )
        driver = StaticDynamicDriver(
            columns=("sid", "wait_seconds"),
            rows=((10, 1), ("unknown", 2)),
        )

        result = await self._service(
            control_plane, driver=driver
        ).execute(self._request())

        self.assertEqual("GAP", result.status)
        self.assertEqual("OUTPUT_COLUMN_TYPE_MISMATCH", result.error_code)

    def test_fixed_and_dynamic_grants_cannot_cross_entrypoints(self) -> None:
        token = self.codec.issue_dynamic(self.grant)
        with self.assertRaises(DiagnosticGrantError):
            self.codec.verify(token)


class DynamicQueryPlanningTest(unittest.TestCase):
    def _investigation(self, *, sql: str):
        return InvestigationPlanningOutput.model_validate(
            {
                "input_envelope": {
                    "materials": [
                        {
                            "item_no": 1,
                            "material_kind": "QUESTION",
                            "summary": "检查等待会话",
                            "confidence": 1,
                        }
                    ],
                    "explicit_question": "哪些会话正在等待？",
                },
                "task_frame": {
                    "objectives": ["DIAGNOSE"],
                    "problem_statement": "确认当前等待会话",
                    "success_criteria": ["取得当前会话快照"],
                },
                "plan": {
                    "revision_no": 1,
                    "actions": [
                        {
                            "action_id": "a1",
                            "question": "当前有哪些等待会话？",
                            "tool_id": "db.oracle.readonly_query",
                            "input": {
                                "sql": sql,
                                "parameters": {"status": "ACTIVE"},
                            },
                            "expected_evidence_kind": "DATABASE",
                            "measurement_semantics": "CURRENT_ACTIVITY",
                        }
                    ],
                },
            }
        )

    def test_planning_normalizes_and_compiles_dynamic_task(self) -> None:
        investigation, frozen = prepare_dynamic_queries(
            self._investigation(
                sql=(
                    "SELECT sid, event FROM v$session "
                    "WHERE type = 'USER' "
                    "AND (status = :status OR username IS NOT NULL)"
                )
            )
        )
        registry = PlaybookRegistry.load()
        compiled = InvestigationTaskCompiler(registry).compile(
            DbaPlaybookPlan(
                catalog_hash=registry.catalog_hash,
                items=(),
            ),
            investigation_actions=investigation.plan.actions,
        )

        self.assertEqual(("dynamic:a1",), compiled.dynamic_task_keys)
        self.assertIn(
            "FETCH FIRST 200 ROWS ONLY",
            investigation.plan.actions[0].input["sql"],
        )
        self.assertEqual("a1", frozen[0]["action_id"])
        assessment = next(
            item
            for item in compiled.tasks
            if item.task_key == "evidence:assess"
        )
        self.assertIn("dynamic:a1", assessment.depends_on)

    def test_planning_rejects_unsafe_dynamic_query(self) -> None:
        with self.assertRaises(InvestigationPlanValidationError) as raised:
            prepare_dynamic_queries(
                self._investigation(
                    sql=(
                        "SELECT custom_function(sid) AS result "
                        "FROM v$session"
                    )
                )
            )
        self.assertIn("DYNAMIC_SQL_FUNCTION_FORBIDDEN", str(raised.exception))
        self.assertIn("CUSTOM_FUNCTION", str(raised.exception))

    def test_dynamic_tool_exposes_exact_function_allowlist(self) -> None:
        snapshot_builder = SimpleNamespace(discover_tools=lambda _: ())
        capabilities = DbaCapabilitySnapshot(
            agent_id=str(uuid7()),
            agent_version_id=str(uuid7()),
            target_id=str(uuid7()),
            database_type="ORACLE",
            database_version="19c",
            target_enabled=True,
            target_reachable=True,
            target_capabilities=("DB_READONLY",),
        )

        tools = available_tools(snapshot_builder, capabilities)
        dynamic_tool = next(
            item
            for item in tools
            if item["tool_id"] == "db.oracle.readonly_query"
        )

        self.assertEqual(
            list(DynamicQueryPolicySnapshot().allowed_functions),
            dynamic_tool["policy"]["allowed_functions"],
        )
        self.assertIn(
            "policy.allowed_functions", dynamic_tool["description"]
        )
        self.assertEqual(
            "AUTO_EXECUTE_BOUNDED",
            dynamic_tool["policy"]["star_projection_behavior"],
        )
        self.assertEqual(
            ["CREATE SESSION", "SELECT ANY DICTIONARY"],
            dynamic_tool["database_access"]["granted_system_privileges"],
        )
        self.assertEqual(
            ["CURRENT", "AWR", "ASH"],
            dynamic_tool["database_access"]["diagnostic_scopes"],
        )
        self.assertFalse(dynamic_tool["database_access"]["license_gating"])

    def test_fixed_tool_exposes_complete_parameter_constraints(self) -> None:
        diagnostic_registry = DiagnosticRegistry.load()
        playbook_registry = PlaybookRegistry.load()
        snapshot_builder = ToolExecutionSnapshotBuilder(
            playbook_registry=playbook_registry,
            diagnostic_registry=diagnostic_registry,
        )
        capabilities = DbaCapabilitySnapshot(
            agent_id=str(uuid7()),
            agent_version_id=str(uuid7()),
            target_id=str(uuid7()),
            database_type="ORACLE",
            database_version="19c",
            target_enabled=True,
            target_reachable=True,
            target_capabilities=("DB_READONLY", "dynamic_performance_views"),
        )

        tools = available_tools(snapshot_builder, capabilities)
        top_sql = next(
            item for item in tools if item["tool_id"] == "db.sql.top_current"
        )

        self.assertEqual(
            {
                "type": "integer",
                "required": False,
                "minimum": 1,
                "maximum": 50,
                "enum": [],
                "default": 10,
            },
            top_sql["input"]["limit"],
        )


class DynamicQueryPlanningRepairTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _context():
        return SimpleNamespace(
            turn_id=uuid7(),
            content=(),
            recent_context=(),
            target_context={
                "target_id": "target-1",
                "display_name": "订单生产库",
                "db_type": "ORACLE",
                "selection_status": "BOUND",
            },
            prompt_snapshot={"frozen": {}},
            source_run_evidence=None,
            deadline=None,
        )

    async def test_policy_rejection_triggers_one_corrective_plan(self) -> None:
        factory = DynamicQueryPlanningTest()
        rejected = factory._investigation(
            sql="SELECT custom_function(sid) AS result FROM v$session"
        )
        repaired = factory._investigation(
            sql=(
                "SELECT COUNT(sid) AS session_count FROM v$session "
                "WHERE status = :status"
            )
        )
        reasoner = SimpleNamespace(
            repair_policy_invalid_plan=AsyncMock(
                return_value=StructuredModelResult(
                    output=repaired,
                    receipt=SimpleNamespace(name="repaired"),
                )
            )
        )
        service = object.__new__(TurnPlanningService)
        service._investigation_reasoner = reasoner
        context = SimpleNamespace(
            turn_id=uuid7(),
            content=(),
            recent_context=(),
            target_context={
                "target_id": "target-1",
                "display_name": "订单生产库",
                "db_type": "ORACLE",
                "selection_status": "BOUND",
            },
            prompt_snapshot={"frozen": {}},
            source_run_evidence=None,
            deadline=None,
        )

        planned, investigation, frozen, source_queries = (
            await service._prepare_queries_with_repair(
                context=context,
                planned=StructuredModelResult(
                    output=rejected,
                    receipt=SimpleNamespace(name="rejected"),
                ),
                available_tools=(
                    {
                        "tool_id": "db.oracle.readonly_query",
                        "policy": {"allowed_functions": ["COUNT"]},
                    },
                ),
                available_playbooks=(),
                model_snapshot={"technical_name": "test"},
                revision_no=1,
            )
        )

        self.assertEqual("repaired", planned.receipt.name)
        self.assertIn(
            "COUNT(sid)", investigation.plan.actions[0].input["sql"]
        )
        self.assertEqual("a1", frozen[0]["action_id"])
        self.assertEqual(
            {
                "ad_hoc_prometheus_queries": [],
                "ad_hoc_log_queries": [],
            },
            source_queries,
        )
        call = reasoner.repair_policy_invalid_plan.await_args.kwargs
        self.assertIn("CUSTOM_FUNCTION", call["validation_error"])
        self.assertEqual(context.target_context, call["target_context"])
        reasoner.repair_policy_invalid_plan.assert_awaited_once()

    async def test_second_rejection_keeps_independent_valid_action(self) -> None:
        factory = DynamicQueryPlanningTest()
        rejected = factory._investigation(
            sql="SELECT custom_function(sid) AS result FROM v$session"
        )
        valid_action = rejected.plan.actions[0].model_copy(
            update={
                "action_id": "a2",
                "question": "当前活动会话数是多少？",
                "input": {
                    "sql": (
                        "SELECT COUNT(sid) AS session_count FROM v$session "
                        "WHERE status = :status"
                    ),
                    "parameters": {"status": "ACTIVE"},
                },
            }
        )
        repaired = rejected.model_copy(
            update={
                "plan": rejected.plan.model_copy(
                    update={
                        "actions": (rejected.plan.actions[0], valid_action)
                    }
                )
            }
        )
        reasoner = SimpleNamespace(
            repair_policy_invalid_plan=AsyncMock(
                return_value=StructuredModelResult(
                    output=repaired,
                    receipt=SimpleNamespace(name="partially-repaired"),
                )
            )
        )
        service = object.__new__(TurnPlanningService)
        service._investigation_reasoner = reasoner

        _, investigation, frozen, _ = (
            await service._prepare_queries_with_repair(
                context=self._context(),
                planned=StructuredModelResult(
                    output=rejected,
                    receipt=SimpleNamespace(name="rejected"),
                ),
                available_tools=({"tool_id": "db.oracle.readonly_query"},),
                available_playbooks=(),
                model_snapshot={"technical_name": "test"},
                revision_no=2,
            )
        )

        self.assertEqual(
            ["a2"],
            [action.action_id for action in investigation.plan.actions],
        )
        self.assertEqual(["a2"], [item["action_id"] for item in frozen])

    async def test_second_rejection_removes_dependent_action(self) -> None:
        factory = DynamicQueryPlanningTest()
        rejected = factory._investigation(
            sql="SELECT custom_function(sid) AS result FROM v$session"
        )
        dependent = rejected.plan.actions[0].model_copy(
            update={
                "action_id": "a2",
                "question": "汇总上一动作的会话证据",
                "input": {
                    "sql": "SELECT COUNT(*) AS sample_count FROM v$session",
                    "parameters": {},
                },
                "depends_on": ("a1",),
            }
        )
        repaired = rejected.model_copy(
            update={
                "plan": rejected.plan.model_copy(
                    update={"actions": (rejected.plan.actions[0], dependent)}
                )
            }
        )
        reasoner = SimpleNamespace(
            repair_policy_invalid_plan=AsyncMock(
                return_value=StructuredModelResult(
                    output=repaired,
                    receipt=SimpleNamespace(name="invalid"),
                )
            )
        )
        service = object.__new__(TurnPlanningService)
        service._investigation_reasoner = reasoner

        with self.assertRaises(InvestigationPlanValidationError) as raised:
            await service._prepare_queries_with_repair(
                context=self._context(),
                planned=StructuredModelResult(
                    output=rejected,
                    receipt=SimpleNamespace(name="rejected"),
                ),
                available_tools=({"tool_id": "db.oracle.readonly_query"},),
                available_playbooks=(),
                model_snapshot={"technical_name": "test"},
                revision_no=2,
            )

        self.assertIn("没有可执行动作", str(raised.exception))
        self.assertIn("依赖的调查动作不可执行", str(raised.exception))

    async def test_fixed_tool_parameter_rejection_triggers_repair(self) -> None:
        factory = DynamicQueryPlanningTest()
        template = factory._investigation(
            sql="SELECT sid FROM v$session WHERE status = :status"
        )
        rejected_action = template.plan.actions[0].model_copy(
            update={
                "tool_id": "db.sql.top_current",
                "input": {"limit": 1000},
            }
        )
        repaired_action = rejected_action.model_copy(
            update={"input": {"limit": 20}}
        )
        rejected = template.model_copy(
            update={
                "plan": template.plan.model_copy(
                    update={"actions": (rejected_action,)}
                )
            }
        )
        repaired = rejected.model_copy(
            update={
                "plan": rejected.plan.model_copy(
                    update={"actions": (repaired_action,)}
                )
            }
        )
        reasoner = SimpleNamespace(
            repair_policy_invalid_plan=AsyncMock(
                return_value=StructuredModelResult(
                    output=repaired,
                    receipt=SimpleNamespace(name="repaired"),
                )
            )
        )
        diagnostic_registry = DiagnosticRegistry.load()
        playbook_registry = PlaybookRegistry.load()
        service = object.__new__(TurnPlanningService)
        service._investigation_reasoner = reasoner
        service._tool_snapshot_builder = ToolExecutionSnapshotBuilder(
            playbook_registry=playbook_registry,
            diagnostic_registry=diagnostic_registry,
        )
        context = SimpleNamespace(
            turn_id=uuid7(),
            content=(),
            recent_context=(),
            target_context={
                "target_id": "target-1",
                "display_name": "订单生产库",
                "db_type": "ORACLE",
                "selection_status": "BOUND",
            },
            prompt_snapshot={"frozen": {}},
            source_run_evidence=None,
            deadline=None,
            capabilities=DbaCapabilitySnapshot(
                agent_id=str(uuid7()),
                agent_version_id=str(uuid7()),
                target_id=str(uuid7()),
                database_type="ORACLE",
                database_version="19c",
                target_enabled=True,
                target_reachable=True,
                target_capabilities=(
                    "DB_READONLY",
                    "dynamic_performance_views",
                ),
            ),
        )

        planned, investigation, frozen, source_queries = (
            await service._prepare_queries_with_repair(
                context=context,
                planned=StructuredModelResult(
                    output=rejected,
                    receipt=SimpleNamespace(name="rejected"),
                ),
                available_tools=available_tools(
                    service._tool_snapshot_builder,
                    context.capabilities,
                ),
                available_playbooks=(),
                model_snapshot={"technical_name": "test"},
                revision_no=1,
            )
        )

        self.assertEqual("repaired", planned.receipt.name)
        self.assertEqual(
            ["db.instance.identity", "db.sql.top_current"],
            [action.tool_id for action in investigation.plan.actions],
        )
        self.assertEqual(20, investigation.plan.actions[1].input["limit"])
        self.assertEqual((), frozen)
        self.assertEqual(
            {
                "ad_hoc_prometheus_queries": [],
                "ad_hoc_log_queries": [],
            },
            source_queries,
        )
        validation_error = reasoner.repair_policy_invalid_plan.await_args.kwargs[
            "validation_error"
        ]
        self.assertIn("参数 limit 大于最大值", validation_error)


class GapDynamicExecutorClient:
    def __init__(
        self,
        *,
        retryable: bool = False,
        error_code: str | None = None,
    ) -> None:
        self.request = None
        self.retryable = retryable
        self.error_code = error_code

    async def execute_dynamic_diagnostic(self, request, *, trace_id):
        self.request = request
        return ReadDiagnosticResult(
            executor_request_id=request.executor_request_id,
            status="GAP",
            error_code=self.error_code
            or (
                "TARGET_CONNECTION_TIMEOUT"
                if self.retryable
                else "PRIVILEGE_MISSING"
            ),
            retryable=self.retryable,
        )


class DynamicQueryInvocationHandlerTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_signs_frozen_query_and_returns_auditable_gap(
        self,
    ) -> None:
        codec = DiagnosticGrantCodec(
            secret="w" * 32,
            issuer="kbot-aiops-worker",
            audience="kbot-aiops-db-executor",
        )
        policy_snapshot = DynamicQueryPolicySnapshot(max_rows=10)
        validated = OracleDynamicQueryPolicy(policy_snapshot).validate(
            "SELECT sid FROM v$session"
        )
        client = GapDynamicExecutorClient()
        handler = DynamicQueryInvocationHandler(
            executor_client=client,
            grant_codec=codec,
            grant_issuer="kbot-aiops-worker",
            grant_audience="kbot-aiops-db-executor",
            grant_ttl_seconds=45,
        )
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="dynamic:a1",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-worker-dynamic",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "investigation_execution": {
                    "capability_snapshot_hash": "c" * 64,
                    "database": {
                        "domain_id": 100,
                        "target_row_version": 1,
                        "connection_profile": {
                            "host": "db.internal",
                            "port": 1521,
                            "service": "PDB01",
                            "tls_enabled": False,
                        },
                        "diagnostic_credential_id": str(uuid7()),
                        "automatic_access_enabled": True,
                    },
                    "dynamic_invocations": {
                        "dynamic:a1": {
                            "action_id": "a1",
                            "measurement_semantics": "CURRENT_ACTIVITY",
                            "policy_snapshot": policy_snapshot.model_dump(
                                mode="json"
                            ),
                            "validated_query": validated.model_dump(
                                mode="json"
                            ),
                            "limits": {
                                "statement_timeout_seconds": 10,
                                "max_result_rows": 10,
                                "max_result_bytes": 65536,
                                "max_columns": 16,
                                "max_cell_chars": 1024,
                            },
                        }
                    },
                }
            },
            policy_snapshot={},
            input_artifacts=(),
            lease_token="lease-1",
            lease_until=(
                datetime.now(UTC) + timedelta(seconds=60)
            ).isoformat(),
        )

        result = await handler.execute(context)

        self.assertEqual("FAILED", result.status)
        self.assertEqual(
            "PRIVILEGE_MISSING", result.tool_outcomes[0].gap.code
        )
        assert client.request is not None
        grant = codec.verify_dynamic(client.request.grant)
        self.assertEqual(validated.query_sha256, grant.query_sha256)

        retryable_handler = DynamicQueryInvocationHandler(
            executor_client=GapDynamicExecutorClient(retryable=True),
            grant_codec=codec,
            grant_issuer="kbot-aiops-worker",
            grant_audience="kbot-aiops-db-executor",
            grant_ttl_seconds=45,
        )
        retry_context = replace(context, attempt=1, max_attempts=2)
        with self.assertRaises(RetryableTaskError):
            await retryable_handler.execute(retry_context)
        final = await retryable_handler.execute(
            replace(retry_context, attempt=2)
        )
        self.assertEqual(
            "TARGET_CONNECTION_TIMEOUT",
            final.tool_outcomes[0].gap.code,
        )

    async def test_worker_preserves_output_validation_reason(self) -> None:
        codec = DiagnosticGrantCodec(
            secret="w" * 32,
            issuer="kbot-aiops-worker",
            audience="kbot-aiops-db-executor",
        )
        policy_snapshot = DynamicQueryPolicySnapshot(max_rows=10)
        validated = OracleDynamicQueryPolicy(policy_snapshot).validate(
            "SELECT sid FROM v$session"
        )
        client = GapDynamicExecutorClient(
            error_code="OUTPUT_COLUMNS_MISMATCH"
        )
        handler = DynamicQueryInvocationHandler(
            executor_client=client,
            grant_codec=codec,
            grant_issuer="kbot-aiops-worker",
            grant_audience="kbot-aiops-db-executor",
            grant_ttl_seconds=45,
        )
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="dynamic:a1",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-worker-output-gap",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "investigation_execution": {
                    "capability_snapshot_hash": "c" * 64,
                    "database": {
                        "domain_id": 100,
                        "target_row_version": 1,
                        "connection_profile": {
                            "host": "db.internal",
                            "port": 1521,
                            "service": "PDB01",
                            "tls_enabled": False,
                        },
                        "diagnostic_credential_id": str(uuid7()),
                        "automatic_access_enabled": True,
                    },
                    "dynamic_invocations": {
                        "dynamic:a1": {
                            "action_id": "a1",
                            "measurement_semantics": "CURRENT_ACTIVITY",
                            "policy_snapshot": policy_snapshot.model_dump(
                                mode="json"
                            ),
                            "validated_query": validated.model_dump(
                                mode="json"
                            ),
                            "limits": {
                                "statement_timeout_seconds": 10,
                                "max_result_rows": 10,
                                "max_result_bytes": 65536,
                                "max_columns": 16,
                                "max_cell_chars": 1024,
                            },
                        }
                    },
                }
            },
            policy_snapshot={},
            input_artifacts=(),
            lease_token="lease-output-gap",
            lease_until=(
                datetime.now(UTC) + timedelta(seconds=60)
            ).isoformat(),
        )

        result = await handler.execute(context)

        gap = result.tool_outcomes[0].gap
        assert gap is not None
        self.assertEqual("OUTPUT_COLUMNS_MISMATCH", gap.code)
        self.assertIn("查询已执行", gap.detail)
        self.assertNotIn("权限", gap.detail)


if __name__ == "__main__":
    unittest.main()
