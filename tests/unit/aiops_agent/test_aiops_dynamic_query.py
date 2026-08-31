"""Oracle 动态只读 SQL 的 AST 安全策略测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from aiops_agent.executor import DynamicDiagnosticExecutorService
from aiops_agent.executor.drivers import DriverQueryResult
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticLimits,
    DynamicDiagnosticExecutionGrant,
    DynamicReadDiagnosticRequest,
    OracleDynamicQueryPolicyGrant,
    ReadDiagnosticResult,
)
from aiops_agent.application.investigation import prepare_dynamic_queries
from aiops_agent.application.investigation.reasoner import (
    InvestigationPlanValidationError,
)
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import InvestigationTaskCompiler
from aiops_agent.workers.database_handlers import (
    DynamicQueryInvocationHandler,
)
from aiops_agent.workers.handlers import TaskExecutionContext
from platform_core.contracts.aiops import InvestigationPlanningOutput
from platform_core.contracts.aiops.playbooks import DbaPlaybookPlan
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

    def test_dml_and_multiple_statements_are_rejected(self) -> None:
        self._assert_rejected(
            "DELETE FROM v$session",
            "DYNAMIC_SQL_NOT_SELECT",
        )
        self._assert_rejected(
            "SELECT sid FROM v$session; SELECT 1 AS value FROM dual",
            "DYNAMIC_SQL_MULTIPLE_STATEMENTS",
        )

    def test_lock_star_database_link_and_application_table_are_rejected(
        self,
    ) -> None:
        self._assert_rejected(
            "SELECT sid FROM v$session FOR UPDATE",
            "DYNAMIC_SQL_LOCK_FORBIDDEN",
        )
        self._assert_rejected(
            "SELECT * FROM v$session",
            "DYNAMIC_SQL_STAR_FORBIDDEN",
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

    def test_sensitive_source_column_is_rejected_even_with_alias(self) -> None:
        self._assert_rejected(
            "SELECT sql_text AS sample FROM v$sqlstats",
            "DYNAMIC_SQL_SENSITIVE_COLUMN_FORBIDDEN",
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

    def _service(self, control_plane):
        return DynamicDiagnosticExecutorService(
            grant_codec=self.codec,
            control_plane=control_plane,
            oracle_driver=FakeDynamicDriver(),
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
                    "WHERE status = :status"
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
        with self.assertRaises(InvestigationPlanValidationError):
            prepare_dynamic_queries(
                self._investigation(sql="SELECT * FROM v$session")
            )


class GapDynamicExecutorClient:
    def __init__(self) -> None:
        self.request = None

    async def execute_dynamic_diagnostic(self, request, *, trace_id):
        self.request = request
        return ReadDiagnosticResult(
            executor_request_id=request.executor_request_id,
            status="GAP",
            error_code="PRIVILEGE_MISSING",
            retryable=False,
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


if __name__ == "__main__":
    unittest.main()
