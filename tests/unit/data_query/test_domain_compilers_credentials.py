"""Data Query 状态、编译器与凭据安全边界。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import oracledb

from data_query import entities as _data_query_entities  # noqa: F401
from data_query.connectors import compile_dialect_query
from data_query.adapters.query_executor import DataSourceExecutorResolver
from data_query.connectors.postgresql import compile_postgresql_query
from data_query.contracts import (
    AgentBindingMatch,
    DataQueryPlanV1,
    DataSourceCredentialStatus,
    DataSourceCreate,
    DataSourceCredentials,
    DataSourceEndpoint,
    DatasetDefinition,
    DimensionDefinition,
    MeasureDefinition,
    PlanFilter,
    PlanMeasure,
    PlanOrderBy,
    SemanticModelDefinition,
)
from data_query.domain import (
    DataQueryRunStatus,
    DataSourceStatus,
    QueryPlanValidationError,
    SchemaSnapshotStatus,
    SemanticModelVersionStatus,
    can_transition,
    validate_query_plan,
)
from platform_core.identity import uuid7
from platform_core.managed_credentials import ManagedCredentialCipher
from platform_core.notifications.catalog import event_definition
from platform_core.persistence.orm import BaseEntity


def _model() -> SemanticModelDefinition:
    return SemanticModelDefinition(
        datasets=(DatasetDefinition(
            name="sales",
            display_name="销售",
            physical_schema="analytics",
            physical_object="orders",
        ),),
        dimensions=(DimensionDefinition(
            name="region",
            dataset="sales",
            physical_column="region",
            value_type="STRING",
        ),),
        measures=(MeasureDefinition(
            name="amount",
            dataset="sales",
            physical_column="amount",
            aggregation="SUM",
            value_type="DECIMAL",
        ),),
    )


def _plan(*, limit: int = 20, field: str = "region") -> DataQueryPlanV1:
    return DataQueryPlanV1(
        semantic_model_id=uuid7(),
        semantic_model_version=1,
        dataset="sales",
        measures=(PlanMeasure(name="amount", aggregation="SUM"),),
        dimensions=("region",),
        filters=(PlanFilter(
            field=field,
            operator="EQ",
            values=("华东'; DROP TABLE x;--",),
        ),),
        limit=limit,
    )


class DataQueryDomainCompilerCredentialTest(unittest.TestCase):
    def test_new_data_source_enables_schema_discovery_by_default(self):
        source = DataSourceCreate(
            display_name="业务库",
            source_type="POSTGRESQL",
            endpoint=DataSourceEndpoint(
                host="db.example.com",
                port=5432,
                database="warehouse",
                allowed_schemas=("analytics",),
            ),
            credentials=DataSourceCredentials(
                username="readonly", password="secret"
            ),
        )
        self.assertTrue(source.auto_discover_schema)

    def test_active_binding_match_can_check_any_model_for_exact_agent_version(self):
        match = AgentBindingMatch(
            consumer_app_id="knowledge_retrieval",
            agent_id=uuid7(),
            agent_version_id=uuid7(),
        )
        self.assertEqual((), match.semantic_model_ids)

    def test_data_query_workflow_notification_events_are_registered(self):
        event_types = (
            "data_query.schema.discovery_started",
            "data_query.schema.selection_required",
            "data_query.schema.capture_started",
            "data_query.schema.capture_progress",
            "data_query.schema.capture_completed",
            "data_query.schema.capture_partial",
            "data_query.schema.capture_failed",
            "data_query.semantic_model.generation_started",
            "data_query.semantic_model.generation_completed",
            "data_query.semantic_model.generation_failed",
            "data_query.semantic_model.review_requested",
            "data_query.semantic_model.returned",
            "data_query.semantic_model.published",
            "data_query.semantic_model.retired",
            "data_query.validation.started",
            "data_query.validation.completed",
            "data_query.validation.failed",
        )
        for event_type in event_types:
            with self.subTest(event_type=event_type):
                self.assertEqual(event_type, event_definition(event_type).event_type)

    def test_data_source_credential_status_never_exposes_reference(self):
        fields = DataSourceCredentialStatus.model_fields
        self.assertNotIn("credential_id", fields)
        self.assertNotIn("username", fields)
        self.assertNotIn("ciphertext", fields)

    def test_state_machines_reject_terminal_transition(self):
        self.assertTrue(can_transition(DataSourceStatus.DRAFT, DataSourceStatus.VALIDATING))
        self.assertTrue(can_transition(SchemaSnapshotStatus.CAPTURING, SchemaSnapshotStatus.READY))
        self.assertTrue(can_transition(SemanticModelVersionStatus.REVIEW, SemanticModelVersionStatus.ACTIVE))
        self.assertFalse(can_transition(DataQueryRunStatus.COMPLETED, DataQueryRunStatus.EXECUTING))

    def test_all_oracle_entities_are_registered(self):
        tables = {name for name in BaseEntity.metadata.tables if name.startswith("KBOT_DQ_")}
        self.assertEqual(14, len(tables))
        self.assertNotIn("KBOT_DQ_CREDENTIAL", tables)
        self.assertIn("KBOT_DQ_AUDIT", tables)

    def test_plan_rejects_unknown_field_and_budget_overage(self):
        with self.assertRaisesRegex(QueryPlanValidationError, "FILTER_FIELD_NOT_FOUND"):
            validate_query_plan(plan=_plan(field="raw_sql"), model=_model(), policy_max_limit=100)
        with self.assertRaisesRegex(QueryPlanValidationError, "POLICY_LIMIT_EXCEEDED"):
            validate_query_plan(plan=_plan(limit=101), model=_model(), policy_max_limit=100)

    def test_all_dialects_bind_untrusted_values(self):
        postgresql = compile_postgresql_query(plan=_plan(), model=_model(), policy_max_limit=100)
        mysql = compile_dialect_query(dialect="MYSQL", plan=_plan(), model=_model(), policy_max_limit=100)
        oracle = compile_dialect_query(dialect="ORACLE", plan=_plan(), model=_model(), policy_max_limit=100)
        self.assertIn("$1", postgresql.sql)
        self.assertIn("%s", mysql.sql)
        self.assertIn(":p1", oracle.sql)
        for compiled in (postgresql, mysql, oracle):
            self.assertNotIn("DROP TABLE", compiled.sql)

    def test_contains_accepts_multiple_values_as_parameterized_or(self):
        plan = _plan().model_copy(update={
            "filters": (PlanFilter(
                field="region",
                operator="CONTAINS",
                values=("finance", "financial"),
            ),),
        })
        postgresql = compile_postgresql_query(
            plan=plan, model=_model(), policy_max_limit=100
        )
        mysql = compile_dialect_query(
            dialect="MYSQL", plan=plan, model=_model(),
            policy_max_limit=100,
        )
        oracle = compile_dialect_query(
            dialect="ORACLE", plan=plan, model=_model(),
            policy_max_limit=100,
        )

        self.assertIn("$1", postgresql.sql)
        self.assertIn("$2", postgresql.sql)
        self.assertIn(" OR ", postgresql.sql)
        self.assertEqual(("finance", "financial", 20), postgresql.parameters)
        self.assertEqual(("finance", "financial", 20), mysql.parameters)
        self.assertEqual(("finance", "financial", 20), oracle.parameters)
        self.assertIn(" OR ", mysql.sql)
        self.assertIn(" OR ", oracle.sql)

    def test_order_by_uses_semantic_model_physical_expressions(self):
        model = SemanticModelDefinition(
            datasets=(DatasetDefinition(
                name="assets",
                display_name="Asset",
                physical_schema="app",
                physical_object="KBOT_V_KM_ASSET_CURRENT",
            ),),
            dimensions=(DimensionDefinition(
                name="asset_date",
                dataset="assets",
                physical_column="ASSET_DATE_VALUE",
                value_type="DATE",
            ),),
            measures=(MeasureDefinition(
                name="asset_count",
                dataset="assets",
                physical_column=None,
                aggregation="COUNT",
                value_type="INTEGER",
            ),),
        )
        plan = DataQueryPlanV1(
            semantic_model_id=uuid7(),
            semantic_model_version=1,
            dataset="assets",
            measures=(PlanMeasure(
                name="asset_count", aggregation="COUNT",
            ),),
            dimensions=("asset_date",),
            order_by=(PlanOrderBy(
                field="asset_date", direction="DESC",
            ),),
            limit=11,
        )

        oracle = compile_dialect_query(
            dialect="ORACLE", plan=plan, model=model,
            policy_max_limit=100,
        )
        mysql = compile_dialect_query(
            dialect="MYSQL", plan=plan, model=model,
            policy_max_limit=100,
        )
        postgresql = compile_postgresql_query(
            plan=plan, model=model, policy_max_limit=100,
        )

        self.assertIn('ORDER BY "ASSET_DATE_VALUE" DESC', oracle.sql)
        self.assertNotIn('ORDER BY "asset_date"', oracle.sql)
        self.assertIn("ORDER BY `ASSET_DATE_VALUE` DESC", mysql.sql)
        self.assertIn('ORDER BY "ASSET_DATE_VALUE" DESC', postgresql.sql)

    def test_order_by_rejects_field_missing_from_projection(self):
        plan = _plan().model_copy(update={
            "dimensions": (),
            "order_by": (PlanOrderBy(
                field="region", direction="DESC",
            ),),
        })

        with self.assertRaisesRegex(
            QueryPlanValidationError, "ORDER_FIELD_NOT_SELECTED",
        ):
            validate_query_plan(
                plan=plan, model=_model(), policy_max_limit=100,
            )

    def test_cipher_uses_aad_and_rejects_tampering(self):
        cipher = ManagedCredentialCipher(
            key=b"k" * 32, key_version="2026-08"
        )
        credential_id = uuid7()
        encrypted = cipher.encrypt(
            {"username": "readonly", "password": "secret"},
            domain_id=20,
            namespace="data_query",
            credential_kind="database",
            credential_id=credential_id,
        )
        self.assertEqual(
            {"username": "readonly", "password": "secret"},
            cipher.decrypt(
                encrypted,
                domain_id=20,
                namespace="data_query",
                credential_kind="database",
                credential_id=credential_id,
            ),
        )
        with self.assertRaises(Exception):
            cipher.decrypt(
                encrypted,
                domain_id=21,
                namespace="data_query",
                credential_kind="database",
                credential_id=credential_id,
            )


class DataSourceExecutorErrorBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_oracle_query_error_is_not_reported_as_connection_failure(self):
        cursor = SimpleNamespace(
            execute=AsyncMock(side_effect=[
                None,
                oracledb.DatabaseError("ORA-00933"),
            ]),
            description=(("VALUE",),),
            fetchall=AsyncMock(return_value=[]),
            close=lambda: None,
        )
        connection = SimpleNamespace(
            call_timeout=0,
            cursor=lambda: cursor,
            rollback=AsyncMock(),
            close=AsyncMock(),
        )
        endpoint = SimpleNamespace(
            host="db.example.com",
            port=1521,
            database="KBot",
            tls_enabled=False,
        )
        compiled = SimpleNamespace(sql="SELECT 1 FROM dual", parameters=())

        with patch(
            "data_query.adapters.query_executor.oracledb.connect_async",
            new=AsyncMock(return_value=connection),
        ):
            with self.assertRaisesRegex(
                ValueError, "DATA_QUERY_EXECUTION_FAILED"
            ):
                await DataSourceExecutorResolver._execute_oracle(
                    endpoint, "readonly", "secret", {}, compiled
                )

    async def test_oracle_connect_error_remains_connection_failure(self):
        endpoint = SimpleNamespace(
            host="db.example.com",
            port=1521,
            database="KBot",
            tls_enabled=False,
        )
        compiled = SimpleNamespace(sql="SELECT 1 FROM dual", parameters=())

        with patch(
            "data_query.adapters.query_executor.oracledb.connect_async",
            new=AsyncMock(
                side_effect=oracledb.DatabaseError("DPY-6005")
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "DATA_SOURCE_CONNECTION_FAILED"
            ):
                await DataSourceExecutorResolver._execute_oracle(
                    endpoint, "readonly", "secret", {}, compiled
                )


if __name__ == "__main__":
    unittest.main()
