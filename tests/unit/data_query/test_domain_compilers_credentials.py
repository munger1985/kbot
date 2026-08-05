"""Data Query 状态、编译器与凭据安全边界。"""

import unittest

from data_query.adapters.credential_cipher import CredentialCipher, DataQueryCredentialError
from data_query.connectors import compile_dialect_query
from data_query.connectors.postgresql import compile_postgresql_query
from data_query.contracts import (
    DataQueryPlanV1,
    DataSourceCredentialStatus,
    DatasetDefinition,
    DimensionDefinition,
    MeasureDefinition,
    PlanFilter,
    PlanMeasure,
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
from data_query.entities import CredentialEntity
from platform_core.identity import uuid7
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
        self.assertEqual(15, len(tables))
        self.assertIn("KBOT_DQ_CREDENTIAL", tables)
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

    def test_cipher_uses_aad_and_rejects_tampering(self):
        cipher = CredentialCipher(key=b"k" * 32, key_version="2026-08")
        source_id = uuid7()
        encrypted = cipher.encrypt(
            domain_id=20,
            data_source_id=source_id,
            credential_version=1,
            username="readonly",
            password="secret",
        )
        row = CredentialEntity(
            domain_id=20,
            data_source_id=source_id,
            credential_version=1,
            username_ciphertext=encrypted.username_ciphertext,
            username_nonce=encrypted.username_nonce,
            password_ciphertext=encrypted.password_ciphertext,
            password_nonce=encrypted.password_nonce,
            key_version=encrypted.key_version,
            status="ACTIVE",
            created_by="test",
            updated_by="test",
        )
        self.assertEqual(
            ("readonly", "secret"),
            cipher.decrypt(domain_id=20, data_source_id=source_id, credential_version=1, row=row),
        )
        with self.assertRaises(DataQueryCredentialError):
            cipher.decrypt(domain_id=21, data_source_id=source_id, credential_version=1, row=row)
        row.password_ciphertext = bytes([row.password_ciphertext[0] ^ 1]) + row.password_ciphertext[1:]
        with self.assertRaises(DataQueryCredentialError):
            cipher.decrypt(domain_id=20, data_source_id=source_id, credential_version=1, row=row)


if __name__ == "__main__":
    unittest.main()
