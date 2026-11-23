"""全服务 Oracle Entity Schema 检查器测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sqlalchemy import Boolean, CHAR, Column, String
from sqlalchemy.dialects.oracle import RAW

from tests.acceptance.check_aiops_entity_schema import (
    AIOPS_ENTITY_CLASSES,
    catalog_column_contract,
    compare_schema_columns,
    entity_column_contract,
)
from tests.acceptance.check_oracle_entity_schema import _compatible, _entity_classes, _orm_tables
from tests.acceptance.check_oracle_schema import SERVICE_TABLES


class OracleEntitySchemaCheckerTest(unittest.TestCase):
    def test_all_service_entities_are_in_scope(self) -> None:
        entities = _entity_classes()
        names = [table.name for table in _orm_tables()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set().union(*SERVICE_TABLES.values()), set(names))

    def test_vector_columns_use_vector_family(self) -> None:
        contracts = [
            entity_column_contract(column).family
            for entity in _entity_classes()
            for column in entity.__table__.columns
        ]
        self.assertIn("VECTOR", contracts)
        self.assertIn("JSON", contracts)

    def test_char_is_not_varchar_and_keeps_character_length(self) -> None:
        row = SimpleNamespace(data_type="CHAR", data_length=4, char_length=1,
                              data_precision=None, data_scale=None, nullable="N")
        actual = catalog_column_contract(row)
        self.assertTrue(_compatible(entity_column_contract(Column("flag", CHAR(1), nullable=False)), actual))
        self.assertFalse(_compatible(entity_column_contract(Column("flag", String(1), nullable=False)), actual))
        flags = [column for entity in _entity_classes() for column in entity.__table__.columns
                 if isinstance(column.type, CHAR)]
        self.assertEqual(5, len(flags))

    def test_managed_credential_nonce_is_raw12_not_blob(self) -> None:
        entity = next(entity for entity in _entity_classes() if entity.__tablename__ == "KBOT_MANAGED_CREDENTIAL")
        nonce = entity.__table__.c.nonce
        self.assertIsInstance(nonce.type, RAW)
        self.assertEqual(("RAW", 12), (entity_column_contract(nonce).family, entity_column_contract(nonce).length))

    def test_notification_boolean_uses_canonical_smallint(self) -> None:
        contract = entity_column_contract(Column("enabled", Boolean, nullable=False))
        self.assertEqual(("NUMBER", None, 0, False),
                         (contract.family, contract.precision, contract.scale, contract.nullable))

    def test_retired_protection_table_is_schema_drift_not_missing_orm_mapping(self) -> None:
        canonical = {entity.__tablename__ for entity in AIOPS_ENTITY_CLASSES}
        self.assertTrue({"KBOT_OPS_PROTECTION_PLAN", "KBOT_OPS_PROTECTION_RUN"} <= canonical)
        column_attrs = dict(data_type="NUMBER", data_length=22, char_length=0,
                            data_precision=19, data_scale=0, nullable="N")
        errors = compare_schema_columns([
            SimpleNamespace(table_name="KBOT_OPS_BACKUP_FIRE", column_name="LEGACY_ID", data_type="旧表独有类型"),
            SimpleNamespace(table_name="KBOT_OPS_PROTECTION_RUN", column_name="EXTRA_COLUMN", **column_attrs),
        ])
        self.assertIn("Oracle 存在规范外对象：['KBOT_OPS_BACKUP_FIRE']", errors)
        self.assertIn("Oracle 已映射表出现规范外列：[('KBOT_OPS_PROTECTION_RUN', 'EXTRA_COLUMN')]", errors)


if __name__ == "__main__":
    unittest.main()
