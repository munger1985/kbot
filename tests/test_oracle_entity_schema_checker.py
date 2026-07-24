"""全服务 Oracle Entity Schema 检查器测试。"""

from __future__ import annotations

import unittest

from scripts.check_aiops_entity_schema import entity_column_contract
from scripts.check_oracle_entity_schema import _entity_classes


class OracleEntitySchemaCheckerTest(unittest.TestCase):
    def test_all_service_entities_are_in_scope(self) -> None:
        entities = _entity_classes()
        self.assertEqual(42, len(entities))
        self.assertEqual(
            838,
            sum(len(entity.__table__.columns) for entity in entities),
        )

    def test_vector_columns_use_vector_family(self) -> None:
        contracts = {
            entity_column_contract(column).family
            for entity in _entity_classes()
            for column in entity.__table__.columns
        }
        self.assertIn("VECTOR", contracts)


if __name__ == "__main__":
    unittest.main()
