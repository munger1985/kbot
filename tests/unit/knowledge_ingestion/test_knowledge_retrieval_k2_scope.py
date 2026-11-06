"""知识检索定向验收的数据库写权限回归。"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/deployment"))
from knowledge_retrieval_k2 import (
    assert_write_allowed, index_columns_match, knowledge_tables, oracle_column_type,
)
from sqlalchemy import Column, DateTime, Index, Integer, MetaData, Table, delete, insert, text, update
from sqlalchemy.schema import CreateTable, DropTable


class ScopedKnowledgeWritesTests(unittest.TestCase):
    def setUp(self):
        self.tables = knowledge_tables()
        self.table = self.tables[0]

    def test_fixed_manifest_has_no_aiops(self):
        self.assertEqual(len(self.tables), 18)
        self.assertTrue(all("ops" not in table.name.lower() for table in self.tables))

    def test_only_canonical_structured_writes(self):
        for statement in (CreateTable(self.table), insert(self.table),
                          update(self.table).where(self.table.c.domain_id == 1),
                          delete(self.table).where(self.table.c.domain_id == 1)):
            assert_write_allowed(statement, self.tables)

    def test_other_app_drop_unscoped_and_raw_writes_rejected(self):
        other = Table("aiops_forbidden", MetaData(), Column("id", Integer))
        foreign_schema = Table(self.table.name, MetaData(), Column("id", Integer), schema="other")
        for statement in (CreateTable(other), insert(other), DropTable(self.table),
                          insert(foreign_schema), delete(self.table), update(self.table),
                          text("DELETE FROM anything")):
            with self.assertRaises(RuntimeError):
                assert_write_allowed(statement, self.tables)

    def test_oracle_native_json_and_number_are_not_legacy_text(self):
        row = dict(data_type="JSON", data_precision=None, data_scale=None,
                   data_length=0, char_used=None, char_length=0)
        self.assertEqual(oracle_column_type(row), "JSON")
        self.assertEqual(oracle_column_type(dict(row, data_type="NUMBER",
                         data_precision=38, data_scale=0)), "NUMBER(38,0)")
        self.assertEqual(oracle_column_type(dict(row, data_type="VARCHAR2",
                         char_used="C", char_length=512)), "VARCHAR2(512CHAR)")
        self.assertNotEqual(oracle_column_type(dict(row, data_type="CLOB")), "JSON")

    def test_only_oracle_timezone_indexes_accept_utc_expression(self):
        table = Table("fixture", MetaData(), Column("expires_at", DateTime(timezone=True)),
                      Column("count", Integer))
        index = Index("fixture_index", table.c.expires_at)
        actual = {"column_names": [None], "expressions": ['SYS_EXTRACT_UTC("EXPIRES_AT")']}
        self.assertTrue(index_columns_match(actual, index, oracle=True))
        self.assertFalse(index_columns_match(actual, index))
        self.assertFalse(index_columns_match(actual, Index("count_index", table.c.count), oracle=True))
        actual["expressions"] = ['SOME_FUNCTION("EXPIRES_AT")']
        self.assertFalse(index_columns_match(actual, index, oracle=True))


if __name__ == "__main__":
    unittest.main()
