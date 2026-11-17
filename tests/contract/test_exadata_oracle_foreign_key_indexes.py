"""Exadata 报告表的外键必须有独立前导列普通索引。"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ExadataOracleForeignKeyIndexesTest(unittest.TestCase):
    def test_report_foreign_keys_have_leading_column_indexes(self) -> None:
        sql = (ROOT / "database/oracle/aiops_agent/015_ops_exadata.sql").read_text()
        foreign_keys = re.findall(
            r"FOREIGN KEY\s*\((\w+)\)",
            sql.split("CREATE TABLE KBOT_OPS_EXADATA_REPORT", 1)[1],
        )
        indexes = re.findall(
            r"CREATE INDEX \w+ ON KBOT_OPS_EXADATA_REPORT\s*\((\w+)\)", sql,
        )
        self.assertEqual(set(foreign_keys), {"RESOURCE_ID", "BINDING_ID"})
        self.assertTrue(set(foreign_keys).issubset(indexes))
        self.assertNotIn("CREATE UNIQUE INDEX", sql.upper())


if __name__ == "__main__":
    unittest.main()
