"""Oracle 全量建库工具的纯本地测试。"""

import unittest

from scripts.apply_oracle_schema import (
    load_schema_statements,
    split_oracle_statements,
)


class OracleSchemaRunnerTest(unittest.TestCase):
    def test_split_preserves_semicolons_in_strings_and_comments(self) -> None:
        sql = """
        -- 注释中的分号 ; 不应拆分
        CREATE TABLE DEMO (VALUE VARCHAR2(20 CHAR) DEFAULT 'a;b');
        /* 块注释中的 ; 也不应拆分 */
        COMMENT ON TABLE DEMO IS 'x;y';
        """

        statements = split_oracle_statements(sql)

        self.assertEqual(2, len(statements))
        self.assertIn("'a;b'", statements[0])
        self.assertIn("'x;y'", statements[1])

    def test_loads_complete_service_order(self) -> None:
        statements = load_schema_statements()

        self.assertEqual(79, len(statements))
        self.assertIn(
            "CREATE TABLE KBOT_PLATFORM_DOMAIN",
            statements[0].sql,
        )
        self.assertIn(
            "CREATE INDEX IX_AGENT_DELEGATION_RUN",
            statements[-1].sql,
        )


if __name__ == "__main__":
    unittest.main()
