"""Agent 问数模式既有 Schema 对齐脚本测试。"""

import unittest

from scripts.db.apply_agent_data_query_schema import (
    ColumnMetadata,
    build_alignment_statements,
)


class AgentDataQuerySchemaScriptTest(unittest.TestCase):
    def test_missing_objects_receive_columns_and_constraint(self):
        statements = build_alignment_statements(
            columns={},
            constraint_exists=False,
            constraint_enabled=False,
        )

        sql = "\n".join(statements)
        self.assertIn("DATA_QUERY_MODE VARCHAR2(16 CHAR)", sql)
        self.assertIn("DATA_PROFILE_NAME VARCHAR2(256 CHAR)", sql)
        self.assertIn("CK_AGENT_DEF_DQ_MODE", sql)

    def test_existing_objects_only_refresh_comments(self):
        statements = build_alignment_statements(
            columns={
                "DATA_QUERY_MODE": ColumnMetadata("VARCHAR2", 16, True),
                "DATA_PROFILE_NAME": ColumnMetadata(
                    "VARCHAR2", 256, True
                ),
            },
            constraint_exists=True,
            constraint_enabled=True,
        )

        self.assertEqual(2, len(statements))
        self.assertTrue(
            all(statement.startswith("COMMENT ON COLUMN") for statement in statements)
        )

    def test_incompatible_existing_column_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Canonical DDL"):
            build_alignment_statements(
                columns={
                    "DATA_QUERY_MODE": ColumnMetadata(
                        "VARCHAR2", 32, True
                    )
                },
                constraint_exists=False,
                constraint_enabled=False,
            )


if __name__ == "__main__":
    unittest.main()
