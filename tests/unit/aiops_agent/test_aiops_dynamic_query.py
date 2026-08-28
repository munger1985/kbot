"""Oracle 动态只读 SQL 的 AST 安全策略测试。"""

from __future__ import annotations

import unittest

from aiops_agent.diagnostics.dynamic_query import (
    DynamicQueryPolicySnapshot,
    DynamicQueryRejected,
    OracleDynamicQueryPolicy,
)


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

    def _assert_rejected(
        self,
        sql: str,
        code: str,
        parameters: dict | None = None,
    ) -> None:
        with self.assertRaises(DynamicQueryRejected) as raised:
            self.policy.validate(sql, parameters)
        self.assertEqual(raised.exception.code, code)


if __name__ == "__main__":
    unittest.main()
