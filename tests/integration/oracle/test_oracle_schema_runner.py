"""Oracle 全量建库工具的纯本地测试。"""

import unittest
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.db.apply_oracle_schema import (
    FOUNDATION_VALIDATION_EXIT_CODE,
    PLATFORM_FOUNDATION_PERMISSIONS,
    FoundationValidationError,
    _apply_platform_foundation,
    _assert_views_queryable,
    load_schema_statements,
    load_service_selection,
    main,
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

        self.assertGreater(len(statements), 79)
        self.assertIn(
            "CREATE TABLE KBOT_PLATFORM_DOMAIN",
            statements[0].sql,
        )
        self.assertTrue(
            any(
                "CREATE INDEX IX_OPS_IMG_EVID_OUTPUT" in statement.sql
                for statement in statements
            )
        )
        self.assertIn("IX_OPS_RESOURCE_LEASE_SOURCE", statements[-1].sql)

    def test_json_fields_use_oracle_native_json(self) -> None:
        sql = "\n".join(
            statement.sql for statement in load_schema_statements()
        )
        self.assertEqual(164, sql.count(" JSON"))
        # 平台审计摘要在 Entity 中明确为 Text；其余 JSON 字段仍必须使用原生类型。
        clob_fields = [
            (str(item.script), re.findall(r"\b[A-Z0-9_]+_JSON\s+CLOB\b", item.sql))
            for item in load_schema_statements()
            if re.search(r"\b[A-Z0-9_]+_JSON\s+CLOB\b", item.sql)
        ]
        self.assertEqual(
            [("database/oracle/main_api/001_access_control.sql", ["SUMMARY_JSON CLOB"])],
            clob_fields,
        )
        self.assertNotIn(" IS JSON", sql)

    def test_user_upload_content_role_is_allowed(self) -> None:
        sql = "\n".join(
            statement.sql for statement in load_schema_statements()
        )
        self.assertIn(
            "DOCUMENT_ROLE IN "
            "('MANIFEST', 'CONTENT', 'ATTACHMENT', 'SUPPLEMENT', 'DERIVED')",
            sql,
        )
        self.assertIn(
            "'PENDING_REVIEW', 'ACCEPTED', 'REJECTED', "
            "'FAILED', 'CLEANUP_PENDING'",
            sql,
        )

    def test_config_selects_services_and_keeps_platform_core(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "init.ini"
            config_path.write_text(
                """
                [services]
                model_serving = false
                knowledge_core = true
                agent_runtime = false
                aiops_agent = false
                """,
                encoding="utf-8",
            )

            selection = load_service_selection(config_path)
            statements = load_schema_statements(selection.ordered)

        self.assertEqual(("platform_core",), selection.required)
        self.assertEqual(("knowledge_core",), selection.enabled)
        sql = "\n".join(statement.sql for statement in statements)
        self.assertIn("CREATE TABLE KBOT_PLATFORM_DOMAIN", sql)
        self.assertIn("CREATE TABLE KBOT_KC_COLLECTION", sql)
        self.assertNotIn("CREATE TABLE KBOT_AI_MODEL", sql)
        self.assertNotIn("CREATE TABLE KBOT_AGENT_DEFINITION", sql)
        self.assertNotIn("CREATE TABLE KBOT_OPS_TARGET", sql)

    def test_aiops_foreign_keys_have_required_leading_indexes(self) -> None:
        sql = "\n".join(
            statement.sql for statement in load_schema_statements()
        )
        expected = {
            "IX_OPS_TARGET_FACT_OWNER": "TARGET_ID, DOMAIN_ID",
            "IX_OPS_SIG_DOMAIN": "DOMAIN_ID",
            "IX_OPS_SIG_BINDING": "SOURCE_BINDING_ID",
            "IX_OPS_SITUATION_DOMAIN": "DOMAIN_ID",
            "IX_OPS_REPORT_RUN": "OPS_RUN_ID",
            "IX_OPS_PLAY_INV_RUN": "OPS_RUN_ID",
            "IX_OPS_TOOL_INV_RUN": "OPS_RUN_ID",
            "IX_OPS_KNOW_ASSET_BASE": "KNOWLEDGE_BASE_ID",
            "IX_OPS_KNOW_ASSET_CUR_VER": "CURRENT_VERSION_ID",
            "IX_OPS_KNOW_SCOPE_TAG": "TAG_ID",
        }
        for index_name, columns in expected.items():
            with self.subTest(index_name=index_name):
                self.assertRegex(
                    sql,
                    rf"CREATE INDEX {index_name}\s+ON [A-Z0-9_]+\s*"
                    rf"\(\s*{columns.replace(', ', r'\s*,\s*')}\s*\)",
                )

    def test_config_rejects_required_service_selection(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "init.ini"
            config_path.write_text(
                """
                [services]
                platform_core = false
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "必建基础层不需要配置",
            ):
                load_service_selection(config_path)

    def test_config_rejects_unknown_service(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "init.ini"
            config_path.write_text(
                """
                [services]
                unknown_service = true
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "未知服务"):
                load_service_selection(config_path)


class _DictionaryResult:
    def __init__(self, *, row=None, scalar=None):
        self._row = row
        self._scalar = scalar

    def first(self):
        return self._row

    def scalar_one_or_none(self):
        return self._scalar

    def all(self):
        return self._row or []


class _ViewConnection:
    def __init__(self):
        self.statements: list[str] = []

    async def exec_driver_sql(self, statement):
        self.statements.append(statement)


class OracleViewValidationTest(unittest.IsolatedAsyncioTestCase):
    async def test_views_are_queried_to_trigger_lazy_compilation(self):
        connection = _ViewConnection()

        await _assert_views_queryable(
            connection,
            {"KBOT_V_OPS_RUN", "KBOT_V_OPS_REPORT"},
        )

        self.assertEqual(
            [
                'SELECT * FROM "KBOT_V_OPS_REPORT" WHERE 1 = 0',
                'SELECT * FROM "KBOT_V_OPS_RUN" WHERE 1 = 0',
            ],
            connection.statements,
        )

    async def test_illegal_view_name_is_rejected(self):
        connection = _ViewConnection()

        with self.assertRaisesRegex(RuntimeError, "规范视图名称非法"):
            await _assert_views_queryable(connection, {'KBOT_V"BAD'})


class _FoundationConnection:
    def __init__(self, *, column, constraint_status):
        self._results = iter(
            (
                _DictionaryResult(row=column),
                _DictionaryResult(scalar=constraint_status),
            )
        )
        self.statements: list[str] = []
        self.role_permission_mappings: list[dict[str, str]] = []
        self.committed = False

    async def execute(self, statement, parameters=None):
        sql = str(statement)
        self.statements.append(sql)
        if isinstance(parameters, list):
            self.role_permission_mappings.extend(parameters)
            return _DictionaryResult()
        if parameters is not None:
            return _DictionaryResult()
        return next(self._results)

    async def exec_driver_sql(self, statement):
        self.statements.append(statement)

    async def commit(self):
        self.committed = True


class PlatformFoundationMaintenanceTest(unittest.IsolatedAsyncioTestCase):
    async def test_existing_not_null_security_column_is_not_modified(self):
        connection = _FoundationConnection(
            column=("N", "1 "),
            constraint_status="ENABLED",
        )

        await _apply_platform_foundation(connection)

        security_alters = [
            statement
            for statement in connection.statements
            if statement.startswith("ALTER TABLE KBOT_PLATFORM_USER")
        ]
        self.assertEqual([], security_alters)
        self.assertIn(
            {
                "app_id": "knowledge_retrieval",
                "role_code": "app_admin",
                "permission_code": "knowledge_retrieval:api_key_manage",
            },
            connection.role_permission_mappings,
        )
        self.assertIn(
            {
                "app_id": "aiops",
                "role_code": "app_user",
                "permission_code": "aiops:use",
            },
            connection.role_permission_mappings,
        )
        foundation_sql = "\n".join(connection.statements)
        self.assertIn("MERGE INTO KBOT_PLATFORM_APP", foundation_sql)
        self.assertIn("MERGE INTO KBOT_PERMISSION", foundation_sql)
        self.assertIn("MERGE INTO KBOT_APP_ROLE", foundation_sql)
        self.assertIn(
            "SELECT 'aiops', 'app_user', 'App 用户'",
            foundation_sql,
        )
        self.assertIn("DELETE FROM KBOT_PERMISSION", foundation_sql)
        self.assertIn(
            "DELETE FROM KBOT_APP_ROLE_PERMISSION",
            foundation_sql,
        )
        for permission_code in sorted(
            code
            for code in PLATFORM_FOUNDATION_PERMISSIONS
            if code.startswith(("knowledge_retrieval:", "media_studio:"))
        ):
            self.assertTrue(any(
                mapping["permission_code"] == permission_code
                for mapping in connection.role_permission_mappings
            ), permission_code)
        self.assertTrue(connection.committed)

    async def test_noncanonical_security_column_is_rejected(self):
        connection = _FoundationConnection(
            column=("Y", None),
            constraint_status="DISABLED",
        )

        with self.assertRaisesRegex(
            FoundationValidationError,
            "MAX_SECURITY_LEVEL 与规范 Schema 不一致",
        ):
            await _apply_platform_foundation(connection)


class PlatformFoundationCliTest(unittest.TestCase):
    @patch(
        "sys.argv",
        ["apply_oracle_schema.py", "--check-foundation"],
    )
    def test_check_returns_dedicated_validation_exit_code(self):
        def reject(awaitable):
            awaitable.close()
            raise FoundationValidationError("缺少默认 Domain")

        with patch(
            "scripts.db.apply_oracle_schema.asyncio.run",
            side_effect=reject,
        ) as mocked_run:
            self.assertEqual(FOUNDATION_VALIDATION_EXIT_CODE, main())
            mocked_run.assert_called_once()

    @patch(
        "sys.argv",
        ["apply_oracle_schema.py", "--check-foundation"],
    )
    def test_check_keeps_runtime_failure_as_general_error(self):
        def reject(awaitable):
            awaitable.close()
            raise RuntimeError("数据库配置无效")

        with patch(
            "scripts.db.apply_oracle_schema.asyncio.run",
            side_effect=reject,
        ) as mocked_run:
            self.assertEqual(1, main())
            mocked_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
