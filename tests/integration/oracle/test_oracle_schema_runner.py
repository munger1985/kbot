"""Oracle 全量建库工具的纯本地测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.db.apply_oracle_schema import (
    FOUNDATION_VALIDATION_EXIT_CODE,
    FoundationValidationError,
    _apply_platform_foundation,
    _repair_aiops_foreign_key_indexes,
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
        self.assertIn(
            "CREATE INDEX IX_OPS_IMG_EVID_OUTPUT",
            statements[-1].sql,
        )

    def test_json_fields_use_oracle_native_json(self) -> None:
        sql = "\n".join(
            statement.sql for statement in load_schema_statements()
        )
        self.assertEqual(109, sql.count(" JSON"))
        self.assertNotRegex(sql, r"\b[A-Z0-9_]+_JSON\s+CLOB\b")
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
        del statement
        if parameters is not None:
            self.role_permission_mappings.extend(parameters)
            return _DictionaryResult()
        return next(self._results)

    async def exec_driver_sql(self, statement):
        self.statements.append(statement)

    async def commit(self):
        self.committed = True


class PlatformFoundationRepairTest(unittest.IsolatedAsyncioTestCase):
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
                "permission_code": "knowledge_retrieval:operations_manage",
            },
            connection.role_permission_mappings,
        )
        self.assertTrue(connection.committed)

    async def test_nullable_security_column_is_repaired_incrementally(self):
        connection = _FoundationConnection(
            column=("Y", None),
            constraint_status="DISABLED",
        )

        await _apply_platform_foundation(connection)

        self.assertIn(
            "ALTER TABLE KBOT_PLATFORM_USER MODIFY ("
            "MAX_SECURITY_LEVEL DEFAULT 1)",
            connection.statements,
        )
        self.assertIn(
            "UPDATE KBOT_PLATFORM_USER SET MAX_SECURITY_LEVEL = 1 "
            "WHERE MAX_SECURITY_LEVEL IS NULL",
            connection.statements,
        )
        self.assertIn(
            "ALTER TABLE KBOT_PLATFORM_USER MODIFY ("
            "MAX_SECURITY_LEVEL NOT NULL)",
            connection.statements,
        )
        self.assertIn(
            "ALTER TABLE KBOT_PLATFORM_USER ENABLE CONSTRAINT "
            "CK_PLATFORM_USER_SECURITY",
            connection.statements,
        )


class _AIOpsIndexConnection:
    def __init__(self, *, indexes):
        self._results = iter(
            (
                _DictionaryResult(
                    row=[
                        (
                            "FK_OPS_SAMPLE_PARENT",
                            "KBOT_OPS_SAMPLE",
                            "PARENT_ID,DOMAIN_ID",
                        )
                    ]
                ),
                _DictionaryResult(row=indexes),
            )
        )
        self.statements: list[str] = []

    async def execute(self, statement):
        del statement
        return next(self._results)

    async def exec_driver_sql(self, statement):
        self.statements.append(statement)


class AIOpsIndexRepairTest(unittest.IsolatedAsyncioTestCase):
    manifest = {
        "foreign_key_indexes": [
            {
                "constraint": "FK_OPS_SAMPLE_PARENT",
                "index": "IX_OPS_SAMPLE_PARENT",
                "table": "KBOT_OPS_SAMPLE",
                "columns": ["PARENT_ID", "DOMAIN_ID"],
            }
        ]
    }

    async def test_creates_manifest_index_when_foreign_key_is_uncovered(self):
        connection = _AIOpsIndexConnection(indexes=[])

        await _repair_aiops_foreign_key_indexes(
            connection,
            aiops_manifest=self.manifest,
        )

        self.assertEqual(
            [
                "CREATE INDEX IX_OPS_SAMPLE_PARENT ON KBOT_OPS_SAMPLE "
                "(PARENT_ID, DOMAIN_ID)"
            ],
            connection.statements,
        )

    async def test_keeps_existing_prefix_covering_index(self):
        connection = _AIOpsIndexConnection(
            indexes=[
                (
                    "KBOT_OPS_SAMPLE",
                    "IX_CUSTOM_SAMPLE_PARENT",
                    "PARENT_ID,DOMAIN_ID,STATUS",
                )
            ]
        )

        await _repair_aiops_foreign_key_indexes(
            connection,
            aiops_manifest=self.manifest,
        )

        self.assertEqual([], connection.statements)


class PlatformFoundationCliTest(unittest.TestCase):
    @patch(
        "sys.argv",
        ["apply_oracle_schema.py", "--finalize-existing"],
    )
    @patch(
        "scripts.db.apply_oracle_schema.apply_schema",
        new_callable=AsyncMock,
    )
    def test_finalize_existing_uses_existing_schema_recovery_mode(
        self, mocked_apply
    ):
        self.assertEqual(0, main())
        mocked_apply.assert_awaited_once()
        self.assertTrue(
            mocked_apply.await_args.kwargs["finalize_existing"]
        )

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
