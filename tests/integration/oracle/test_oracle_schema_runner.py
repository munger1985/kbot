"""Oracle 全量建库工具的纯本地测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.db.apply_oracle_schema import (
    load_schema_statements,
    load_service_selection,
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
            "COMMENT ON COLUMN KBOT_OPS_TASK.OUTPUT_ARTIFACT_ID",
            statements[-1].sql,
        )

    def test_json_fields_use_oracle_native_json(self) -> None:
        sql = "\n".join(
            statement.sql for statement in load_schema_statements()
        )
        self.assertEqual(71, sql.count(" JSON"))
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


if __name__ == "__main__":
    unittest.main()
