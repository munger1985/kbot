"""知识检索与多媒体创作工作台原地升级脚本合同测试。"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_knowledge_retrieval_and_media_studio.sql"
)
KR_AGENTS = ROOT / "database" / "oracle" / "knowledge_retrieval_app" / "001_agents.sql"
KR_X_SEARCH = ROOT / "database" / "oracle" / "knowledge_retrieval_app" / "002_x_search.sql"
MEDIA_RUNTIME = ROOT / "database" / "oracle" / "media_studio_app" / "001_media_runtime.sql"


class ProductAppSchemaUpgradeScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sql = SCRIPT.read_text(encoding="utf-8")
        self.normalized = self.sql.upper()

    def test_converges_knowledge_retrieval_version_columns(self) -> None:
        self.assertIn("KBOT_KR_AGENT_VERSION", self.normalized)
        self.assertIn("KNOWLEDGE_CORE_ID", self.normalized)
        self.assertIn("DATA_MODEL_IDS_JSON", self.normalized)
        self.assertIn("DROP COLUMN ENABLED_CAPABILITIES_JSON", self.normalized)
        self.assertIn("JSON_ARRAY()", self.sql)
        agents = KR_AGENTS.read_text(encoding="utf-8")
        self.assertIn("KNOWLEDGE_CORE_ID RAW(16)", agents)
        self.assertIn("DATA_MODEL_IDS_JSON JSON NOT NULL", agents)
        self.assertNotIn("ENABLED_CAPABILITIES_JSON", agents)

    def test_creates_x_search_and_media_runtime_tables(self) -> None:
        for table_name in (
            "KBOT_KR_RESEARCH_RUN",
            "KBOT_KR_RESEARCH_EVENT",
            "KBOT_KR_X_SOURCE",
            "KBOT_MEDIA_MODEL_BINDING",
            "KBOT_MEDIA_RUN",
            "KBOT_MEDIA_RUN_EVENT",
            "KBOT_MEDIA_PROMPT_REVISION",
            "KBOT_MEDIA_ASSET",
        ):
            self.assertIn(f"CREATE TABLE {table_name}", self.normalized)
        x_search = KR_X_SEARCH.read_text(encoding="utf-8").upper()
        media = MEDIA_RUNTIME.read_text(encoding="utf-8").upper()
        self.assertIn("CREATE TABLE KBOT_KR_RESEARCH_RUN", x_search)
        self.assertIn("CREATE TABLE KBOT_MEDIA_ASSET", media)

    def test_drops_retired_assistant_tables_without_history_migration(self) -> None:
        for table_name in (
            "KBOT_ASST_MEDIA_ASSET",
            "KBOT_ASST_PROMPT_REVISION",
            "KBOT_ASST_X_SOURCE",
            "KBOT_ASST_RUN_EVENT",
            "KBOT_ASST_RUN",
            "KBOT_ASST_MODEL_BINDING",
            "KBOT_ASST_AGENT_VERSION",
            "KBOT_ASST_AGENT",
        ):
            self.assertIn(table_name, self.normalized)
        self.assertIn("CASCADE CONSTRAINTS PURGE", self.normalized)
        self.assertNotIn("INSERT INTO KBOT_MEDIA_", self.normalized)
        self.assertNotIn("INSERT INTO KBOT_KR_RESEARCH_", self.normalized)
        self.assertIn("不迁移历史运行或图片元数据", self.sql)


    def test_alternative_quotes_do_not_contain_closers(self) -> None:
        self.assertNotIn("q'[", self.sql)
        self.assertNotIn("JSON('[]')", self.sql)
        opener = "q'{"
        closer = "}'"
        start = 0
        count = 0
        while True:
            begin = self.sql.find(opener, start)
            if begin < 0:
                break
            body_start = begin + len(opener)
            end = self.sql.find(closer, body_start)
            self.assertGreater(end, body_start, "q-quote 未闭合")
            body = self.sql[body_start:end]
            self.assertNotIn("}", body)
            start = end + len(closer)
            count += 1
        self.assertGreaterEqual(count, 10)

    def test_defers_permission_catalog_and_initial_admin_to_followup(self) -> None:
        self.assertNotIn("INSERT INTO KBOT_PERMISSION", self.normalized)
        self.assertNotIn("INSERT INTO KBOT_APP_ROLE", self.normalized)
        self.assertIn("--foundation-only", self.sql)
        self.assertIn("database/oracle/bootstrap/knowledge_retrieval/initial_admin.sql", self.sql)
        self.assertIn("database/oracle/bootstrap/media_studio/initial_admin.sql", self.sql)


if __name__ == "__main__":
    unittest.main()
