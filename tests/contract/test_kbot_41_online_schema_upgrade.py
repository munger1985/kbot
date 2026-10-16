"""KBot 4.1 线上 Schema 单文件追平脚本合同测试。"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "kbot_41"
    / "apply_kbot_41_from_aiops_schema_42.sql"
)
RESUME_OUTPUT_PATH = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "kbot_41"
    / "apply_kbot_41_from_aiops_schema_43.sql"
)
SOURCE_SCRIPTS = (
    ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_43.sql",
    ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_44.sql",
    ROOT / "database" / "oracle" / "operations" / "apply_aiops_schema_45.sql",
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "converge_kbot_41_shared_views.sql",
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "upgrade_kbot_41_retire_km_asset.sql",
)


class KBot41OnlineSchemaUpgradeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sql = OUTPUT_PATH.read_text(encoding="utf-8")
        self.upper = self.sql.upper()

    def test_generated_script_is_current_and_self_contained(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "tools/db/render_kbot_41_online_upgrade.py",
                "--check",
            ],
            cwd=ROOT,
            check=True,
        )
        self.assertNotRegex(self.sql, r"(?m)^\s*@@?")
        sections = re.findall(
            r"^-- ===== 开始升级阶段：(.+\.sql) =====$",
            self.sql,
            re.MULTILINE,
        )
        self.assertEqual(
            [item.relative_to(ROOT).as_posix() for item in SOURCE_SCRIPTS],
            sections,
        )

    def test_upgrade_covers_aiops_42_through_45(self) -> None:
        guards = (
            "L_SCHEMA_VERSION <> 42",
            "L_SCHEMA_VERSION <> 43",
            "L_SCHEMA_VERSION <> 44",
        )
        for guard in guards:
            self.assertIn(guard, self.upper)
        self.assertIn("45 AS SCHEMA_VERSION", self.upper)
        self.assertIn(
            "AIOPS-ORACLE-V45-SHARED-MONITOR-LOCATOR",
            self.upper,
        )
        self.assertLess(
            self.upper.index("43 AS SCHEMA_VERSION"),
            self.upper.index("44 AS SCHEMA_VERSION"),
        )
        self.assertLess(
            self.upper.index("44 AS SCHEMA_VERSION"),
            self.upper.index("45 AS SCHEMA_VERSION"),
        )

    def test_upgrade_includes_explicit_kbot_41_km_retirement(self) -> None:
        self.assertIn("KBot 4.1 KM Asset 退役预检", self.sql)
        self.assertIn("DROP TABLE ", self.upper)
        self.assertIn("CASCADE CONSTRAINTS PURGE", self.upper)
        self.assertIn("APP_ID = 'KM_ASSET'", self.upper)
        self.assertIn("DATA PUMP", self.upper)
        self.assertIn("继续运行 KM 4.0 的环境不得执行本脚本", self.sql)

    def test_schema_43_resume_script_skips_completed_schema_43_stage(self) -> None:
        sql = RESUME_OUTPUT_PATH.read_text(encoding="utf-8")
        sections = re.findall(
            r"^-- ===== 开始升级阶段：(.+\.sql) =====$",
            sql,
            re.MULTILINE,
        )
        self.assertEqual(
            [item.relative_to(ROOT).as_posix() for item in SOURCE_SCRIPTS[1:]],
            sections,
        )
        self.assertIn("唯一起点：AIOPS / 43 / aiops-oracle-v33", sql)
        self.assertNotIn("L_SCHEMA_VERSION <> 42", sql.upper())
        self.assertIn("L_SCHEMA_VERSION <> 43", sql.upper())
        self.assertIn("L_SCHEMA_VERSION <> 44", sql.upper())
        self.assertIn("KBOT_KM_AGENT_GRANT", sql.upper())
        self.assertIn("CREATE OR REPLACE VIEW KBOT_V_PLATFORM_DOMAIN", sql.upper())


if __name__ == "__main__":
    unittest.main()
