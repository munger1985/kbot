"""用隔离数据验证迁移谓词，不连接 Oracle 或真实环境。"""

import re
import sqlite3
import unittest
from pathlib import Path


SQL_FILE = (
    Path(__file__).resolve().parents[2] /
    "database/oracle/operations/upgrade_platform_user_management.sql"
)


class PlatformUserMigrationTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE KBOT_PLATFORM_USER (
                USER_ID TEXT, ACCOUNT_ORIGIN TEXT, OWNER_APP_ID TEXT,
                IS_PROTECTED TEXT, STATUS TEXT, UPDATED_AT TEXT
            );
            CREATE TABLE KBOT_PLATFORM_USER_ROLE (USER_ID TEXT);
            CREATE TABLE KBOT_APP_MEMBER (
                APP_ID TEXT, USER_ID TEXT, MEMBER_SOURCE TEXT, IS_INITIAL_ADMIN TEXT,
                STATUS TEXT, UPDATED_AT TEXT
            );
            INSERT INTO KBOT_PLATFORM_USER VALUES
                ('ordinary', 'APP', 'aiops', 'N', 'ACTIVE', NULL),
                ('paused', 'APP', 'aiops', 'N', 'DISABLED', NULL),
                ('initial', 'APP', 'aiops', 'Y', 'ACTIVE', NULL),
                ('platform', 'PLATFORM', NULL, 'N', 'ACTIVE', NULL);
            INSERT INTO KBOT_APP_MEMBER VALUES
                ('aiops', 'ordinary', 'APP_CREATED', 'N', 'ACTIVE', NULL),
                ('aiops', 'paused', 'APP_CREATED', 'N', 'DISABLED', NULL),
                ('aiops', 'initial', 'APP_INITIAL_ADMIN', 'Y', 'ACTIVE', NULL),
                ('aiops', 'platform', 'PLATFORM_GRANT', 'N', 'ACTIVE', NULL);
        """)
        self.sql = SQL_FILE.read_text(encoding="utf-8")
        preflight = re.search(r"SELECT COUNT\(\*\) INTO unsafe_count.*?;", self.sql, re.S)
        self.preflight = preflight.group().replace(" INTO unsafe_count", "")

    def test_existing_memberships_keep_effective_access_and_protected_account_is_unchanged(self):
        self.assertEqual(0, self.db.execute(self.preflight).fetchone()[0])
        dml = self.sql[self.sql.index("UPDATE KBOT_APP_MEMBER"):].removesuffix("COMMIT;\n")
        dml = dml.replace("KBOT_APP_MEMBER m", "KBOT_APP_MEMBER AS m", 1).replace("SYSTIMESTAMP", "CURRENT_TIMESTAMP")
        for _ in range(2):
            self.db.executescript(dml)
            self.assertEqual([
                ("initial", "APP", "aiops", "ACTIVE", "APP_INITIAL_ADMIN", "ACTIVE"),
                ("ordinary", "PLATFORM", None, "ACTIVE", "PLATFORM_GRANT", "ACTIVE"),
                ("paused", "PLATFORM", None, "DISABLED", "PLATFORM_GRANT", "ACTIVE"),
                ("platform", "PLATFORM", None, "ACTIVE", "PLATFORM_GRANT", "ACTIVE"),
            ], self.db.execute("""
                SELECT u.USER_ID, u.ACCOUNT_ORIGIN, u.OWNER_APP_ID, u.STATUS, m.MEMBER_SOURCE, m.STATUS
                FROM KBOT_PLATFORM_USER u JOIN KBOT_APP_MEMBER m ON m.USER_ID = u.USER_ID
                ORDER BY u.USER_ID
            """).fetchall())

    def test_preflight_flags_latent_platform_or_cross_app_authorizations(self):
        self.db.execute("INSERT INTO KBOT_PLATFORM_USER_ROLE VALUES ('ordinary')")
        self.db.execute("INSERT INTO KBOT_APP_MEMBER VALUES ('other_app', 'paused', 'PLATFORM_GRANT', 'N', 'ACTIVE', NULL)")
        self.assertEqual(2, self.db.execute(self.preflight).fetchone()[0])
        self.assertIn("WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK", self.sql)
        self.assertIn("RAISE_APPLICATION_ERROR(-20091", self.sql)
