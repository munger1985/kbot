"""仓库运维脚本与测试工具的目录边界检查。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"


class RepositoryScriptLayoutTest(unittest.TestCase):
    def test_scripts_contains_only_operational_entry_points(self):
        entries = {
            str(path.relative_to(SCRIPTS_ROOT))
            for path in SCRIPTS_ROOT.rglob("*.py")
            if path.name != "__init__.py"
        }
        self.assertEqual(
            {
                "db/apply_oracle_schema.py",
                "db/sync_prompt_catalog.py",
                "deployment/check_deployment.py",
                "deployment/ensure_workspace_packages.py",
                "deployment/init_local_env.py",
                "deployment/models/download_colqwen_model.py",
                "deployment/models/download_easyocr_model.py",
                "deployment/models/download_qwen_model.py",
                "deployment/workspace_fingerprint.py",
                "release/verify_release.py",
            },
            entries,
        )
        self.assertTrue(
            (SCRIPTS_ROOT / "deployment" / "install_workspace.sh").is_file()
        )
        self.assertTrue(
            (SCRIPTS_ROOT / "deployment" / "bootstrap_kbot.sh").is_file()
        )
        self.assertTrue(
            (SCRIPTS_ROOT / "db" / "init_services.ini").is_file()
        )
        self.assertTrue(
            (
                SCRIPTS_ROOT
                / "db"
                / "bootstrap_platform_foundation.sql"
            ).is_file()
        )
        self.assertFalse(
            (ROOT / "database" / "oracle" / "init_services.ini").exists()
        )

    def test_workspace_installer_uses_package_boundaries(self):
        installer = (
            SCRIPTS_ROOT / "deployment" / "install_workspace.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('-m pip install -r requirements.txt', installer)
        self.assertIn('-m pip install --no-deps -e "$member"', installer)
        self.assertIn('-m pip wheel --no-deps', installer)
        self.assertIn('check_workspace_packages.py', installer)
        self.assertIn('KBOT_CONDA_ENV', installer)
        self.assertIn('"$conda_bin" run -n "$selected_conda_env"', installer)
        self.assertIn('conda_env_exists "$conda_bin" "kbot4"', installer)
        self.assertNotIn('conda_env_exists "$conda_bin" "kbot3"', installer)

        startup = (ROOT / "start_kbot.sh").read_text(encoding="utf-8")
        self.assertIn("当前 Conda 环境缺少 KBot 内部包", startup)
        self.assertIn("scripts/deployment/install_workspace.sh", startup)
        self.assertIn(
            "scripts/deployment/ensure_workspace_packages.py", startup
        )
        self.assertIn("KBOT_AUTO_INSTALL_PACKAGES", (
            SCRIPTS_ROOT
            / "deployment"
            / "ensure_workspace_packages.py"
        ).read_text(encoding="utf-8"))
        self.assertTrue(
            (SCRIPTS_ROOT / "deployment" / "run_service.sh").is_file()
        )
        self.assertIn("--check-foundation", startup)
        self.assertIn("--foundation-only", startup)
        self.assertIn('dev:3|development:3|debug:3)', startup)
        self.assertIn('FOUNDATION_MODE="--foundation-only"', startup)
        self.assertIn('FOUNDATION_MODE="--check-foundation"', startup)
        self.assertIn('FOUNDATION_CHECK_STATUS" -eq 0', startup)
        self.assertIn("这不代表系统未初始化", startup)
        self.assertIn("main_api/runtime.log", startup)
        self.assertIn('conda_env_exists "kbot4"', startup)
        self.assertNotIn('conda_env_exists "kbot3"', startup)

        bootstrap = (
            SCRIPTS_ROOT / "deployment" / "bootstrap_kbot.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("scripts/deployment/install_workspace.sh", bootstrap)
        self.assertIn("tests/acceptance/check_process_topology.py", bootstrap)
        self.assertIn("tests/acceptance/check_oracle_schema.py", bootstrap)
        self.assertIn("scripts/deployment/check_deployment.py", bootstrap)
        self.assertIn("scripts/db/apply_oracle_schema.py", bootstrap)
        self.assertIn("--check-foundation", bootstrap)
        self.assertIn("--schema-dry-run", bootstrap)
        self.assertNotIn("reset_kbot_schema.sql", bootstrap)

        foundation = (
            SCRIPTS_ROOT / "db" / "bootstrap_platform_foundation.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("MERGE INTO KBOT_PLATFORM_DOMAIN", foundation)
        self.assertIn("MERGE INTO KBOT_PLATFORM_USER", foundation)
        self.assertIn("'ADMIN' USER_ID", foundation)
        self.assertIn("'platform_admin' ROLE_CODE", foundation)
        self.assertIn("KBOT_PLATFORM_USER_ROLE", foundation)
        self.assertNotIn("KBOT_APP_MEMBER_ROLE target", foundation)
        self.assertIn("target.MAX_SECURITY_LEVEL = 3", foundation)
        self.assertIn("MEMBER_SOURCE <> 'PLATFORM_GRANT'", foundation)
        self.assertIn("IS_INITIAL_ADMIN <> 'N'", foundation)
        credential_merge = foundation.split(
            "MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL", 1
        )[1].split("MERGE INTO KBOT_PLATFORM_USER_ROLE", 1)[0]
        self.assertNotIn("WHEN MATCHED", credential_merge)
        self.assertIn("WHEN NOT MATCHED", credential_merge)

        schema_runner = (
            SCRIPTS_ROOT / "db" / "apply_oracle_schema.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CK_PLATFORM_USER_SECURITY", schema_runner)
        self.assertIn("MAX_SECURITY_LEVEL BETWEEN 0 AND 3", schema_runner)
        self.assertIn("FOUNDATION_VALIDATION_EXIT_CODE = 3", schema_runner)
        self.assertIn("PDB={pdb_name}，Schema={schema_name}", schema_runner)
        self.assertIn("MEMBER_SOURCE <> 'PLATFORM_GRANT'", schema_runner)

        access_schema = (
            ROOT
            / "database"
            / "oracle"
            / "main_api"
            / "001_access_control.sql"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "MAX_SECURITY_LEVEL NUMBER(3) DEFAULT 1 NOT NULL",
            access_schema,
        )
        self.assertIn(
            "MAX_SECURITY_LEVEL BETWEEN 0 AND 3", access_schema
        )

    def test_operational_python_scripts_do_not_modify_import_path(self):
        scripts = (
            "db/apply_oracle_schema.py",
            "db/sync_prompt_catalog.py",
            "deployment/check_deployment.py",
        )

        for relative_path in scripts:
            content = (SCRIPTS_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("sys.path", content)

    def test_explicit_test_tools_live_under_tests(self):
        expected = (
            ROOT / "tests" / "acceptance" / "check_4_0_boundaries.py",
            ROOT / "tests" / "acceptance" / "check_oracle_schema.py",
            ROOT / "tests" / "evaluation" / "evaluate_kc_parser.py",
            ROOT / "tests" / "smoke" / "smoke_oracle_service_uow.py",
            ROOT / "tests" / "support" / "oracle_preflight.py",
        )
        self.assertTrue(all(path.is_file() for path in expected))

if __name__ == "__main__":
    unittest.main()
