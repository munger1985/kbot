"""仓库运维脚本与测试工具的目录边界检查。"""

from pathlib import Path
import re
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
                "db/apply_notification_schema.py",
                "db/repair_model_serving_s4.py",
                "deployment/check_deployment.py",
                "deployment/init_local_env.py",
                "deployment/models/download_colqwen_model.py",
                "deployment/models/download_easyocr_model.py",
                "deployment/models/download_qwen_model.py",
                "release/verify_release.py",
                "security/generate_portal_api_key.py",
            },
            entries,
        )
        self.assertTrue(
            (SCRIPTS_ROOT / "deployment" / "install_workspace.sh").is_file()
        )
        self.assertTrue(
            (SCRIPTS_ROOT / "db" / "init_services.ini").is_file()
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
        self.assertIn('conda_env_exists "kbot4"', startup)
        self.assertNotIn('conda_env_exists "kbot3"', startup)

    def test_operational_python_scripts_do_not_modify_import_path(self):
        scripts = (
            "db/apply_oracle_schema.py",
            "deployment/check_deployment.py",
            "security/generate_portal_api_key.py",
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

    def test_aiops_knowledge_schema_alignment_has_no_data_migration(self):
        script = (
            SCRIPTS_ROOT / "db" / "align_ammolite_aiops_knowledge_schema.sql"
        ).read_text(encoding="utf-8")

        for required in (
            "CREATE TABLE KBOT_MANAGED_CREDENTIAL",
            "CREATE TABLE KBOT_KR_AGENT",
            "CREATE TABLE KBOT_OPS_AGENT",
            "CREATE TABLE KBOT_OPS_CONVERSATION",
            "CREATE TABLE KBOT_PLATFORM_USER",
            "EXECUTION_SPEC_JSON JSON",
            "SUBJECT_SELECTOR_JSON JSON",
            "PARSE_POLICY_JSON JSON",
            "ENABLE NOVALIDATE",
        ):
            self.assertIn(required, script)

        self.assertIsNone(
            re.search(
                r"^(?:INSERT|UPDATE|DELETE|MERGE|DROP TABLE|@@)",
                script,
                flags=re.MULTILINE,
            )
        )
        phase_three = script.index("PROMPT [3/4]")
        drop_index_declaration = script.index(
            "PROCEDURE drop_index_if_exists",
            phase_three,
        )
        drop_index_call = script.index(
            "drop_index_if_exists('UX_DQ_BINDING_ACTIVE')",
            phase_three,
        )
        self.assertLess(drop_index_declaration, drop_index_call)

    def test_default_app_roles_do_not_create_users_or_memberships(self):
        script = (
            SCRIPTS_ROOT / "db" / "seed_aiops_knowledge_roles_permissions.sql"
        ).read_text(encoding="utf-8")

        self.assertEqual(3, len(re.findall(r"^MERGE INTO ", script, re.MULTILINE)))
        self.assertIn("MERGE INTO KBOT_PERMISSION", script)
        self.assertIn("MERGE INTO KBOT_APP_ROLE", script)
        self.assertIn("MERGE INTO KBOT_APP_ROLE_PERMISSION", script)
        self.assertIn("knowledge_retrieval:operations_manage", script)
        self.assertIn("aiops:proposal:approve", script)
        self.assertNotRegex(
            script,
            r"(?:INSERT|MERGE)\s+INTO\s+KBOT_PLATFORM_USER",
        )
        self.assertNotRegex(
            script,
            r"(?:INSERT|MERGE)\s+INTO\s+KBOT_APP_MEMBER_ROLE",
        )


if __name__ == "__main__":
    unittest.main()
