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
                "db/apply_notification_schema.py",
                "db/apply_composition_schema.py",
                "db/repair_agent_data_query_mode.py",
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


if __name__ == "__main__":
    unittest.main()
