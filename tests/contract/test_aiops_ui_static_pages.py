"""AIOps 正式页面静态契约检查。"""

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
AIOPS_ROOT = ROOT / "ui" / "aiops"


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"])
        if tag == "link" and values.get("href"):
            self.assets.append(values["href"])


class AIOpsUiStaticPagesTest(unittest.TestCase):
    pages = {
        "dashboard", "chat", "situations", "runs", "run-detail",
        "reports", "report-detail", "inspections", "changes",
        "notifications", "targets", "target-detail",
        "diagnostic-sources", "diagnostic-source-detail", "knowledge-core",
        "agents",
        "policies", "inspection-plans", "inspection-plan-detail",
        "report-templates", "notification-subscriptions", "api-clients",
        "login",
    }

    def test_exact_page_inventory_and_assets(self):
        actual = {path.stem for path in AIOPS_ROOT.glob("*.html")}
        self.assertEqual(self.pages, actual)
        for page in AIOPS_ROOT.glob("*.html"):
            parser = _Parser()
            parser.feed(page.read_text(encoding="utf-8"))
            for reference in parser.assets:
                asset = reference.partition("?")[0]
                self.assertTrue(
                    (page.parent / asset).resolve().is_file(),
                    f"{page.name} 缺少资源 {reference}",
                )

    def test_javascript_syntax_and_public_boundary(self):
        scripts = list((AIOPS_ROOT / "js").glob("*.js"))
        self.assertEqual(6, len(scripts))
        source = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
        self.assertIn("/api/v1/apps/aiops", source)
        self.assertNotIn("/internal/v1", source)
        for script in scripts:
            result = subprocess.run(
                ["node", "--check", str(script)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_pages_do_not_embed_demo_records_or_api_keys(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in AIOPS_ROOT.rglob("*") if path.is_file()
        )
        self.assertNotIn("140.238.44.208", source)
        self.assertNotIn("kbot_ak_", source)
        self.assertNotIn("/metrics", source)
        self.assertNotIn("Idempotency-Key':crypto.randomUUID()", source)
        self.assertNotIn("client_file_id: crypto.randomUUID()", source)
        self.assertIn("KBotAIOpsAuth.uuid()", source)

    def test_knowledge_core_page_uses_fixed_public_bff(self):
        page = (AIOPS_ROOT / "knowledge-core.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-knowledge-core.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="model-form"', page)
        self.assertIn('id="manual-form"', page)
        self.assertIn("/knowledge-core/models", script)
        self.assertIn("/knowledge-core/manuals", script)
        self.assertIn("embedding_change_allowed", script)
        self.assertIn("visual_embedding_change_allowed", script)
        self.assertNotIn("/api/v1/knowledge", script)

    def test_login_uses_fixed_aiops_domain_contract(self):
        login = (AIOPS_ROOT / "login.html").read_text(encoding="utf-8")
        auth = (AIOPS_ROOT / "js" / "aiops-auth.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("aiops_portal", login)
        self.assertNotIn('name="domain_id"', login)
        self.assertIn("/api/v1/apps/aiops/auth/login", auth)

    def test_request_id_falls_back_without_crypto_random_uuid(self):
        script = """
const fs = require("node:fs");
const vm = require("node:vm");
const sandbox = {
  crypto: {},
  sessionStorage: {getItem: () => null, setItem: () => {}, removeItem: () => {}},
  location: {replace: () => {}},
  FormData: class {},
};
vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const value = sandbox.KBotAIOpsAuth.uuid();
if (!/^ui-[0-9]+-[0-9a-f]+$/.test(value)) process.exit(1);
"""
        result = subprocess.run(
            ["node", "-e", script, str(AIOPS_ROOT / "js" / "aiops-auth.js")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_target_create_form_uses_public_contract_fields(self):
        page = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-targets.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="target-dialog"', page)
        self.assertIn('id="target-form"', page)
        self.assertIn('id="test-target-connection"', page)
        self.assertIn("diagnostic_credential", script)
        self.assertIn("Idempotency-Key", script)
        self.assertIn("/targets/test-connection", script)
        self.assertIn('oracle ? "service" : "database"', script)
        self.assertIn('method: "PATCH"', script)
        self.assertIn("diagnostic-credential:rotate", script)
        self.assertIn("openEdit", script)
        self.assertIn("db_type", script)
        self.assertNotIn("engine_type", script)

    def test_configuration_pages_open_real_create_and_edit_dialogs(self):
        pages = {
            "diagnostic-sources.html": "diagnostic-source-dialog",
            "policies.html": "policy-dialog",
            "inspection-plans.html": "inspection-plan-dialog",
        }
        for filename, dialog_id in pages.items():
            source = (AIOPS_ROOT / filename).read_text(encoding="utf-8")
            self.assertIn(f'id="{dialog_id}"', source)
            self.assertIn("aiops-configurations.js", source)
        script = (AIOPS_ROOT / "js" / "aiops-configurations.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("/diagnostic-sources/test-connection", script)
        self.assertIn('method: editing ? "PATCH" : "POST"', script)
        self.assertIn("基于当前策略创建新版本", script)
        self.assertIn('"If-Match"', script)
        self.assertIn("openEdit", script)


if __name__ == "__main__":
    unittest.main()
