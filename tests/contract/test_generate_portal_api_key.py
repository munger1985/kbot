"""Portal API Key 生成脚本的配置写入测试。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "security" / "generate_portal_api_key.py"
SPEC = importlib.util.spec_from_file_location("generate_portal_api_key", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GeneratePortalApiKeyTest(unittest.TestCase):
    def test_replaces_existing_key_id_without_creating_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "kbot.toml"
            config_path.write_text(
                'environment = "production"\n\n'
                '[[portal_api_keys]]\n'
                'key_id = "portal-prod"\n'
                'client_id = "old-client"\n'
                'key_digest = "old-digest"\n',
                encoding="utf-8",
            )

            MODULE.upsert_portal_api_key(
                config_path=config_path,
                key_id="portal-prod",
                client_id="portal",
                key_digest="new-digest",
            )

            content = config_path.read_text(encoding="utf-8")
            self.assertEqual(1, content.count("[[portal_api_keys]]"))
            self.assertIn('client_id = "portal"', content)
            self.assertIn('key_digest = "new-digest"', content)

    def test_appends_new_key_id(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "kbot.toml"
            config_path.write_text('environment = "production"\n', encoding="utf-8")

            MODULE.upsert_portal_api_key(
                config_path=config_path,
                key_id="portal-prod",
                client_id="portal",
                key_digest="new-digest",
            )

            self.assertIn(
                '[[portal_api_keys]]\nkey_id = "portal-prod"',
                config_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
