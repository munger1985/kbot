"""安装器数据库编排的隔离测试；所有数据库与业务初始化依赖均替换为桩。"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("installation_bootstrap", ROOT / "scripts/db/initialize_installation.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class InstallationBootstrapTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.accounts_path = Path(temporary.name) / ".kbot/initial-accounts.json"
        self.catalog = bootstrap.tomllib.loads((ROOT / "installation/catalog.toml").read_text())

    def dependencies(self, apps):
        expected = set(self.catalog["base_pack"]["service_configs"]) | {"platform_core"}
        for app in apps:
            expected.update(self.catalog["app_packs"][app]["service_configs"])
        self.schema = SimpleNamespace(
            load_service_selection=Mock(return_value=SimpleNamespace(ordered=tuple(sorted(expected)))),
            load_schema_statements=Mock(return_value=[SimpleNamespace(label="CREATE TABLE KBOT_SAMPLE"), SimpleNamespace(label="CREATE OR REPLACE VIEW KBOT_SAMPLE_VIEW")]),
            apply_schema=AsyncMock(), maintain_platform_foundation=AsyncMock(), _validate_schema=AsyncMock(),
        )
        self.initialize_app = AsyncMock()
        self.initialize_aiops = AsyncMock()
        self.seed = AsyncMock()
        users = ["ADMIN"] + [{"knowledge_retrieval": "knowledgeadmin", "media_studio": "mediaadmin", "aiops": "aiopsadmin"}[app] for app in apps]
        self.credentials = [SimpleNamespace(user_id=user, password_hash="old", must_change_password="N") for user in users]
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.execute = AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self.credentials)))
        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=connection)
        self.runtime = SimpleNamespace(
            session_factory=Mock(return_value=session),
            engine=SimpleNamespace(connect=Mock(return_value=connection)), close=AsyncMock(),
        )
        statement = MagicMock()
        statement.where.return_value = statement
        statement.with_for_update.return_value = statement
        modules = {
            "bcrypt": SimpleNamespace(hashpw=Mock(return_value=b"random-hash"), gensalt=Mock(return_value=b"salt")),
            "sqlalchemy": SimpleNamespace(select=Mock(return_value=statement)),
            "main_api.entities.access_control": SimpleNamespace(PlatformUserCredentialEntity=SimpleNamespace(user_id=MagicMock())),
            "platform_core.config": SimpleNamespace(get_settings=Mock(return_value=object())),
            "platform_core.database.oracle": SimpleNamespace(create_database_runtime=Mock(return_value=self.runtime)),
            "scripts.db": SimpleNamespace(apply_oracle_schema=self.schema),
            "scripts.db.initialize_aiops": SimpleNamespace(initialize_aiops=self.initialize_aiops),
            "scripts.db.initialize_product_app": SimpleNamespace(
                APP_BOOTSTRAPS={
                    "knowledge_retrieval": SimpleNamespace(admin_user="knowledgeadmin"),
                    "media_studio": SimpleNamespace(admin_user="mediaadmin"),
                }, initialize_app=self.initialize_app,
            ),
            "scripts.db.seed_aiops_inspection_templates": SimpleNamespace(seed_default_inspection_templates=self.seed),
        }
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", modules).start()
        patch("builtins.print").start()

    def test_initialize_uses_canonical_schema_and_only_selected_apps(self):
        self.dependencies(["knowledge_retrieval"])
        asyncio.run(bootstrap.initialize(["knowledge_retrieval"], check_only=False, accounts_path=self.accounts_path))
        self.schema.apply_schema.assert_awaited_once()
        self.assertFalse(self.schema.apply_schema.call_args.kwargs["dry_run"])
        self.initialize_app.assert_awaited_once()
        self.assertFalse(self.initialize_app.call_args.kwargs["check_only"])
        self.initialize_aiops.assert_not_awaited()
        self.seed.assert_not_awaited()
        accounts = json.loads(self.accounts_path.read_text())["accounts"]
        self.assertEqual({"ADMIN", "knowledgeadmin"}, set(accounts))
        self.assertNotEqual(accounts["ADMIN"], accounts["knowledgeadmin"])
        self.assertTrue(all(len(password) >= 24 for password in accounts.values()))
        self.assertTrue(all(item.password_hash == "random-hash" and item.must_change_password == "Y" for item in self.credentials))

    def test_aiops_initialization_seeds_templates_and_randomizes_account(self):
        self.dependencies(["aiops"])
        asyncio.run(bootstrap.initialize(["aiops"], check_only=False, accounts_path=self.accounts_path))
        self.initialize_aiops.assert_awaited_once_with(check_only=False)
        self.seed.assert_awaited_once()
        self.initialize_app.assert_not_awaited()
        self.assertIn("aiopsadmin", json.loads(self.accounts_path.read_text())["accounts"])

    def test_adoption_is_read_only_and_passes_aiops_manifest(self):
        self.dependencies(["aiops"])
        asyncio.run(bootstrap.initialize(["aiops"], check_only=True, accounts_path=self.accounts_path))
        self.schema.apply_schema.assert_not_awaited()
        self.schema.maintain_platform_foundation.assert_awaited_once_with(repair=False)
        self.initialize_aiops.assert_awaited_once_with(check_only=True)
        self.seed.assert_not_awaited()
        self.runtime.session_factory.assert_not_called()
        self.assertFalse(self.accounts_path.exists())
        validation = self.schema._validate_schema.call_args.kwargs
        self.assertEqual({"KBOT_SAMPLE"}, validation["expected_tables"])
        self.assertEqual({"KBOT_SAMPLE_VIEW"}, validation["expected_views"])
        self.assertIsInstance(validation["aiops_manifest"], dict)

    def test_adoption_refuses_schema_mismatch_without_repairing(self):
        self.dependencies(["media_studio"])
        self.schema._validate_schema.side_effect = RuntimeError("对象范围不一致")
        with self.assertRaisesRegex(RuntimeError, "不一致"):
            asyncio.run(bootstrap.initialize(["media_studio"], check_only=True, accounts_path=self.accounts_path))
        self.schema.apply_schema.assert_not_awaited()
        self.schema.maintain_platform_foundation.assert_not_awaited()
        self.initialize_app.assert_not_awaited()
        self.assertFalse(self.accounts_path.exists())

    def test_existing_account_handoff_blocks_reinitialization(self):
        self.dependencies(["media_studio"])
        bootstrap.write_accounts(self.accounts_path, {"ADMIN": "original"})
        with self.assertRaisesRegex(RuntimeError, "重复初始化"):
            asyncio.run(bootstrap.initialize(["media_studio"], check_only=False, accounts_path=self.accounts_path))
        self.schema.apply_schema.assert_not_awaited()
        self.initialize_app.assert_not_awaited()
        self.assertEqual("original", json.loads(self.accounts_path.read_text())["accounts"]["ADMIN"])

    def test_app_schema_selection_mismatch_is_rejected_before_writes(self):
        self.dependencies(["knowledge_retrieval"])
        self.schema.load_service_selection.return_value = SimpleNamespace(ordered=("platform_core",))
        with self.assertRaisesRegex(RuntimeError, "范围不一致"):
            asyncio.run(bootstrap.initialize(["knowledge_retrieval"], check_only=False, accounts_path=self.accounts_path))
        self.schema.apply_schema.assert_not_awaited()
        self.initialize_app.assert_not_awaited()
        self.assertFalse(self.accounts_path.exists())


if __name__ == "__main__":
    unittest.main()
