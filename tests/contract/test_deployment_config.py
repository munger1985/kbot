"""生产部署配置与 Secret 预检测试。"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.deployment.check_deployment import check_deployment
from scripts.deployment.check_deployment import TOPOLOGY_PATH


class DeploymentConfigTest(unittest.TestCase):
    def test_deployment_summary_uses_canonical_topology(self):
        self.assertTrue(TOPOLOGY_PATH.is_file())
        self.assertNotIn(
            "14个进程",
            Path("scripts/deployment/check_deployment.py").read_text(
                encoding="utf-8"
            ),
        )

    def _write_production_config(self, root: Path) -> Path:
        path = root / "kbot.toml"
        path.write_text(
            "environment='production'\n"
            "data_dir='/var/lib/kbot'\n"
            "log_dir='/var/log/kbot'\n"
            "embedding_dimension=2048\n"
            "[database]\n"
            "host='db.internal'\n"
            "username='kbot'\n"
            "service_name='kbot4'\n"
            "[[portal_api_keys]]\n"
            "key_id='portal-prod'\n"
            "client_id='portal'\n"
            f"key_digest='{'1' * 64}'\n",
            encoding="utf-8",
        )
        return path

    def test_production_requires_database_password_and_master_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_production_config(Path(directory))
            with patch.dict(
                os.environ,
                {"ENV_FILE": str(Path(directory) / "missing.env")},
                clear=True,
            ):
                errors = check_deployment(path)

        self.assertIn("未设置KBOT_ORACLE_PASSWORD", errors)
        self.assertIn("KBOT_MASTER_KEY必须至少32字节", errors)

    def test_production_accepts_minimal_secret_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_production_config(Path(directory))
            with patch.dict(
                os.environ,
                {
                    "KBOT_ORACLE_PASSWORD": "secret",
                    "KBOT_MASTER_KEY": "m" * 32,
                    "ENV_FILE": str(Path(directory) / "missing.env"),
                },
                clear=True,
            ):
                errors = check_deployment(path)

        self.assertEqual([], errors)


if __name__ == "__main__":
    unittest.main()
