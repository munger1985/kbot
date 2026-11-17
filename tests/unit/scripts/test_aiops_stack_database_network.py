from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
LOADER = importlib.machinery.SourceFileLoader("aiops_stack_database_network", str(ROOT / "scripts/aiops-stack"))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC and SPEC.loader
STACK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STACK
SPEC.loader.exec_module(STACK)


class DatabaseNetworkTest(unittest.TestCase):
    def settings(self, directory: str, version: str = "5.7", network: str = "test-databases"):
        path = Path(directory) / "aiops-stack.ini"
        path.write_text(
            "[deployment]\ndeployment_id = test-local\nrole = all-in-one\nlocal_access = true\n"
            "[metrics]\nenabled = true\n"
            "[mysql:mysql-test]\nenabled = true\n"
            f"version = {version}\ndocker_network = {network}\n"
            "host = mysql-test\nusername = monitor\npassword = test-password\n",
            encoding="utf-8",
        )
        path.chmod(0o600)
        return STACK._load_settings(path)

    def test_mysql_57_and_84_have_explicit_supported_versions(self):
        for version in ("5.7", "8.4"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                settings = self.settings(directory, version=version)
                self.assertEqual(settings.values.get("mysql:mysql-test", "version"), version)

    def test_unverified_mysql_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "version"):
                self.settings(directory, version="9.7")

    def test_rejects_invalid_network_names(self):
        for network in ("../network", "database network", "-invalid"):
            with self.subTest(network=network), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "docker_network"):
                    self.settings(directory, network=network)

    def test_only_database_collector_joins_external_network(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(directory)
            STACK._prepare_runtime(settings)
            generated = json.loads((settings.runtime_dir / "compose.generated.yaml").read_text())
            self.assertEqual(
                generated["networks"],
                {"database-mysql-test": {"external": True, "name": "test-databases"}},
            )
            exporter = generated["services"]["mysql-mysql-test-exporter"]
            self.assertEqual(exporter["networks"], ["observability", "outbound", "database-mysql-test"])
            self.assertNotIn("prometheus", generated["services"])
            self.assertNotIn("test-password", (settings.runtime_dir / "stack.env").read_text())
            self.assertEqual(len(exporter["secrets"]), 1)

    def test_unconfigured_network_keeps_original_networks(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(directory, network="")
            STACK._prepare_runtime(settings)
            generated = json.loads((settings.runtime_dir / "compose.generated.yaml").read_text())
            self.assertNotIn("networks", generated)
            exporter = generated["services"]["mysql-mysql-test-exporter"]
            self.assertEqual(exporter["networks"], ["observability", "outbound"])

    def test_preflight_checks_existing_network(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(directory)
            with patch.object(STACK.subprocess, "run") as run:
                STACK._preflight(settings)
            self.assertTrue(any(call.args[0][:3] == ["docker", "network", "inspect"] for call in run.call_args_list))

    def test_preflight_rejects_missing_external_network(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(directory)
            def run(command, **kwargs):
                if command[:3] == ["docker", "network", "inspect"]:
                    raise subprocess.CalledProcessError(1, command)
            with patch.object(STACK.subprocess, "run", side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):
                    STACK._preflight(settings)


if __name__ == "__main__":
    unittest.main()
