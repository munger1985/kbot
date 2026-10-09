"""KBot 容器交付工具的离线契约测试。"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "kbot_release", ROOT / "installation" / "release.py"
)
assert SPEC and SPEC.loader
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseToolingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog, self.topology = release.validate_catalog()

    def test_catalog_covers_every_topology_process(self) -> None:
        image_ids = {item["id"] for item in self.catalog["images"]}
        self.assertEqual(9, len(image_ids))
        self.assertEqual(25, len(self.topology["processes"]))
        self.assertTrue(
            all(
                process["service_config"] in image_ids
                for process in self.topology["processes"]
            )
        )

    def test_cpu_torch_is_limited_to_model_and_knowledge_images(self) -> None:
        cpu_images = {
            item["id"]
            for item in self.catalog["images"]
            if item.get("cpu_torch")
        }
        self.assertEqual({"model_serving", "knowledge_core"}, cpu_images)

    def test_runtime_config_uses_compose_dns_for_every_endpoint(self) -> None:
        values = self._values(Path("/tmp/oracle"), Path("/tmp/master"))
        rendered = release.render_runtime_toml(self.topology, values)
        processes = {
            process["process_key"]: process for process in self.topology["processes"]
        }
        for endpoint, process_key in self.topology["endpoints"].items():
            process = processes[process_key]
            expected = (
                f'{endpoint} = "http://{release.compose_name(process_key)}:'
                f'{process["port"]}"'
            )
            self.assertIn(expected, rendered)
        endpoints_section = rendered.split("[endpoints]", maxsplit=1)[1]
        self.assertNotIn("127.0.0.1", endpoints_section)

    def test_only_ui_publishes_a_host_port(self) -> None:
        values = self._values(Path("/tmp/oracle"), Path("/tmp/master"))
        rendered = release.render_compose(
            self.catalog, self.topology, values
        )
        self.assertEqual(1, rendered.count("    ports:\n"))
        self.assertIn('      - "8080:8080"', rendered)
        self.assertNotIn("network_mode: host", rendered)

    def test_every_process_has_one_compose_service(self) -> None:
        values = self._values(Path("/tmp/oracle"), Path("/tmp/master"))
        rendered = release.render_compose(
            self.catalog, self.topology, values
        )
        for process in self.topology["processes"]:
            self.assertEqual(
                1,
                rendered.count(
                    f"  {release.compose_name(process['process_key'])}:\n"
                ),
            )

    def test_app_selection_limits_processes_and_schema(self) -> None:
        values = self._values(Path("/tmp/oracle"), Path("/tmp/master"))
        values["enabled_apps"] = ["aiops"]
        rendered = release.render_compose(
            self.catalog, self.topology, values
        )
        self.assertIn("  aiops-api:\n", rendered)
        self.assertIn("  knowledge-core-api:\n", rendered)
        self.assertNotIn("  media-studio-app-api:\n", rendered)
        self.assertNotIn("  km-asset-app-api:\n", rendered)
        self.assertNotIn("  knowledge-retrieval-app-api:\n", rendered)
        schema = release.render_schema_services(self.catalog, values)
        self.assertIn("aiops_agent = true", schema)
        self.assertIn("media_studio_app = false", schema)
        self.assertIn("km_asset_app = false", schema)

    def test_rendered_secrets_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            source.write_text("secret-value\n", encoding="utf-8")
            release.private_copy(source, target)
            self.assertEqual("secret-value\n", target.read_text(encoding="utf-8"))
            self.assertEqual(
                stat.S_IRUSR | stat.S_IWUSR,
                stat.S_IMODE(target.stat().st_mode),
            )

    def test_deployment_form_must_be_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "deployment.ini"
            config.write_text("[deployment]\n", encoding="utf-8")
            config.chmod(0o644)
            with self.assertRaisesRegex(release.ReleaseError, "0600"):
                release.deployment_values(config, self.catalog)

    def test_render_deployment_writes_private_secret_copies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "deployment.ini"
            secret_dir = root / "input-secrets"
            secret_dir.mkdir()
            (secret_dir / "oracle").write_text("oracle-secret\n", encoding="utf-8")
            (secret_dir / "master").write_text("m" * 40 + "\n", encoding="utf-8")
            config.write_text(
                self._deployment_ini(root), encoding="utf-8"
            )
            config.chmod(0o600)
            generated = root / "generated"
            with patch.object(release, "deployment_directory", return_value=generated):
                release.render_deployment(config)
            for name in ("oracle_password", "master_key"):
                path = generated / "secrets" / name
                self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))
            env_text = (generated / ".env").read_text(encoding="utf-8")
            self.assertIn(f"KBOT_UID={os.getuid()}", env_text)

    def test_observability_is_rendered_from_the_same_form(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret_dir = root / "input-secrets"
            secret_dir.mkdir()
            (secret_dir / "oracle").write_text("oracle-secret\n", encoding="utf-8")
            (secret_dir / "master").write_text("m" * 40 + "\n", encoding="utf-8")
            config = root / "deployment.ini"
            config.write_text(
                self._deployment_ini(root)
                + """
[observability]
enabled = true
deployment_id = test-observability
role = all-in-one
local_access = false
minimum_free_gb = 1

[observability.metrics]
enabled = true
prometheus_retention = 7d
alertmanager_retention = 24h
""",
                encoding="utf-8",
            )
            config.chmod(0o600)
            generated = root / "generated"
            with patch.object(release, "deployment_directory", return_value=generated):
                release.render_deployment(config)
            observability = generated / "observability"
            self.assertEqual(
                0o600,
                stat.S_IMODE((observability / "aiops-stack.ini").stat().st_mode),
            )
            self.assertTrue(
                (observability / "generated/prometheus/prometheus.yml").is_file()
            )
            manifest = json.loads(
                (generated / "installation-input.json").read_text()
            )
            self.assertEqual(
                ["alertmanager", "prometheus"],
                manifest["observability"]["services"],
            )

    def test_aiops_stack_accepts_generated_config_via_environment(self) -> None:
        expected = "/tmp/kbot-generated-aiops-stack.ini"
        with patch.dict(
            os.environ, {"KBOT_AIOPS_STACK_CONFIG_FILE": expected}
        ):
            module = release.load_aiops_stack_module()
        self.assertEqual(Path(expected), module.CONFIG_FILE)

    def test_deploy_validates_compose_before_starting_services(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
            with (
                patch.object(release, "render_deployment", return_value=output),
                patch.object(release.subprocess, "run") as runner,
                patch("builtins.print"),
            ):
                release.deploy(Path("deployment.ini"))
            commands = [call.args[0] for call in runner.call_args_list]
            self.assertEqual(["config", "--quiet"], commands[0][-2:])
            self.assertEqual(
                ["up", "--detach", "--remove-orphans"], commands[1][-3:]
            )

    @staticmethod
    def _values(oracle_secret: Path, master_secret: Path) -> dict[str, object]:
        return {
            "project_name": "kbot4-test",
            "version": "4.0.0-test",
            "registry": "registry.test/kbot",
            "ui_port": 8080,
            "public_base_url": "http://127.0.0.1:8080",
            "database_host": "oracle.test",
            "database_port": 1521,
            "database_service_name": "kbot4",
            "database_username": "kbot",
            "embedding_dimension": 2048,
            "enabled_apps": [
                "knowledge_retrieval",
                "media_studio",
                "km_asset",
                "aiops",
            ],
            "aiops_agent_execution_enabled": False,
            "aiops_mutation_enabled": False,
            "observability_enabled": False,
            "observability_sections": {},
            "oracle_password_file": oracle_secret,
            "master_key_file": master_secret,
            "data_dir": Path("/tmp/kbot-data"),
            "log_dir": Path("/tmp/kbot-log"),
        }

    @staticmethod
    def _deployment_ini(root: Path) -> str:
        return f"""[deployment]
project_name = kbot4-test
registry = registry.test/kbot
image_version = 4.0.0-test
ui_port = 8080
public_base_url = http://127.0.0.1:8080
embedding_dimension = 2048

[apps]
knowledge_retrieval = true
media_studio = true
km_asset = true
aiops = true

[database]
host = oracle.test
port = 1521
service_name = kbot4
username = kbot

[paths]
data_dir = {root / 'data'}
log_dir = {root / 'log'}

[secrets]
oracle_password_file = input-secrets/oracle
master_key_file = input-secrets/master

[aiops]
agent_execution_enabled = false
mutation_enabled = false
"""


if __name__ == "__main__":
    unittest.main()
