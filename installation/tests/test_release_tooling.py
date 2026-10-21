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
        self.assertEqual(8, len(image_ids))
        self.assertEqual(23, len(self.topology["processes"]))
        self.assertTrue(
            all(
                process["service_config"] in image_ids
                for process in self.topology["processes"]
            )
        )

    def test_formal_image_lock_covers_complete_catalog(self) -> None:
        lock = release.load_image_lock("1.0", self.catalog)
        self.assertEqual(
            release.catalog_image_names(self.catalog),
            [item["name"] for item in lock["images"]],
        )
        self.assertEqual("linux/amd64", lock["platform"])
        self.assertEqual(40, len(lock["source_commit"]))

    def test_cpu_torch_is_limited_to_model_and_knowledge_images(self) -> None:
        cpu_images = {
            item["id"]
            for item in self.catalog["images"]
            if item.get("cpu_torch")
        }
        self.assertEqual({"model_serving", "knowledge_core"}, cpu_images)

    def test_main_api_image_declares_direct_runtime_dependencies(self) -> None:
        pyproject = release.tomllib.loads(
            (ROOT / "services/main_api/pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        dependencies = set(pyproject["project"]["dependencies"])
        self.assertTrue(
            {
                "bcrypt==5.0.0",
                "loguru==0.7.3",
                "python-jose==3.5.0",
                "python-multipart==0.0.32",
            }.issubset(dependencies)
        )

    def test_aiops_image_packages_inspection_catalog(self) -> None:
        pyproject = release.tomllib.loads(
            (ROOT / "services/aiops_agent/pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        package_data = set(
            pyproject["tool"]["setuptools"]["package-data"]["aiops_agent"]
        )
        self.assertIn("actions/catalog/**/*.txt", package_data)
        self.assertIn("application/inspections/*.json", package_data)

    def test_model_serving_declares_platform_clients_dependency(self) -> None:
        pyproject = release.tomllib.loads(
            (ROOT / "services/model_serving/pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        dependencies = set(pyproject["project"]["dependencies"])
        self.assertIn("kbot-platform-clients==4.0.0", dependencies)

    def test_knowledge_core_declares_tokenizer_dependency(self) -> None:
        pyproject = release.tomllib.loads(
            (ROOT / "services/knowledge_core/pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        dependencies = set(pyproject["project"]["dependencies"])
        self.assertIn("jieba==0.42.1", dependencies)

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

    def test_ui_runtime_config_uses_public_base_url(self) -> None:
        values = self._values(Path("/tmp/oracle"), Path("/tmp/master"))
        rendered = release.render_ui_runtime_config(values)
        self.assertIn(
            '"mainApiBaseUrl":"http://127.0.0.1:8080"',
            rendered,
        )
        self.assertIn(
            '"enabledApps":["knowledge_retrieval","media_studio","aiops"]',
            rendered,
        )

    def test_only_ui_publishes_a_host_port(self) -> None:
        values = self._values(Path("/tmp/oracle"), Path("/tmp/master"))
        rendered = release.render_compose(
            self.catalog, self.topology, values
        )
        self.assertEqual(1, rendered.count("    ports:\n"))
        self.assertIn('      - "8080:8080"', rendered)
        self.assertNotIn("network_mode: host", rendered)

    def test_dynamic_uid_services_publish_runtime_identity(self) -> None:
        values = self._values(Path("/tmp/oracle"), Path("/tmp/master"))
        rendered = release.render_compose(
            self.catalog, self.topology, values
        )
        dynamic_user_count = rendered.count(
            '    user: "${KBOT_UID}:${KBOT_GID}"'
        )
        self.assertGreater(dynamic_user_count, 1)
        self.assertEqual(
            dynamic_user_count,
            rendered.count('      HOME: "/var/lib/kbot"'),
        )
        self.assertEqual(
            dynamic_user_count,
            rendered.count('      LOGNAME: "kbot"'),
        )
        self.assertEqual(
            dynamic_user_count,
            rendered.count('      USER: "kbot"'),
        )

    def test_ui_root_redirect_preserves_external_port_mapping(self) -> None:
        template = (
            ROOT / "installation/common/docker/nginx.conf.template"
        ).read_text(encoding="utf-8")
        self.assertIn("absolute_redirect off;", template)

    def test_version_labels_follow_expensive_image_layers(self) -> None:
        python_dockerfile = (
            ROOT / "installation/common/docker/Dockerfile.python"
        ).read_text(encoding="utf-8")
        python_label = python_dockerfile.index(
            'LABEL org.opencontainers.image.title="KBot 4.0"'
        )
        self.assertGreater(
            python_label,
            python_dockerfile.index("python -m pip install --find-links"),
        )
        self.assertGreater(
            python_label,
            python_dockerfile.index(
                "COPY installation/common/docker/python-entrypoint.sh"
            ),
        )

        ui_dockerfile = (
            ROOT / "installation/common/docker/Dockerfile.ui"
        ).read_text(encoding="utf-8")
        self.assertGreater(
            ui_dockerfile.index(
                'LABEL org.opencontainers.image.title="KBot UI"'
            ),
            ui_dockerfile.index(
                "COPY installation/common/docker/nginx.conf.template"
            ),
        )

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
        self.assertNotIn("  knowledge-retrieval-app-api:\n", rendered)
        schema = release.render_schema_services(self.catalog, values)
        self.assertIn("aiops_agent = true", schema)
        self.assertIn("media_studio_app = false", schema)

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

    def test_prepare_secrets_prompts_for_password_and_generates_master_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "deployment.ini"
            config.write_text(
                "[secrets]\n"
                "oracle_password_file = secrets/oracle_password\n"
                "master_key_file = secrets/master_key\n",
                encoding="utf-8",
            )
            config.chmod(0o600)
            with (
                patch.object(
                    release.getpass,
                    "getpass",
                    side_effect=["oracle-secret", "oracle-secret"],
                ),
                patch("builtins.print"),
            ):
                oracle_secret, master_secret = release.prepare_secrets(config)
            self.assertEqual("oracle-secret\n", oracle_secret.read_text())
            self.assertGreaterEqual(
                len(master_secret.read_text().strip().encode("utf-8")), 32
            )
            self.assertEqual(0o600, stat.S_IMODE(oracle_secret.stat().st_mode))
            self.assertEqual(0o600, stat.S_IMODE(master_secret.stat().st_mode))

    def test_prepare_secrets_preserves_existing_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret_dir = root / "secrets"
            secret_dir.mkdir()
            oracle_secret = secret_dir / "oracle_password"
            master_secret = secret_dir / "master_key"
            oracle_secret.write_text("existing-oracle\n", encoding="utf-8")
            master_secret.write_text("m" * 40 + "\n", encoding="utf-8")
            config = root / "deployment.ini"
            config.write_text(
                "[secrets]\n"
                "oracle_password_file = secrets/oracle_password\n"
                "master_key_file = secrets/master_key\n",
                encoding="utf-8",
            )
            config.chmod(0o600)
            with (
                patch.object(release.getpass, "getpass") as password_reader,
                patch("builtins.print"),
            ):
                release.prepare_secrets(config)
            password_reader.assert_not_called()
            self.assertEqual("existing-oracle\n", oracle_secret.read_text())
            self.assertEqual("m" * 40 + "\n", master_secret.read_text())

    def test_prepare_secrets_does_not_write_mismatched_password(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "deployment.ini"
            config.write_text(
                "[secrets]\n"
                "oracle_password_file = secrets/oracle_password\n"
                "master_key_file = secrets/master_key\n",
                encoding="utf-8",
            )
            config.chmod(0o600)
            with patch.object(
                release.getpass,
                "getpass",
                side_effect=["first", "second"],
            ):
                with self.assertRaisesRegex(release.ReleaseError, "不一致"):
                    release.prepare_secrets(config)
            self.assertFalse((root / "secrets/oracle_password").exists())
            self.assertFalse((root / "secrets/master_key").exists())

    def test_prepare_secrets_replaces_only_oracle_password(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret_dir = root / "secrets"
            secret_dir.mkdir()
            oracle_secret = secret_dir / "oracle_password"
            master_secret = secret_dir / "master_key"
            oracle_secret.write_text("old-oracle\n", encoding="utf-8")
            master_secret.write_text("m" * 40 + "\n", encoding="utf-8")
            config = root / "deployment.ini"
            config.write_text(
                "[secrets]\n"
                "oracle_password_file = secrets/oracle_password\n"
                "master_key_file = secrets/master_key\n",
                encoding="utf-8",
            )
            config.chmod(0o600)
            with (
                patch.object(
                    release.getpass,
                    "getpass",
                    side_effect=["new-oracle", "new-oracle"],
                ),
                patch("builtins.print"),
            ):
                release.prepare_secrets(
                    config, replace_oracle_password=True
                )
            self.assertEqual("new-oracle\n", oracle_secret.read_text())
            self.assertEqual("m" * 40 + "\n", master_secret.read_text())

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

    def test_offline_deploy_forbids_registry_pull(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
            with (
                patch.object(release, "render_deployment", return_value=output),
                patch.object(release.subprocess, "run") as runner,
                patch("builtins.print"),
            ):
                release.deploy(Path("deployment.ini"), offline=True)
            command = runner.call_args_list[1].args[0]
            self.assertEqual(
                ["up", "--detach", "--remove-orphans", "--pull", "never"],
                command[-5:],
            )

    def test_export_offline_bundle_is_self_contained_and_has_no_secrets(self) -> None:
        lock = release.load_image_lock("1.0", self.catalog)
        registry = lock["registry"]
        names = release.catalog_image_names(self.catalog)
        image_id = "sha256:" + "a" * 64
        details = [
            {
                "Id": image_id,
                "RepoDigests": [
                    f"{registry}/{item['name']}@{item['digest']}"
                ],
            }
            for item in lock["images"]
        ]

        def save_images(command: list[str], **_: object) -> None:
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"docker-image-archive")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "kbot-1.0"
            with (
                patch.object(
                    release,
                    "release_values",
                    return_value={
                        "version": "1.0",
                        "registry": registry,
                        "platform": "linux/amd64",
                        "revision": "0123456789ab",
                    },
                ),
                patch.object(release, "inspect_images", return_value=details),
                patch.object(release.subprocess, "run", side_effect=save_images),
            ):
                release.export_offline_bundle(Path("release.ini"), output)

            self.assertTrue((output / "kbot-offline").is_file())
            self.assertTrue((output / "images/kbot-images.tar").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            self.assertFalse((output / "runtime").exists())
            self.assertFalse(any(output.rglob("*password*")))
            manifest = json.loads(
                (output / "bundle-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(names, [item["name"] for item in manifest["images"]])
            self.assertTrue(
                all(item["image_id"] == image_id for item in manifest["images"])
            )

    def test_verify_offline_images_compares_loaded_image_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "bundle-manifest.json"
            image_id = "sha256:" + "b" * 64
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "kbot-offline-bundle.v1",
                        "images": [
                            {
                                "reference": "registry.test/kbot/main:1.0",
                                "image_id": image_id,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(
                    release,
                    "inspect_images",
                    return_value=[{"Id": image_id}],
                ),
                patch("builtins.print"),
            ):
                release.verify_offline_images(manifest_path)

    @staticmethod
    def _values(oracle_secret: Path, master_secret: Path) -> dict[str, object]:
        return {
            "project_name": "kbot4-test",
            "version": "4.1.0-test",
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
image_version = 4.1.0-test
ui_port = 8080
public_base_url = http://127.0.0.1:8080
embedding_dimension = 2048

[apps]
knowledge_retrieval = true
media_studio = true
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
