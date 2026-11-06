"""统一安装入口的离线回归测试；不连接 Docker、Registry 或 Oracle。"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "installation"))
import deployment
import release


class DeploymentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lock = release.load_image_lock("1.0")
        self.catalog, self.topology = release.validate_catalog()
        self.source = self.root / "models-source"
        for item in release.load_docling_model_spec("1.0")["models"]:
            directory = self.source / item["directory"]
            directory.mkdir(parents=True)
            (directory / "model.bin").write_bytes(b"test-model")
        self.bundle = self.root / "bundle"
        with (
            patch.object(release, "release_values", return_value={
                "version": "1.0", "registry": self.lock["registry"],
                "platform": "linux/amd64", "revision": self.lock["source_commit"][:12],
            }),
            patch.object(release, "assert_release_sources"),
        ):
            release.export_deployment_bundle(Path("release.ini"), self.bundle, self.source)
        self.manifest = deployment.verify_package(self.bundle)
        self.config = self.bundle / "deployment.ini"
        text = (self.bundle / "deployment.ini.example").read_text().replace("CHANGE_ME", "oracle-test-secret%with#symbols")
        self.config.write_text(text, encoding="utf-8")
        self.config.chmod(0o600)
        self.parser, self.state_dir = deployment.load_input(self.config, self.manifest)

    def prepared(self):
        self.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        effective, values = deployment.prepare_config(self.parser, self.state_dir, None)
        values["image_digests"] = {record["name"]: record["digest"] for record in self.manifest["images"]}
        return effective, values

    def pipeline(self, *, initialize: bool = False, adopt: bool = False):
        text = self.config.read_text()
        if initialize:
            text = text.replace("initialize_empty_schema = false", "initialize_empty_schema = true")
        if adopt:
            text = text.replace("adopt_existing_schema = false", "adopt_existing_schema = true")
        self.config.write_text(text)
        contexts = contextlib.ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(patch.object(deployment, "preflight"))
        contexts.enter_context(patch.object(deployment, "acquire_images"))
        contexts.enter_context(patch.object(deployment, "installer"))
        contexts.enter_context(patch.object(deployment.release, "install_offline_docling_models"))
        runner = contexts.enter_context(patch.object(deployment, "run", return_value=subprocess.CompletedProcess([], 0, "", "")))
        bootstrap = contexts.enter_context(patch.object(deployment, "bootstrap_database"))
        acceptance = contexts.enter_context(patch.object(deployment, "accept_deployment"))
        contexts.enter_context(patch("builtins.print"))
        return runner, bootstrap, acceptance

    def test_online_bundle_is_self_contained_without_images_secrets_or_source(self):
        self.assertTrue((self.bundle / "kbot-deploy").stat().st_mode & stat.S_IXUSR)
        self.assertTrue((self.bundle / "models/docling-models.tar").is_file())
        self.assertTrue((self.bundle / "scripts/aiops-stack").is_file())
        self.assertFalse((self.bundle / "images").exists())
        self.assertFalse((self.bundle / "services").exists())
        self.assertFalse((self.bundle / "configuration").exists())
        self.assertEqual(release.DEPLOYMENT_BUNDLE_SCHEMA, self.manifest["schema"])

    def test_bundle_tool_runs_without_repository_or_python_packages(self):
        self.config.unlink()
        result = subprocess.run(["sh", str(self.bundle / "kbot-deploy")], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("已创建唯一配置", result.stdout)
        self.assertEqual(0o600, stat.S_IMODE(self.config.stat().st_mode))
        self.assertFalse(self.state_dir.exists())

    def test_zero_argument_entry_rejects_flags(self):
        result = subprocess.run(["sh", str(self.bundle / "kbot-deploy"), "--force"], capture_output=True, text=True)
        self.assertEqual(2, result.returncode)
        self.assertIn("不接受参数", result.stderr)

    def test_package_detects_modified_controller(self):
        path = self.bundle / "installation/deployment.py"
        path.write_text(path.read_text() + "\n# 篡改\n")
        with self.assertRaisesRegex(release.ReleaseError, "校验失败"):
            deployment.verify_package(self.bundle)

    def test_checksum_manifest_must_cover_controller(self):
        path = self.bundle / "SHA256SUMS"
        path.write_text("\n".join(line for line in path.read_text().splitlines() if not line.endswith("installation/deployment.py")) + "\n")
        with self.assertRaisesRegex(release.ReleaseError, "必要工件"):
            deployment.verify_package(self.bundle)

    def test_rejects_untrusted_checksum_paths(self):
        with (self.bundle / "SHA256SUMS").open("a") as output:
            output.write("a" * 64 + "  ../outside\n")
        with self.assertRaisesRegex(release.ReleaseError, "不安全"):
            deployment.verify_package(self.bundle)

    def test_rejects_world_readable_config(self):
        self.config.chmod(0o644)
        with self.assertRaisesRegex(release.ReleaseError, "600"):
            deployment.load_input(self.config, self.manifest)

    def test_rejects_placeholder_password(self):
        self.config.write_text(self.config.read_text().replace("oracle-test-secret%with#symbols", "CHANGE_ME"))
        with self.assertRaisesRegex(release.ReleaseError, "Oracle 密码"):
            deployment.load_input(self.config, self.manifest)

    def test_rejects_unknown_configuration_fields(self):
        self.config.write_text(self.config.read_text().replace("ui_port = 8080", "ui_prt = 8080"))
        with self.assertRaisesRegex(release.ReleaseError, "未知"):
            deployment.load_input(self.config, self.manifest)

    def test_rejects_conflicting_database_authorizations(self):
        self.config.write_text(self.config.read_text().replace("initialize_empty_schema = false", "initialize_empty_schema = true").replace("adopt_existing_schema = false", "adopt_existing_schema = true"))
        with self.assertRaisesRegex(release.ReleaseError, "同时"):
            deployment.load_input(self.config, self.manifest)

    def test_single_config_generates_secrets_without_leaking_password(self):
        effective, values = self.prepared()
        secret = values["oracle_password_file"]
        self.assertEqual("oracle-test-secret%with#symbols\n", secret.read_text())
        self.assertNotIn("oracle-test-secret", effective.read_text())
        self.assertGreaterEqual(len(values["master_key_file"].read_text().strip()), 32)
        self.assertEqual(0o600, stat.S_IMODE(secret.stat().st_mode))
        self.assertEqual(0o600, stat.S_IMODE(effective.stat().st_mode))

    def test_repeated_prepare_preserves_master_key(self):
        _, first = self.prepared()
        original = first["master_key_file"].read_bytes()
        deployment.prepare_config(self.parser, self.state_dir, {"phase": "ready"})
        self.assertEqual(original, first["master_key_file"].read_bytes())

    def test_missing_master_key_in_existing_installation_is_fatal(self):
        self.state_dir.mkdir(parents=True)
        with self.assertRaisesRegex(release.ReleaseError, "原主密钥"):
            deployment.prepare_config(self.parser, self.state_dir, {"phase": "ready"})
        self.assertFalse((self.state_dir / "master_key").exists())

    def test_adoption_requires_original_master_key(self):
        self.parser["installation"]["adopt_existing_schema"] = "true"
        self.state_dir.mkdir(parents=True)
        with self.assertRaisesRegex(release.ReleaseError, "原主密钥"):
            deployment.prepare_config(self.parser, self.state_dir, None)

    def test_initialization_requires_explicit_authorization(self):
        _, values = self.prepared()
        with self.assertRaisesRegex(release.ReleaseError, "initialize_empty_schema"):
            deployment.check_installation(None, deployment.installation_identity(values, self.manifest), self.parser)

    def test_same_version_deploy_never_replays_initialization(self):
        _, values = self.prepared()
        identity = deployment.installation_identity(values, self.manifest)
        self.parser["installation"]["initialize_empty_schema"] = "true"
        state = {"identity": identity, "phase": "ready"}
        self.assertEqual("restart", deployment.check_installation(state, identity, self.parser))

    def test_changed_apps_database_images_or_master_key_are_rejected(self):
        _, values = self.prepared()
        identity = deployment.installation_identity(values, self.manifest)
        for field in ("enabled_apps", "database", "images", "master_key_sha256", "project_name", "data_dir", "embedding_dimension", "docling_models_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(identity)
                changed[field] = "changed"
                with self.assertRaisesRegex(release.ReleaseError, "受控升级"):
                    deployment.check_installation({"identity": identity, "phase": "ready"}, changed, self.parser)

    def test_partial_initialization_never_replays_ddl(self):
        _, values = self.prepared()
        identity = deployment.installation_identity(values, self.manifest)
        self.parser["installation"]["initialize_empty_schema"] = "true"
        with self.assertRaisesRegex(release.ReleaseError, "未完成"):
            deployment.check_installation({"identity": identity, "phase": "initializing"}, identity, self.parser)

    def test_exclusive_lock_prevents_concurrent_deployment(self):
        with deployment.installation_lock(self.state_dir):
            with self.assertRaisesRegex(release.ReleaseError, "正在执行"):
                with deployment.installation_lock(self.state_dir):
                    self.fail("不应获取第二把锁")

    def test_compose_pins_digest_and_checks_every_http_process(self):
        _, values = self.prepared()
        rendered = release.render_compose(self.catalog, self.topology, values)
        self.assertNotIn(":1.0\"", rendered)
        self.assertIn("@sha256:", rendered)
        selected = release.selected_service_configs(self.catalog, values)
        expected = sum(item["kind"] == "http" and item["service_config"] in selected for item in self.topology["processes"])
        self.assertEqual(expected + 1, rendered.count("    healthcheck:"))
        self.assertNotIn("USER_TABLES", rendered)
        self.assertNotIn("USER_OBJECTS", rendered)
        self.assertEqual(1, rendered.count("    ports:"))

    def test_unknown_digest_is_never_replaced_with_mutable_tag(self):
        _, values = self.prepared()
        del values["image_digests"]["kbot-main-api"]
        with self.assertRaisesRegex(release.ReleaseError, "Digest"):
            release.image_reference(values, "kbot-main-api")

    def test_empty_digest_lock_is_not_a_tag_fallback(self):
        _, values = self.prepared()
        values["image_digests"] = {}
        with self.assertRaisesRegex(release.ReleaseError, "Digest"):
            release.image_reference(values, "kbot-main-api")

    @unittest.skipUnless(shutil.which("docker"), "需要本机 Docker CLI")
    def test_generated_compose_passes_real_cli_validation_without_daemon(self):
        effective, values = self.prepared()
        output = self.root / "rendered"
        release.render_deployment(effective, output=output, values=values)
        result = subprocess.run(
            [*deployment.compose_command(output, values["project_name"]), "--profile", "tools", "config", "--format", "json"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        definition = json.loads(result.stdout)
        self.assertIn("installer", definition["services"])
        self.assertTrue(all("@sha256:" in item["image"] for item in definition["services"].values()))
        self.assertIn("healthcheck", definition["services"]["main-api"])

    def test_online_pulls_only_selected_locked_images(self):
        _, values = self.prepared()
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            reference = command[-1]
            details = [{"RepoDigests": [reference], "Architecture": "amd64", "Os": "linux"}]
            return subprocess.CompletedProcess(command, 0, json.dumps(details), "")
        with patch.object(deployment, "run", side_effect=fake_run):
            deployment.acquire_images(self.bundle, values, self.manifest)
        pulls = [command for command in calls if command[:3] == ["docker", "image", "pull"]]
        self.assertTrue(pulls)
        self.assertTrue(all("@sha256:" in command[-1] for command in pulls))
        self.assertFalse(any("kbot-media-studio-app" in command[-1] for command in pulls))

    def test_offline_image_acquisition_never_pulls(self):
        _, values = self.prepared()
        manifest = copy.deepcopy(self.manifest)
        manifest["schema"] = release.OFFLINE_BUNDLE_SCHEMA
        manifest["image_archive"] = "images.tar"
        (self.bundle / "images.tar").write_bytes(b"test")
        for item in manifest["images"]:
            item["image_id"] = "sha256:" + "a" * 64
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, json.dumps([{"Id": "sha256:" + "a" * 64}]), "")
        with patch.object(deployment, "run", side_effect=fake_run):
            deployment.acquire_images(self.bundle, values, manifest)
        self.assertFalse(any("pull" in command for command in calls))
        self.assertEqual("load", calls[0][2])

    def test_offline_full_pipeline_uses_content_ids_and_never_requests_pull(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["schema"] = release.OFFLINE_BUNDLE_SCHEMA
        manifest["image_archive"] = "images.tar"
        (self.bundle / "images.tar").write_bytes(b"test-image-archive")
        for record in manifest["images"]:
            record["registry_digest"] = record.pop("digest")
            record["image_id"] = "sha256:" + "a" * 64
        manifest_path = self.bundle / "bundle-manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        checksum_path = self.bundle / "SHA256SUMS"
        checksums = [line for line in checksum_path.read_text().splitlines() if not line.endswith("  bundle-manifest.json")]
        checksums.extend([
            release.sha256_file(manifest_path) + "  bundle-manifest.json",
            release.sha256_file(self.bundle / "images.tar") + "  images.tar",
        ])
        checksum_path.write_text("\n".join(checksums) + "\n")
        runner, _, _ = self.pipeline(initialize=True)
        deployment.deploy(self.bundle)
        output = Path(deployment.read_state(self.state_dir)["deployment_dir"])
        text = (output / "compose.yaml").read_text()
        self.assertIn('image: "sha256:', text)
        self.assertNotIn("@sha256:", text)
        for call in runner.call_args_list:
            command = call.args[0]
            if "--pull" in command:
                self.assertEqual("never", command[command.index("--pull") + 1])

    def test_fresh_install_runs_bootstrap_before_up_and_acceptance(self):
        runner, bootstrap, acceptance = self.pipeline(initialize=True)
        phases = []
        bootstrap.side_effect = lambda *a, **k: phases.append(deployment.read_state(self.state_dir)["phase"])
        deployment.deploy(self.bundle)
        self.assertEqual(["initializing"], phases)
        self.assertEqual("ready", deployment.read_state(self.state_dir)["phase"])
        bootstrap.assert_called_once()
        acceptance.assert_called_once()
        ups = [call.args[0] for call in runner.call_args_list if "up" in call.args[0]]
        self.assertIn("--wait", ups[0])
        self.assertIn("--no-build", ups[0])
        self.assertEqual("never", ups[0][ups[0].index("--pull") + 1])

    def test_repeated_deploy_preserves_snapshots_and_does_not_bootstrap(self):
        _, bootstrap, _ = self.pipeline(initialize=True)
        deployment.deploy(self.bundle)
        old_output = Path(deployment.read_state(self.state_dir)["deployment_dir"])
        master = (self.state_dir / "master_key").read_bytes()
        deployment.deploy(self.bundle)
        bootstrap.assert_called_once()
        self.assertTrue((old_output / "compose.yaml").is_file())
        self.assertNotEqual(str(old_output), deployment.read_state(self.state_dir)["deployment_dir"])
        self.assertEqual(master, (self.state_dir / "master_key").read_bytes())

    def test_failed_initialization_blocks_retry_before_ddl(self):
        _, bootstrap, _ = self.pipeline(initialize=True)
        bootstrap.side_effect = release.ReleaseError("模拟数据库初始化中断")
        with self.assertRaisesRegex(release.ReleaseError, "中断"):
            deployment.deploy(self.bundle)
        self.assertEqual("initializing", deployment.read_state(self.state_dir)["phase"])
        with self.assertRaisesRegex(release.ReleaseError, "未完成"):
            deployment.deploy(self.bundle)
        bootstrap.assert_called_once()

    def test_failed_startup_can_retry_without_resetting_accounts(self):
        _, bootstrap, acceptance = self.pipeline(initialize=True)
        acceptance.side_effect = release.ReleaseError("模拟未就绪")
        with self.assertRaisesRegex(release.ReleaseError, "未就绪"):
            deployment.deploy(self.bundle)
        self.assertEqual("initialized", deployment.read_state(self.state_dir)["phase"])
        acceptance.side_effect = None
        deployment.deploy(self.bundle)
        bootstrap.assert_called_once()
        self.assertEqual("ready", deployment.read_state(self.state_dir)["phase"])

    def test_adoption_checks_without_initializing_or_replacing_master(self):
        _, bootstrap, _ = self.pipeline(adopt=True)
        self.state_dir.mkdir(parents=True)
        release.write_private_secret(self.state_dir / "master_key", "m" * 40)
        deployment.deploy(self.bundle)
        self.assertTrue(bootstrap.call_args.kwargs["adopt"])
        self.assertEqual("m" * 40 + "\n", (self.state_dir / "master_key").read_text())

    def test_failed_read_only_adoption_can_be_rechecked_without_ddl(self):
        _, bootstrap, _ = self.pipeline(adopt=True)
        self.state_dir.mkdir(parents=True)
        release.write_private_secret(self.state_dir / "master_key", "m" * 40)
        bootstrap.side_effect = release.ReleaseError("校验失败")
        with self.assertRaisesRegex(release.ReleaseError, "校验失败"):
            deployment.deploy(self.bundle)
        self.assertIsNone(deployment.read_state(self.state_dir))
        bootstrap.side_effect = None
        deployment.deploy(self.bundle)
        self.assertEqual("ready", deployment.read_state(self.state_dir)["phase"])

    def test_no_authorization_never_changes_database_or_starts_containers(self):
        runner, bootstrap, _ = self.pipeline()
        with self.assertRaisesRegex(release.ReleaseError, "没有安装记录"):
            deployment.deploy(self.bundle)
        bootstrap.assert_not_called()
        runner.assert_not_called()

    def test_command_failure_does_not_echo_secret_output(self):
        result = subprocess.CompletedProcess([], 1, "oracle-test-secret", "database-password")
        with patch.object(deployment.subprocess, "run", return_value=result):
            with self.assertRaises(release.ReleaseError) as raised:
                deployment.run(["docker", "compose", "run"])
        self.assertNotIn("oracle-test-secret", str(raised.exception))
        self.assertNotIn("database-password", str(raised.exception))

    def test_command_failure_exposes_only_safe_error_codes(self):
        result = subprocess.CompletedProcess([], 1, "INSTALLATION_ERROR=DatabaseError", "ORA-01017 password=do-not-echo")
        with patch.object(deployment.subprocess, "run", return_value=result):
            with self.assertRaises(release.ReleaseError) as raised:
                deployment.run(["docker", "compose", "run"])
        self.assertIn("ORA-01017", str(raised.exception))
        self.assertIn("INSTALLATION_ERROR=DatabaseError", str(raised.exception))
        self.assertNotIn("do-not-echo", str(raised.exception))

    def test_process_acceptance_refuses_missing_workers(self):
        container = {"Id": "c1", "RestartCount": 0, "Config": {"Labels": {"com.docker.compose.service": "main-api"}}, "State": {"Status": "running", "Health": {"Status": "healthy"}}}
        with patch.object(deployment, "run", side_effect=[subprocess.CompletedProcess([], 0, "c1", ""), subprocess.CompletedProcess([], 0, json.dumps([container]), "")]):
            with self.assertRaisesRegex(release.ReleaseError, "不完整"):
                deployment.verify_running(["docker", "compose"], {"main-api", "worker"})

    def test_process_acceptance_refuses_unhealthy_http_service(self):
        container = {"Id": "c1", "RestartCount": 0, "Config": {"Labels": {"com.docker.compose.service": "main-api"}}, "State": {"Status": "running", "Health": {"Status": "unhealthy"}}}
        with patch.object(deployment, "run", side_effect=[subprocess.CompletedProcess([], 0, "c1", ""), subprocess.CompletedProcess([], 0, json.dumps([container]), "")]):
            with self.assertRaisesRegex(release.ReleaseError, "未就绪"):
                deployment.verify_running(["docker", "compose"], {"main-api"})

    def test_new_install_never_replaces_an_existing_compose_project(self):
        _, values = self.prepared()
        with patch.object(deployment, "run", return_value=subprocess.CompletedProcess([], 0, "existing-container", "")):
            with self.assertRaisesRegex(release.ReleaseError, "同名"):
                deployment.guard_project(values, action="initialize")

    def test_source_drift_blocks_publishing(self):
        with patch.object(release.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
            with self.assertRaisesRegex(release.ReleaseError, "新镜像锁"):
                release.assert_release_sources(self.lock)

    def test_account_handoff_is_private_and_never_overwritten(self):
        spec = importlib.util.spec_from_file_location("initialize_installation", ROOT / "scripts/db/initialize_installation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        path = self.root / "private/accounts.json"
        module.write_accounts(path, {"ADMIN": "random-test"})
        self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))
        with self.assertRaisesRegex(RuntimeError, "拒绝覆盖"):
            module.write_accounts(path, {"ADMIN": "replacement"})
        self.assertEqual("random-test", json.loads(path.read_text())["accounts"]["ADMIN"])


if __name__ == "__main__":
    unittest.main()
