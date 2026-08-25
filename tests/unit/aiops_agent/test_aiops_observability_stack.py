from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STACK = ROOT / "services/aiops_agent/deployment/observability"
SCRIPT = ROOT / "scripts/aiops-stack"


def _load_collector():
    path = STACK / "oracle_alert_collector/collector.py"
    spec = importlib.util.spec_from_file_location("oracle_alert_collector", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_upstream_images_are_version_and_digest_locked() -> None:
    assignments = [
        line
        for line in (STACK / "images.env").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    upstream = [line for line in assignments if "kbot/" not in line]
    assert upstream
    assert all("@sha256:" in line for line in upstream)
    assert all(":latest" not in line.lower() for line in assignments)


def test_compose_keeps_oem_external_and_has_no_public_default_ports() -> None:
    compose = (STACK / "compose.yaml").read_text(encoding="utf-8")
    assert "oem:" not in compose.lower()
    assert "ports:" not in compose
    assert "profiles: [metrics]" in compose
    assert "profiles: [logs]" in compose
    assert "profiles: [oracle]" in compose
    assert "oracle-alert-log:/var/lib/kbot/oracle-alert:ro" in compose
    assert "internal: ${AIOPS_INTERNAL_NETWORK:-true}" in compose
    assert "networks: [observability, outbound]" in compose


def test_oracle_configure_keeps_password_out_of_generated_env(tmp_path: Path) -> None:
    password = "p@ss:/with-specials"
    password_file = tmp_path / "input-password"
    password_file.write_text(password, encoding="utf-8")
    state = tmp_path / "state"
    subprocess.run(
        [
            str(SCRIPT),
            "configure",
            "--preset",
            "oracle-lite",
            "--state-dir",
            str(state),
            "--target-key",
            "oracle-test-01",
            "--oracle-host",
            "db.example.internal",
            "--oracle-service",
            "FREEPDB1",
            "--oracle-user",
            "kbot_monitor",
            "--oracle-password-file",
            str(password_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    env_text = (state / "stack.env").read_text(encoding="utf-8")
    assert password not in env_text
    assert password not in (state / "deployment.json").read_text(encoding="utf-8")
    assert (state / "secrets/oracle_password").read_text().strip() == password
    assert stat.S_IMODE((state / "secrets/oracle_password").stat().st_mode) == 0o640
    dsn = (state / "secrets/oracle_exporter_dsn").read_text().strip()
    assert "p%40ss%3A%2Fwith-specials" in dsn
    targets = json.loads((state / "prometheus/targets/kbot.json").read_text())
    assert {item["labels"]["job"] for item in targets} == {"node", "oracle"}


def test_oracle_collector_uses_durable_ordered_checkpoint() -> None:
    collector = _load_collector()
    assert "V$DIAG_ALERT_EXT" in collector.QUERY
    assert "ORIGINATING_TIMESTAMP > :last_timestamp" in collector.QUERY
    assert "RECORD_ID > :last_record_id" in collector.QUERY
    assert "ORDER BY ORIGINATING_TIMESTAMP, RECORD_ID" in collector.QUERY
    assert collector.Settings.__dataclass_params__.frozen is True
