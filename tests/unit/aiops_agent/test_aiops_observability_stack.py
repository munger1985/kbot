from __future__ import annotations

import importlib.util
import json
import stat
import sys
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STACK = ROOT / "scripts/deployment/aiops_observability"
SCRIPT = ROOT / "scripts/aiops-stack"


def _load_stack_script():
    loader = SourceFileLoader("aiops_stack", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    assert "oracle-exporter:" not in compose
    assert "internal: ${AIOPS_INTERNAL_NETWORK:-true}" in compose
    assert "networks: [observability, outbound]" in compose
    assert "kbot-webhook-token" not in compose
    assert "${AIOPS_STACK_SECRET_DIR}:/run/secrets" not in compose


def test_single_config_enables_oracle_and_keeps_password_out_of_env(
    tmp_path: Path,
) -> None:
    stack = _load_stack_script()
    password = "p@ss:/with-specials"
    config = tmp_path / "aiops-stack.ini"
    config.write_text(
        stack.CONFIG_TEMPLATE.replace(
            "deployment_id = CHANGE_ME", "deployment_id = oracle-test-01"
        )
        .replace("[metrics]\n# enabled = true", "[metrics]\nenabled = true")
        .replace("[logs]\n# enabled = true", "[logs]\nenabled = true")
        .replace("[host]\n# enabled = true", "[host]\nenabled = true")
        .replace("# target_key = host-prod-01", "target_key = host-test-01")
        .replace(
            "[oracle:oracle-prod-01]\n# enabled = true",
            "[oracle:oracle-prod-01]\nenabled = true",
        )
        .replace("# host = 10.0.0.20", "host = db.example.internal")
        .replace("# service = ORCLPDB1", "service = FREEPDB1")
        .replace("# username = kbot_monitor", "username = kbot_monitor", 1)
        .replace("# password = CHANGE_ME", f"password = {password}", 1),
        encoding="utf-8",
    )
    config.chmod(0o600)
    settings = stack._load_settings(config)
    stack._prepare_runtime(settings)
    state = settings.runtime_dir
    env_text = (state / "stack.env").read_text(encoding="utf-8")
    assert "AIOPS_PROMETHEUS_CONFIG_REVISION=" in env_text
    assert password not in env_text
    assert password not in (state / "deployment.json").read_text(encoding="utf-8")
    password_file = state / "secrets/oracle-oracle-prod-01_password"
    assert password_file.read_text().strip() == password
    assert stat.S_IMODE(password_file.stat().st_mode) == 0o640
    dsn = (state / "secrets/oracle-oracle-prod-01_exporter_dsn").read_text().strip()
    assert "p%40ss%3A%2Fwith-specials" in dsn
    targets = json.loads((state / "prometheus/targets/kbot.json").read_text())
    assert {item["labels"]["job"] for item in targets} == {"node", "oracle"}
    generated = json.loads((state / "compose.generated.yaml").read_text())
    oracle_exporter = generated["services"]["oracle-oracle-prod-01-exporter"]
    assert "--query.timeout=15" in oracle_exporter["command"]
    assert any("kbot-custom-metrics.yaml" in item for item in oracle_exporter["volumes"])
    assert len(oracle_exporter["environment"]["KBOT_CUSTOM_METRICS_REVISION"]) == 64
    overrides = json.loads(
        (state / "prometheus/kbot-aiops-query-overrides.json").read_text()
    )["prometheus_queries"]
    assert set(overrides) == {
        "db.availability",
        "db.cpu.utilization",
        "db.connection.active",
        "db.connection.utilization",
        "db.transaction.throughput",
        "db.response.latency",
        "db.storage.utilization",
        "db.storage.free_bytes",
        "db.storage.max_bytes",
        "db.error.rate",
        "host.cpu.utilization",
        "host.memory.utilization",
        "host.filesystem.utilization",
        "host.disk.io.utilization",
        "host.network.throughput",
    }


def test_template_has_one_required_section_and_modules_are_disabled() -> None:
    stack = _load_stack_script()
    assert "[deployment]" in stack.CONFIG_TEMPLATE
    assert "deployment_id = CHANGE_ME" in stack.CONFIG_TEMPLATE
    assert "OEM不在此部署文件中配置" in stack.CONFIG_TEMPLATE
    assert stack.CONFIG_TEMPLATE.count("# enabled = true") == 8


def test_single_config_generates_every_enabled_module(tmp_path: Path) -> None:
    stack = _load_stack_script()
    config = tmp_path / "aiops-stack.ini"
    config.write_text(
        """[deployment]
deployment_id = all-modules-01
role = all-in-one
local_access = true

[metrics]
enabled = true
kbot_webhook_url = https://kbot.example.com
kbot_webhook_key = whk-acceptance-key-long-enough-for-kbot
kbot_webhook_secret = actual-webhook-credential-value

[logs]
enabled = true
loki_retention = 168h

[dashboard]
enabled = true
grafana_admin_user = kbot-admin
grafana_admin_password = grafana-secret

[host]
enabled = true
target_key = host-all-modules-01

[oracle:oracle-all-modules-01]
enabled = true
host = oracle.internal
service = FREEPDB1
username = oracle-monitor
password = oracle-secret

[mysql:mysql-all-modules-01]
enabled = true
host = mysql.internal
username = mysql-monitor
password = mysql-secret

[postgres:postgres-all-modules-01]
enabled = true
uri = postgres.internal:5432/postgres?sslmode=require
username = postgres-monitor
password = postgres-secret
""",
        encoding="utf-8",
    )
    config.chmod(0o600)
    settings = stack._load_settings(config)
    stack._prepare_runtime(settings)
    assert set(settings.profiles) == {
        *stack.MODULE_PROFILES.values(),
        "oracle",
        "mysql",
        "postgres",
    }
    assert (
        settings.runtime_dir / "secrets/grafana_admin_password"
    ).read_text().strip() == "grafana-secret"
    assert "mysql-secret" in (
        settings.runtime_dir / "secrets/mysql-mysql-all-modules-01_exporter_cnf"
    ).read_text()
    assert (
        settings.runtime_dir / "secrets/postgres-postgres-all-modules-01_password"
    ).read_text().strip() == "postgres-secret"
    assert "168h" in (settings.runtime_dir / "loki/loki.yml").read_text()
    loki_config = (settings.runtime_dir / "loki/loki.yml").read_text()
    assert "ruler:" in loki_config
    assert "alertmanager_url: http://alertmanager:9093" in loki_config
    assert "rules_directory: /etc/loki/rules" in loki_config
    loki_rules = (
        settings.runtime_dir / "loki/rules/fake/kbot-oracle-alerts.yml"
    ).read_text()
    assert "alert: OracleAlertLogProblemDetected" in loki_rules
    assert 'severity=~"critical|warning"' in loki_rules
    assert "ORA-00060" not in loki_rules
    assert "event_class: database.alert_log_problem" in loki_rules
    alloy_config = (STACK / "configuration/alloy/config.alloy").read_text()
    assert 'severity       = "severity"' in alloy_config
    revision = next(
        line.split("=", 1)[1]
        for line in (settings.runtime_dir / "stack.env").read_text().splitlines()
        if line.startswith("AIOPS_LOKI_CONFIG_REVISION=")
    )
    assert len(revision) == 64
    alloy_revision = next(
        line.split("=", 1)[1]
        for line in (settings.runtime_dir / "stack.env").read_text().splitlines()
        if line.startswith("AIOPS_ALLOY_CONFIG_REVISION=")
    )
    assert len(alloy_revision) == 64
    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in settings.runtime_dir.rglob("*")
        if path.is_file() and "secrets" not in path.parts
    )
    assert "actual-webhook-credential-value" not in generated_text
    assert "oracle-secret" not in generated_text
    assert "mysql-secret" not in generated_text
    assert "postgres-secret" not in generated_text


def test_loki_ruler_is_disabled_without_local_alertmanager(tmp_path: Path) -> None:
    stack = _load_stack_script()
    config = stack._loki_config("168h", alerting_enabled=False)
    assert "ruler:" not in config
    assert "alertmanager_url:" not in config
    ini = tmp_path / "aiops-stack.ini"
    ini.write_text(
        """[deployment]
deployment_id = logs-only
role = all-in-one
local_access = false

[logs]
enabled = true
""",
        encoding="utf-8",
    )
    ini.chmod(0o600)
    settings = stack._load_settings(ini)
    stale_rule = settings.runtime_dir / "loki/rules/fake/kbot-oracle-alerts.yml"
    stale_rule.parent.mkdir(parents=True, exist_ok=True)
    stale_rule.write_text("stale", encoding="utf-8")
    stack._prepare_runtime(settings)
    assert not stale_rule.exists()


def test_multiple_oracle_targets_generate_isolated_services_and_labels(
    tmp_path: Path,
) -> None:
    stack = _load_stack_script()
    config = tmp_path / "aiops-stack.ini"
    config.write_text(
        """[deployment]
deployment_id = multi-oracle
role = all-in-one
local_access = false

[metrics]
enabled = true

[logs]
enabled = true

[oracle:oracle-prod-01]
enabled = true
host = db01.internal
service = PROD1
username = monitor01
password = secret01

[oracle:oracle-prod-02]
enabled = true
host = db02.internal
service = PROD2
username = monitor02
password = secret02
""",
        encoding="utf-8",
    )
    config.chmod(0o600)
    settings = stack._load_settings(config)
    stack._prepare_runtime(settings)
    generated = json.loads(
        (settings.runtime_dir / "compose.generated.yaml").read_text()
    )
    assert {
        "oracle-oracle-prod-01-exporter",
        "oracle-oracle-prod-01-alert-collector",
        "oracle-oracle-prod-02-exporter",
        "oracle-oracle-prod-02-alert-collector",
    } <= set(generated["services"])
    targets = json.loads(
        (settings.runtime_dir / "prometheus/targets/kbot.json").read_text()
    )
    assert {item["labels"]["target_key"] for item in targets} == {
        "oracle-prod-01",
        "oracle-prod-02",
    }
    assert (
        settings.runtime_dir / "secrets/oracle-oracle-prod-01_password"
    ).read_text().strip() == "secret01"
    assert (
        settings.runtime_dir / "secrets/oracle-oracle-prod-02_password"
    ).read_text().strip() == "secret02"


def test_central_role_generates_remote_database_targets(tmp_path: Path) -> None:
    stack = _load_stack_script()
    config = tmp_path / "aiops-stack.ini"
    config.write_text(
        """[deployment]
deployment_id = central-prod
role = central
local_access = false

[metrics]
enabled = true

[prometheus_target:oracle-prod-01]
enabled = true
engine = oracle
address = 10.20.0.11:19161
environment = production

[prometheus_target:oracle-prod-02]
enabled = true
engine = oracle
address = 10.20.0.12:19161
environment = production
""",
        encoding="utf-8",
    )
    config.chmod(0o600)
    settings = stack._load_settings(config)
    stack._prepare_runtime(settings)
    targets = json.loads(
        (settings.runtime_dir / "prometheus/targets/kbot.json").read_text()
    )
    assert [item["targets"][0] for item in targets] == [
        "10.20.0.11:19161",
        "10.20.0.12:19161",
    ]
    assert all(item["labels"]["environment"] == "production" for item in targets)


def test_collector_role_maps_each_exporter_to_declared_endpoint(
    tmp_path: Path,
) -> None:
    stack = _load_stack_script()
    config = tmp_path / "aiops-stack.ini"
    config.write_text(
        """[deployment]
deployment_id = collector-prod
role = collector
local_access = false

[host]
enabled = true
target_key = host-prod-01
exporter_bind_address = 10.20.0.11
exporter_port = 19100

[oracle:oracle-prod-01]
enabled = true
host = oracle.internal
service = PROD1
username = monitor
password = collector-secret
exporter_bind_address = 10.20.0.11
exporter_port = 19161
""",
        encoding="utf-8",
    )
    config.chmod(0o600)
    settings = stack._load_settings(config)
    stack._prepare_runtime(settings)
    generated = json.loads(
        (settings.runtime_dir / "compose.generated.yaml").read_text()
    )
    assert generated["services"]["node-exporter"]["ports"] == [
        "10.20.0.11:19100:9100"
    ]
    assert generated["services"]["oracle-oracle-prod-01-exporter"]["ports"] == [
        "10.20.0.11:19161:9161"
    ]


def test_webhook_signer_uses_dynamic_hmac_headers() -> None:
    signer = (STACK / "webhook_signer/signer.py").read_text(encoding="utf-8")
    assert 'timestamp.encode("ascii") + b"." + body' in signer
    assert '"X-KBot-Timestamp": timestamp' in signer
    assert '"X-KBot-Signature": f"sha256={signature}"' in signer
    assert "Authorization" not in signer


def test_oracle_collector_uses_durable_ordered_checkpoint() -> None:
    collector = _load_collector()
    assert "V$DIAG_ALERT_EXT" in collector.QUERY
    assert "ORIGINATING_TIMESTAMP > :last_timestamp" in collector.QUERY
    assert "RECORD_ID > :last_record_id" in collector.QUERY
    assert "ORDER BY ORIGINATING_TIMESTAMP, RECORD_ID" in collector.QUERY
    assert collector.Settings.__dataclass_params__.frozen is True


def test_oracle_collector_classifies_all_structured_alert_types() -> None:
    collector = _load_collector()
    assert collector._diagnostic_severity(2, 8) == "critical"
    assert collector._diagnostic_severity(3, 8) == "critical"
    assert collector._diagnostic_severity(1, 1) == "critical"
    assert collector._diagnostic_severity(4, 16) == "warning"
    assert collector._diagnostic_severity(5, 16) == "info"
    assert (
        collector._diagnostic_severity(
            5,
            16,
            'ORA-12012: error on auto execute of job "SYS"."DBMS_JOB$_5"',
        )
        == "critical"
    )
    assert collector._diagnostic_severity(5, 16, "ABC_1-4567: 任意组件错误") == (
        "critical"
    )


def test_oracle_collector_writes_normalized_severity(tmp_path: Path) -> None:
    collector = _load_collector()
    settings = collector.Settings(
        host="oracle.internal",
        port=1521,
        service="PDB01",
        target_key="oracle-test",
        poll_seconds=15,
        initial_lookback_seconds=900,
        max_rows=1000,
        username_file=tmp_path / "username",
        password_file=tmp_path / "password",
        output_file=tmp_path / "alert.jsonl",
        checkpoint_file=tmp_path / "checkpoint.json",
        health_file=tmp_path / "health.json",
    )
    timestamp = datetime.now(timezone.utc)
    collector._append_rows(
        settings,
        [
            "ORIGINATING_TIMESTAMP",
            "RECORD_ID",
            "MESSAGE_TYPE",
            "MESSAGE_LEVEL",
            "MESSAGE_TEXT",
        ],
        [(timestamp, 1, 5, 16, "ORA-12012: 自动任务执行失败")],
    )
    payload = json.loads(settings.output_file.read_text(encoding="utf-8"))
    assert payload["target_key"] == "oracle-test"
    assert payload["severity"] == "critical"
    assert payload["message_text"] == "ORA-12012: 自动任务执行失败"


def test_oracle_rules_use_exporter_metric_contract_without_double_percentage() -> None:
    stack = _load_stack_script()
    rules = stack._prometheus_rules()
    assert "oracledb_kbot_cpu_utilization_percent" in rules
    assert "oracledb_kbot_errors_total" in rules
    assert "oracledb_kbot_connection_current_sessions" in rules
    assert "oracledb_kbot_connection_limit_sessions" in rules
    assert "oracledb_resource_current_utilization" not in rules
    assert "oracledb_tablespace_free_bytes" in rules
    assert 'expr: oracledb_tablespace_used_percent{job="oracle"}' in rules
    assert 'oracledb_tablespace_used_percent{job="oracle"} * 100' not in rules


def test_oracle_custom_metrics_use_pdb_compatible_sources() -> None:
    custom_metrics = (STACK / "configuration/oracle/custom-metrics.yaml").read_text(
        encoding="utf-8"
    )
    assert "v$rsrcpdbmetric" in custom_metrics
    assert "avg_cpu_utilization" in custom_metrics
    assert "v$parameter" in custom_metrics
    assert "max_pdb_sessions" in custom_metrics
    assert "v$resource_limit" not in custom_metrics
    assert "Host CPU Utilization (%)" not in custom_metrics
