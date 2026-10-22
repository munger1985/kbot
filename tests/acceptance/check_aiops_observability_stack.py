#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STACK = ROOT / "scripts/deployment/aiops_observability"
SCRIPT = ROOT / "scripts/aiops-stack"


def _load_stack_script():
    loader = SourceFileLoader("aiops_stack_acceptance", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if not spec or not spec.loader:
        raise RuntimeError("无法加载AIOps观测栈脚本")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    stack = _load_stack_script()
    with tempfile.TemporaryDirectory(prefix="kbot-aiops-stack-") as temporary:
        config = Path(temporary) / "aiops-stack.ini"
        config.write_text(
            stack.CONFIG_TEMPLATE.replace(
                "deployment_id = CHANGE_ME", "deployment_id = acceptance-oracle"
            )
            .replace("local_access = false", "local_access = true", 1)
            .replace("[metrics]\n# enabled = true", "[metrics]\nenabled = true")
            .replace("[logs]\n# enabled = true", "[logs]\nenabled = true")
            .replace("[host]\n# enabled = true", "[host]\nenabled = true")
            .replace("# target_key = host-prod-01", "target_key = host-acceptance-01")
            .replace(
                "[oracle:oracle-prod-01]\n# enabled = true",
                "[oracle:oracle-prod-01]\nenabled = true",
            )
            .replace("# version = 19c", "version = 19c")
            .replace("# host = 10.0.0.20", "host = 127.0.0.1")
            .replace("# service = ORCLPDB1", "service = FREEPDB1")
            .replace("# username = kbot_monitor", "username = kbot_monitor", 1)
            .replace(
                "# password = CHANGE_ME", "password = acceptance-only-password", 1
            ),
            encoding="utf-8",
        )
        config.chmod(0o600)
        settings = stack._load_settings(config)
        stack._prepare_runtime(settings)
        state = settings.runtime_dir
        rendered_compose = json.loads(
            subprocess.check_output(
                [*stack._compose_command(settings), "config", "--format", "json"],
                text=True,
            )
        )
        for service_name in ("prometheus", "alertmanager", "loki"):
            if not rendered_compose["services"][service_name].get("group_add"):
                raise RuntimeError(f"{service_name}没有获得运行配置文件所属组")
        if "grafana" in rendered_compose["services"]:
            raise RuntimeError("部署栈仍包含Grafana服务")
        for service_name in ("prometheus", "loki"):
            service_networks = rendered_compose["services"][service_name]["networks"]
            if "management" not in service_networks:
                raise RuntimeError(f"{service_name}只连接内部网络，维护端口无法发布")
        loki_healthcheck = rendered_compose["services"]["loki"]["healthcheck"][
            "test"
        ]
        if "wget" in loki_healthcheck or "-verify-config=true" not in loki_healthcheck:
            raise RuntimeError("Loki健康检查依赖镜像中不存在的外部工具")
        loki_environment = rendered_compose["services"]["loki"]["environment"]
        if not loki_environment.get("AIOPS_LOKI_CONFIG_REVISION"):
            raise RuntimeError("Loki配置变化不会触发容器重建")
        alertmanager_environment = rendered_compose["services"]["alertmanager"][
            "environment"
        ]
        if not alertmanager_environment.get("AIOPS_ALERTMANAGER_CONFIG_REVISION"):
            raise RuntimeError("Alertmanager配置变化不会触发容器重建")
        alloy_environment = rendered_compose["services"]["alloy"]["environment"]
        if not alloy_environment.get("AIOPS_ALLOY_CONFIG_REVISION"):
            raise RuntimeError("Alloy配置变化不会触发容器重建")
        loki_config = (state / "loki/loki.yml").read_text(encoding="utf-8")
        for required_text in (
            "ruler:",
            "instance_addr: 127.0.0.1",
            "alertmanager_url: http://alertmanager:9093",
            "rules_directory: /etc/loki/rules",
            "rule_path: /loki/ruler",
        ):
            if required_text not in loki_config:
                raise RuntimeError(f"Loki Ruler配置缺少约束：{required_text}")
        loki_rules = (
            state / "loki/rules/fake/kbot-database-alerts.yml"
        ).read_text(encoding="utf-8")
        for required_text in (
            "alert: OracleAlertLogProblemDetected",
            "alert: MySQLAlertLogProblemDetected",
            "alert: PostgreSQLAlertLogProblemDetected",
            'severity=~"critical|warning"',
            'job="mysql_error"',
            'job="postgresql_log"',
            'json source_event_id="record_id"',
            "event_class: database.alert_log_problem",
        ):
            if required_text not in loki_rules:
                raise RuntimeError(f"数据库通用异常日志规则缺少约束：{required_text}")
        if "ORA-00060" in loki_rules:
            raise RuntimeError("Oracle异常日志规则不得维护错误码白名单")
        if os.stat(state / "secrets/oracle-oracle-prod-01_password").st_mode & 0o037:
            raise RuntimeError("Oracle Secret权限过宽")
        generated = json.loads((state / "compose.generated.yaml").read_text())
        if "oracle-oracle-prod-01-exporter" not in generated["services"]:
            raise RuntimeError("动态Compose没有生成Oracle Exporter")
        oracle_service = generated["services"]["oracle-oracle-prod-01-exporter"]
        if not any(
            "kbot-custom-metrics.yaml" in item
            for item in oracle_service.get("volumes", [])
        ):
            raise RuntimeError("Oracle扩展指标没有挂载到Exporter")
        overrides = json.loads(
            (state / "prometheus/kbot-aiops-query-overrides.json").read_text()
        )["prometheus_queries"]
        expected_overrides = {
            "db.availability",
            "db.cpu.utilization",
            "db.connection.active",
            "db.connection.utilization",
            "db.transaction.throughput",
            "db.response.latency",
            "db.storage.utilization",
            "db.storage.used_bytes",
            "db.storage.free_bytes",
            "db.storage.max_bytes",
            "db.error.rate",
            "host.cpu.utilization",
            "host.memory.utilization",
            "host.filesystem.utilization",
            "host.filesystem.used_bytes",
            "host.filesystem.available_bytes",
            "host.filesystem.size_bytes",
            "host.disk.io.utilization",
            "host.network.throughput",
            "runtime.cpu.utilization",
            "runtime.memory.utilization",
        }
        if set(overrides) != expected_overrides:
            raise RuntimeError("Oracle AIOps指标查询映射不完整")
        if shutil.which("promtool"):
            subprocess.run(
                [
                    "promtool",
                    "check",
                    "rules",
                    str(state / "prometheus/rules/kbot-observability.yml"),
                ],
                check=True,
            )

        multi_config = Path(temporary) / "aiops-stack-multi.ini"
        multi_config.write_text(
            """[deployment]
deployment_id = acceptance-central
role = central
local_access = false

[metrics]
enabled = true
kbot_webhook_url = https://kbot.example.com
kbot_webhook_key = whk-acceptance-key-long-enough-for-kbot
kbot_webhook_secret = acceptance-webhook-secret
kbot_uat_webhook_url = https://kbot-uat.example.com
kbot_uat_webhook_key = whk-acceptance-key-long-enough-for-kbot-uat
kbot_uat_webhook_secret = acceptance-uat-webhook-secret
ammolite_webhook_url = https://ammolite.example.com
ammolite_webhook_key = whk-acceptance-key-long-enough-for-ammolite
ammolite_webhook_secret = acceptance-ammolite-webhook-secret
ammolite_webhook_target_keys = oracle-prod-01,oracle-prod-02

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
        multi_config.chmod(0o600)
        central = stack._load_settings(multi_config)
        stack._prepare_runtime(central)
        subprocess.run(
            [*stack._compose_command(central), "config", "--quiet"],
            check=True,
        )
        targets = json.loads(
            (central.runtime_dir / "prometheus/targets/kbot.json").read_text()
        )
        if {item["labels"]["target_key"] for item in targets} != {
            "oracle-prod-01",
            "oracle-prod-02",
        }:
            raise RuntimeError("Prometheus多数据库Target生成错误")
        central_compose = json.loads(
            (central.runtime_dir / "compose.generated.yaml").read_text()
        )
        if "kbot-webhook-signer" not in central_compose["services"]:
            raise RuntimeError("KBot Webhook签名桥没有生成")
        if "kbot-uat-webhook-signer" not in central_compose["services"]:
            raise RuntimeError("KBot UAT Webhook签名桥没有生成")
        if "ammolite-webhook-signer" not in central_compose["services"]:
            raise RuntimeError("Ammolite Webhook签名桥没有生成")
        alertmanager_config = (
            central.runtime_dir / "alertmanager/alertmanager.yml"
        ).read_text(encoding="utf-8")
        for signer in (
            "kbot-webhook-signer",
            "kbot-uat-webhook-signer",
            "ammolite-webhook-signer",
        ):
            if f"http://{signer}:8080/alertmanager" not in alertmanager_config:
                raise RuntimeError(f"Alertmanager缺少{signer}双发配置")
        signer_environment = central_compose["services"][
            "kbot-webhook-signer"
        ].get("environment", {})
        if not signer_environment.get(
            "AIOPS_WEBHOOK_SIGNER_CONFIG_REVISION"
        ):
            raise RuntimeError("Webhook签名桥缺少凭据修订号")
    if "@sha256:" not in (STACK / "images.env").read_text(encoding="utf-8"):
        raise RuntimeError("镜像清单没有固定Digest")
    oracle_user_script = (
        STACK / "oracle/create_kbot_monitor.sql"
    ).read_text(encoding="utf-8")
    for required_text in (
        "CURRENT_USER",
        "CDB$ROOT",
        "ACCEPT KBOT_MONITOR_PASSWORD",
        "GRANT CREATE SESSION TO kbot_monitor",
        "GRANT SELECT ANY DICTIONARY TO kbot_monitor",
        "system_grant_count <> 2",
    ):
        if required_text not in oracle_user_script:
            raise RuntimeError(f"Oracle监控用户脚本缺少约束：{required_text}")
    for forbidden_text in (
        "GRANT DBA TO",
        "GRANT SELECT ANY TABLE TO",
        "GRANT SELECT_CATALOG_ROLE TO",
        "GRANT SELECT ON SYS.",
    ):
        if forbidden_text in oracle_user_script:
            raise RuntimeError(f"Oracle监控用户脚本包含过宽授权：{forbidden_text}")
    oracle_grant_script = (
        STACK / "oracle/grant_kbot_monitor.sql"
    ).read_text(encoding="utf-8")
    for required_text in (
        "CURRENT_USER",
        "CDB$ROOT",
        "DBA_USERS",
        "GRANT CREATE SESSION TO kbot_monitor",
        "GRANT SELECT ANY DICTIONARY TO kbot_monitor",
        "system_grant_count <> 2",
    ):
        if required_text not in oracle_grant_script:
            raise RuntimeError(f"Oracle完整授权脚本缺少约束：{required_text}")
    for forbidden_text in (
        "CREATE USER",
        "GRANT DBA TO",
        "GRANT SELECT ANY TABLE TO",
        "GRANT SELECT_CATALOG_ROLE TO",
        "GRANT SELECT ON SYS.",
    ):
        if forbidden_text in oracle_grant_script:
            raise RuntimeError(f"Oracle完整授权脚本包含禁用内容：{forbidden_text}")
    grant_pattern = re.compile(r"^\s*GRANT\s+.+?;", re.IGNORECASE | re.MULTILINE)
    expected_grants = {
        "GRANT CREATE SESSION TO KBOT_MONITOR;",
        "GRANT SELECT ANY DICTIONARY TO KBOT_MONITOR;",
    }
    for script_name, script in (
        ("Oracle建用户脚本", oracle_user_script),
        ("Oracle完整授权脚本", oracle_grant_script),
    ):
        grants = {
            " ".join(statement.upper().split())
            for statement in grant_pattern.findall(script)
        }
        if grants != expected_grants:
            raise RuntimeError(f"{script_name}必须且只能授予两项数据库诊断权限")
    print(
        "AIOps观测栈检查通过：角色、Compose、Secret、多数据库目标、"
        "Loki通用日志告警、签名桥和原生实时监控基础配置有效"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
