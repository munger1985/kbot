#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import os
import json
import re
import shutil
import subprocess
import sys
import tempfile
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
            .replace(
                "[dashboard]\n# enabled = true",
                "[dashboard]\nenabled = true",
            )
            .replace(
                "# grafana_bind_address = 127.0.0.1",
                "grafana_bind_address = 10.0.0.190",
            )
            .replace(
                "# grafana_admin_user = kbot-admin",
                "grafana_admin_user = kbot-admin",
            )
            .replace(
                "# grafana_admin_password = CHANGE_ME",
                "grafana_admin_password = acceptance-grafana-password",
            )
            .replace("[host]\n# enabled = true", "[host]\nenabled = true")
            .replace("# target_key = host-prod-01", "target_key = host-acceptance-01")
            .replace(
                "[oracle:oracle-prod-01]\n# enabled = true",
                "[oracle:oracle-prod-01]\nenabled = true",
            )
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
        stack_env = (state / "stack.env").read_text(encoding="utf-8")
        if "GRAFANA_BIND_ADDRESS=10.0.0.190" not in stack_env:
            raise RuntimeError("Grafana管理网监听地址没有写入Compose环境")
        rendered_compose = json.loads(
            subprocess.check_output(
                [*stack._compose_command(settings), "config", "--format", "json"],
                text=True,
            )
        )
        for service_name in ("prometheus", "alertmanager", "loki"):
            if not rendered_compose["services"][service_name].get("group_add"):
                raise RuntimeError(f"{service_name}没有获得运行配置文件所属组")
        for service_name in ("prometheus", "loki", "grafana"):
            service_networks = rendered_compose["services"][service_name]["networks"]
            if "management" not in service_networks:
                raise RuntimeError(f"{service_name}只连接内部网络，维护端口无法发布")
        grafana_mount_targets = {
            item["target"]
            for item in rendered_compose["services"]["grafana"]["volumes"]
        }
        if "/var/lib/grafana/dashboards" not in grafana_mount_targets:
            raise RuntimeError("Grafana没有挂载受控Dashboard目录")
        grafana_environment = rendered_compose["services"]["grafana"]["environment"]
        if grafana_environment.get("GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH") != (
            "/var/lib/grafana/dashboards/oracle-overview.json"
        ):
            raise RuntimeError("Grafana没有把Oracle总览设置为默认首页")
        loki_healthcheck = rendered_compose["services"]["loki"]["healthcheck"][
            "test"
        ]
        if "wget" in loki_healthcheck or "-verify-config=true" not in loki_healthcheck:
            raise RuntimeError("Loki健康检查依赖镜像中不存在的外部工具")
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
        if len(overrides) != 15:
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
        "SYS.V_$SYSMETRIC",
        "SYS.V_$SGA",
        "SYS.V_$PGASTAT",
        "SYS.V_$RECOVERY_FILE_DEST",
        "SYS.V_$DIAG_ALERT_EXT",
        "SYS.GV_$TRANSACTION",
    ):
        if required_text not in oracle_user_script:
            raise RuntimeError(f"Oracle监控用户脚本缺少约束：{required_text}")
    for forbidden_text in (
        "GRANT DBA TO",
        "GRANT SELECT ANY TABLE TO",
        "GRANT SELECT_CATALOG_ROLE TO",
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
        "SYS.V_$INSTANCE",
        "SYS.V_$DATABASE",
        "SYS.V_$SQLSTATS",
        "SYS.GV_$SESSION",
        "SYS.GV_$TRANSACTION",
        "SYS.DBA_DATA_FILES",
        "SYS.DBA_FREE_SPACE",
        "SYS.V_$SYSMETRIC",
        "SYS.V_$SGA",
        "SYS.V_$PGASTAT",
        "SYS.V_$RECOVERY_FILE_DEST",
        "SYS.V_$DIAG_ALERT_EXT",
        "object_grant_count <> 22",
    ):
        if required_text not in oracle_grant_script:
            raise RuntimeError(f"Oracle完整授权脚本缺少约束：{required_text}")
    for forbidden_text in (
        "CREATE USER",
        "GRANT DBA TO",
        "GRANT SELECT ANY TABLE TO",
        "GRANT SELECT_CATALOG_ROLE TO",
    ):
        if forbidden_text in oracle_grant_script:
            raise RuntimeError(f"Oracle完整授权脚本包含禁用内容：{forbidden_text}")
    oracle_catalog = json.loads(
        (
            ROOT
            / "services/aiops_agent/src/aiops_agent/diagnostics/catalog/oracle/manifest.json"
        ).read_text(encoding="utf-8")
    )
    catalog_privileges = {
        privilege
        for tool in oracle_catalog["tools"]
        for privilege in tool.get("required_privileges", [])
    }
    grant_pattern = re.compile(
        r"GRANT\s+SELECT\s+ON\s+SYS\.([A-Z0-9_$]+)\s+TO\s+kbot_monitor\s*;",
        re.IGNORECASE,
    )
    create_grants = {
        value.upper() for value in grant_pattern.findall(oracle_user_script)
    }
    existing_user_grants = {
        value.upper() for value in grant_pattern.findall(oracle_grant_script)
    }
    for script_name, granted in (
        ("Oracle建用户脚本", create_grants),
        ("Oracle完整授权脚本", existing_user_grants),
    ):
        missing = catalog_privileges - granted
        if missing:
            raise RuntimeError(
                f"{script_name}未覆盖当前诊断目录权限：{', '.join(sorted(missing))}"
            )
    if create_grants != existing_user_grants:
        raise RuntimeError("Oracle建用户脚本与完整授权脚本的对象权限不一致")
    datasource_config = (
        STACK / "configuration/grafana/provisioning/datasources/aiops.yml"
    ).read_text(encoding="utf-8")
    for datasource_uid in (
        "kbot-prometheus",
        "kbot-loki",
        "kbot-alertmanager",
    ):
        if f"uid: {datasource_uid}" not in datasource_config:
            raise RuntimeError(f"Grafana数据源缺少固定UID：{datasource_uid}")
    if "implementation: prometheus" not in datasource_config:
        raise RuntimeError("Grafana没有按Prometheus实现配置Alertmanager数据源")
    dashboard_provider = (
        STACK / "configuration/grafana/provisioning/dashboards/aiops.yml"
    ).read_text(encoding="utf-8")
    for required_text in (
        "folderUid: kbot-aiops",
        "editable: false",
        "path: /var/lib/grafana/dashboards",
    ):
        if required_text not in dashboard_provider:
            raise RuntimeError(f"Grafana Dashboard Provider缺少约束：{required_text}")

    dashboard_dir = STACK / "configuration/grafana/dashboards"
    expected_dashboards = {
        "oracle-overview.json": ("kbot-oracle-overview", "kbot-prometheus"),
        "oracle-storage.json": ("kbot-oracle-storage", "kbot-prometheus"),
        "oracle-alerts.json": ("kbot-oracle-alerts", "kbot-prometheus"),
        "oracle-alert-log.json": ("kbot-oracle-alert-log", "kbot-loki"),
        "host-overview.json": ("kbot-host-overview", "kbot-prometheus"),
    }
    actual_dashboards = {path.name for path in dashboard_dir.glob("*.json")}
    if actual_dashboards != set(expected_dashboards):
        raise RuntimeError("Grafana受控Dashboard清单与交付契约不一致")
    dashboard_uids: set[str] = set()
    dashboard_promql: list[str] = []
    for file_name, (expected_uid, datasource_uid) in expected_dashboards.items():
        dashboard_path = dashboard_dir / file_name
        raw_dashboard = dashboard_path.read_text(encoding="utf-8")
        dashboard = json.loads(raw_dashboard)
        if dashboard.get("uid") != expected_uid:
            raise RuntimeError(f"Dashboard UID不稳定：{file_name}")
        if expected_uid in dashboard_uids:
            raise RuntimeError(f"Dashboard UID重复：{expected_uid}")
        dashboard_uids.add(expected_uid)
        if not dashboard.get("panels") or len(dashboard["panels"]) < 4:
            raise RuntimeError(f"Dashboard缺少可用面板：{file_name}")
        variables = {
            item.get("name") for item in dashboard.get("templating", {}).get("list", [])
        }
        if "target_key" not in variables:
            raise RuntimeError(f"Dashboard缺少target_key目标切换：{file_name}")
        if datasource_uid not in raw_dashboard:
            raise RuntimeError(f"Dashboard没有引用固定数据源：{file_name}")
        for panel in dashboard["panels"]:
            panel_datasource = panel.get("datasource", {}).get("uid")
            for target in panel.get("targets", []):
                target_datasource = target.get("datasource", {}).get(
                    "uid", panel_datasource
                )
                expression = target.get("expr")
                if target_datasource == "kbot-prometheus" and expression:
                    dashboard_promql.append(expression.replace("$target_key", ".*"))
        for forbidden_text in ("CHANGE_ME", "140.238.44.208", "welcome1"):
            if forbidden_text in raw_dashboard:
                raise RuntimeError(f"Dashboard包含环境专属或敏感内容：{file_name}")
    if shutil.which("promtool"):
        with tempfile.TemporaryDirectory(prefix="kbot-dashboard-promql-") as temporary:
            rule_file = Path(temporary) / "dashboard-promql.yml"
            rule_lines = ["groups:", "  - name: kbot-dashboard-promql", "    rules:"]
            for index, expression in enumerate(dashboard_promql, start=1):
                rule_lines.extend(
                    [
                        f"      - record: kbot_dashboard_query_{index}",
                        f"        expr: {json.dumps(expression)}",
                    ]
                )
            rule_file.write_text("\n".join(rule_lines) + "\n", encoding="utf-8")
            subprocess.run(["promtool", "check", "rules", str(rule_file)], check=True)
    print(
        "AIOps观测栈检查通过：角色、Compose、Secret、多数据库目标、"
        "签名桥和Grafana看板配置有效"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
