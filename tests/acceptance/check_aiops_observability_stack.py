#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import os
import json
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
        subprocess.run(
            [*stack._compose_command(settings), "config", "--quiet"],
            check=True,
        )
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
        if len(overrides) != 10:
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
        "SYS.V_$DIAG_ALERT_EXT",
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
    print("AIOps观测栈检查通过：角色、Compose、Secret、多数据库目标和签名桥配置有效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
