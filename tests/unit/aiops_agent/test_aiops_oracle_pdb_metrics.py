from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from importlib.machinery import SourceFileLoader
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
STACK = ROOT / "scripts/deployment/aiops_observability"
BRAND = "kbot"


def _stack_module():
    loader = SourceFileLoader(f"{BRAND}_pdb_stack", str(ROOT / "scripts/aiops-stack"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics():
    default = tomllib.loads((STACK / "configuration/oracle/default-metrics.toml").read_text())
    custom = yaml.safe_load((STACK / "configuration/oracle/custom-metrics.yaml").read_text())
    return (
        {metric["context"]: metric for metric in default["metric"]},
        {metric["context"]: metric for metric in custom["metrics"]},
    )


def test_only_optional_default_sources_accept_empty_results():
    default, _ = _metrics()
    assert set(default) == {
        "sessions", "resource", "asm_diskgroup", "activity", "process", "wait_time", "tablespace",
    }
    assert {name for name, metric in default.items() if metric.get("ignorezeroresult")} == {
        "resource", "asm_diskgroup", "wait_time",
    }
    assert default["resource"]["labels"] == ["resource_name"]
    assert default["wait_time"]["fieldtoappend"] == "wait_class"
    assert "v$resource_limit" in default["resource"]["request"]
    assert "v$waitclassmetric" in default["wait_time"]["request"]


def test_cpu_fallback_uses_current_container_counter_and_host_capacity():
    _, custom = _metrics()
    cpu = custom[f"{BRAND}_cpu"]
    assert cpu["ignorezeroresult"] is True
    assert "end_time >= SYSDATE - 2 / 1440" in cpu["request"]
    fallback = custom[f"{BRAND}_cpu_time"]
    assert fallback["metricstype"]["seconds_total"] == "counter"
    assert "v$sys_time_model" in fallback["request"]
    assert "t.value / 1000000" in fallback["request"]
    assert "t.stat_name = 'DB CPU'" in fallback["request"]
    assert "o.stat_name = 'NUM_CPUS'" in fallback["request"]
    assert "o.value > 0" in fallback["request"]
    assert "'cpu_count'" not in fallback["request"].lower()
    assert not fallback.get("ignorezeroresult")
    assert "nvl" not in fallback["request"].lower()


def test_wait_counter_and_source_coverage_do_not_fabricate_usage():
    _, custom = _metrics()
    wait = custom[f"{BRAND}_wait"]
    assert wait["labels"] == ["wait_class"]
    assert wait["metricstype"] == {"seconds_total": "counter"}
    assert "SUM(time_waited_micro) / 1000000" in wait["request"]
    assert "wait_class <> 'Idle'" in wait["request"]
    coverage = custom[f"{BRAND}_source"]
    assert coverage["labels"] == ["source"]
    assert list(coverage["metricsdesc"]) == ["row_count"]
    assert coverage["request"].count("COUNT(*) AS row_count") == 3
    assert not coverage.get("ignorezeroresult")


def test_cpu_rule_is_target_scoped_and_missing_data_still_alerts():
    groups = yaml.safe_load(_stack_module()._prometheus_rules())["groups"]
    rules = {rule.get("record", rule.get("alert")): rule for group in groups for rule in group["rules"]}
    cpu = rules[f"{BRAND}_db_cpu_utilization_percent"]["expr"]
    assert "or on (instance, target_key)" in cpu
    assert f"rate(oracledb_{BRAND}_cpu_time_seconds_total" in cpu
    assert f'oracledb_{BRAND}_cpu_time_host_cpu_count{{job="oracle"}} > 0' in cpu
    assert "vector(0)" not in cpu
    assert rules["OracleExporterScrapeFailed"]["expr"] == 'oracledb_exporter_last_scrape_error{job="oracle"} > 0'
    missing = rules["OracleCpuMetricUnavailable"]
    assert missing["for"] == "2m"
    assert missing["labels"]["metric_code"] == "db.cpu.utilization"
    assert "unless on (instance, target_key)" in missing["expr"]


def test_default_metric_changes_trigger_exporter_recreation(tmp_path, monkeypatch):
    module = _stack_module()
    config = tmp_path / "aiops-stack.ini"
    config.write_text(
        "[deployment]\ndeployment_id = pdb-test\nrole = all-in-one\nlocal_access = false\n"
        "[metrics]\nenabled = true\n"
        "[oracle:pdb-test]\nenabled = true\nversion = 19c\nhost = db.internal\n"
        "service = ORCLPDB1\nusername = test_monitor\npassword = test-only\n"
    )
    config.chmod(0o600)
    settings = module._load_settings(config)
    asset_root = tmp_path / "assets"
    oracle = asset_root / "configuration/oracle"
    oracle.mkdir(parents=True)
    for name in ["default-metrics.toml", "custom-metrics.yaml"]:
        (oracle / name).write_bytes((STACK / "configuration/oracle" / name).read_bytes())
    monkeypatch.setattr(module, "STACK_ROOT", asset_root)
    first, _ = module._database_generated_compose(settings)
    default = oracle / "default-metrics.toml"
    default.write_text(default.read_text() + "\n# 回归测试：采集定义已变化\n")
    second, _ = module._database_generated_compose(settings)
    key = "KBOT_DEFAULT_METRICS_REVISION"
    service = "oracle-pdb-test-exporter"
    assert first["services"][service]["environment"][key] != second["services"][service]["environment"][key]


def promtool_suite(rule_file: str) -> dict:
    """生成可直接交给promtool的数值、重置、隔离和缺失告警回归用例。"""
    target = 'instance="pdb-a",job="oracle",target_key="pdb-a"'
    other = 'instance="pdb-b",job="oracle",target_key="pdb-b"'
    cpu = f"oracledb_{BRAND}_cpu_utilization_percent"
    total = f"oracledb_{BRAND}_cpu_time_seconds_total"
    cpus = f"oracledb_{BRAND}_cpu_time_host_cpu_count"
    recorded = f"{BRAND}_db_cpu_utilization_percent"
    labels = f'{recorded}{{instance="pdb-a",job="oracle",target_key="pdb-a"}}'

    def case(name, series, value):
        return {
            "name": name,
            "interval": "15s",
            "input_series": [{"series": key, "values": values} for key, values in series],
            "promql_expr_test": [{
                "expr": f'{recorded}{{instance="pdb-a"}}',
                "eval_time": "2m",
                "exp_samples": [] if value is None else [{"labels": labels, "value": value}],
            }],
        }

    tests = [
        case("原生窗口优先于累计采样", [
            (f"{cpu}{{{target}}}", "7+0x12"),
            (f"{total}{{{target}}}", "0+60x12"),
            (f"{cpus}{{{target}}}", "8+0x12"),
        ], 7),
        case("PDB无窗口时按主机容量计算百分比", [
            (f"{total}{{{target}}}", "0+60x12"),
            (f"{cpus}{{{target}}}", "8+0x12"),
        ], 50),
        case("真实零CPU不被当作无数据", [
            (f"{total}{{{target}}}", "0+0x12"),
            (f"{cpus}{{{target}}}", "8+0x12"),
        ], 0),
        case("无有效CPU分母不补零", [
            (f"{total}{{{target}}}", "0+60x12"),
            (f"{cpus}{{{target}}}", "0+0x12"),
        ], None),
        case("不同目标的CPU分母不可串用", [
            (f"{total}{{{target}}}", "0+60x12"),
            (f"{cpus}{{{other}}}", "8+0x12"),
        ], None),
        case("计数器重置后仍正确计算", [
            (f"{total}{{{target}}}", "0 60 120 180 240 300 0 60 120 180 240 300 360"),
            (f"{cpus}{{{target}}}", "8+0x12"),
        ], 100 / 3),
    ]
    tests.append({
        "name": "数据库在线但CPU指标缺失必须告警",
        "interval": "15s",
        "input_series": [{"series": f"oracledb_up{{{target}}}", "values": "1+0x16"}],
        "alert_rule_test": [{
            "eval_time": "3m",
            "alertname": "OracleCpuMetricUnavailable",
            "exp_alerts": [{
                "exp_labels": {
                    "instance": "pdb-a", "job": "oracle", "target_key": "pdb-a",
                    "severity": "warning", "event_class": "diagnostic_source.metric_missing",
                    "metric_code": "db.cpu.utilization", "diagnostic_window_seconds": "120",
                },
                "exp_annotations": {
                    "summary": "Oracle CPU指标缺失",
                    "description": "pdb-a数据库可连接，但原生CPU窗口和累计CPU采样均未产生有效使用率，不能据此判断CPU健康。",
                },
            }],
        }],
    })
    return {
        "rule_files": [rule_file], "evaluation_interval": "15s",
        "fuzzy_compare": True, "tests": tests,
    }


if __name__ == "__main__":
    print(json.dumps(promtool_suite(sys.argv[1]), ensure_ascii=False))
