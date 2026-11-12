"""从真实导出的定义列建立独立目录；监控与诊断共用，不修改默认基线。"""

import hashlib
import json
from importlib.resources import files

from platform_core.contracts.aiops.exadata import ExadataStatusDefinition


def load_catalog():
    root = files("aiops_agent").joinpath("resources/exadata_oem")
    raw = root.joinpath("metrics.v1.json").read_bytes()
    metrics = json.loads(raw)
    devices_raw = root.joinpath("device_metrics.v1.json").read_bytes()
    metrics["metrics"].extend(json.loads(devices_raw)["metrics"])
    monitoring = json.loads(root.joinpath("monitoring_profiles.v1.json").read_text())
    diagnosis = json.loads(root.joinpath("diagnosis_profiles.v1.json").read_text())
    status_raw = root.joinpath("status.v1.json").read_bytes()
    statuses = json.loads(status_raw)
    by_code = {item["code"]: item for item in metrics["metrics"]}
    if len(by_code) != len(metrics["metrics"]):
        raise ValueError("Exadata 指标编码重复")
    reviewed = set()
    for group in statuses["groups"]:
        status = ExadataStatusDefinition.model_validate(group["status"]).model_dump(mode="json")
        for code in group["metric_codes"]:
            if code not in by_code or code in reviewed:
                raise ValueError("Exadata 状态目录引用未知或重复的指标")
            reviewed.add(code)
            by_code[code]["status"] = status
    for profile in (*monitoring["profiles"], *diagnosis["profiles"]):
        if any(code not in by_code for code in profile["metric_codes"]):
            raise ValueError("Exadata 场景引用了未登记指标")
    digest = hashlib.sha256(raw + devices_raw + status_raw + json.dumps([monitoring, diagnosis], sort_keys=True).encode()).hexdigest()
    return metrics["version"], digest, by_code, monitoring["profiles"], diagnosis["profiles"]
