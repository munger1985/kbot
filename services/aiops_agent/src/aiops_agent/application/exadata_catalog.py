"""从真实导出的定义列建立独立目录；监控与诊断共用，不修改默认基线。"""

import hashlib
import json
from importlib.resources import files


def load_catalog():
    root = files("aiops_agent").joinpath("resources/exadata_oem")
    raw = root.joinpath("metrics.v1.json").read_bytes()
    metrics = json.loads(raw)
    monitoring = json.loads(root.joinpath("monitoring_profiles.v1.json").read_text())
    diagnosis = json.loads(root.joinpath("diagnosis_profiles.v1.json").read_text())
    by_code = {item["code"]: item for item in metrics["metrics"]}
    if len(by_code) != len(metrics["metrics"]):
        raise ValueError("Exadata 指标编码重复")
    for profile in (*monitoring["profiles"], *diagnosis["profiles"]):
        if any(code not in by_code for code in profile["metric_codes"]):
            raise ValueError("Exadata 场景引用了未登记指标")
    digest = hashlib.sha256(raw + json.dumps([monitoring, diagnosis], sort_keys=True).encode()).hexdigest()
    return metrics["version"], digest, by_code, monitoring["profiles"], diagnosis["profiles"]
