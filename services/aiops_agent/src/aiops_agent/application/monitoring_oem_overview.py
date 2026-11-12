"""OEM 面板精确匹配和历史采样均值，不猜测插件名称或跨维度聚合。"""

from decimal import Decimal, localcontext
from importlib.resources import files
import json

from platform_core.contracts.aiops.monitoring_native import (
    MonitoringNativeBaselineSeries, MonitoringNativePanel,
)


def overview_panels(scope, metrics, *, target_type=None):
    profile_key = "PDB" if scope == "DATABASE" and target_type == "oracle_pdb" else scope
    profile = json.loads(files("aiops_agent.resources").joinpath(
        "monitoring/oem_overview.v1.json").read_text(encoding="utf-8"))["profiles"][profile_key]
    indexed = {metric.name: metric for metric in metrics}
    panels = []
    for definition in profile:
        expected = definition["metric"]
        metric = indexed.get(expected["name"]) if expected else None
        reason = None
        if metric is None:
            reason = definition.get("unavailable_reason", "当前 OEM 插件未提供此总览指标；可在更多指标中核查。")
        elif (metric.native_value_type != expected["native_value_type"] or metric.unit != expected["unit"]
              or set(metric.dimensions) != set(expected["dimensions"])):
            reason = "指标类型、单位或维度与已审核面板不符，未自动套用。"
        if reason:
            metric = None
        panels.append(MonitoringNativePanel(
            panel_id=definition["panel_id"], title=definition["title"],
            description=definition["description"], metric=metric,
            baseline_supported=bool(metric and metric.native_value_type == "NUMBER" and definition.get("baseline", True)),
            ranking=definition.get("ranking", "DESC"),
            mode=definition.get("mode", "HISTORY"), unavailable_reason=reason))
    return tuple(panels)


def freshness_seconds(scope, metric, *, target_type=None):
    """审核展示时效，不冒充 OEM 在线采集调度；未审核指标不参与当前比较。"""
    panels = overview_panels(scope, [metric], target_type=target_type)
    if not any(panel.metric is not None for panel in panels):
        return None
    profile_key = "PDB" if scope == "DATABASE" and target_type == "oracle_pdb" else scope
    policy = json.loads(files("aiops_agent.resources").joinpath(
        "monitoring/oem_overview.v1.json").read_text(encoding="utf-8"))
    return policy["current_freshness_seconds"][profile_key]


def baseline_series(summary):
    if summary["truncated"]:
        return ()
    with localcontext() as context:
        context.prec = 40
        return tuple(MonitoringNativeBaselineSeries(
            dimensions=series["dimensions"], sample_count=series["sample_count"],
            average=str(Decimal(series["total"]) / series["sample_count"]))
            for series in summary["series"])
