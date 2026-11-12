"""Exadata 历史基线的有界用例，缓存不能替代授权和在线成员复核。"""

import asyncio
from datetime import UTC, datetime, timedelta

from platform_core.contracts.aiops.exadata import ExadataBaseline, ExadataSample
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeBaseline
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.ports.exadata import ExadataMetricRead
from .errors import validation_failed, state_conflict
from .monitoring_oem_overview import baseline_series
from .exadata_analysis import comparisons
from .exadata_catalog import guid_rest_key_alternatives


def baseline_dimensions_match(definition, latest, series):
    """GUID提升键可省略，但快照、历史及缓存必须采用同一种维度表示。"""
    allowed = {frozenset(definition["keys"]),
        *(frozenset(keys) for keys in guid_rest_key_alternatives(definition, definition["target_type"]))}
    current = {frozenset(labels) for labels in latest["dimensions"]}
    historical = {frozenset(item["dimensions"]) for item in series}
    return (len(current) <= 1 and len(historical) <= 1 and (current | historical) <= allowed
        and (not current or not historical or current == historical))


class ExadataBaselineMixin:
    async def baseline(self, *, scope, system_id, resource_id, source_id, query):
        definition = self.metrics.get(query.metric_code)
        if definition is None or not definition.get("baseline_supported"):
            raise validation_failed("此 Exadata 指标未经均值比较审核；状态、计数和阈值不求均值")
        frozen, adapter = await self._snapshot(scope, system_id, resource_id, source_id)
        members = [frozen["inventory"]["root"], *frozen["inventory"]["members"]]
        member = next((item for item in members if item["target_id"] == query.component_id
                       and item["target_type"] == definition["target_type"]), None)
        if member is None:
            raise validation_failed("所选组件/指标不属于已确认的 Exadata 整机")
        now = datetime.now(UTC)
        days = 7 if query.period == "1w" else 30
        # 基线与用于比较的最新值各计一次读取；页面主快照不等待本接口。
        request = ExadataMetricRead(source_locator_key=query.component_id, window_start=now-timedelta(days=days),
            window_end=now, baseline_days=days)
        key = (str(scope.domain_id), str(getattr(scope, "tenant_id", None)), str(getattr(scope, "actor_id", None)),
               system_id, resource_id, source_id, query.component_id, query.metric_code, query.period, self.digest)
        async with asyncio.timeout(55):
            await self._validate_live_inventory(adapter, frozen)
            latest = await adapter.exadata_latest(ExadataMetricRead(source_locator_key=query.component_id), definition, member["target_type"])
            samples = tuple(ExadataSample(component_id=query.component_id, metric_code=query.metric_code,
                selector=definition["selector"], observed_at=when, value=value, dimensions=labels,
                unit=latest["unit"], native_value_type=latest["native_value_type"], response_sha256=latest["response_sha256"],
                quality="TRUNCATED" if latest["truncated"] else "EMPTY" if value in (None, "") else
                    "STALE" if (datetime.now(UTC)-when).total_seconds() > definition["max_age_seconds"] else "OBSERVED")
                for (when, value), labels in zip(latest["points"], latest["dimensions"], strict=True))
            cached = self._baseline_cache.get(key)
            if (cached and cached[0] > now and cached[1] == frozen and cached[3] == latest["unit"]
                    and baseline_dimensions_match(definition, latest,
                        ({"dimensions": row.dimensions} for row in cached[2].series))):
                result = cached[2]
            else:
                try:
                    # 失败只影响基线；不吞掉最终成员或绑定变化错误。
                    async with asyncio.timeout(30):
                        summary = await adapter.exadata_baseline(request, definition, member["target_type"])
                    if summary["unit"] != latest["unit"] or not baseline_dimensions_match(
                            definition, latest, summary["series"]):
                        raise DiagnosticSourceAdapterError("SOURCE_METADATA_CHANGED", "OEM 基线元数据发生变化")
                    result = MonitoringNativeBaseline(period=query.period, start=request.window_start, end=now,
                        series=baseline_series(summary), truncated=summary["truncated"], response_sha256=summary["response_sha256"],
                        unavailable_reason="历史结果截断，不生成不完整均值。" if summary["truncated"] else
                            None if summary["series"] else "基线窗口内没有有效留存采样。")
                    if len(self._baseline_cache) >= 64:
                        self._baseline_cache.clear()
                    self._baseline_cache[key] = (now+timedelta(minutes=5), frozen, result, latest["unit"])
                except (DiagnosticSourceAdapterError, TimeoutError) as exc:
                    code = getattr(exc, "code", "SOURCE_TIMEOUT")
                    result = MonitoringNativeBaseline(period=query.period, start=request.window_start, end=now,
                        unavailable_reason=(
                            "OEM 原厂 EMCLI 此读取通道最多支持 7 天历史，无法计算 30 天均值；未以 7 天替代。"
                            if code == "SOURCE_BASELINE_WINDOW_UNSUPPORTED" else
                            f"OEM 历史基线读取未完成（{code}）；当前快照不受影响，可重试。"))
            await self._validate_live_inventory(adapter, frozen)
        current, _ = await self._snapshot(scope, system_id, resource_id, source_id)
        if current != frozen:
            raise state_conflict("基线查询期间资源、来源或绑定版本变化，请重试")
        return ExadataBaseline(resource_id=resource_id, component_id=query.component_id, metric_code=query.metric_code,
            binding_id=frozen["binding_id"], binding_version=frozen["binding_version"], source_version=frozen["source_version"],
            catalog_sha256=self.digest, captured_at=datetime.now(UTC), unit=latest["unit"], baseline=result,
            current_samples=samples, comparisons=comparisons(samples, result))
