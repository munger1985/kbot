"""普通 OEM 有维度数值历史的显式 EMCLI 通道，不做失败后的隐式切换。"""
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from math import ceil

from aiops_agent.contracts.evidence import NativeMetricSamples
from .oem_exadata_emcli import parse_values
from .oem_native import _fail, _hash


async def query_native_emcli(adapter, request, *, target, target_hash, metadata_hash,
        metric_name, unit, keys, limit, summary_only):
    now = datetime.now(UTC)
    # 原厂仅按最近整天读取；7天边界允许一次调用的秒级耗时，不扩大业务投影窗口。
    age = (now - request.window_start).total_seconds()
    span = (request.window_end - request.window_start).total_seconds()
    if request.window_end > now or span > 7 * 86400 or age > 7 * 86400 + 60:
        _fail("SOURCE_HISTORY_WINDOW_UNSUPPORTED")
    client = adapter._exadata_emcli_client
    if client is None:
        _fail("SOURCE_EMCLI_UNAVAILABLE")
    definition = {"target_type": target["targetType"], "selector": metric_name,
        "native_value_type": "NUMBER", "keys": list(keys)}
    days = min(7, max(1, ceil(age / 86400)))
    payload, digest = await client.read_native(adapter.context, target, definition,
        history=True, max_bytes=request.max_response_bytes // 2, days=days)
    captured = datetime.now(UTC)
    points, dimensions, truncated = parse_values(payload, target=target, definition=definition,
        timezone=adapter.context.config["native_emcli_timezone"], captured_at=captured,
        start=request.window_start, end=request.window_end, limit=10000 if summary_only else limit,
        strict_history=True)
    kept = [(point, labels) for point, labels in zip(points, dimensions, strict=True)
        if request.window_start <= point[0] < request.window_end]
    response_hash = _hash(target_hash, metadata_hash, digest)
    if summary_only:
        totals = {}
        with localcontext() as context:
            context.prec = 4096
            for (_, value), labels in kept:
                if value in (None, ""):
                    continue
                identity = tuple(sorted(labels.items()))
                total, count = totals.get(identity, (Decimal(0), 0))
                totals[identity] = total + Decimal(value), count + 1
        return {"series": tuple({"dimensions": dict(identity), "total": str(total), "sample_count": count}
            for identity, (total, count) in totals.items()), "unit": unit, "truncated": truncated,
            "response_sha256": response_hash}
    return NativeMetricSamples(action_id=request.action_id, target_id=request.target_id,
        binding_id=request.binding_id, source_id=adapter.context.source_id, source_type="OEM",
        metric_name=metric_name, native_value_type="NUMBER", unit=unit,
        history_resolution="PROVIDER_MANAGED", sample_dimensions=tuple(labels for _, labels in kept),
        source_version=f"{adapter.context.adapter_version}:{adapter.context.config_version}",
        window_start=request.window_start, window_end=request.window_end, captured_at=captured,
        points=tuple(point for point, _ in kept), truncated=truncated, response_sha256=response_hash)
