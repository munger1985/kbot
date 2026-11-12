"""OEM 整机成员发现，只接受官方显式包含关系，拒绝名称推断和不完整目录。"""

import asyncio
import hashlib
import json
import re
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from functools import wraps
import aiohttp
from .base import DiagnosticSourceAdapterError

from .oem_target_identity import normalize_target, target_error
from .oem_latest import query_latest
from .oem_exadata_inventory import verified_members
from aiops_agent.ports.diagnostic_source import CAPABILITY_METRIC_QUERY_RANGE

_GUID_COMPONENT_TYPES = frozenset({'oracle_si_hw_component_pdu', 'oracle_si_hw_component_server',
    'oracle_si_hw_component_switch', 'oracle_si_network_port_server', 'oracle_si_network_port_switch'})


def use_emcli(context, definition, target_type):
    """子组件仅有可核验的 GUID 接口，不送入原厂不支持的同名目标 CLI。"""
    return (context.config.get('exadata_metric_transport', 'OEM_REST') == 'EMCLI'
        and bool(definition['keys']) and target_type not in _GUID_COMPONENT_TYPES)


def inventory_digest(root, members):
    return hashlib.sha256(json.dumps({"root": root, "members": members},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exadata_read(operation):
    """仅归一专用接口的网络异常，不影响普通 OEM Adapter 行为。"""
    @wraps(operation)
    async def guarded(*args, **kwargs):
        try:
            return await operation(*args, **kwargs)
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise DiagnosticSourceAdapterError("SOURCE_UNREACHABLE", "OEM Exadata 读取超时或连接失败", retryable=True) from exc
        except ValueError as exc:
            raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "OEM Exadata 响应或分页无效") from exc
    return guarded


def exadata_metric_read(operation):
    """仅区分专用指标查询的 HTTP 5xx；成员发现及普通 OEM 的错误合同不变。"""
    @wraps(operation)
    async def guarded(*args, **kwargs):
        try:
            return await operation(*args, **kwargs)
        except DiagnosticSourceAdapterError as exc:
            # 原生读取仅将 HTTP 5xx 转成此码；网络异常交给外层 exadata_read。
            if exc.code == "SOURCE_UNREACHABLE":
                raise DiagnosticSourceAdapterError("SOURCE_PROVIDER_ERROR",
                    "OEM 服务端返回 HTTP 5xx，读取失败；不能解释为没有采样", retryable=True) from exc
            raise
    return exadata_read(guarded)


class OEMExadata:
    @exadata_read
    async def exadata_inventory(self, root_id, *, component_manifest=()):
        if not re.fullmatch(r"[0-9a-f]{32}", root_id):
            target_error("SOURCE_TARGET_LOCATOR_INVALID")
        self._remaining_response_bytes = 4 * 1024 * 1024
        async with asyncio.timeout(30):
            payload, _ = await self._native_get("/targets/" + root_id, max_bytes=65536)
            root = normalize_target(payload, expected_guid=root_id)
            if root["targetType"] != "oracle_dbmachine":
                target_error("SOURCE_TARGET_MISMATCH")
            ordered = await verified_members(self, root)
            if component_manifest:
                from .oem_exadata_components import confirm_components
                ordered = await confirm_components(self, ordered, component_manifest)
            identity = {"target_id": root_id, "target_type": "oracle_dbmachine",
                "display_name": root["targetName"]}
            return identity, ordered, inventory_digest(identity, ordered)

    @exadata_read
    async def discover_exadata(self):
        self._remaining_response_bytes = 4 * 1024 * 1024
        roots = []
        async with asyncio.timeout(45):
            seen = set()
            async for rows in self._target_pages("/targets"):
                for row in rows:
                    item = normalize_target(row)
                    if item["targetId"] in seen or len(seen) >= 2000:
                        target_error("SOURCE_DISCOVERY_LIMIT")
                    seen.add(item["targetId"])
                    if item["targetType"] == "oracle_dbmachine":
                        roots.append(item["targetId"])
                        if len(roots) > 16:
                            target_error("SOURCE_DISCOVERY_LIMIT")
            result = []
            for guid in sorted(roots):
                result.append(await self.exadata_inventory(guid))
            return result

    @exadata_metric_read
    async def exadata_latest(self, request, definition, target_type):
        if CAPABILITY_METRIC_QUERY_RANGE not in self.context.declared_capabilities:
            target_error("SOURCE_CAPABILITY_FORBIDDEN")
        if definition["semantics"] == "OEM_TARGET_AVAILABILITY":
            from .oem_exadata_availability import query_availability
            return await query_availability(self, request, definition, target_type)
        # 原厂 EMCLI 不支持部分无维度计算列；按审核键结构固定分流，不做错误回退。
        if use_emcli(self.context, definition, target_type):
            from .oem_exadata_emcli import query_emcli
            return await query_emcli(self, request, definition, target_type, history=False)
        return await query_latest(self, request, metric_name=definition["selector"], limit=256,
            expected_target_type=target_type, expected_keys=tuple(definition["keys"]),
            expected_value_type=definition["native_value_type"], allow_empty_number=True)

    @exadata_metric_read
    async def exadata_history(self, request, definition, target_type):
        if CAPABILITY_METRIC_QUERY_RANGE not in self.context.declared_capabilities:
            target_error("SOURCE_CAPABILITY_FORBIDDEN")
        if definition["semantics"] == "OEM_TARGET_AVAILABILITY":
            target_error("SOURCE_METRIC_HISTORY_UNSUPPORTED")
        if definition["native_value_type"] != "NUMBER":
            target_error("SOURCE_METRIC_HISTORY_UNSUPPORTED")
        if use_emcli(self.context, definition, target_type):
            from .oem_exadata_emcli import query_emcli
            return await query_emcli(self, request, definition, target_type, history=True)
        return await self.query_native_metric(request, metric_name=definition["selector"], limit=256,
            _expected_target_type=target_type, _expected_keys=tuple(definition["keys"]),
            _paginate=True, _resource_samples=True)

    @exadata_metric_read
    async def exadata_baseline(self, request, definition, target_type):
        if CAPABILITY_METRIC_QUERY_RANGE not in self.context.declared_capabilities:
            target_error("SOURCE_CAPABILITY_FORBIDDEN")
        if not definition.get("baseline_supported") or type(request.baseline_days) is not int or request.baseline_days not in (7, 30):
            target_error("SOURCE_QUERY_INVALID")
        if (request.window_start is None or request.window_end is None or request.window_start.tzinfo is None
                or request.window_end.tzinfo is None or request.window_end-request.window_start != timedelta(days=request.baseline_days)):
            target_error("SOURCE_QUERY_INVALID")
        if use_emcli(self.context, definition, target_type):
            from .oem_exadata_emcli import query_emcli
            return await query_emcli(self, request, definition, target_type, history=True, baseline=True)
        # OEM 按请求跨度切换留存层；计算列长窗可能没有聚合值。固定分片而非失败回退，
        # 六天片段避开七天边界的毫秒扩窗；投影仍按原始半开区间，边界不重复计数。
        self._remaining_response_bytes = request.max_response_bytes
        lower = request.window_start
        totals, hashes, unit, truncated = {}, [], None, False
        while lower < request.window_end:
            upper = min(lower + timedelta(days=6), request.window_end)
            summary = await self.query_native_metric(replace(request, window_start=lower, window_end=upper),
                metric_name=definition["selector"], limit=1000, _expected_target_type=target_type,
                _expected_keys=tuple(definition["keys"]), _paginate=True, _summary_only=True, _shared_budget=True)
            if hashes and summary["unit"] != unit:
                target_error("SOURCE_METADATA_CHANGED")
            unit = summary["unit"]
            hashes.append(summary["response_sha256"])
            truncated = truncated or summary["truncated"]
            with localcontext() as context:
                context.prec = 4096
                for series in summary["series"]:
                    identity = tuple(sorted(series["dimensions"].items()))
                    total, count = totals.get(identity, (Decimal(0), 0))
                    totals[identity] = (total + Decimal(series["total"]), count + series["sample_count"])
            lower = upper
        from .oem_native import _hash
        return {"series": tuple({"dimensions": dict(identity), "total": str(total), "sample_count": count}
                for identity, (total, count) in totals.items()), "unit": unit, "truncated": truncated,
            "response_sha256": _hash(*hashes)}
