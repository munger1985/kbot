"""OEM 完整原生目录的有界读取；续页只提取固定端点的页令牌。"""

import re
from urllib.parse import parse_qs, urlsplit

from .oem_native import _collection, _fail, _hash, _name, metric_selector
from aiops_agent.contracts.evidence import DiscoveredMetric


async def read_pages(adapter, suffix, *, params, max_bytes, max_pages=32, validate_payload=None):
    items, hashes, seen = [], [], set()
    frozen_params = dict(params)
    page_size = params.get("limit")
    for _ in range(max_pages):
        payload, digest = await adapter._native_get(suffix, params=params, max_bytes=max_bytes)
        page, more = _collection(payload)
        if validate_payload is not None:
            validate_payload(payload)
        if page_size is not None and len(page) > page_size:
            _fail()
        items.extend(page)
        hashes.append(digest)
        if not more:
            return items, False, _hash(*hashes)
        link = payload["links"]["next"]
        href = link.get("href") if isinstance(link, dict) else None
        if not isinstance(href, str) or len(href) > 16384:
            _fail("SOURCE_PAGINATION_INVALID")
        parsed = urlsplit(href)
        try:
            query = parse_qs(parsed.query, strict_parsing=True)
        except ValueError:
            _fail("SOURCE_PAGINATION_INVALID")
        if (parsed.scheme or parsed.netloc or parsed.fragment
            or parsed.path != urlsplit(adapter._native_api()).path + suffix
            or "page" not in query or any(len(values) != 1 for values in query.values())
            or set(query) - {"page"} - set(frozen_params)
            or any(values[0] != str(frozen_params[name]) for name, values in query.items() if name != "page")):
            _fail("SOURCE_PAGINATION_INVALID")
        token = query["page"][0]
        if not re.fullmatch(r"[A-Za-z0-9_+/=-]{1,8192}", token) or token in seen:
            _fail("SOURCE_PAGINATION_INVALID")
        seen.add(token)
        params = {"page": token} if set(query) == {"page"} else {**frozen_params, "page": token}
    return items, True, _hash(*hashes)


def catalog_entries(groups):
    entries, seen = [], set()
    for group in groups:
        group_name = _name(group.get("name"))
        keys, metrics = group.get("keys", []), group.get("metrics")
        if (not isinstance(keys, list) or len(keys) > 16 or not isinstance(metrics, list)
            or any(not isinstance(key, dict) for key in keys)):
            _fail()
        names = tuple(_name(key.get("name")) for key in keys)
        if len(set(names)) != len(names):
            _fail()
        for metric in metrics:
            if not isinstance(metric, dict) or not isinstance(metric.get("dataType"), str):
                _fail()
            selector = metric_selector(group_name, metric.get("name"))
            if len(selector) > 256:
                _fail()
            if selector in seen:
                _fail("SOURCE_ITEM_AMBIGUOUS")
            seen.add(selector)
            unit = metric.get("unitDisplayName")
            if unit is not None and (not isinstance(unit, str) or len(unit) > 64):
                _fail()
            entries.append(DiscoveredMetric(
                name=selector, label_names=names, metric_type="UNKNOWN", unit=unit,
                native_value_type=metric["dataType"][:64],
                help_text=str(metric.get("displayName") or selector)[:2000],
                metadata_state="VERIFIED", query_supported=metric["dataType"] == "NUMBER",
            ))
    return tuple(entries)


async def discover_catalog(adapter, request):
    adapter._remaining_response_bytes = request.max_response_bytes
    target, target_hash = await adapter._native_target(request)
    if (request.scope == "HOST") != (target["targetType"] == "host"):
        _fail("SOURCE_TARGET_MISMATCH")
    groups, more, digest = await read_pages(adapter,
        "/targets/" + request.source_locator_key + "/metricGroups",
        params={"include": "metrics", "limit": 100}, max_bytes=request.max_response_bytes)
    entries = catalog_entries(groups)
    return entries, more, _hash(target_hash, digest)
