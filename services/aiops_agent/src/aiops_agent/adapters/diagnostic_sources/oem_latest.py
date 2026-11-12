"""OEM 官方 latestData 快照；文本与数值保留真实采集时间，不冒充历史时序。"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

from .oem_catalog import read_pages
from .oem_native import _fail, _hash, _name, parse_metric_selector


async def query_latest(adapter, request, *, metric_name, limit=1000, expected_target_type=None,
                       expected_keys=None, expected_value_type=None, allow_empty_number=False):
    if type(limit) is not int or not 1 <= limit <= 1000:
        _fail("SOURCE_QUERY_INVALID")
    group, metric = parse_metric_selector(metric_name)
    adapter._remaining_response_bytes = request.max_response_bytes
    target, target_hash = await adapter._native_target(request)
    if expected_target_type is not None and target["targetType"] != expected_target_type:
        _fail("SOURCE_TARGET_MISMATCH")
    metadata, metadata_hash = await adapter._native_get(
        "/targets/" + request.source_locator_key + "/metricGroups/" + quote(group, safe=""),
        max_bytes=request.max_response_bytes // 4)
    if (not isinstance(metadata, dict) or metadata.get("name") != group
        or not isinstance(metadata.get("metrics"), list) or not isinstance(metadata.get("keys", []), list)):
        _fail()
    matches = [item for item in metadata["metrics"] if isinstance(item, dict) and item.get("name") == metric]
    if len(matches) != 1:
        _fail("SOURCE_ITEM_AMBIGUOUS")
    native_type, unit = matches[0].get("dataType"), matches[0].get("unitDisplayName")
    if native_type not in {"NUMBER", "STRING"}:
        _fail("SOURCE_METRIC_TYPE_MISMATCH")
    if unit is not None and (not isinstance(unit, str) or len(unit) > 64):
        _fail()
    keys = metadata.get("keys", [])
    if len(keys) > 16 or any(not isinstance(key, dict) for key in keys):
        _fail()
    names = tuple(_name(key.get("name")) for key in keys)
    if len(set(names)) != len(names):
        _fail()
    # Exadata 目录约束维度身份；OEM 元数据的字段顺序不代表维度身份。
    if expected_keys is not None and set(names) != set(expected_keys):
        _fail("SOURCE_METRIC_DIMENSION_MISMATCH")
    if expected_value_type is not None and native_type != expected_value_type:
        _fail("SOURCE_METRIC_TYPE_MISMATCH")
    captured_at = datetime.now(UTC)
    timestamps = []
    def validate_page(payload):
        if (payload.get("targetName") != target["targetName"]
            or payload.get("targetTypeName") != target["targetType"]
            or str(payload.get("targetId", "")).casefold() != request.source_locator_key.casefold()
            or payload.get("metricGroupName") != group):
            _fail("SOURCE_TARGET_MISMATCH")
        if not payload.get("items"):
            return
        try:
            when = datetime.fromisoformat(payload["timeCollected"].replace("Z", "+00:00"))
            if when.tzinfo is None or when > captured_at + timedelta(seconds=5):
                raise ValueError
        except (KeyError, ValueError, TypeError, AttributeError):
            _fail()
        if timestamps and timestamps[0] != when:
            _fail("SOURCE_COLLECTION_CHANGED")
        timestamps.append(when)
    rows, truncated, digest = await read_pages(adapter,
        "/targets/" + request.source_locator_key + "/metricGroups/" + quote(group, safe="") + "/latestData",
        params={"metricName": metric, "limit": 100}, max_bytes=request.max_response_bytes // 2,
        max_pages=16, validate_payload=validate_page)
    points, dimensions, seen = [], [], set()
    for row in rows:
        labels = {}
        for name in names:
            value = row.get(name)
            if not isinstance(value, str) or len(value) > 256:
                _fail()
            labels[name] = value
        identity = tuple(sorted(labels.items()))
        if identity in seen:
            _fail("SOURCE_ITEM_AMBIGUOUS")
        seen.add(identity)
        if metric not in row:
            continue
        value = row[metric]
        if value is not None:
            if native_type == "NUMBER" and type(value) is int:
                value = str(value)
            if not isinstance(value, str) or len(value) > 8192:
                _fail()
            if native_type == "NUMBER" and not (allow_empty_number and value == ""):
                try:
                    numeric = Decimal(value)
                    if not numeric.is_finite() or abs(numeric.as_tuple().exponent) > 1024:
                        raise ValueError
                except (ValueError, InvalidOperation):
                    _fail()
        if len(points) < limit:
            points.append((timestamps[0].astimezone(UTC), value))
            dimensions.append(labels)
        else:
            truncated = True
    return {"points": tuple(points), "dimensions": tuple(dimensions), "unit": unit, "native_value_type": native_type,
        "captured_at": captured_at, "truncated": truncated,
        "response_sha256": _hash(target_hash, metadata_hash, digest)}
