"""核验原厂 EMCLI 指标身份、具名维度和原始时间；不生成模拟历史。"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .oem_native import _fail, _hash, parse_metric_selector


def parse_values(payload, *, target, definition, timezone, captured_at, start=None, end=None, limit=256):
    if payload == {}:
        return (), (), False
    group, column = parse_metric_selector(definition["selector"])
    if (payload.get("Entity Type"), payload.get("Entity Name"), payload.get("Metric Group Name")) != (
            target["targetType"], target["targetName"], group):
        _fail("SOURCE_TARGET_MISMATCH")
    names = [payload.get(f"keyColumn_{index}_name", "") for index in range(1, 8)]
    active = [(index, name) for index, name in enumerate(names, 1) if name]
    if (any(not isinstance(name, str) for name in names) or len({name for _, name in active}) != len(active)
            or {name for _, name in active} != set(definition["keys"])):
        _fail("SOURCE_METRIC_DIMENSION_MISMATCH")
    zone = ZoneInfo(timezone)

    def stamp(raw):
        if not isinstance(raw, str) or not 1 <= len(raw) <= 64:
            _fail("SOURCE_RESPONSE_INVALID")
        try:
            value = datetime.fromisoformat(raw)
        except ValueError:
            _fail("SOURCE_RESPONSE_INVALID")
        if value.tzinfo is None:
            # 不在夏令时重复/不存在的本地时刻静默猜测一个 UTC 时间。
            if value.replace(tzinfo=zone, fold=0).utcoffset() != value.replace(tzinfo=zone, fold=1).utcoffset():
                _fail("SOURCE_TIMEZONE_AMBIGUOUS")
            value = value.replace(tzinfo=zone)
        return value.astimezone(UTC)

    columns = payload.get("data_columns")
    if not isinstance(columns, list) or len(columns) > 4096:
        _fail("SOURCE_RESPONSE_INVALID")
    values = {}
    latest = {}
    for item in columns:
        if not isinstance(item, dict) or item.get("Metric Column Name") != column:
            _fail("SOURCE_METRIC_MISMATCH")
        rows = item.get("keyValues")
        if not isinstance(rows, list) or len(rows) > 4096:
            _fail("SOURCE_RESPONSE_INVALID")
        for row in rows:
            if not isinstance(row, dict):
                _fail("SOURCE_RESPONSE_INVALID")
            labels = {name: row.get(f"key{index}") for index, name in active}
            if any(not isinstance(value, str) or len(value) > 256 for value in labels.values()):
                _fail("SOURCE_RESPONSE_INVALID")
            if any(row.get(f"key{index}") not in (None, "", " ") for index, name in enumerate(names, 1) if not name):
                _fail("SOURCE_METRIC_DIMENSION_MISMATCH")
            identity = tuple(sorted(labels.items()))
            samples = row.get("values")
            if not isinstance(samples, list) or len(samples) > 10000:
                _fail("SOURCE_RESPONSE_INVALID")
            for sample in samples:
                if not isinstance(sample, dict):
                    _fail("SOURCE_RESPONSE_INVALID")
                first, last = stamp(sample.get("First Collection Time")), stamp(sample.get("Last Collection Time"))
                if first > last or last > captured_at + timedelta(seconds=5):
                    _fail("SOURCE_RESPONSE_INVALID")
                value = sample.get("Value")
                if definition["native_value_type"] == "NUMBER" and type(value) is int:
                    value = str(value)
                if value is not None and (not isinstance(value, str) or len(value) > 8192):
                    _fail("SOURCE_RESPONSE_INVALID")
                if definition["native_value_type"] == "NUMBER" and value not in (None, ""):
                    try:
                        number = Decimal(value)
                        if not number.is_finite() or abs(number.as_tuple().exponent) > 1024:
                            raise ValueError
                    except (ValueError, InvalidOperation):
                        _fail("SOURCE_RESPONSE_INVALID")
                if start is not None:
                    if start <= last <= end:
                        values[(identity, last)] = (last, value, labels)
                elif identity not in latest or last >= latest[identity][0]:
                    # 原厂同一时间末条空值仍是空值，不能回填前一条正常值。
                    latest[identity] = (last, value, labels)
    ordered = sorted(values.values() if start is not None else latest.values(),
                     key=lambda item: (tuple(sorted(item[2].items())), item[0]))
    return tuple((when, value) for when, value, _ in ordered[:limit]), tuple(labels for _, _, labels in ordered[:limit]), len(ordered) > limit


async def query_emcli(adapter, request, definition, target_type, *, history):
    client = getattr(adapter, "_exadata_emcli_client", None)
    if client is None:
        _fail("SOURCE_EMCLI_UNAVAILABLE")
    if definition["target_type"] != target_type:
        _fail("SOURCE_TARGET_MISMATCH")
    if history and definition["native_value_type"] != "NUMBER":
        _fail("SOURCE_METRIC_HISTORY_UNSUPPORTED")
    if history and (request.window_start is None or request.window_end is None
            or request.window_start.tzinfo is None or request.window_end.tzinfo is None
            or not 0 < (request.window_end - request.window_start).total_seconds() <= 3600):
        _fail("SOURCE_QUERY_INVALID")
    adapter._remaining_response_bytes = request.max_response_bytes
    target, target_hash = await adapter._native_target(request)
    if target["targetType"] != target_type:
        _fail("SOURCE_TARGET_MISMATCH")
    group, column = parse_metric_selector(definition["selector"])
    metadata, metadata_hash = await adapter._native_get(
        "/targets/" + request.source_locator_key + "/metricGroups/" + quote(group, safe=""),
        max_bytes=request.max_response_bytes // 4)
    if (not isinstance(metadata, dict) or metadata.get("name") != group
            or not isinstance(metadata.get("metrics"), list) or not isinstance(metadata.get("keys", []), list)):
        _fail("SOURCE_RESPONSE_INVALID")
    matches = [item for item in metadata["metrics"] if isinstance(item, dict) and item.get("name") == column]
    if len(matches) != 1:
        _fail("SOURCE_ITEM_AMBIGUOUS")
    native_type, unit = matches[0].get("dataType"), matches[0].get("unitDisplayName")
    if native_type != definition["native_value_type"]:
        _fail("SOURCE_METRIC_TYPE_MISMATCH")
    if unit is not None and (not isinstance(unit, str) or len(unit) > 64):
        _fail("SOURCE_RESPONSE_INVALID")
    keys = metadata.get("keys", [])
    if (any(not isinstance(key, dict) or not isinstance(key.get("name"), str) for key in keys)
            or len({key["name"] for key in keys}) != len(keys)
            or {key["name"] for key in keys} != set(definition["keys"])):
        _fail("SOURCE_METRIC_DIMENSION_MISMATCH")
    payload, digest = await client.read(adapter.context, target, definition, history=history,
                                       max_bytes=request.max_response_bytes // 2)
    captured_at = datetime.now(UTC)
    points, dimensions, truncated = parse_values(payload, target=target, definition=definition,
        timezone=adapter.context.config["exadata_emcli_timezone"], captured_at=captured_at,
        start=request.window_start if history else None, end=request.window_end if history else None)
    return {"points": points, "dimensions": dimensions, "unit": unit, "native_value_type": native_type,
        "captured_at": captured_at, "truncated": truncated, "response_sha256": _hash(target_hash, metadata_hash, digest)}
