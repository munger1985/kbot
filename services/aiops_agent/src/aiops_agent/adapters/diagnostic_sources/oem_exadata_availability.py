"""读取 OEM 目标的当前可用性，不把 NO_DATA Response 集合伪装成指标采样。"""

from datetime import UTC, datetime

from .oem_native import _fail


async def query_availability(adapter, request, definition, target_type):
    if (target_type != "host" or definition["code"] != "host.OEMTarget.status"
            or definition["selector"] != "OEMTarget/status" or definition["keys"]):
        _fail("SOURCE_QUERY_INVALID")
    adapter._remaining_response_bytes = request.max_response_bytes
    target, digest = await adapter._native_target(request)
    if target["targetType"] != target_type:
        _fail("SOURCE_TARGET_MISMATCH")
    status = target.get("status")
    if status is not None and (not isinstance(status, str) or not 1 <= len(status) <= 64
                               or any(ord(char) < 32 or ord(char) == 127 for char in status)):
        _fail("SOURCE_RESPONSE_INVALID")
    # 此时间是当前状态接口的取证时间，不冒充 Response 指标的设备采集时间。
    captured_at = datetime.now(UTC)
    return {"points": ((captured_at, status),) if status else (),
        "dimensions": ({},) if status else (), "unit": None,
        "native_value_type": "STRING", "captured_at": captured_at,
        "truncated": False, "response_sha256": digest}
