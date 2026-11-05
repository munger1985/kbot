"""OEM 目标身份解析；不同 OEM API 字段归一后仍严格核验 GUID。"""

import re

from .base import DiagnosticSourceAdapterError


def target_error(code="SOURCE_RESPONSE_INVALID"):
    raise DiagnosticSourceAdapterError(code, "OEM 目标目录或数据库身份无法核验")


def target_text(value, maximum=256):
    if (not isinstance(value, str) or not value or len(value) > maximum
        or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        target_error()
    return value


def normalize_target(payload, *, expected_guid=None):
    if not isinstance(payload, dict):
        target_error()
    modern = any(key in payload for key in ("id", "name", "typeName"))
    fields = ("id", "name", "typeName") if modern else ("targetId", "targetName", "targetType")
    guid = payload.get(fields[0], expected_guid if not modern else None)
    if not isinstance(guid, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", guid):
        target_error("SOURCE_TARGET_MISMATCH")
    if expected_guid is not None and guid.casefold() != expected_guid.casefold():
        target_error("SOURCE_TARGET_MISMATCH")
    name = target_text(payload.get(fields[1]))
    kind = target_text(payload.get(fields[2]), 64)
    for key, value in zip(("targetId", "targetName", "targetType"), (guid, name, kind)):
        if key in payload and (
            str(payload[key]).casefold() != value.casefold() if key == "targetId"
            else payload[key] != value
        ):
            target_error("SOURCE_TARGET_MISMATCH")
    return {**payload, "targetId": guid.lower(), "targetName": name, "targetType": kind}
