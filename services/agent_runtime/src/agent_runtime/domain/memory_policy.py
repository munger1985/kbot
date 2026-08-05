"""长期记忆的确定性低敏作用域策略。"""

import re
from collections.abc import Mapping
from typing import Any, Literal


MemoryScope = Literal["USER_AGENT", "USER_SHARED"]

AUTOMATICALLY_SHARED_MEMORY_KEYS = frozenset(
    {
        "user.preference.accessibility",
        "user.preference.response_format",
        "user.preference.response_language",
        "user.preference.timezone",
        "user.preference.unit_system",
    }
)

_RESPONSE_FORMATS = frozenset(
    {
        "CONCISE",
        "DETAILED",
        "MARKDOWN",
        "PLAIN_TEXT",
        "BULLET_POINTS",
        "TABLE_FIRST",
    }
)
_UNIT_SYSTEMS = frozenset({"METRIC", "IMPERIAL"})
_ACCESSIBILITY_PREFERENCES = frozenset(
    {"LARGE_TEXT", "SCREEN_READER", "HIGH_CONTRAST", "REDUCED_MOTION"}
)
_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_TIMEZONE_PATTERN = re.compile(
    r"^(?:UTC|[A-Za-z_]+(?:/[A-Za-z0-9_+.-]+)+)$"
)


def _single_string(value: Mapping[str, Any], field: str) -> str | None:
    if set(value) != {field}:
        return None
    item = value.get(field)
    return item.strip() if isinstance(item, str) and item.strip() else None


def _is_valid_shared_value(
    canonical_key: str,
    value: Mapping[str, Any],
) -> bool:
    if canonical_key == "user.preference.response_language":
        language = _single_string(value, "language")
        return bool(language and _LANGUAGE_PATTERN.fullmatch(language))
    if canonical_key == "user.preference.response_format":
        response_format = _single_string(value, "format")
        return bool(
            response_format and response_format.upper() in _RESPONSE_FORMATS
        )
    if canonical_key == "user.preference.timezone":
        timezone = _single_string(value, "timezone")
        return bool(timezone and _TIMEZONE_PATTERN.fullmatch(timezone))
    if canonical_key == "user.preference.unit_system":
        unit_system = _single_string(value, "unit_system")
        return bool(unit_system and unit_system.upper() in _UNIT_SYSTEMS)
    if canonical_key == "user.preference.accessibility":
        preferences = value.get("preferences")
        return bool(
            set(value) == {"preferences"}
            and isinstance(preferences, list)
            and preferences
            and all(
                isinstance(item, str)
                and item.upper() in _ACCESSIBILITY_PREFERENCES
                for item in preferences
            )
        )
    return False


def memory_scope(
    canonical_key: str,
    value: Mapping[str, Any],
) -> MemoryScope:
    """仅固定白名单键和严格低敏值可跨 Agent 自动共享。"""
    if (
        canonical_key in AUTOMATICALLY_SHARED_MEMORY_KEYS
        and _is_valid_shared_value(canonical_key, value)
    ):
        return "USER_SHARED"
    return "USER_AGENT"
