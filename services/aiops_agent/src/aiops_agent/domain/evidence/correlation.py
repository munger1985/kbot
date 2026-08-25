"""SignalEvent 到 Situation 的确定性关联规则。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


_CANONICAL_EVENT_CLASS = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class SituationCorrelationDecision:
    """一次可审计、可复现的 Situation 关联决定。"""

    canonical_event_class: str
    correlation_key: str
    correlation_hash: str
    correlation_version: str
    method: str
    detail: dict[str, Any]


def validate_event_class_map(mapping_overrides: dict[str, Any] | None) -> None:
    """校验显式跨来源事件类别映射，不猜测产品语义。"""
    if not mapping_overrides or "event_class_map" not in mapping_overrides:
        return
    event_class_map = mapping_overrides["event_class_map"]
    if not isinstance(event_class_map, dict) or len(event_class_map) > 256:
        raise ValueError("event_class_map 必须是最多 256 项的对象")
    for source_class, canonical_class in event_class_map.items():
        if not isinstance(source_class, str) or not source_class.strip():
            raise ValueError("event_class_map 来源类别必须是非空字符串")
        if not isinstance(canonical_class, str) or not _CANONICAL_EVENT_CLASS.fullmatch(
            canonical_class
        ):
            raise ValueError(
                "event_class_map 目标类别必须是规范的小写事件类别"
            )


def correlate_signal_event(
    *,
    target_id: str,
    source_event_class: str,
    mapping_overrides: dict[str, Any] | None,
) -> SituationCorrelationDecision:
    """按 Target 和规范事件类别关联，允许 Binding 提供显式语义映射。"""
    validate_event_class_map(mapping_overrides)
    event_class_map = (
        dict(mapping_overrides.get("event_class_map", {}))
        if mapping_overrides
        else {}
    )
    mapped = event_class_map.get(source_event_class)
    if mapped is None:
        canonical = re.sub(
            r"[^a-z0-9._-]+", ".", source_event_class.strip().lower()
        ).strip("._-")
        if not canonical:
            raise ValueError("事件类别无法规范化")
        canonical = canonical[:128]
        method = "EXACT"
    else:
        canonical = mapped
        method = "RULE"
    correlation_key = f"{target_id}:{canonical}"
    return SituationCorrelationDecision(
        canonical_event_class=canonical,
        correlation_key=correlation_key[:256],
        correlation_hash=hashlib.sha256(
            correlation_key.encode("utf-8")
        ).hexdigest(),
        correlation_version="target-event-class.v1",
        method=method,
        detail={
            "source_event_class": source_event_class,
            "canonical_event_class": canonical,
            "mapping_applied": mapped is not None,
        },
    )
