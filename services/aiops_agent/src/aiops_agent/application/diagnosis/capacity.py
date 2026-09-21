"""根据 Finding 与已确认 Target Fact 决定容量方案，不自动加文件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from platform_core.contracts.aiops import FindingCard, FindingType


CapacityKind = Literal["OBSERVE", "AUTOEXTEND", "RESIZE"]

CAPACITY_ADD_ACTION_IDS = frozenset(
    {
        "db.storage.datafile.add",
        "db.storage.tempfile.add",
    }
)
CAPACITY_AUTOEXTEND_ACTION_IDS = (
    "db.storage.datafile.autoextend",
    "db.storage.tempfile.autoextend",
)
CAPACITY_RESIZE_ACTION_IDS = (
    "db.storage.datafile.resize",
    "db.storage.tempfile.resize",
)
CAPACITY_GROW_ACTION_IDS = frozenset(
    CAPACITY_AUTOEXTEND_ACTION_IDS + CAPACITY_RESIZE_ACTION_IDS
)
CAPACITY_STORAGE_ACTION_IDS = CAPACITY_GROW_ACTION_IDS | CAPACITY_ADD_ACTION_IDS
_PLACEMENT_FACT_TYPES = frozenset({"ASM_DISKGROUP", "DATAFILE_PATH"})


@dataclass(frozen=True)
class CapacityDecision:
    tablespace_name: str
    kind: CapacityKind
    autoextend_headroom_mb: float | None
    allowed_action_ids: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tablespace_name": self.tablespace_name,
            "kind": self.kind,
            "autoextend_headroom_mb": self.autoextend_headroom_mb,
            "allowed_action_ids": list(self.allowed_action_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CapacityPlan:
    decisions: tuple[CapacityDecision, ...]
    allowed_action_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": [item.as_dict() for item in self.decisions],
            "allowed_action_ids": list(self.allowed_action_ids),
            "blocked_action_ids": sorted(CAPACITY_ADD_ACTION_IDS),
        }


def decide_capacity_actions(
    *,
    findings: Sequence[FindingCard] | Iterable[FindingCard],
    target_facts: Sequence[Mapping[str, Any]]
    | Iterable[Mapping[str, Any]]
    | None,
) -> CapacityPlan:
    """容量 Finding 只能观察 / AUTOEXTEND / RESIZE，永不 ADD DATAFILE。"""
    has_placement = _has_active_placement_fact(target_facts)
    decisions: list[CapacityDecision] = []
    allowed: list[str] = []
    seen_allowed: set[str] = set()
    for card in findings:
        if card.finding_type != FindingType.TABLESPACE:
            continue
        decision = _decide_tablespace(card, has_placement=has_placement)
        decisions.append(decision)
        for action_id in decision.allowed_action_ids:
            if (
                action_id in CAPACITY_ADD_ACTION_IDS
                or action_id in seen_allowed
            ):
                continue
            seen_allowed.add(action_id)
            allowed.append(action_id)
    return CapacityPlan(
        decisions=tuple(decisions),
        allowed_action_ids=tuple(allowed),
    )


def allows_capacity_action(plan: CapacityPlan, action_template_id: str) -> bool:
    """存储增长模板必须命中容量方案；加文件永远不允许。"""
    if action_template_id in CAPACITY_ADD_ACTION_IDS:
        return False
    if action_template_id not in CAPACITY_STORAGE_ACTION_IDS:
        return True
    return action_template_id in plan.allowed_action_ids


def _has_active_placement_fact(
    target_facts: Sequence[Mapping[str, Any]]
    | Iterable[Mapping[str, Any]]
    | None,
) -> bool:
    for item in target_facts or ():
        fact_type = str(item.get("fact_type") or "")
        status = str(item.get("status") or "ACTIVE")
        if fact_type in _PLACEMENT_FACT_TYPES and status == "ACTIVE":
            return True
    return False


def _decide_tablespace(
    card: FindingCard,
    *,
    has_placement: bool,
) -> CapacityDecision:
    fields = dict(card.fields or {})
    tablespace_name = str(
        fields.get("tablespace_name")
        or getattr(card.object_ref, "object_name", None)
        or ""
    )
    headroom = _autoextend_headroom_mb(fields)
    if not has_placement:
        return CapacityDecision(
            tablespace_name=tablespace_name,
            kind="OBSERVE",
            autoextend_headroom_mb=headroom,
            allowed_action_ids=(),
            reason=(
                "缺少已确认的 ASM 磁盘组或数据文件路径事实，"
                "容量方案只能观察，不加文件。"
            ),
        )
    if headroom is not None and headroom > 0:
        return CapacityDecision(
            tablespace_name=tablespace_name,
            kind="AUTOEXTEND",
            autoextend_headroom_mb=headroom,
            allowed_action_ids=CAPACITY_AUTOEXTEND_ACTION_IDS,
            reason="已确认放置事实，且自动扩展仍有余量，方案为 AUTOEXTEND。",
        )
    if headroom is None:
        reason = "已确认放置事实，但无法确认自动扩展余量，方案为 RESIZE。"
    else:
        reason = "已确认放置事实，但自动扩展没有余量，方案为 RESIZE。"
    return CapacityDecision(
        tablespace_name=tablespace_name,
        kind="RESIZE",
        autoextend_headroom_mb=headroom,
        allowed_action_ids=CAPACITY_RESIZE_ACTION_IDS,
        reason=reason,
    )


def _autoextend_headroom_mb(fields: Mapping[str, Any]) -> float | None:
    maximum_mb = _as_number(fields.get("maximum_mb"))
    allocated_mb = _as_number(fields.get("allocated_mb"))
    if maximum_mb is not None and allocated_mb is not None:
        return maximum_mb - allocated_mb
    maximum_headroom_mb = _as_number(fields.get("maximum_headroom_mb"))
    free_mb = _as_number(fields.get("free_mb"))
    if maximum_headroom_mb is not None and free_mb is not None:
        return maximum_headroom_mb - free_mb
    return None


def _as_number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
