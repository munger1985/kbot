"""区分巡检检查项的证据覆盖与固定工具的执行覆盖。"""

from __future__ import annotations

from typing import Any


def inspection_evidence_coverage(
    checks: tuple[dict[str, Any], ...],
    gaps: tuple[dict[str, Any], ...],
) -> dict[str, int | float]:
    """一份观测可覆盖多个检查项，但有缺口的观测不算完整证据。"""
    all_check_ids: set[str] = set()
    unsatisfied_check_ids: set[str] = set()
    check_ids_by_tool: dict[str, set[str]] = {}
    for check in checks:
        tool_id = str(check.get("tool_id") or "")
        check_ids = {
            str(value) for value in check.get("check_ids") or () if str(value)
        } or ({tool_id} if tool_id else set())
        all_check_ids.update(check_ids)
        check_ids_by_tool.setdefault(tool_id, set()).update(check_ids)
        if not check.get("evidence_refs") or check.get("check_status") in {
            "PARTIAL", "GAP", "MISSING",
        }:
            unsatisfied_check_ids.update(check_ids)
    for gap in gaps:
        unsatisfied_check_ids.update(
            check_ids_by_tool.get(str(gap.get("step_id") or ""), set())
        )
    covered = len(all_check_ids.difference(unsatisfied_check_ids))
    total = len(all_check_ids)
    executed = sum(bool(check.get("evidence_refs")) for check in checks)
    return {
        "evidence_covered_count": covered,
        "evidence_total_count": total,
        "evidence_coverage_percent": round(covered * 100 / total, 2) if total else 0.0,
        "tool_execution_covered_count": executed,
        "tool_execution_total_count": len(checks),
        "tool_execution_coverage_percent": (
            round(executed * 100 / len(checks), 2) if checks else 0.0
        ),
    }
