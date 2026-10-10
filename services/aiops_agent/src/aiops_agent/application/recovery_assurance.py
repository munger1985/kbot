"""把恢复目标与演练记录冻结为可诊断、可追溯的证据快照。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aiops_agent.application.configuration.recovery_service import (
    recovery_drill_view,
    recovery_profile_view,
)


_ASSURANCE_RANK = {
    "BACKUP_METADATA": 1,
    "RESTORE_VALIDATE": 2,
    "DATABASE_OPEN": 3,
    "APPLICATION_VALIDATED": 4,
}


async def build_recovery_assurance_snapshot(
    *, recovery_repository, target, now: datetime | None = None
) -> dict:
    """保留最新尝试与合格成功；较早成功不能遮蔽最新失败。"""
    captured_at = now or datetime.now(UTC)
    profile = await recovery_repository.get_active_profile(
        target_id=target.target_id, domain_id=int(target.domain_id)
    )
    drills = await recovery_repository.list_recent_drills(
        target_id=target.target_id, domain_id=int(target.domain_id), limit=100
    )
    latest_attempt = drills[0] if drills else None
    required_rank = _ASSURANCE_RANK.get(
        getattr(profile, "required_assurance_level", ""), 0
    )
    verified_passes = [
        row for row in drills if row.status == "VERIFIED" and row.result == "PASS"
    ]
    latest_success = next(
        (
            row
            for row in verified_passes
            if _ASSURANCE_RANK.get(row.assurance_level, 0) >= required_rank
        ),
        None,
    )
    restore_demonstrated = bool(
        latest_success
        and _ASSURANCE_RANK.get(latest_success.assurance_level, 0)
        >= _ASSURANCE_RANK["DATABASE_OPEN"]
    )
    stale = bool(
        profile
        and latest_success
        and latest_success.simulated_failure_at
        < captured_at - timedelta(days=int(profile.required_drill_interval_days))
    )
    profile_changed = bool(
        profile
        and latest_success
        and (
            latest_success.recovery_profile_id != profile.recovery_profile_id
            or int(latest_success.recovery_profile_version or 0)
            != int(profile.version_no)
        )
    )
    covered_sources = sorted({row.backup_source_type for row in verified_passes})
    required_sources = (
        list(profile.required_backup_source_types_json) if profile else []
    )
    uncovered_sources = sorted(set(required_sources) - set(covered_sources))
    gaps: list[dict[str, object]] = []
    if profile is None:
        gaps.extend(
            (
                {"code": "RPO_NOT_CONFIGURED", "detail": "Target 未配置业务 RPO"},
                {"code": "RTO_NOT_CONFIGURED", "detail": "Target 未配置业务 RTO"},
            )
        )
    if not restore_demonstrated:
        gaps.append(
            {
                "code": "RESTORE_NOT_DEMONSTRATED",
                "detail": "没有已审核且至少达到 DATABASE_OPEN 的成功恢复演练",
            }
        )
    if stale:
        gaps.append({"code": "DRILL_STALE", "detail": "最近合格成功演练已超过要求周期"})
    if profile_changed:
        gaps.append(
            {
                "code": "POLICY_CHANGED_SINCE_DRILL",
                "detail": "恢复目标版本在最近合格成功演练后已变更",
            }
        )
    if latest_attempt and latest_attempt.result == "FAIL":
        gaps.append({"code": "LATEST_DRILL_FAILED", "detail": "最近一次恢复演练失败"})
    if profile and latest_success:
        if latest_success.achieved_rpo_seconds is None:
            gaps.append({"code": "RPO_MEASUREMENT_MISSING", "detail": "合格演练缺少实测 RPO"})
        elif int(latest_success.achieved_rpo_seconds) > int(profile.rpo_seconds):
            gaps.append({
                "code": "RPO_TARGET_BREACHED",
                "detail": "恢复演练实测 RPO 超出业务目标",
                "target_seconds": int(profile.rpo_seconds),
                "achieved_seconds": int(latest_success.achieved_rpo_seconds),
            })
        if latest_success.achieved_rto_seconds is None:
            gaps.append({"code": "RTO_MEASUREMENT_MISSING", "detail": "合格演练缺少实测 RTO"})
        elif int(latest_success.achieved_rto_seconds) > int(profile.rto_seconds):
            gaps.append({
                "code": "RTO_TARGET_BREACHED",
                "detail": "恢复演练实测 RTO 超出业务目标",
                "target_seconds": int(profile.rto_seconds),
                "achieved_seconds": int(latest_success.achieved_rto_seconds),
            })
    if uncovered_sources:
        gaps.append({
            "code": "BACKUP_SOURCE_NOT_VERIFIED",
            "detail": "必需备份来源尚无已审核成功演练覆盖",
            "backup_source_types": uncovered_sources,
        })
    gap_codes = {str(item["code"]) for item in gaps}
    assurance_status = (
        "TARGET_NOT_CONFIGURED"
        if profile is None
        else "NOT_DEMONSTRATED"
        if "RESTORE_NOT_DEMONSTRATED" in gap_codes
        else "LATEST_DRILL_FAILED"
        if "LATEST_DRILL_FAILED" in gap_codes
        else "STALE_DRILL"
        if "DRILL_STALE" in gap_codes
        else "POLICY_CHANGED"
        if "POLICY_CHANGED_SINCE_DRILL" in gap_codes
        else "OBJECTIVE_BREACHED"
        if {"RPO_TARGET_BREACHED", "RTO_TARGET_BREACHED"} & gap_codes
        else "MEASUREMENT_INCOMPLETE"
        if {"RPO_MEASUREMENT_MISSING", "RTO_MEASUREMENT_MISSING"} & gap_codes
        else "BACKUP_SOURCE_UNVERIFIED"
        if "BACKUP_SOURCE_NOT_VERIFIED" in gap_codes
        else "DEMONSTRATED_COMPLIANT"
    )
    return {
        "schema_version": "TARGET_RECOVERY_ASSURANCE.v1",
        "target_id": str(target.target_id),
        "db_type": str(target.db_type),
        "captured_at": captured_at.isoformat(),
        "assurance_status": assurance_status,
        "profile": recovery_profile_view(profile).model_dump(mode="json") if profile else None,
        "latest_attempt": (
            recovery_drill_view(latest_attempt).model_dump(mode="json")
            if latest_attempt
            else None
        ),
        "latest_verified_success": (
            recovery_drill_view(latest_success).model_dump(mode="json")
            if latest_success
            else None
        ),
        "restore_demonstrated": restore_demonstrated,
        "drill_stale": stale,
        "profile_changed_since_success": profile_changed,
        "covered_backup_source_types": covered_sources,
        "uncovered_backup_source_types": uncovered_sources,
        "gaps": gaps,
    }
