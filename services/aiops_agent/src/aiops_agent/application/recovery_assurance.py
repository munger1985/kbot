"""把恢复目标和演练记录冻结为可诊断的公共证据快照。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aiops_agent.application.configuration.recovery_service import (
    _profile_view,
    recovery_drill_view,
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
    """返回最新尝试和范围匹配的最新成功；旧成功不能覆盖新失败。"""
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
    latest_success = next(
        (
            row
            for row in drills
            if row.status == "VERIFIED"
            and row.result == "PASS"
            and _ASSURANCE_RANK.get(row.assurance_level, 0) >= required_rank
        ),
        None,
    )
    restore_verified = bool(
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
    gaps: list[dict[str, object]] = []
    if profile is None:
        gaps.extend(
            (
                {"code": "RPO_NOT_CONFIGURED", "detail": "Target 未配置业务 RPO"},
                {"code": "RTO_NOT_CONFIGURED", "detail": "Target 未配置业务 RTO"},
            )
        )
    if not restore_verified:
        gaps.append(
            {
                "code": "RESTORE_NOT_VERIFIED",
                "detail": "没有已审核且达到 DATABASE_OPEN 的成功恢复演练",
            }
        )
    if stale:
        gaps.append({"code": "DRILL_STALE", "detail": "最近成功演练已超过要求周期"})
    if profile_changed:
        gaps.append(
            {
                "code": "POLICY_CHANGED_SINCE_DRILL",
                "detail": "恢复目标版本在最近成功演练后已变更",
            }
        )
    capabilities = dict(getattr(target, "capabilities_json", None) or {})
    backup_provider = dict(capabilities.get("backup_provider") or {})
    if not backup_provider.get("configured"):
        gaps.append(
            {
                "code": "EXTERNAL_BACKUP_PROVIDER_NOT_CONFIGURED",
                "detail": "未配置可验证外部副本及恢复链的备份平台连接",
            }
        )
    return {
        "schema_version": "TARGET_RECOVERY_ASSURANCE.v1",
        "target_id": str(target.target_id),
        "db_type": str(target.db_type),
        "captured_at": captured_at.isoformat(),
        "profile": _profile_view(profile).model_dump(mode="json") if profile else None,
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
        "restore_verified": restore_verified,
        "drill_stale": stale,
        "profile_changed_since_success": profile_changed,
        "covered_backup_source_types": sorted(
            {row.backup_source_type for row in drills if row.status == "VERIFIED"}
        ),
        "gaps": gaps,
    }
