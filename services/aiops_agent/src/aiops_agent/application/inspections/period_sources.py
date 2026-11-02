"""自然周期巡检范围读取，使用历史批次冻结清单而非当前系统成员。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID


async def inspection_period_sources(
    *, uow, run, inspection: dict, domain_id, period_start: datetime, period_end: datetime,
):
    """返回期内运行及实际冻结成员，保留未生成运行的目标。"""
    system_id = inspection.get("business_system_id")
    if not system_id:
        runs = await uow.runs.list_completed_inspection_runs(
            domain_id=domain_id, target_id=run.target_id,
            period_start=period_start, period_end=period_end,
        )
        return runs, None, []
    rows = await uow.inspections.list_system_period_sources(
        domain_id=domain_id, business_system_id=UUID(str(system_id)),
        period_start=period_start, period_end=period_end,
    )
    members, runs, seen_runs, missing = {}, [], set(), []
    system_name = str(inspection.get("business_system_name") or "业务系统")
    by_fire = {}
    for fire, source_run in rows:
        by_fire.setdefault(fire.inspection_fire_id, [fire, []])[1].extend(
            [source_run] if source_run is not None else []
        )
        if source_run is not None and source_run.ops_run_id not in seen_runs:
            seen_runs.add(source_run.ops_run_id)
            runs.append(source_run)
    for fire, fire_runs in by_fire.values():
        snapshot = dict(fire.plan_snapshot_json or {})
        scope = dict(snapshot.get("system_scope") or {})
        system_name = str(scope.get("business_system_name") or system_name)
        resources = list(scope.get("resources") or ())
        expected = set(str(value) for value in scope.get("target_ids") or ())
        actual = {str(item.target_id) for item in fire_runs}
        for resource in resources:
            target_id = str(resource.get("target_id") or "")
            if not target_id:
                continue
            expected.add(target_id)
            member = members.setdefault(target_id, {
                "target_id": target_id, "display_name": str(resource.get("display_name") or "未记录"),
                "database_type": str(resource.get("db_type") or ""),
                "first_scheduled_at": fire.scheduled_for.isoformat(),
                "last_scheduled_at": fire.scheduled_for.isoformat(),
            })
            member["first_scheduled_at"] = min(member["first_scheduled_at"], fire.scheduled_for.isoformat())
            member["last_scheduled_at"] = max(member["last_scheduled_at"], fire.scheduled_for.isoformat())
        for item in fire_runs:
            target_id = str(item.target_id)
            frozen = dict(item.plan_snapshot_json or {})
            target = dict(frozen.get("target") or {})
            members.setdefault(target_id, {
                "target_id": target_id, "display_name": str(target.get("display_name") or "未记录"),
                "database_type": str(target.get("db_type") or ""),
            })
        for target_id in sorted(expected):
            member = members.setdefault(target_id, {
                "target_id": target_id, "display_name": "未记录",
                "database_type": "", "first_scheduled_at": fire.scheduled_for.isoformat(),
                "last_scheduled_at": fire.scheduled_for.isoformat(),
            })
            if target_id in actual:
                continue
            missing.append({
                "code": "INSPECTION_TARGET_RUN_MISSING",
                "detail": f"{member.get('display_name') or '纳管数据库'}在 {fire.scheduled_for.isoformat()} 批次未形成运行记录。",
                "target_id": target_id, "inspection_fire_id": str(fire.inspection_fire_id),
            })
    scope = {
        "scope_type": "BUSINESS_SYSTEM", "business_system_id": str(system_id),
        "scope_name": system_name, "target_count": len(members),
        "database_types": sorted({item["database_type"] for item in members.values() if item["database_type"]}),
        "member_targets": list(members.values()),
    }
    return runs, scope, missing
