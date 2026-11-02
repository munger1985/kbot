"""按真实备份作业成功时间核验策略，不以日志链或恢复演练替代备份记录。"""

from datetime import UTC, datetime, timedelta

from .daily_baseline import _number, _time
from .monitoring_observations import snapshot_is_fresh


BACKUP_METRIC_CODES = ("backup.last_success.timestamp", "backup.last_run.status")


def backup_policy_result(*, policy, database_type, facts, evidence_by_tool, reference_time):
    """逐一核验约定来源；缺失、不可信时间、失败和超时分别返回事实。"""
    reference = _time(reference_time)
    required = set(policy.get("source_types") or ())
    if not required and database_type == "ORACLE":
        required = {"ORACLE_RMAN"}
    evidence_refs, successes, failed_sources = [], {}, set()
    for fact in facts:
        role = fact.get("semantic_role")
        if fact.get("kind") != "inspection_monitoring_metric" or role not in {
            "BACKUP_LAST_SUCCESS_TIMESTAMP", "BACKUP_LAST_RUN_STATUS",
        }:
            continue
        source = dict(fact.get("metric_dimensions") or {}).get("backup_source_type")
        if source not in required or not fact.get("evidence_ref"):
            continue
        if fact.get("assessment_mode") != "CURRENT_SNAPSHOT" or not snapshot_is_fresh(fact):
            continue
        number = _number(fact.get("latest"))
        if number is None or not reference:
            continue
        if role == "BACKUP_LAST_RUN_STATUS":
            if number == 0:
                failed_sources.add(source)
                evidence_refs.append(fact["evidence_ref"])
            continue
        try:
            stamp = datetime.fromtimestamp(number, UTC)
        except (ValueError, OverflowError, OSError):
            continue
        if stamp > reference or number <= 0:
            continue
        successes[source] = max(successes.get(source, stamp), stamp)
        evidence_refs.append(fact["evidence_ref"])
    if database_type == "ORACLE" and "ORACLE_RMAN" in required and reference:
        for observation in evidence_by_tool.get("db.backup.recent_jobs", ()):
            indexes = {str(column.get("name") or "").lower(): i for i, column in enumerate(observation.columns)}
            if "status" not in indexes or "end_time" not in indexes:
                continue
            for row in observation.rows:
                if str(row[indexes["status"]] or "").upper() != "COMPLETED":
                    continue
                stamp = _time(row[indexes["end_time"]])
                if stamp and stamp <= reference:
                    successes["ORACLE_RMAN"] = max(successes.get("ORACLE_RMAN", stamp), stamp)
                    evidence_refs.append(observation.evidence_ref)
    missing = sorted(required - successes.keys())
    stale = sorted(source for source, stamp in successes.items()
                   if reference and reference - stamp > timedelta(hours=int(policy["success_max_age_hours"])))
    status = "BREACHED" if failed_sources or stale else "NOT_OBSERVABLE" if missing or not required or not reference else "COMPLIANT"
    if failed_sources:
        summary = "最近一次备份作业失败：" + "、".join(sorted(failed_sources)) + "。"
    elif stale:
        summary = "最近成功备份超过策略时效：" + "、".join(stale) + "。"
    elif status == "COMPLIANT":
        summary = "已核对策略约定的备份成功记录，均在允许时效内；不代表已验证可恢复。"
    else:
        summary = "备份成功状态未核验：尚未取得" + ("、".join(missing) or "策略约定来源") + "的有效成功记录；Binlog/归档正常不能代替备份成功。"
    if missing and status == "BREACHED":
        summary += " 另有来源未核验：" + "、".join(missing) + "。"
    fact = {
        "kind": "backup_policy_check", "title": "备份策略校验", "summary": summary,
        "check_status": status, "tool_id": "inspection.backup-policy",
        "check_ids": ["backup.policy.success"], "required": True,
        "evidence_refs": list(dict.fromkeys(evidence_refs)), "policy": policy,
        "last_success_by_source": {key: stamp.isoformat() for key, stamp in successes.items()},
        "unverified_source_types": missing,
        "observed_at": reference.isoformat() if reference else None,
    }
    if status == "BREACHED":
        fact["findings"] = [{"severity": "HIGH", "finding_type": "BACKUP_POLICY", "impact": summary}]
    if missing and status == "BREACHED":
        # 已知失败不得被缺失来源掩盖，仍单独保留未核验限制。
        fact["truncated"] = True
    gaps = []
    if missing or not required or not reference:
        # 已确认的失败或超时属于巡检问题，不冒充证据缺口。
        detail = summary if status == "NOT_OBSERVABLE" else "备份成功状态未核验：尚未取得" + "、".join(missing) + "的有效成功记录。"
        gaps.append({"source_id": "inspection.backup-policy", "step_id": "inspection.backup-policy",
                     "code": "BACKUP_SUCCESS_NOT_OBSERVABLE", "detail": detail, "retryable": False, "category": "DATA_SOURCE"})
    return fact, gaps
