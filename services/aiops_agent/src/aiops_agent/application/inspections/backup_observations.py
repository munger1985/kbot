"""核验备份保障：无有效成功记录是业务风险，查询失败才是检查限制。"""

from datetime import UTC, datetime, timedelta

from .daily_baseline import _number, _time
from .monitoring_observations import snapshot_rejection_reason


BACKUP_METRIC_CODES = ("backup.last_success.timestamp", "backup.last_run.status")


def _required_sources(policy, database_type):
    required = set(policy.get("source_types") or ())
    if not required and database_type == "ORACLE":
        required = {"ORACLE_RMAN"}
    return required


def backup_record_review_is_complete(fact):
    """缺失记录风险可以完成策略审查，但必须冻结逐来源结果，不能放开普通缺证据门禁。"""
    if fact.get("kind") != "backup_policy_check" or not _time(fact.get("observed_at")):
        return False
    policy = fact.get("policy") or {}
    required = _required_sources(policy, str(fact.get("database_type") or ""))
    results = fact.get("source_results") or {}
    return bool(policy.get("enabled") and required and set(results) == required and all(
        result.get("status") == "NO_SUCCESS_RECORD"
        or (result.get("status") in {"COMPLIANT", "FAILED", "OVERDUE"} and result.get("evidence_refs"))
        for result in results.values()
    ))


def _query_errors(gaps, monitoring, database_type):
    """只关联备份查询失败；空结果、历史参考和无关指标错误不冒充备份执行失败。"""
    bindings = {str(item.get("binding_id")) for item in (monitoring or {}).get("bindings", ())
                if any(metric.get("metric_code") in BACKUP_METRIC_CODES for metric in item.get("metrics", ()))}
    errors = []
    for gap in gaps:
        step, code = str(gap.get("step_id") or ""), str(gap.get("code") or "")
        if code.startswith("REFERENCE_") or code == "SOURCE_NO_DATA":
            continue
        if (step in BACKUP_METRIC_CODES
            or database_type == "ORACLE" and step == "db.backup.recent_jobs"
            or not step and str(gap.get("binding_id")) in bindings):
            errors.append(str(gap.get("detail") or "备份来源查询失败"))
    return errors


def backup_policy_result(*, policy, database_type, facts, evidence_by_tool, reference_time,
                         source_gaps=(), monitoring=None):
    """按来源分别记录成功、失败、超期、缺记录和技术性核验失败。"""
    reference = _time(reference_time)
    required = _required_sources(policy, database_type)
    refs = {source: [] for source in required}
    successes, failed_sources, invalid = {}, set(), {}
    errors = _query_errors(source_gaps, monitoring, database_type)
    for gap in source_gaps:
        if gap.get("code") == "SOURCE_NO_DATA" and gap.get("step_id") in BACKUP_METRIC_CODES and gap.get("evidence_ref"):
            for source in required:
                refs[source].append(gap["evidence_ref"])
    for fact in facts:
        role = fact.get("semantic_role")
        if fact.get("kind") != "inspection_monitoring_metric" or role not in {
            "BACKUP_LAST_SUCCESS_TIMESTAMP", "BACKUP_LAST_RUN_STATUS",
        }:
            continue
        source = dict(fact.get("metric_dimensions") or {}).get("backup_source_type")
        if source not in required:
            continue
        reason = snapshot_rejection_reason(fact)
        number = _number(fact.get("latest"))
        if fact.get("assessment_mode") != "CURRENT_SNAPSHOT" or reason or not fact.get("evidence_ref") or number is None:
            invalid[source] = reason or "备份来源返回无效或不可追溯的当前记录"
            continue
        if role == "BACKUP_LAST_RUN_STATUS":
            if number not in {0, 1}:
                invalid[source] = "备份作业状态值无效"
            elif number == 0:
                failed_sources.add(source)
                refs[source].append(fact["evidence_ref"])
            continue
        # 显式零值表示没有成功记录，而不是无效日期或一次成功作业。
        if number == 0:
            continue
        try:
            stamp = datetime.fromtimestamp(number, UTC)
        except (ValueError, OverflowError, OSError):
            invalid[source] = "备份成功时间无效"
            continue
        if not reference or stamp > reference or number < 0:
            invalid[source] = "备份成功时间无效或位于核验时刻之后"
            continue
        successes[source] = max(successes.get(source, stamp), stamp)
        refs[source].append(fact["evidence_ref"])
    if database_type == "ORACLE" and "ORACLE_RMAN" in required and reference:
        for observation in evidence_by_tool.get("db.backup.recent_jobs", ()):
            indexes = {str(column.get("name") or "").lower(): i for i, column in enumerate(observation.columns)}
            if observation.truncated:
                invalid["ORACLE_RMAN"] = "备份作业查询结果已截断"
            if "status" not in indexes or "end_time" not in indexes:
                if observation.row_count:
                    invalid["ORACLE_RMAN"] = "备份作业结果缺少状态或结束时间"
                continue
            for row in observation.rows:
                if str(row[indexes["status"]] or "").upper() != "COMPLETED":
                    continue
                stamp = _time(row[indexes["end_time"]])
                if stamp and stamp <= reference and observation.evidence_ref:
                    successes["ORACLE_RMAN"] = max(successes.get("ORACLE_RMAN", stamp), stamp)
                    refs["ORACLE_RMAN"].append(observation.evidence_ref)
                else:
                    invalid["ORACLE_RMAN"] = "备份成功时间无效、来自未来或缺少证据引用"
    results, findings, limitations = {}, [], []
    for source in sorted(required):
        state, reason = "COMPLIANT", "成功记录在策略允许时效内"
        if source in failed_sources:
            state, reason = "FAILED", "最近一次备份作业失败"
        elif source in successes and reference and reference - successes[source] > timedelta(hours=int(policy["success_max_age_hours"])):
            state, reason = "OVERDUE", "最近成功备份超过策略允许时效"
        elif source in invalid or (source not in successes and errors):
            state, reason = "UNVERIFIABLE", invalid.get(source) or errors[0]
        elif source not in successes:
            state, reason = "NO_SUCCESS_RECORD", "未提供有效备份成功记录，备份保障不符合已启用策略"
        results[source] = {"status": state, "reason": reason, "evidence_refs": list(dict.fromkeys(refs[source]))}
        if state == "UNVERIFIABLE":
            limitations.append(f"{source}：{reason}")
        elif state != "COMPLIANT":
            findings.append({
                "severity": "MEDIUM" if state == "NO_SUCCESS_RECORD" else "HIGH",
                "finding_type": "BACKUP_SUCCESS_RECORD_MISSING" if state == "NO_SUCCESS_RECORD" else "BACKUP_POLICY",
                "object_ref": {"object_name": source}, "impact": f"{source}：{reason}。",
                "evidence_refs": results[source]["evidence_refs"],
            })
    if not required or not reference:
        limitations.append("备份策略来源或核验时刻无效")
    high = any(item["severity"] == "HIGH" for item in findings)
    status = "BREACHED" if high else "FINDING" if findings else "NOT_OBSERVABLE" if limitations else "COMPLIANT"
    summary = "；".join(str(item["impact"]).rstrip("。") for item in findings)
    if limitations:
        summary += ("；" if summary else "") + "备份核验未完成：" + "；".join(limitations)
    summary = summary + "。" if summary else "策略约定来源的备份成功记录均在允许时效内；不代表已验证可恢复。"
    fact = {
        "kind": "backup_policy_check", "title": "备份策略校验", "summary": summary,
        "check_status": status, "tool_id": "inspection.backup-policy",
        "check_ids": ["backup.policy.success"], "required": True,
        "evidence_refs": list(dict.fromkeys(ref for values in refs.values() for ref in values)),
        "policy": policy, "database_type": database_type, "source_results": results,
        "basis": "按已启用策略逐来源审查有效成功记录；无成功记录列为备份保障风险，不据此断言从未执行备份。",
        "last_success_by_source": {key: stamp.isoformat() for key, stamp in successes.items()},
        "missing_success_source_types": [key for key, result in results.items() if result["status"] == "NO_SUCCESS_RECORD"],
        "unverified_source_types": [key for key, result in results.items() if result["status"] == "UNVERIFIABLE"],
        "observed_at": reference.isoformat() if reference else None,
        "findings": findings,
    }
    if limitations:
        fact["truncated"] = True
        fact["incomplete_reason"] = "；".join(limitations)
    gaps = [{"source_id": "inspection.backup-policy", "step_id": "inspection.backup-policy",
             "code": "BACKUP_VERIFICATION_FAILED", "detail": detail, "retryable": True, "category": "DATA_SOURCE"}
            for detail in limitations]
    return fact, gaps
