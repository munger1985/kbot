"""月度业务评估与问题台账，不从缺失观测推导恢复事实。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


_RANK = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}
_LABEL = {"NORMAL": "正常", "ATTENTION": "需关注", "ABNORMAL": "异常", "INCOMPLETE": "检查未完成"}


def monthly_business_review(
    *, facts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    period_start: datetime, period_end: datetime, timezone: str,
    members: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """按库隔离问题身份，巡检观测次数不冒充事故次数。"""
    tz = ZoneInfo(timezone)
    start, end = period_start.astimezone(tz).date(), period_end.astimezone(tz).date()
    expected_days = max(0, (end - start).days)
    targets: dict[str, dict[str, Any]] = {}
    issues: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    actions: list[str] = []
    for member in members:
        target_id = str(member.get("target_id") or "")
        targets[target_id] = {
            "target_id": target_id, "target_name": str(member.get("display_name") or "未记录"),
            "assessments": {}, "expected_days": expected_days,
            "first_scheduled_at": member.get("first_scheduled_at"),
            "last_scheduled_at": member.get("last_scheduled_at"),
        }
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        stamp = str(fact.get("_source_observed_at") or "")
        try:
            moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                continue
            stamp = moment.astimezone(ZoneInfo("UTC")).isoformat()
            date = moment.astimezone(tz).date()
        except ValueError:
            continue
        if not start <= date < end:
            continue
        target_id = str(fact.get("_source_target_id") or "")
        target = targets.setdefault(target_id, {
            "target_id": target_id,
            "target_name": str(fact.get("_source_target_name") or "未记录"),
            "assessments": {}, "expected_days": expected_days,
        })
        day = date.isoformat()
        kind = str(fact.get("kind") or "")
        if kind == "inspection_assessment":
            assessment = dict(fact.get("assessment") or {})
            previous = target["assessments"].get(day)
            if previous is None or stamp >= previous[0]:
                target["assessments"][day] = (stamp, assessment)
            actions.extend(str(value) for value in assessment.get("recommendations") or ())
            continue
        if kind not in {"inspection_check", "inspection_metric_check", "backup_policy_check"}:
            continue
        status = str(fact.get("check_status") or "MISSING")
        counts[status] = counts.get(status, 0) + 1
        findings = list(fact.get("findings") or ())
        if kind == "backup_policy_check" and status == "BREACHED" and not findings:
            findings = [{"finding_type": "BACKUP_POLICY", "severity": "HIGH", "impact": fact.get("summary")}]
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            identity = finding.get("finding_id") or json.dumps({
                "type": finding.get("finding_type"), "tool": fact.get("tool_id"),
                "object": finding.get("object_ref") or {},
                "metric": fact.get("metric_code"), "dimensions": fact.get("dimensions"),
            }, ensure_ascii=False, sort_keys=True, default=str)
            fingerprint = hashlib.sha256(f"{target_id}|{identity}".encode()).hexdigest()[:24]
            issue = issues.setdefault(fingerprint, {
                "finding_id": fingerprint, "source_finding_id": finding.get("finding_id"),
                "target_id": target_id, "target_name": target["target_name"],
                "title": str(finding.get("title") or fact.get("title") or "巡检发现"),
                "summary": str(finding.get("impact") or finding.get("summary") or fact.get("summary") or ""),
                "severity": str(finding.get("severity") or "INFO"),
                "dates": set(), "observations": set(), "events": set(),
                "verification": None, "latest_observed_at": "",
                "confirmation": str(finding.get("confirmation") or "CONFIRMED"),
            })
            severity = str(finding.get("severity") or "INFO")
            if _RANK.get(severity, 0) > _RANK.get(issue["severity"], 0):
                issue["severity"] = severity
            issue["dates"].add(day)
            issue["observations"].add(str(fact.get("_source_run_id") or stamp))
            if finding.get("event_id"):
                issue["events"].add(str(finding["event_id"]))
            if stamp >= issue["latest_observed_at"]:
                issue["latest_observed_at"] = stamp
                issue["verification"] = None
                verification = dict(finding.get("verification") or {})
                try:
                    verified_at = datetime.fromisoformat(str(verification.get("verified_at") or "").replace("Z", "+00:00"))
                    verified = verified_at.tzinfo is not None and verified_at <= moment
                except (ValueError, TypeError):
                    verified = False
                if verified and verification.get("result") == "RESOLVED" and verification.get("evidence_refs"):
                    issue["verification"] = verification
    issue_rows = []
    for issue in issues.values():
        dates = sorted(issue.pop("dates"))
        observations = issue.pop("observations")
        events = issue.pop("events")
        resolved = bool(issue["verification"])
        issue_rows.append({
            **issue, "first_seen": dates[0], "last_seen": dates[-1],
            "observed_days": len(dates), "observation_count": len(observations),
            "occurrence_count": len(events) if events else None,
            "lifecycle": "VERIFIED_RESOLVED" if resolved else "PENDING_VERIFICATION",
            "lifecycle_display": "已验证恢复" if resolved else "待验证",
            "repeat_display": "多日重复发现" if len(dates) > 1 else "本月发现",
        })
    issue_rows.sort(key=lambda item: (-_RANK.get(item["severity"], 0), item["target_name"], item["finding_id"]))
    target_rows, limitations = [], []
    for target in targets.values():
        observations = sorted(target.pop("assessments").values(), key=lambda item: item[0])
        assessments = [item[1] for item in observations]
        own_issues = [item for item in issue_rows if item["target_id"] == target["target_id"]]
        required_complete_days = sum(item.get("completion_status") == "COMPLETE" for item in assessments)
        own_limitations = []
        if required_complete_days < target["expected_days"]:
            own_limitations.append(
                f"{target['target_name']}：仅 {required_complete_days}/{target['expected_days']} 天有完整必要检查，未覆盖日期不能判定正常。"
            )
            if target.get("first_scheduled_at"):
                own_limitations.append(
                    f"{target['target_name']}：期内首次批次为 {target['first_scheduled_at']}；"
                    "批次时间不等于纳管生效时间，未据此缩短自然月判断范围。"
                )
        rank = max((_RANK.get(item["severity"], 0) for item in own_issues if item.get("confirmation") != "UNKNOWN"), default=0)
        health = "ABNORMAL" if rank >= 4 else "ATTENTION" if own_issues else "INCOMPLETE" if own_limitations else "NORMAL"
        latest = assessments[-1] if assessments else {}
        latest_health = str(latest.get("health_status") or "INCOMPLETE")
        target_rows.append({
            **target, "health_status": health, "health_status_display": _LABEL[health],
            "latest_health_status": latest_health,
            "latest_health_display": _LABEL.get(latest_health, "检查未完成"),
            "latest_checked_at": observations[-1][0] if observations else None,
            "issue_count": len(own_issues), "completed_days": required_complete_days,
            "limitations": own_limitations,
            "first_observed_at": observations[0][0] if observations else None,
            "last_observed_at": observations[-1][0] if observations else None,
        })
        limitations.extend(own_limitations)
    if not targets:
        limitations.append("本月没有可用于业务判断的完整基础检查结果。")
    rank = max((_RANK.get(item["severity"], 0) for item in issue_rows if item.get("confirmation") != "UNKNOWN"), default=0)
    health = "ABNORMAL" if rank >= 4 else "ATTENTION" if issue_rows else "INCOMPLETE" if limitations else "NORMAL"
    if issue_rows:
        resolved = sum(item["lifecycle"] == "VERIFIED_RESOLVED" for item in issue_rows)
        text = f"本月发现 {len(issue_rows)} 项问题，已验证恢复 {resolved} 项，其余仍需跟踪或验证；不能以期末未再出现认定已解决。"
    elif limitations:
        text = "已记录的巡检未发现明确问题，但必要检查存在未覆盖日期，尚不足以认定全月运行正常。"
    else:
        text = "本月纳管数据库必要巡检均已完成，未发现异常或需关注风险，数据库运行正常。"
    if limitations:
        text += " 统计限制：" + "；".join(limitations[:3])
    return {
        "conclusion": {"text": text, "health_status": health,
                       "health_status_display": _LABEL[health],
                       "completion_display": "部分完成" if limitations else "完整",
                       "issue_count": len(issue_rows)},
        "issues": issue_rows, "target_results": target_rows,
        "limitations": limitations,
        "recommendations": list(dict.fromkeys(actions)),
        "database_checks": {"status_counts": counts, "observation_count": sum(counts.values())},
    }

def merge_monthly_comparison(summary: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    """往期问题未再发现时保留待验证事项，不伪造本月发生或已经恢复。"""
    pending = [
        {**item, "from_previous_period": True, "title": "往期待验证：" + str(item.get("title") or "巡检问题")}
        for item in comparison.get("pending_verification_issues") or ()
        if not summary.get("target_results") or item.get("target_id") in {
            target.get("target_id") for target in summary["target_results"]
        }
    ]
    if not pending:
        return summary
    conclusion = dict(summary.get("conclusion") or {})
    conclusion["text"] = str(conclusion.get("text") or "") + f" 另有 {len(pending)} 项往期问题仍待同对象复查验证。"
    if conclusion.get("health_status") == "NORMAL":
        conclusion.update(health_status="ATTENTION", health_status_display="需关注")
    issues = [*summary.get("issues", ()), *pending]
    conclusion["issue_count"] = len(issues)
    targets = []
    for target in summary.get("target_results") or ():
        own_pending = [item for item in pending if item.get("target_id") == target.get("target_id")]
        row = {**target, "issue_count": int(target.get("issue_count") or 0) + len(own_pending)}
        if own_pending and row.get("health_status") == "NORMAL":
            row.update(health_status="ATTENTION", health_status_display="需关注")
        targets.append(row)
    actions = [
        *summary.get("recommendations", ()),
        *(f"复查{item.get('target_name') or '对应数据库'}的“{item.get('title') or '往期问题'}”，取得处理后同一对象的验证记录。"
          for item in pending),
    ]
    return {**summary, "conclusion": conclusion, "issues": issues,
            "target_results": targets, "recommendations": list(dict.fromkeys(actions))}
