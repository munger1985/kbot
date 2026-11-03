"""跨数据库日报分区投影：只整理冻结事实，不重判检查或调用取证。"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from platform_core.contracts.aiops.daily_report import DailyReportLayout

_SEVERITY = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}
_NORMAL = {"CLEAR", "COMPLIANT", "NOT_APPLICABLE"}
_LABELS = {"NORMAL": "正常", "ATTENTION": "需关注", "ABNORMAL": "异常", "INCOMPLETE": "检查未完成",
           "NOT_APPLICABLE": "不适用", "NOT_INCLUDED": "未纳入本次检查", "OBSERVED": "仅观测"}


def daily_section_catalog() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("daily_sections.json").read_text(encoding="utf-8"))


def normalize_daily_layout(value=None) -> dict[str, Any]:
    return DailyReportLayout.model_validate(value or {}).model_dump(mode="json")


def _check_section(check, check_sections, metric_sections, roles):
    section = check.get("report_section")
    if section:
        return section
    if check.get("kind") == "backup_policy_check":
        return "BACKUP"
    if check.get("semantic_role") in roles:
        return roles[check["semantic_role"]]
    if check.get("metric_code") in metric_sections:
        return metric_sections[check["metric_code"]]
    mapped = {check_sections[key] for key in check.get("check_ids") or () if key in check_sections}
    return next(iter(mapped)) if len(mapped) == 1 else "PERFORMANCE"


def _section_status(checks, metrics, findings):
    if findings:
        return "ABNORMAL" if any(_SEVERITY.get(item.get("severity"), 0) >= 4 for item in findings) else "ATTENTION"
    if any(item.get("status") in {"FINDING", "BREACHED"} for item in checks):
        return "ABNORMAL" if any(_SEVERITY.get(item.get("severity"), 0) >= 4 for item in checks) else "ATTENTION"
    if any(item.get("status") not in _NORMAL | {"OBSERVED"} for item in checks):
        return "INCOMPLETE"
    if any(item.get("status_display") == "未完成" for item in metrics):
        return "INCOMPLETE"
    if checks and all(item.get("status") == "NOT_APPLICABLE" for item in checks) and not metrics:
        return "NOT_APPLICABLE"
    if any(item.get("status_display") in {"异常", "需关注"} for item in metrics):
        return "ABNORMAL" if any(item.get("status_display") == "异常" for item in metrics) else "ATTENTION"
    if checks or metrics:
        if all(item.get("status") in _NORMAL for item in checks) and all(
            item.get("status_display") == "正常" for item in metrics):
            return "NORMAL"
        return "OBSERVED"
    return "NOT_INCLUDED"


def project_daily_sections(data: dict[str, Any], *, layout=None, catalog=None) -> dict[str, Any]:
    """共享目录归类三库及未来引擎；未知内容保留，不根据类型或中文标题分流。"""
    from .check_catalog import load_check_catalog
    from aiops_agent.application.display_dictionary import display_field_name
    from .sql_observations import observation_field

    result = deepcopy(data)
    custom_title = layout is not None
    layout = normalize_daily_layout(layout)
    if custom_title:
        result["header"]["title"] = layout["title"]
    catalog = deepcopy(catalog or daily_section_catalog())
    definitions = {item["section_id"]: item for item in catalog["sections"]}
    check_sections = {check.check_id: check.report_section
                      for group in load_check_catalog().groups for check in group.checks}
    roles = {role: item["section_id"] for item in catalog["sections"] for role in item.get("semantic_roles") or ()}
    metric_sections = {}
    for metric in result.get("metrics") or ():
        metric["report_section"] = metric.get("report_section") or roles.get(metric.get("semantic_role"), "PERFORMANCE")
        metric_sections[metric["metric_code"]] = metric["report_section"]
    for check in result.get("checks") or ():
        check["report_section"] = _check_section(check, check_sections, metric_sections, roles)
        check["detail_records"] = [[{"label": display_field_name(key),
            "value": observation_field(key, value).split("：", 1)[-1] if value is not None else "未记录"}
            for key, value in row.items()] for row in check.get("observations") or () if isinstance(row, dict)]
        if check["report_section"] not in definitions:
            check["report_section"] = "PERFORMANCE"
            check["classification_note"] = "该检查未声明有效区域，保留在此供核查。"
    issues = sorted(result.get("findings") or (), key=lambda item: -_SEVERITY.get(item.get("severity"), 0))
    for issue in issues:
        section_id = issue.get("report_section")
        if not section_id:
            match = next((check for check in result.get("checks") or ()
                          if set(issue.get("check_ids") or ()) & set(check.get("check_ids") or ())), None)
            section_id = match["report_section"] if match else "OVERVIEW"
        issue["report_section"] = section_id if section_id in definitions else "OVERVIEW"
    result["findings"] = issues
    result["finding_index"] = [{**issue, "anchor": issue.get("finding_id") if issue["report_section"] == "SQL" and issue.get("finding_id") else "daily-finding-" + str(index)}
                               for index, issue in enumerate(issues, 1)]
    actions = []
    for index, issue in enumerate(result["finding_index"], 1):
        actions.append({"action_id": f"action-{index}", "priority": issue.get("severity", "MEDIUM"),
            "section_id": issue["report_section"], "finding_id": issue.get("finding_id"),
            "finding_anchor": issue["anchor"], "object_name": issue.get("object_name") or issue.get("title"),
            "action": issue.get("recommendation") or "核对对应问题并按变更流程处理。",
            "validation": issue.get("basis") or "复查同一对象，确认问题判定条件已解除。"})
    for check in result.get("checks") or ():
        if check.get("required") and check.get("status") not in _NORMAL | {"FINDING", "BREACHED"}:
            actions.append({"action_id": f"action-{len(actions)+1}", "priority": "MEDIUM",
                "section_id": check["report_section"], "finding_id": None, "finding_anchor": "",
                "object_name": check["title"], "action": "完成该项检查限制的核查后重新执行。",
                "validation": check.get("summary") or "取得完整可追溯结果并按原规则复核。"})
    represented = {str(item["action"]) for item in actions}
    for group in result.get("sql_details") or ():
        for statement in group.get("statements") or ():
            review = statement.get("agent_review") or {}
            if review.get("status") != "COMPLETE":
                continue
            for advice in review.get("recommendations") or ():
                actions.append({"action_id": f"action-{len(actions)+1}", "priority": statement.get("severity") or "MEDIUM",
                    "section_id": "SQL", "finding_id": statement.get("problem_id"),
                    "finding_anchor": statement.get("problem_id"), "object_name": f"问题 SQL {statement.get('ordinal')}",
                    "action": advice["action"] + ("；实施前须审批。" if advice.get("requires_approval") else ""),
                    "validation": advice["validation"]})
    for advice in result.get("recommendations") or ():
        if advice not in represented:
            actions.append({"action_id": f"action-{len(actions)+1}", "priority": "INFO",
                "section_id": "OVERVIEW", "finding_id": None, "finding_anchor": "",
                "object_name": "补充建议", "action": advice, "validation": "按建议完成后复核对应检查。"})
            represented.add(advice)
    result["actions"] = actions
    sections = []
    for settings in layout["sections"]:
        key = settings["section_id"]
        checks = [item for item in result.get("checks") or () if item["report_section"] == key]
        metrics = [item for item in result.get("metrics") or () if item["report_section"] == key]
        findings = [item for item in result["finding_index"] if item["report_section"] == key]
        sql = result.get("sql_details", []) if key == "SQL" else []
        comparisons = [item for item in result.get("metrics") or () if item.get("baseline_applicable")] if key == "BASELINE" else []
        state = _section_status(checks, metrics, findings)
        if key == "SQL" and sql:
            if any(item.get("status") != "COMPLETE" for item in sql):
                state = "ATTENTION" if findings else "INCOMPLETE"
            elif not findings:
                state = "NORMAL"
        if key == "BASELINE":
            state = "OBSERVED" if comparisons else "NOT_INCLUDED"
        if key == "OVERVIEW":
            state = result.get("conclusion", {}).get("health_status", "INCOMPLETE")
            findings = [item for item in result["finding_index"] if item["report_section"] == "OVERVIEW"]
        if key == "ACTIONS":
            state = "ATTENTION" if actions else ("NORMAL" if result.get("conclusion", {}).get("health_status") == "NORMAL" else "OBSERVED")
        has_data = bool(checks or metrics or findings or sql or comparisons)
        force = key in {"OVERVIEW", "ACTIONS"} or bool(findings) or any(item.get("required") for item in checks) or state == "INCOMPLETE"
        if settings["visibility"] == "WHEN_RELEVANT" and not has_data and not force:
            continue
        if key == "OVERVIEW":
            summary = result.get("conclusion", {}).get("text") or "尚未形成完整巡检结论。"
        elif key == "ACTIONS":
            summary = f"共 {len(actions)} 项处置与复查事项，按问题优先级执行。" if actions else "无额外处置事项。"
        elif state == "NOT_INCLUDED":
            summary = "本报告未记录该区域的检查结果，不据此判断正常或不适用。"
        elif state == "NOT_APPLICABLE":
            summary = "本次检查已确认该区域不适用，依据见检查明细。"
        elif key == "SQL" and sql:
            summary = "；".join(item["summary"] for item in sql)
        elif key == "BASELINE":
            summary = "对比口径见各指标；历史不足只影响趋势比较，不改变本期健康结论。"
        else:
            incomplete = sum(item.get("status") not in _NORMAL | {"FINDING", "BREACHED", "OBSERVED"} for item in checks)
            summary = f"本区域 {len(checks)} 项检查，{len(metrics)} 项指标，{len(findings)} 项问题" + (f"，{incomplete} 项检查未完成。" if incomplete else "。")
        sections.append({**definitions[key], **settings, "anchor": "daily-" + key.lower(),
            "status": state, "status_display": _LABELS.get(state, "检查未完成"), "summary": summary,
            "checks": checks, "metrics": metrics, "findings": findings, "sql_details": sql,
            "comparisons": comparisons, "actions": actions if key == "ACTIONS" else []})
    for ordinal, section in enumerate(sections, 1):
        section["ordinal"] = ordinal
    result.update(sections=sections, layout=layout, section_catalog_version=catalog["version"])
    return result
