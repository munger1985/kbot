"""专业巡检业务判定：规则、检查结果和报告结论独立于采集流水。"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from aiops_agent.application.display_dictionary import (
    display_field_name, display_metric_name, display_text, format_metric_value,
)
from .sql_observations import SQL_TEXT_COLUMNS, observation_field, select_sql_text


_STATUS_LABELS = {
    "CLEAR": "正常", "COMPLIANT": "正常", "NOT_APPLICABLE": "不适用",
    "FINDING": "需关注", "BREACHED": "异常", "OBSERVED": "仅观测",
    "PARTIAL": "未完成", "GAP": "未完成", "MISSING": "未完成",
    "NOT_OBSERVABLE": "未完成", "NOT_OBSERVED": "未完成",
}
_HEALTH_LABELS = {
    "NORMAL": "正常", "ATTENTION": "需关注", "ABNORMAL": "异常",
    "INCOMPLETE": "检查未完成",
}
_SEVERITY = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}


@lru_cache(maxsize=1)
def report_policy() -> dict[str, Any]:
    """业务规则来自版本化目录，不由模型临时选择。"""
    return json.loads(Path(__file__).with_name("report_policy.json").read_text(encoding="utf-8"))


def freeze_inspection_policy(database_type: str) -> dict[str, Any]:
    """每次执行冻结适用基础集，模板专项不能取消必检项。"""
    from .check_catalog import load_check_catalog

    policy = report_policy()
    required = set(policy["baseline"].get(database_type.upper(), ()))
    additional = set(policy["supplemental"].get(database_type.upper(), ()))
    checks = [
        {"check_id": item.check_id, "title": item.display_name,
         "tool_id": item.tool_id, "required": item.check_id in required}
        for group in load_check_catalog().groups for item in group.checks
        if item.check_id in required | additional
    ]
    return {
        "schema_version": policy["schema_version"], "version": policy["version"],
        "database_type": database_type.upper(), "checks": checks,
        "required_metric_roles": dict(policy["required_metric_roles"]),
        "minimum_monitoring_coverage_ratio": policy["minimum_monitoring_coverage_ratio"],
        "current_monitoring_window_seconds": policy["current_monitoring_window_seconds"],
        "maximum_current_sample_age_seconds": policy["maximum_current_sample_age_seconds"],
        "recommendations": dict(policy["recommendations"]),
    }


def compile_standard_steps(
    steps: tuple[dict[str, Any], ...], *, database_type: str, schedule_type: str,
    policy: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """基础集独立于模板专项，在执行前合并并完整冻结。"""
    from .check_catalog import compile_selected_check_steps

    baseline = compile_selected_check_steps(
        [item["check_id"] for item in policy["checks"]],
        database_type=database_type, schedule_type=schedule_type,
    )
    by_tool: dict[str, dict[str, Any]] = {}
    required_ids = {item["check_id"] for item in policy["checks"] if item["required"]}
    current_tools = {check_id: item["tool_id"] for item in baseline for check_id in item.get("check_ids") or ()}
    for raw in (*baseline, *steps):
        tool_id = str(raw["tool_id"])
        selected_ids = set(raw.get("check_ids") or ())
        # 当前标准集已覆盖的检查以当前工具为准，不重复执行模板中被替代的累计榜单。
        if (selected_ids and selected_ids <= current_tools.keys()
                and tool_id not in {current_tools[check_id] for check_id in selected_ids}):
            continue
        if tool_id not in by_tool:
            by_tool[tool_id] = {**raw, "check_ids": list(raw.get("check_ids") or ())}
        else:
            step = by_tool[tool_id]
            step["check_ids"] = list(dict.fromkeys((*step["check_ids"], *raw.get("check_ids", ()))))
            step["input"] = {**dict(step.get("input") or {}), **dict(raw.get("input") or {})}
        by_tool[tool_id]["required"] = bool(required_ids.intersection(by_tool[tool_id]["check_ids"]))
    from .sql_problems import PROBLEM_SQL_TOOL, problem_sql_inputs
    if PROBLEM_SQL_TOOL in by_tool:
        # 模板不能以独立查询参数覆盖已冻结策略，避免取数和判定阈值不一致。
        step = by_tool[PROBLEM_SQL_TOOL]
        step["input"] = problem_sql_inputs(step["sql_problem_policy"])
        step["measurement_semantics"] = "CUMULATIVE_SINCE_LOAD"
    return tuple(by_tool.values())


def _number(value: object) -> float | None:
    """非数值、布尔值和非有限数均不参与健康判断。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def monitoring_checks(
    facts: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *, profile: dict[str, Any], policy: dict[str, Any],
) -> list[dict[str, Any]]:
    """按冻结 Profile 阈值判读最新样本，不把相对变化或峰值当持续故障。"""
    thresholds = dict(profile.get("metric_thresholds") or {})
    required_roles = dict(policy.get("required_metric_roles") or {})
    result = []
    assessed_roles: set[str] = set()
    for fact in facts:
        if fact.get("kind") == "inspection_monitoring_absence":
            code, role = str(fact["metric_code"]), str(fact.get("semantic_role") or "")
            title = display_metric_name(code)
            summary = f"{title}查询成功但未返回样本，存在监控盲区；不能据此判定数值为零或数据库异常。"
            result.append({
                "kind": "inspection_metric_check", "title": title,
                "check_status": "FINDING", "required": role in required_roles or role == "AVAILABILITY",
                "check_ids": [f"metric:{code}:{fact.get('dimensions') or '-'}"],
                "metric_code": code, "semantic_role": role, "dimensions": fact.get("dimensions"),
                "tool_id": "metric.query_range", "basis": "监控查询执行成功，应返回有效样本；空结果按监控保障风险报告。",
                "summary": summary, "observed_at": fact.get("observed_at"),
                "evidence_refs": [fact["evidence_ref"]] if fact.get("evidence_ref") else [],
                "findings": [{"finding_type": "MONITORING_SAMPLE_MISSING", "severity": "MEDIUM", "impact": summary}],
            })
            assessed_roles.add(role)
            continue
        if fact.get("kind") != "inspection_monitoring_metric":
            continue
        code = str(fact.get("metric_code") or "")
        role = str(fact.get("semantic_role") or "")
        value = _number(fact.get("latest"))
        band = thresholds.get(code)
        availability = role == "AVAILABILITY"
        if not band and not availability:
            continue
        required = role in required_roles or availability
        title = str(fact.get("title") or display_metric_name(code))
        current = format_metric_value(value, str(fact.get("unit") or ""))
        findings = []
        incomplete_reason = ""
        if value is None:
            status, basis = "MISSING", "未取得有效样本，不能完成本项判断。"
            incomplete_reason = basis
        elif availability:
            status = "CLEAR" if value == 1 else "FINDING"
            basis = "可用性最新有效样本应为 1；不代表全观察期连续可用。"
            if status == "FINDING":
                findings = [{"severity": "HIGH", "finding_type": "AVAILABILITY",
                             "impact": "最新可用性观测未达到正常状态。"}]
        else:
            warning = _number(band.get("warning_value"))
            critical = _number(band.get("critical_value"))
            if warning is None or critical is None or band.get("direction", "ABOVE") != "ABOVE":
                status, basis = "MISSING", "冻结阈值不完整，不能完成本项判断。"
                incomplete_reason = basis
            else:
                severity = "HIGH" if value >= critical else "MEDIUM" if value >= warning else None
                status = "FINDING" if severity else "CLEAR"
                basis = (
                    f"最新样本达到 {format_metric_value(warning, str(fact.get('unit') or ''))} 时需关注，"
                    f"达到 {format_metric_value(critical, str(fact.get('unit') or ''))} 时异常；"
                    "单次样本不证明持续超标。"
                )
                if severity:
                    findings = [{"severity": severity, "finding_type": role,
                                 "impact": f"{title}最新样本达到配置阈值。"}]
        coverage = _number(fact.get("observation_coverage_ratio", fact.get("coverage_ratio")))
        snapshot = fact.get("assessment_mode") == "CURRENT_SNAPSHOT"
        if snapshot:
            from .monitoring_observations import snapshot_rejection_reason
            basis += " 本项为当前快照，不代表过去全天连续正常。"
            rejection = snapshot_rejection_reason(fact)
            if rejection:
                status, findings = "MISSING", []
                incomplete_reason = rejection
                basis += " " + rejection
        elif coverage is not None and coverage < float(policy.get("minimum_monitoring_coverage_ratio", 0.8)):
            status = "PARTIAL"
            incomplete_reason = "观察期采样不足，不能据此判断整个观察区间。"
            basis += " " + incomplete_reason
        if status not in {"MISSING", "GAP", "PARTIAL"}:
            assessed_roles.add(role)
        result.append({
            "kind": "inspection_metric_check", "title": title,
            "check_status": status, "required": required,
            "check_ids": [f"metric:{code}:{fact.get('dimensions') or '-'}"],
            "metric_code": code, "semantic_role": role,
            "dimensions": fact.get("dimensions") or "-",
            "tool_id": "metric.query_range", "basis": basis,
            "summary": f"{title}：{current}。", "findings": findings,
            "incomplete_reason": incomplete_reason,
            "evidence_refs": [fact["evidence_ref"]] if fact.get("evidence_ref") else [],
            "observed_at": fact.get("observed_at") or fact.get("window_end"),
        })
    for role, title in required_roles.items():
        if role not in assessed_roles and not any(item.get("semantic_role") == role for item in result):
            result.append({
                "kind": "inspection_metric_check", "title": title, "required": True,
                "semantic_role": role, "check_ids": [f"metric-role:{role}"],
                "check_status": "MISSING", "summary": f"{title}未取得可按冻结规则判读的监控样本。",
                "basis": "标准巡检需要有效样本及明确的指标判断规则。",
            })
    return result


def _incomplete_action(fact, related_gaps, *, title, status):
    """按结构化原因给出下一步，不把程序错误或调度限流都归为权限缺失。"""
    codes = {str(item.get("code") or "") for item in related_gaps}
    prefix = f"“{title}”"
    if codes & {"RESOURCE_QUOTA_TARGET_EXHAUSTED", "RESOURCE_QUOTA_SOURCE_EXHAUSTED", "RESOURCE_QUOTA_GLOBAL_EXHAUSTED"}:
        return f"{prefix}等待共享只读额度释放后重试，并检查失败查询的清理和调度情况；不要直接扩大并发。"
    if codes & {"PRIVILEGE_MISSING", "SOURCE_PRIVILEGE_MISSING"}:
        return f"{prefix}核对该检查依赖的视图和函数，授予必要的最小只读权限后重试。"
    if codes & {"QUERY_SYNTAX_INVALID", "QUERY_COLUMN_INVALID", "QUERY_COLUMN_AMBIGUOUS", "QUERY_OBJECT_UNAVAILABLE"}:
        return f"{prefix}修复诊断 SQL 模板、参数绑定或数据库版本对象适配后重试；无需扩大数据库权限。"
    if codes & {"EXECUTOR_INTERNAL_ERROR"}:
        return f"{prefix}检查受控执行器的对应运行日志并修复执行错误后重试。"
    if codes & {"TIMEOUT", "EXECUTION_DEADLINE_EXCEEDED", "TARGET_UNREACHABLE", "SOURCE_UNREACHABLE"}:
        return f"{prefix}核对目标连接、查询耗时及执行器清理结果，恢复取数链路后重试。"
    if codes & {"AUTH_FAILED", "SOURCE_AUTH_FAILED"}:
        return f"{prefix}核对目标凭据及来源认证配置，连接验证成功后重试。"
    if fact.get("kind") == "inspection_metric_check":
        if status == "PARTIAL":
            return f"{prefix}核对来源采样间隔、查询窗口和本期有效样本覆盖率，恢复采集后重新检查。"
        return f"{prefix}核对目标的监控来源绑定、有效样本及冻结阈值规则，补齐后重新检查。"
    return f"{prefix}核对本项未完成原因及取数链路，修复后重新执行；本次不能据此判定正常。"


def _observation_display(rows):
    """SQL 优先展示同一行来源正文，摘要哈希只用于缺文本时定位；不改变原始观测。"""
    sql_keys = set(SQL_TEXT_COLUMNS) | {"digest"}
    formatted = []
    for row in rows[:3]:
        text = select_sql_text(row)
        fields, sql_added = [], False
        for key, value in row.items():
            if key in sql_keys:
                if sql_added:
                    continue
                sql_added = True
                if text:
                    label, statement = text
                    preview = statement[:400] + "……（展示已截短）" if len(statement) > 400 else statement
                    fields.append(f"{label}：{preview}")
                else:
                    locator = f"（Digest：{row['digest']}）" if row.get("digest") else ""
                    fields.append("SQL 文本：来源未返回 SQL 文本" + locator)
            else:
                fields.append(observation_field(str(key), value))
        formatted.append("，".join(fields[:6]))
    return "；".join(formatted)


def assess_inspection(
    facts: tuple[dict[str, Any], ...] | list[dict[str, Any]], *,
    policy: dict[str, Any], gaps: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> dict[str, Any]:
    """从检查结果形成专业结论，必要检查缺失不掩盖已知异常。"""
    required_specs = [item for item in policy.get("checks") or () if item.get("required")]
    required_ids = {item["check_id"] for item in required_specs}
    raw_checks = [dict(item) for item in facts if item.get("kind") in {
        "inspection_check", "inspection_metric_check", "backup_policy_check",
    }]
    seen_ids = {str(value) for item in raw_checks for value in item.get("check_ids") or ()}
    for spec in required_specs:
        if spec["check_id"] not in seen_ids:
            raw_checks.append({
                **spec, "check_ids": [spec["check_id"]], "check_status": "MISSING",
                "summary": f"{spec['title']}未形成检查结果。",
            })
    checks, issues, limitations, actions = [], [], [], []
    action_catalog = dict(policy.get("recommendations") or {})
    for fact in raw_checks:
        status = str(fact.get("check_status") or "MISSING")
        required = bool(fact.get("required")) or bool(required_ids.intersection(fact.get("check_ids") or ()))
        required = required or fact.get("kind") == "backup_policy_check"
        title = str(fact.get("title") or "巡检检查")
        incomplete_reason = str(fact.get("incomplete_reason") or "")
        if required and status in {"CLEAR", "COMPLIANT", "NOT_APPLICABLE", "FINDING", "BREACHED"}:
            from .backup_observations import backup_record_review_is_complete
            reviewed_backup = backup_record_review_is_complete(fact)
            if (not fact.get("evidence_refs") and not reviewed_backup) or fact.get("truncated"):
                status = "PARTIAL"
                if not incomplete_reason:
                    incomplete_reason = (
                        "检查结果缺少可追溯的证据引用。" if not fact.get("evidence_refs")
                        else str(fact.get("summary") or "备份来源未完成核验。") if fact.get("kind") == "backup_policy_check"
                        else "检查结果不完整或已截断，需要补齐后核验。"
                    )
        findings = [item for item in fact.get("findings") or () if isinstance(item, dict)]
        if status == "FINDING" and not findings:
            findings = [{"severity": "MEDIUM", "impact": fact.get("summary") or "检查规则报告风险，需复核详细对象。",
                         "finding_type": "INSPECTION_RISK"}]
        severity = max((str(item.get("severity") or "INFO") for item in findings),
                       key=lambda value: _SEVERITY.get(value, 0), default="INFO")
        if status == "BREACHED" and not findings:
            severity = "HIGH"
            findings = [{"severity": severity, "impact": fact.get("summary"), "finding_type": "BACKUP_POLICY"}]
        label = "异常" if findings and _SEVERITY.get(severity, 0) >= 4 else _STATUS_LABELS.get(status, "未完成")
        basis = str(fact.get("basis") or "")
        if not basis:
            thresholds = [item.get("threshold") for item in findings if isinstance(item.get("threshold"), dict)]
            basis = "；".join(dict.fromkeys(
                f"{display_field_name(str(item.get('metric') or ''))} {item.get('operator') or ''} {item.get('limit')}"
                for item in thresholds
            )) or str(fact.get("rule_description") or (
                "检查查询成功，未发现符合本项异常判定条件的记录。" if status == "CLEAR"
                else "本项仅展示观测结果，未进行异常判定。" if status == "OBSERVED"
                else "依据本项检查的查询条件与判定规则。"
            ))
        summary = str(fact.get("summary") or "")
        related_gaps = [item for item in gaps
                        if fact.get("tool_id") and item.get("step_id") == fact["tool_id"]]
        gap_details = [str(item["detail"]) for item in related_gaps if item.get("detail")]
        complete = status in {"CLEAR", "COMPLIANT", "NOT_APPLICABLE", "FINDING", "BREACHED"}
        if required and not complete:
            reason = incomplete_reason or (gap_details[0] if gap_details else summary) or "必要检查未完成"
            if reason.startswith(title + "："):
                reason = reason[len(title) + 1:]
            detail = f"{title}：{reason.rstrip('。；; ')}"
            limitations.append(detail)
            if fact.get("kind") == "backup_policy_check":
                actions.append(action_catalog.get("BACKUP_POLICY", "补齐备份成功记录并重新核验。"))
            else:
                actions.append(_incomplete_action(fact, related_gaps, title=title, status=status))
        observation_rows = [
            item for item in fact.get("observations") or () if isinstance(item, dict)
        ]
        actual = _observation_display(observation_rows)
        row = {
            "actual_display": actual or summary,
            "title": title, "status": status, "status_display": label,
            "required": required, "summary": summary, "basis": basis,
            "severity": severity, "check_ids": list(fact.get("check_ids") or ()),
            "observed_at": fact.get("observed_at"), "tool_id": fact.get("tool_id"),
            "metric_code": fact.get("metric_code"), "dimensions": fact.get("dimensions"),
        }
        checks.append(row)
        for finding in findings:
            finding_type = str(finding.get("finding_type") or "")
            object_ref = dict(finding.get("object_ref") or {})
            object_name = str(object_ref.get("object_name") or object_ref.get("sql_id") or "")
            impact = str(finding.get("impact") or summary or "已达到检查规则条件。")
            recommendation = finding.get("recommendation") or action_catalog.get(finding_type) or (
                f"核对“{title}”中{object_name or '对应对象'}的异常原因，按变更流程处理后复查同一对象及规则。"
            )
            issue = {
                "finding_id": str(finding.get("finding_id") or f"F{len(issues) + 1:02d}"),
                "title": title, "status": "FINDING",
                "severity": str(finding.get("severity") or severity),
                "confirmation": str(finding.get("confirmation") or "CONFIRMED"),
                "summary": impact, "impact": impact, "object_name": object_name,
                "basis": finding.get("basis") or basis, "recommendation": recommendation,
                "observed_at": fact.get("observed_at"),
                "evidence_refs": list(finding.get("evidence_refs") or fact.get("evidence_refs") or ()),
            }
            issues.append(issue)
            actions.append(recommendation)
    if not checks:
        limitations.append("本次未形成基础检查结果。")
    max_severity = max((_SEVERITY.get(item["severity"], 0) for item in issues if item["confirmation"] != "UNKNOWN"), default=0)
    health = "ABNORMAL" if max_severity >= 4 else "ATTENTION" if issues else "INCOMPLETE" if limitations else "NORMAL"
    complete = not limitations
    completed_required = sum(item["required"] and item["status"] in {
        "CLEAR", "COMPLIANT", "NOT_APPLICABLE", "FINDING", "BREACHED",
    } for item in checks)
    text = {
        "NORMAL": "数据库运行正常。本次必要检查已完成，未发现异常或需关注风险，无需额外处置。",
        "ATTENTION": f"数据库存在 {len(issues)} 项需关注风险，请按建议复核并处理。",
        "ABNORMAL": f"数据库存在异常，共发现 {len(issues)} 项问题，请优先处理高风险事项。",
        "INCOMPLETE": "本次检查未完成，暂不能形成完整的数据库正常结论。",
    }[health]
    if health == "INCOMPLETE" and completed_required:
        text = f"本次检查未完成；已完成的 {completed_required} 项必要检查未发现异常或需关注风险。"
    if limitations:
        text += f" 另有 {len(limitations)} 项必要检查未完成：" + "；".join(limitations[:3]) + "。"
    return {
        "schema_version": "INSPECTION_ASSESSMENT.v1", "policy_version": policy.get("version"),
        "health_status": health, "health_status_display": _HEALTH_LABELS[health],
        "completion_status": "COMPLETE" if complete else "PARTIAL",
        "completion_display": "完整" if complete else "部分完成",
        "text": text, "checks": checks, "findings": issues,
        "limitations": list(dict.fromkeys(limitations)),
        "recommendations": list(dict.fromkeys(actions)),
        "required_check_count": sum(item["required"] for item in checks),
        "completed_required_count": completed_required,
    }
