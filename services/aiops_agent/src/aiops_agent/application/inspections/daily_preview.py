"""日报模板预览只生成明确标记的模拟事实，不连接数据库或调用模型。"""

from datetime import UTC, datetime, timedelta

from .daily_sections import daily_section_catalog, normalize_daily_layout
from .assessment import assess_inspection


def daily_preview(*, database_type: str, scenario: str, selected_check_ids=(), daily_report_layout=None):
    from aiops_agent.application.reporting import report_presentation, resolve_system_template
    from .check_catalog import load_check_catalog

    catalog = load_check_catalog()
    selected = set(selected_check_ids)
    checks = []
    now = datetime.now(UTC).replace(microsecond=0)
    for section in daily_section_catalog()["sections"]:
        if section["section_id"] in {"OVERVIEW", "BASELINE", "ACTIONS"}:
            continue
        chosen = next((check for group in catalog.groups for check in group.checks
                       if check.report_section == section["section_id"] and database_type in check.supported_db_types
                       and (not selected or check.check_id in selected)), None)
        key = chosen.check_id if chosen else "preview." + section["section_id"].lower()
        title = chosen.display_name if chosen else section["title"]
        checks.append({"kind": "inspection_check", "report_section": section["section_id"],
            "title": title, "tool_id": chosen.tool_id if chosen else "preview.readonly",
            "check_ids": [key], "required": True, "observed_at": now.isoformat(),
            "check_status": "CLEAR", "summary": "【模拟样例】检查完成，未发现异常。",
            "basis": "模拟结果，仅用于版式预览，不代表真实数据库状态。",
            "evidence_refs": ["sample:template-preview"], "observations": []})
    if scenario == "ATTENTION":
        check = next(item for item in checks if item["report_section"] == "CAPACITY")
        check.update(check_status="FINDING", summary="【模拟样例】可用容量达到预警条件。",
            findings=[{"finding_type": "INSPECTION_RISK", "severity": "MEDIUM",
                "impact": "【模拟样例】业务存储余量不足，需要复核增长和容量上限。",
                "recommendation": "核对容量上限及实际可用空间，评估清理或扩容后复查。"}])
    elif scenario == "INCOMPLETE":
        check = next(item for item in checks if item["report_section"] == "RESOURCES")
        check.update(check_status="MISSING", summary="【模拟样例】未取得资源检查结果。")
    # SQL 空态也必须来自明确的模拟完整结果，而非把无结果判断为正常。
    sql_check = next(item for item in checks if item["report_section"] == "SQL")
    from .sql_problems import PROBLEM_SQL_COLUMNS, freeze_problem_sql_check, problem_sql_inputs, problem_sql_policy
    sql_policy = problem_sql_policy()
    row = {key: None for key in PROBLEM_SQL_COLUMNS}
    row["collection_status"] = "AVAILABLE"
    if scenario == "ATTENTION":
        row.update(sql_key="preview-sql", schema_name="sample_app", text_kind="NORMALIZED",
            sql_text="SELECT order_id, status FROM orders WHERE customer_id = ? ORDER BY created_at",
            execution_count=100, total_elapsed_ms=160000, average_elapsed_ms=1600, maximum_elapsed_ms=2200)
    sql_check.update(freeze_problem_sql_check(
        {**sql_check, "tool_id": "db.sql.problem_candidates", "observations": [row]},
        step={"sql_problem_policy": sql_policy, "input": problem_sql_inputs(sql_policy)}, database_type=database_type,
    ))
    checks.append({"kind": "inspection_sql_reviews", "reviews": [{
        "problem_id": problem["problem_id"], "status": "COMPLETE",
        "analysis": "【模拟分析】单次耗时达到筛查条件，需要结合业务响应要求与执行计划复核。",
        "possible_causes": [{"possible_cause": "【模拟】访问路径或排序可能增加耗时。",
            "supporting_metrics": ["average_elapsed_ms"], "verification": "读取执行计划，核对扫描量和排序步骤。"}],
        "recommendations": [{"action": "【模拟】核对过滤列与排序列上的现有索引。",
            "rationale": "确认是否能减少扫描与排序，不能直接据此创建索引。",
            "validation": "同参数比较耗时、扫描量及业务结果。", "requires_approval": True}],
        "limitations": ["模拟样例，未取得真实执行计划，不确认根因。"], "evidence_refs": ["sample:template-preview"],
    } for problem in sql_check.get("sql_problems") or ()]})
    metrics = []
    for role, code, label, value, unit, dimension in (
        ("CPU_UTILIZATION", "host.cpu.utilization", "主机 CPU 使用率", 24, "percent", "host=模拟主机"),
        ("MEMORY_UTILIZATION", "runtime.memory.utilization", "运行环境内存使用率", 35, "percent", "container=模拟运行环境"),
        ("TRANSACTION_THROUGHPUT", "db.transaction.throughput", "事务吞吐量", 120, "transactions_per_second", "database=模拟实例"),
    ):
        metrics.append({"kind": "inspection_monitoring_metric", "metric_code": code, "title": label,
            "semantic_role": role, "latest": value, "average": value-3, "maximum": value+2,
            "unit": unit, "dimensions": dimension, "value_kind": "GAUGE", "coverage_ratio": 1,
            "window_start": (now-timedelta(days=7)).isoformat(), "window_end": now.isoformat(),
            "daily_points": [{"date": (now-timedelta(days=day)).date().isoformat(), "value": value-3}
                for day in range(7, 0, -1)], "evidence_ref": "sample:template-preview"})
    assessment = assess_inspection(checks, policy={})
    payload = {"report_type": "INSPECTION_DAILY", "title": "【模拟预览】数据库日常巡检报告",
        "period_start": (now-timedelta(days=1)).isoformat(), "period_end": now.isoformat(),
        "scope": {"inspection_assessment": assessment, "database_type": database_type,
            "timezone": "Asia/Shanghai", "business_system_name": "模拟业务系统",
            "daily_report_layout": normalize_daily_layout(daily_report_layout),
            "target": {"display_name": "模拟数据库（非真实运行）", "database_type": database_type,
                       "environment": "PREVIEW"}},
        "facts": [*checks, *metrics], "gaps": []}
    presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    presentation["daily_inspection"]["preview"] = True
    return presentation
