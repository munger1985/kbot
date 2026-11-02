"""生成脱离真实数据库的巡检出版样例，用于网页与 PDF 对照验收。"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from aiops_agent.application.inspections.monthly import monthly_business_review
from aiops_agent.application.reporting import report_presentation, resolve_system_template, render_pdf
from tests.unit.aiops_agent.test_aiops_professional_inspection_reports import (
    START, END, assessment_fact, full_assessment, issue_fact, metric_facts,
)


def inspection_samples():
    """业务样例只由同一冻结判定与展示函数生成。"""
    samples = {}
    for name, cpu, incomplete in (
        ("normal", 10, False), ("attention", 75, False),
        ("abnormal", 95, False), ("incomplete", 10, True),
    ):
        from aiops_agent.application.inspections.assessment import assess_inspection
        policy, checks, assessment = full_assessment(cpu)
        for check in checks:
            if check.get("kind") == "inspection_check" and not check.get("required"):
                check.update(check_status="OBSERVED", summary="已取得辅助观测，未设健康判断阈值。")
        if incomplete:
            checks[0]["check_status"] = "MISSING"
            checks[0]["summary"] = "只读权限不足，未完成连接使用率检查。"
        assessment = assess_inspection(checks, policy=policy)
        payload = {
            "report_type": "INSPECTION_DAILY", "title": "数据库日常巡检报告",
            "status": "READY" if not incomplete else "PARTIAL",
            "period_start": "2026-11-01T02:00:00+00:00",
            "period_end": "2026-11-02T02:00:00+00:00",
            "scope": {"inspection_assessment": assessment, "report_policy": policy,
                      "timezone": "Asia/Shanghai", "business_system_name": "生产业务系统",
                      "database_type": "MYSQL",
                      "target": {"display_name": "WMS 生产数据库", "target_id": "wms",
                                 "database_type": "MYSQL", "version_code": "8.4", "environment": "PRODUCTION"}},
            "facts": [*checks, *metric_facts(cpu)], "gaps": [], "evidence_refs": [],
            "summary": assessment["text"], "recommendations": assessment["recommendations"],
        }
        samples[name] = report_presentation(
            payload=payload, template=resolve_system_template("system:inspection.daily"),
            report_context={"report_id": "sample-" + name, "report_version": 1},
        )
    from aiops_agent.application.reporting import monthly_inspection_summary
    facts = []
    members = [{"target_id": target, "display_name": target.upper() + " 生产库"}
               for target in ("erp", "wms", "crm")]
    for member in members:
        target = member["target_id"]
        for day in range(31):
            moment = START + timedelta(days=day)
            facts.append(assessment_fact(moment, target=target))
            for metric in metric_facts():
                facts.append({
                    **metric, "_source_target_id": target, "_source_target_name": member["display_name"],
                    "_source_observed_at": moment.isoformat(),
                    "value_kind": "STATE" if metric["semantic_role"] == "AVAILABILITY" else "GAUGE",
                    "daily_points_timezone": "UTC",
                    "daily_points": [{"date": moment.date().isoformat(),
                                      "value": metric["average"] if metric["semantic_role"] == "AVAILABILITY" else round(metric["average"] + day / 10, 2)}],
                })
    facts.append(issue_fact(START + timedelta(days=5), target="erp"))
    issue_assessment = assessment_fact(START + timedelta(days=5), target="erp", health="ABNORMAL")
    issue_assessment["assessment"]["recommendations"] = ["复查 ERP 阻塞源会话及业务归属，取得同对象处理后的验证记录。"]
    facts.append(issue_assessment)
    summary = monthly_inspection_summary(facts=facts, period_start=START, period_end=END, timezone="UTC")
    summary.update(monthly_business_review(facts=facts, period_start=START, period_end=END, timezone="UTC", members=members))
    payload = {
        "report_type": "INSPECTION_MONTHLY", "title": "数据库月度巡检报告", "status": "READY",
        "period_start": START.isoformat(), "period_end": END.isoformat(),
        "scope": {"monthly_summary": summary, "timezone": "UTC", "scope_type": "BUSINESS_SYSTEM",
                  "scope_name": "生产业务系统", "target_count": 3, "database_types": ["MYSQL"],
                  "member_targets": members},
        "facts": facts, "gaps": [], "evidence_refs": [], "summary": summary["conclusion"]["text"],
    }
    samples["monthly"] = report_presentation(
        payload=payload, template=resolve_system_template("system:inspection.monthly"),
        report_context={"report_id": "sample-monthly", "report_version": 1},
    )
    return samples


def main():
    """样例生成与生产运行分离，不访问业务数据库。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = inspection_samples()
    (args.output / "samples.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    for name, presentation in samples.items():
        (args.output / f"{name}.pdf").write_bytes(render_pdf(presentation))
    print(f"已生成 {len(samples)} 份隔离巡检样例：{args.output}")


if __name__ == "__main__":
    main()
