"""只读验收真实 Prometheus 到巡检监控结论；不创建批次、不验证 SQL 或备份。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import aiohttp

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    for source in reversed((root / "packages/platform_core/src", root / "services/aiops_agent/src")):
        sys.path.insert(0, str(source))

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
from aiops_agent.application.diagnosis.monitoring_calculations import validate_monitoring_trace_sources
from aiops_agent.application.inspections.assessment import assess_inspection, freeze_inspection_policy, monitoring_checks
from aiops_agent.application.inspections.monitoring_observations import classify_monitoring_facts
from aiops_agent.application.reporting import render_pdf, report_presentation, resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.evidence import ObservationSet
from aiops_agent.domain.evidence.metric_time import canonical_metric_window
from aiops_agent.monitoring.catalog import load_monitoring_profile_catalog
from aiops_agent.playbooks import canonical_hash
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, MetricsEvidenceRequest
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7


async def smoke(args):
    catalog = load_metric_catalog()
    profile = load_monitoring_profile_catalog(catalog).overview_for_database(args.database_type, args.database_version)
    policy = freeze_inspection_policy(args.database_type)
    # 此程序仅验收监控链路，不把未执行的 SQL 和备份核验冒充为已完成。
    policy["checks"] = []
    now = datetime.now(UTC)
    start, end = canonical_metric_window(now - timedelta(seconds=policy["current_monitoring_window_seconds"]), now)
    target_id, binding_id, source_id, artifact_id = (str(uuid7()) for _ in range(4))
    context = DiagnosticSourceContext(
        source_id=source_id, source_type="PROMETHEUS", adapter_id="prometheus", adapter_version="1.0.0",
        config_version=1, endpoint=args.endpoint, declared_capabilities={"metric.query_range": {}},
    )
    request = MetricsEvidenceRequest(
        target_id=target_id, binding_id=binding_id, source_locator_key=args.target_key,
        source_locator={"target_key": args.target_key, "host_target_key": args.host_target_key},
        metric_definitions=catalog.select(profile.metric_codes, db_type=args.database_type),
        window_start=start, window_end=end, requested_step_seconds=60, max_response_bytes=1048576,
        trace_id="inspection-current-metrics-smoke",
    )
    async with aiohttp.ClientSession() as session:
        adapter = PrometheusAdapter(context=context, session=session, request_timeout_seconds=15,
                                    webhook_replay_seconds=300)
        result = await adapter.query_metrics(request)
    raw = ObservationSet(target_id=target_id, binding_id=binding_id, source_id=source_id,
                         observations=result.observations, gaps=result.gaps, collected_at=now)
    evidence = DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=artifact_id, result=raw)
    if evidence is None:
        raise RuntimeError("真实来源未返回可判读的监控序列")
    source_payload = raw.model_dump(mode="json")
    artifact = SimpleNamespace(ops_run_id=target_id, schema_version="OBSERVATION_SET.v1",
                              payload_json=source_payload, content_hash=canonical_hash(source_payload))

    async def get_artifact(*, artifact_id):
        return artifact

    replayed = await validate_monitoring_trace_sources(
        uow=SimpleNamespace(runs=SimpleNamespace(get_artifact=get_artifact)),
        run=SimpleNamespace(target_id=target_id, ops_run_id=target_id),
        payload={"evidence": [evidence.model_dump(mode="json")]},
    )
    if not replayed:
        raise RuntimeError("真实监控原始输入与冻结计算回放不一致")
    frozen = AIOpsRuntimeService._inspection_monitoring_facts(SimpleNamespace(evidence=(evidence,)))
    facts = classify_monitoring_facts(frozen, monitoring={
        "current_window": {"start": start.isoformat(), "end": end.isoformat()},
        "snapshot_max_age_seconds": policy["maximum_current_sample_age_seconds"],
    })
    checks = monitoring_checks(facts, profile=profile.model_dump(mode="json"), policy=policy)
    assessment = assess_inspection(checks, policy=policy)
    presentation = report_presentation(payload={
        "report_type": "INSPECTION_DAILY", "title": "巡检监控链路只读验收（非完整巡检）",
        "period_start": (now - timedelta(days=1)).isoformat(), "period_end": now.isoformat(),
        "scope": {"inspection_assessment": assessment}, "facts": [*facts, *checks], "gaps": [],
    }, template=resolve_system_template("system:inspection.daily"))
    pdf = render_pdf(presentation)
    if not pdf.startswith(b"%PDF-"):
        raise RuntimeError("监控判读未能生成有效 PDF")
    print(json.dumps({
        "scope": "MONITORING_ONLY", "boundary": "未执行数据库 SQL 或真实备份核验，非线上巡检批次",
        "window_start": start.isoformat(), "window_end": end.isoformat(),
        "metric_count": len(result.observations), "series_count": len(facts),
        "source_gap_codes": [gap.code for gap in result.gaps],
        "raw_trace_replayed": replayed, "pdf_bytes": len(pdf),
        "completed_required_count": assessment["completed_required_count"],
        "required_check_count": assessment["required_check_count"],
        "health_status": assessment["health_status"], "limitations": assessment["limitations"],
        "checks": [{"metric_code": check.get("metric_code"), "status": check["check_status"],
                    "summary": check["summary"]} for check in checks],
    }, ensure_ascii=False, indent=2))
    if assessment["completion_status"] != "COMPLETE":
        raise RuntimeError("必要监控检查未全部完成，请查看限制原因")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="已授权的无认证只读 Prometheus 地址，可使用 SSH 隧道")
    parser.add_argument("--target-key", required=True)
    parser.add_argument("--host-target-key", required=True)
    parser.add_argument("--database-type", choices=("MYSQL", "ORACLE", "POSTGRESQL"), required=True)
    parser.add_argument("--database-version", required=True, help="目标库登记的标准版本代码")
    asyncio.run(smoke(parser.parse_args()))


if __name__ == "__main__":
    main()
