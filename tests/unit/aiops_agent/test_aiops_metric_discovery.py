"""目标限定指标发现从规划、冻结、执行到后续可引用证据的回归。"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.investigation.query_freezing import prepare_source_queries
from aiops_agent.contracts.evidence import MetricDiscoveryResult, ObservationSet
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, MetricsDiscoveryRequest
from aiops_agent.workers.evidence_handlers import EvidenceObserveHandler
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7


def request(**overrides):
    end = datetime.now(timezone.utc)
    return MetricsDiscoveryRequest(
        action_id="a1", target_id=str(uuid7()), binding_id=str(uuid7()),
        source_locator_key="bound-target", window_start=end - timedelta(minutes=15),
        window_end=end, trace_id="test",
    ).model_copy(update=overrides)


def adapter(payload):
    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    calls = []
    value = PrometheusAdapter.__new__(PrometheusAdapter)
    value.context = DiagnosticSourceContext(
        source_id=str(uuid7()), source_type="PROMETHEUS", adapter_id="prometheus",
        adapter_version="1.0.0", config_version=1, endpoint="http://prometheus.example.com",
    )
    value._session = SimpleNamespace(get=lambda *args, **kwargs: (calls.append(kwargs) or Response()))
    value._timeout = None
    value._response_json = AsyncMock(return_value=(payload, "a" * 64))
    return value, calls


def test_adapter_discovery_is_target_bound_bounded_and_does_not_invent_metric_type():
    value, calls = adapter({"status": "success", "data": [
        {"__name__": "custom_workload_new_metric", "instance": "bound-target", "application": "mes"},
        {"__name__": "db_connections", "target_key": "bound-target"},
    ]})
    result = asyncio.run(value.discover_metrics(request(limit=1)))
    assert len(result.entries) == 1 and result.truncated
    assert result.entries[0].metric_type == "UNKNOWN"
    assert calls[0]["params"]["match[]"] == ['{instance="bound-target"}', '{target_key="bound-target"}']
    assert value._response_json.call_args.kwargs["max_bytes"] == 262144
    assert result.warnings


def test_adapter_rejects_other_target_and_missing_host_mapping():
    value, _ = adapter({"status": "success", "data": [{"__name__": "cpu", "instance": "other-target"}]})
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.discover_metrics(request()))
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.discover_metrics(request(scope="HOST")))


def test_label_catalog_truncation_is_reported():
    value, _ = adapter({"status": "success", "data": [{
        "__name__": "workload_latency", "instance": "bound-target",
        **{f"label_{index}": "value" for index in range(150)},
    }]})
    result = asyncio.run(value.discover_metrics(request()))
    assert result.truncated
    assert len(result.entries[0].label_names) == 128


def test_metric_discovery_requires_explicit_source_capability():
    from aiops_agent.application.investigation.discovery import available_tools
    from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot

    capabilities = DbaCapabilitySnapshot(
        agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
        database_type="POSTGRESQL", database_version="16",
        target_enabled=True, target_reachable=True,
        source_snapshots=({"source_id": str(uuid7()), "source_type": "PROMETHEUS", "enabled": True,
            "reachable": True, "capabilities": ("metric.query_range",)},),
    )
    builder = SimpleNamespace(discover_tools=lambda _: ())
    assert "monitor.discover_metrics" not in {item["tool_id"] for item in available_tools(builder, capabilities)}
    source = capabilities.source_snapshots[0].model_copy(update={"capabilities": ("metric.discover",)})
    allowed = capabilities.model_copy(update={"source_snapshots": (source,)})
    assert "monitor.discover_metrics" in {item["tool_id"] for item in available_tools(builder, allowed)}


def test_label_value_is_escaped_and_bytes_limit_failure_is_not_empty_success():
    value, calls = adapter({"status": "success", "data": []})
    result = asyncio.run(value.discover_metrics(request(source_locator_key='a"b\\c')))
    assert not result.entries
    assert calls[0]["params"]["match[]"][0] == '{instance="a\\"b\\\\c"}'
    value._response_json.side_effect = DiagnosticSourceAdapterError("SOURCE_RESPONSE_TOO_LARGE", "超过字节预算")
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.discover_metrics(request()))


def test_planning_freezes_only_scope_window_and_limit():
    from platform_core.contracts.aiops.investigation import InvestigationPlanningOutput

    value = InvestigationPlanningOutput.model_validate({
        "input_envelope": {"materials": [{"item_no": 1, "material_kind": "QUESTION", "summary": "发现可用指标", "confidence": 1}], "explicit_question": "有哪些指标？"},
        "task_frame": {"objectives": ["DIAGNOSE"], "problem_statement": "确认可用指标", "success_criteria": ["得到目标目录"]},
        "plan": {"revision_no": 1, "actions": [{
            "action_id": "a1", "question": "有哪些指标可用？", "tool_id": "monitor.discover_metrics",
            "input": {"scope": "DATABASE"}, "expected_evidence_kind": "METRIC_CATALOG",
            "measurement_semantics": "NOT_APPLICABLE",
        }]},
    })
    prepared, queries = prepare_source_queries(value)
    assert queries["metric_discoveries"] == [{"action_id": "a1", "scope": "DATABASE", "window_seconds": 900, "limit": 50}]
    action = prepared.plan.actions[0].model_copy(update={"input": {"scope": "DATABASE", "selector": '{instance=~".*"}'}})
    with pytest.raises(ValueError):
        prepare_source_queries(prepared.model_copy(update={"plan": prepared.plan.model_copy(update={"actions": (action,)})}))


def test_worker_checks_binding_authorization_then_projects_catalog_for_replanning():
    async def run():
        req = request()
        source_id = str(uuid7())
        catalog = MetricDiscoveryResult(
            action_id="a1", target_id=req.target_id, binding_id=req.binding_id, source_id=source_id,
            scope="DATABASE", window_start=req.window_start, window_end=req.window_end,
            captured_at=req.window_end, entries=({"name": "brand_new_latency_seconds", "label_names": ("instance",)},),
            response_sha256="a" * 64,
        )
        discovery_adapter = SimpleNamespace(discover_metrics=AsyncMock(return_value=catalog))
        registry = SimpleNamespace(create=lambda *args, **kwargs: discovery_adapter)
        binding = {
            "binding_id": str(req.binding_id), "unsupported_metrics": [], "effective_capabilities": ["metric.discover"],
            "source_locator_key": req.source_locator_key, "source_locator": {},
            "source": {"source_id": str(source_id), "source_type": "PROMETHEUS", "adapter_id": "prometheus",
                "adapter_version": "1.0.0", "config_version": 1, "endpoint": "http://prometheus.example.com",
                "declared_capabilities": {"metric.discover": {}}, "config": {}},
        }
        context = SimpleNamespace(
            target_id=req.target_id, task_key=f"observe:{req.binding_id}", trace_id="test",
            plan_snapshot={"monitoring": {
                "window": {"start": req.window_start.isoformat(), "end": req.window_end.isoformat()},
                "bindings": [binding], "max_response_bytes": 262144,
                "metric_discoveries": [{"action_id": "a1", "scope": "DATABASE", "window_seconds": 900, "limit": 50}],
            }},
        )
        handler = EvidenceObserveHandler(diagnostic_source_registry=registry, secret_store=object())
        result = await handler.execute(context)
        assert result.metric_catalogs[0].entries[0].name == "brand_new_latency_seconds"
        assert not result.observations
        binding["effective_capabilities"] = []
        denied = await handler.execute(context)
        assert not denied.metric_catalogs
        assert denied.gaps[0].code == "METRIC_DISCOVERY_NOT_AUTHORIZED"
        discovery_adapter.discover_metrics.assert_awaited_once()

        # 通过真实充分性投影，目录不是普通性能数值，也不受固定指标 Profile 过滤。
        from aiops_agent.workers.handlers import TaskExecutionContext
        assessment = await DbaEvidenceAssessmentHandler().execute(TaskExecutionContext(
            run_id=str(uuid7()), task_id=str(uuid7()), task_key="assess", target_id=req.target_id,
            agent_id=str(uuid7()), actor_id="tester", trigger_type="USER", trace_id="test",
            attempt=1, deadline_at=None, plan_snapshot={"answer_context": {"task_frame": {"objectives": ["DIAGNOSE"]}}},
            policy_snapshot={}, input_artifacts=({
            "artifact_id": str(uuid7()), "schema_version": "OBSERVATION_SET.v1", "payload": result.model_dump(mode="json"),
        },)))
        fact = next(item for item in assessment.evidence if item.tool_id == "monitor.discover_metrics")
        assert fact.rows[0][0] == "brand_new_latency_seconds"
        assert str(fact.measurement_semantics) == "NOT_APPLICABLE"
        assert fact.trust_level == "SOURCE_VERIFIED"

    asyncio.run(run())
