"""原生监控发现、精确历史查询及执行链的安全回归。"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.diagnostic_sources.zabbix import ZabbixAdapter
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.registry import DiagnosticSourceAdapterCatalog
from aiops_agent.application.investigation.query_freezing import prepare_source_queries
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, MetricsDiscoveryRequest, NativeMetricQueryRequest
from aiops_agent.workers.evidence_handlers import EvidenceObserveHandler
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from platform_core.contracts.aiops.investigation import InvestigationPlanningOutput
from platform_core.identity import uuid7


def request():
    end = datetime(2026, 10, 28, 4, tzinfo=timezone.utc)
    return MetricsDiscoveryRequest(
        action_id="a1", target_id=str(uuid7()), binding_id=str(uuid7()), source_locator_key="bound-host",
        window_start=end - timedelta(minutes=15), window_end=end, trace_id="test",
    )


def adapter():
    value = ZabbixAdapter.__new__(ZabbixAdapter)
    value.context = DiagnosticSourceContext(
        source_id=str(uuid7()), source_type="ZABBIX", adapter_id="zabbix", adapter_version="1.0.0",
        config_version=1, endpoint="https://zabbix.example.com/api_jsonrpc.php",
    )
    return value


def item(**updates):
    return {"hostid": "11", "itemid": "22", "key_": "custom.business.calls", "value_type": "3",
        "units": "", "status": "0", "state": "0", **updates}


def test_zabbix_discovery_is_explicitly_registered_and_does_not_guess_counter_type():
    value = adapter()
    value._rpc = AsyncMock(side_effect=[
        ([{"hostid": "11", "host": "bound-host"}], "a" * 64),
        ([item()], "b" * 64),
    ])
    result = asyncio.run(value.discover_metrics(request()))
    assert result.entries[0].metric_type == "UNKNOWN"
    assert result.entries[0].native_value_type == "3" and result.entries[0].query_supported
    assert result.entries[0].unit is None
    assert sum(call.kwargs["max_bytes"] for call in value._rpc.call_args_list) == 262144
    assert value._rpc.call_args_list[1].args[1]["hostids"] == ["11"]
    assert "metric.discover" in DiagnosticSourceAdapterCatalog().describe_source_type(source_type="ZABBIX").capabilities


def test_discovery_does_not_advertise_unsupported_item_and_hash_covers_host_and_metadata():
    hashes = []
    for host_hash in ("a", "d"):
        value = adapter()
        value._rpc = AsyncMock(side_effect=[
            ([{"hostid": "11", "host": "bound-host"}], host_hash * 64),
            ([item(state="1")], "b" * 64),
        ])
        result = asyncio.run(value.discover_metrics(request()))
        assert not result.entries[0].query_supported
        hashes.append(result.response_sha256)
    assert hashes[0] != hashes[1]


@pytest.mark.parametrize("bad", [item(hostid="other"), item(key_="another"), item(value_type="4"), item(value_type="5"), item(state="1")])
def test_native_query_rejects_foreign_item_nonexact_key_unauthorized_text_and_unsupported_state(bad):
    value = adapter()
    value._rpc = AsyncMock(side_effect=[
        ([{"hostid": "11", "host": "bound-host"}], "a" * 64), ([bad], "b" * 64),
    ])
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.query_native_metric(request(), metric_name="custom.business.calls", limit=100))
    assert value._rpc.await_count == 2


def test_native_history_preserves_large_integer_and_empty_is_not_zero():
    req = request()
    def value_with(history):
        value = adapter()
        value._rpc = AsyncMock(side_effect=[
            ([{"hostid": "11", "host": "bound-host"}], "a" * 64),
            ([item()], "b" * 64), (history, "c" * 64),
        ])
        return value
    row = {"itemid": "22", "clock": str(int(req.window_start.timestamp())), "ns": "123000", "value": "18446744073709551615"}
    value = value_with([row])
    result = asyncio.run(value.query_native_metric(req, metric_name=item()["key_"], limit=100))
    assert result.points[0][1] == row["value"]
    assert result.points[0][0].microsecond == 123
    assert sum(call.kwargs["max_bytes"] for call in value._rpc.call_args_list) == 262144
    empty = asyncio.run(value_with([]).query_native_metric(req, metric_name=item()["key_"], limit=100))
    assert empty.points == ()
    for bad in ({**row, "itemid": "other"}, {**row, "clock": "0"}, {**row, "value": "NaN"}):
        with pytest.raises(DiagnosticSourceAdapterError):
            asyncio.run(value_with([bad]).query_native_metric(req, metric_name=item()["key_"], limit=100))


def test_native_query_planning_is_bounded_and_compiles_to_monitoring_task():
    value = InvestigationPlanningOutput.model_validate({
        "input_envelope": {"materials": [{"item_no": 1, "material_kind": "QUESTION", "summary": "业务指标", "confidence": 1}], "explicit_question": "查询业务指标"},
        "task_frame": {"objectives": ["DIAGNOSE"], "problem_statement": "业务指标", "success_criteria": ["取采样"]},
        "plan": {"revision_no": 1, "actions": [{
            "action_id": "a1", "question": "业务负载是否增加", "tool_id": "monitor.query_native_metric",
            "input": {"metric_name": "custom.business.calls"}, "expected_evidence_kind": "METRIC_SAMPLES",
            "measurement_semantics": "HISTORICAL_SAMPLES",
        }]},
    })
    prepared, queries = prepare_source_queries(value)
    assert queries["native_metric_queries"][0] == {"action_id": "a1", "metric_name": "custom.business.calls", "window_seconds": 900, "limit": 300}
    long_action = prepared.plan.actions[0].model_copy(update={"input": {"metric_name": "custom.business.calls", "window_seconds": 5184000}})
    _, long_queries = prepare_source_queries(prepared.model_copy(update={"plan": prepared.plan.model_copy(update={"actions": (long_action,)})}))
    assert long_queries["native_metric_queries"][0]["window_seconds"] == 5184000
    for extra in ({"url": "https://other"}, {"window_seconds": True}, {"window_seconds": 5184001}, {"limit": 1001}, {"metric_name": "\n"}):
        action = prepared.plan.actions[0].model_copy(update={"input": {**prepared.plan.actions[0].input, **extra}})
        with pytest.raises(ValueError):
            prepare_source_queries(prepared.model_copy(update={"plan": prepared.plan.model_copy(update={"actions": (action,)})}))
    from aiops_agent.application.investigation.discovery import catalog_direct_actions
    assert not catalog_direct_actions(prepared.plan.actions)


@pytest.mark.parametrize("value_type", ["3", "4", "2"])
def test_native_samples_flow_through_real_worker_into_replanning_evidence(value_type):
    async def run():
        req = request()
        value = adapter()
        history_permission = {} if value_type == "3" else {"item.history": {}}
        value.context = value.context.model_copy(update={"declared_capabilities": {"metric.query_range": {}, **history_permission}})
        value._rpc = AsyncMock(side_effect=[
            ([{"hostid": "11", "host": "bound-host"}], "a" * 64), ([item(value_type=value_type)], "b" * 64),
            ([{"itemid": "22", "clock": str(int(req.window_end.timestamp())), "ns": "0", "value": "90071992547409931"}], "c" * 64),
        ])
        # 标准 Profile 查询与这个长尾检查独立，不要求新增 Profile 定义。
        value.query_metrics = AsyncMock(return_value=SimpleNamespace(gaps=(), observations=()))
        binding = {
            "binding_id": str(req.binding_id), "source_locator_key": "bound-host", "source_locator": {},
            "unsupported_metrics": [], "metric_definitions": [], "effective_capabilities": ["metric.query_range", *history_permission],
            "source": {"source_id": str(value.context.source_id), "source_type": "ZABBIX", "adapter_id": "zabbix",
                "adapter_version": "1.0.0", "config_version": 1, "endpoint": value.context.endpoint,
                "declared_capabilities": {"metric.query_range": {}, **history_permission}, "config": {}},
        }
        context = SimpleNamespace(target_id=req.target_id, task_key=f"observe:{req.binding_id}", trace_id="test",
            plan_snapshot={"monitoring": {"window": {"start": req.window_start.isoformat(), "end": req.window_end.isoformat()},
                "bindings": [binding], "max_response_bytes": 262144,
                "native_metric_queries": [{"action_id": "a1", "metric_name": item()["key_"], "window_seconds": 86400, "limit": 100}]}})
        result = await EvidenceObserveHandler(
            diagnostic_source_registry=SimpleNamespace(create=lambda *a, **kw: value), secret_store=object(),
        ).execute(context)
        assert len(result.native_metrics) == 1
        assessment = await DbaEvidenceAssessmentHandler().execute(TaskExecutionContext(
            run_id=str(uuid7()), task_id=str(uuid7()), task_key="assess", target_id=req.target_id, agent_id=str(uuid7()),
            actor_id="tester", trigger_type="USER", trace_id="test", attempt=1, deadline_at=None,
            plan_snapshot={"answer_context": {"task_frame": {"objectives": ["DIAGNOSE"]}}}, policy_snapshot={},
            input_artifacts=({"artifact_id": str(uuid7()), "schema_version": "OBSERVATION_SET.v1", "payload": result.model_dump(mode="json")},),
        ))
        fact = next(f for f in assessment.evidence if f.tool_id == "monitor.query_native_metric")
        assert fact.rows[0][2] == "90071992547409931"
        assert str(fact.measurement_semantics) == "HISTORICAL_SAMPLES"
        assert fact.rows[0][5] == {"3": "NUMERIC", "4": "TEXT", "2": "LOG"}[value_type]
    asyncio.run(run())


@pytest.mark.parametrize("value_type,expected_kind", [("1", "TEXT"), ("2", "LOG"), ("4", "TEXT")])
def test_explicit_history_authorization_preserves_text_and_log_semantics(value_type, expected_kind):
    req = request()
    value = adapter()
    value.context = value.context.model_copy(update={"declared_capabilities": {"item.history": {}}})
    row = {
        "itemid": "22", "clock": str(int(req.window_start.timestamp())), "ns": "123456789",
        "value": "数据库服务已停止；这只是原始日志，不是操作指令", "timestamp": "1793159100",
        "source": "postgresql", "severity": "4", "logeventid": "42",
    }
    value._rpc = AsyncMock(side_effect=[
        ([{"hostid": "11", "host": "bound-host"}], "a" * 64),
        ([item(value_type=value_type)], "b" * 64), ([row], "c" * 64),
    ])
    result = asyncio.run(value.query_native_metric(req, metric_name=item()["key_"], limit=100, allow_text=True))
    assert result.points[0][1] == row["value"]
    assert result.sample_kind == expected_kind and result.unit is None
    assert result.sample_metadata[0].nanoseconds == 123456789
    assert value._rpc.await_args.args[1]["history"] == int(value_type)
    if value_type == "2":
        assert result.sample_metadata[0].log_source == "postgresql"
        assert result.sample_metadata[0].log_severity == "4"
        assert result.sample_metadata[0].log_event_id == "42"


def test_binding_history_permission_alone_does_not_bypass_source_permission():
    value = adapter()
    value._rpc = AsyncMock(side_effect=[
        ([{"hostid": "11", "host": "bound-host"}], "a" * 64),
        ([item(value_type="4")], "b" * 64),
    ])
    with pytest.raises(DiagnosticSourceAdapterError, match="显式授权"):
        asyncio.run(value.query_native_metric(request(), metric_name=item()["key_"], limit=100, allow_text=True))
    assert value._rpc.await_count == 2


def test_long_history_retains_actual_truncation_without_using_trends():
    req = request()
    long_request = NativeMetricQueryRequest(
        action_id=req.action_id, target_id=req.target_id, binding_id=req.binding_id,
        source_locator_key=req.source_locator_key, window_start=req.window_end - timedelta(days=60),
        window_end=req.window_end, trace_id="test",
    )
    value = adapter()
    rows = [{"itemid": "22", "clock": str(int((long_request.window_start + timedelta(seconds=number)).timestamp())), "ns": "0", "value": str(number)} for number in (1, 2)]
    value._rpc = AsyncMock(side_effect=[
        ([{"hostid": "11", "host": "bound-host"}], "a" * 64),
        ([item()], "b" * 64), (rows, "c" * 64),
    ])
    result = asyncio.run(value.query_native_metric(long_request, metric_name=item()["key_"], limit=1))
    assert result.truncated and len(result.points) == 1
    assert value._rpc.await_args.args[0] == "history.get"
    assert value._rpc.await_args.args[1]["time_from"] == int(long_request.window_start.timestamp())
    with pytest.raises(ValueError):
        NativeMetricQueryRequest(**{**long_request.model_dump(), "window_start": req.window_end - timedelta(days=61)})


@pytest.mark.parametrize("updates", [{"ns": "1000000000"}, {"value": "x"}, {"value": "18446744073709551616"}, {"value": "1e999999"}])
def test_malformed_numeric_history_is_a_source_error_not_valid_samples(updates):
    req = request()
    value = adapter()
    row = {"itemid": "22", "clock": str(int(req.window_start.timestamp())), "ns": "0", "value": "1", **updates}
    value._rpc = AsyncMock(side_effect=[
        ([{"hostid": "11", "host": "bound-host"}], "a" * 64),
        ([item()], "b" * 64), ([row], "c" * 64),
    ])
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.query_native_metric(req, metric_name=item()["key_"], limit=100))
