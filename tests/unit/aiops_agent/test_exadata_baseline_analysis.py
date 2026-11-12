"""Exadata 均值比较与只读分析，不连接真实 OEM 或执行器。"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from platform_core.contracts.aiops import AIOpsCapability, TurnPlanStepV1, ScopeType
from platform_core.contracts.aiops.exadata import ExadataBaselineQuery, ExadataSample
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeBaseline, MonitoringNativeBaselineSeries
from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.application.exadata_analysis import comparisons, analyze_view
from aiops_agent.application.exadata_chat import exadata_parameters, ExadataChatReader
from aiops_agent.application.turn_planner import TurnPlannerService, CapabilityResult
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.ports.exadata import ExadataMetricRead
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter as native_adapter
from tests.unit.aiops_agent.test_exadata_chat import collector, lab, resource, make_plan, parameters, CPU, TEMP, CELL
from platform_core.identity import uuid7


def sample(value, dimensions=None, quality="OBSERVED", at=None):
    return ExadataSample(component_id=CELL, metric_code=CPU, selector="Cell_Metric/cpu_utilization",
        observed_at=at or datetime.now(UTC), value=value, dimensions=dimensions or {}, unit="%",
        native_value_type="NUMBER", quality=quality, response_sha256="a"*64)


def baseline_fixture(averages):
    now = datetime.now(UTC)
    return MonitoringNativeBaseline(period="1w", start=now-timedelta(days=7), end=now,
        series=tuple(MonitoringNativeBaselineSeries(dimensions=labels, average=value, sample_count=3)
                     for labels, value in averages))


def active_plan(params=None):
    return make_plan().model_copy(update={"requested_capabilities": (AIOpsCapability.ACTIVE_DIAGNOSIS,),
        "requires_active_diagnosis": True, "steps": (TurnPlanStepV1(step_id="s1", capability=AIOpsCapability.ACTIVE_DIAGNOSIS,
        objective="基于 Exadata 指标分析变化", scope_type=ScopeType.RESOURCE, parameters=params or parameters()),)})


def test_baseline_policy_positive_allowlist_excludes_states_counters_configuration_and_limits():
    metrics = load_catalog()[2]
    for code in [CPU, TEMP, "host.CPUUsageSummary.OverAllUsagePct", "host.Load.memUsedPct",
                 "oracle_exadata.CellDisk_Metric.cd_sm_r_reqs", "oracle_exadata.IORMDB_Metric.db_io_wt_lg_rq"]:
        assert metrics[code]["baseline_supported"]
    for suffix in ["Cell_Metric.temperature_lower", "Cell_Metric.temperature_upper", "Cell_Metric.iorm_mode",
                   "FlashCache_Metric.default_hits", "FlashLog_Metric.flashlog_disk_io_errs"]:
        assert not metrics["oracle_exadata."+suffix]["baseline_supported"]
    assert all(not definition["baseline_supported"] for definition in metrics.values() if definition.get("status"))


def test_decimal_comparison_preserves_dimension_zero_negative_and_precision():
    mean = baseline_fixture([({"disk": "a"}, "9007199254740993.123456789"), ({"disk": "b"}, "0"), ({"disk": "c"}, "-2")])
    result = comparisons([sample("9007199254740994.123456789", {"disk": "a"}), sample("2", {"disk": "b"}),
                          sample("-1", {"disk": "c"}), sample("100", {"disk": "unmatched"})], mean)
    assert len(result) == 3
    assert Decimal(result[0].difference) == 1
    assert result[1].deviation_percent is None and Decimal(result[1].difference) == 2
    assert Decimal(result[2].deviation_percent) == 50


@pytest.mark.parametrize("quality", ["EMPTY", "STALE", "TRUNCATED"])
def test_non_current_data_and_last_empty_sample_do_not_compare(quality):
    now = datetime.now(UTC)
    assert not comparisons([sample("2", quality=quality)], baseline_fixture([({}, "1")]))
    assert not comparisons([sample("2", at=now-timedelta(seconds=1)), sample(None, quality="EMPTY", at=now)],
                           baseline_fixture([({}, "1")]))


def setup_service(**kwargs):
    service, adapter = collector(value="15", unit="%", **kwargs)
    adapter.exadata_baseline = AsyncMock(return_value={"series": ({"dimensions": {}, "total": "30", "sample_count": 3},),
        "unit": "%", "truncated": False, "response_sha256": "b"*64})
    args = dict(scope=NS(domain_id=7, actor_id="tester"), system_id=uuid7(), resource_id=uuid7(), source_id=uuid7(),
                query=ExadataBaselineQuery(component_id=CELL, metric_code=CPU))
    return service, adapter, args


@pytest.mark.parametrize("period,days", [("1w", 7), ("1m", 30)])
def test_baseline_has_fixed_window_sample_count_and_comparison_evidence(period, days):
    async def run():
        service, adapter, args = setup_service()
        args["query"] = args["query"].model_copy(update={"period": period})
        value = await service.baseline(**args)
        assert value.baseline.end-value.baseline.start == timedelta(days=days)
        assert value.baseline.series[0].average == "10" and value.baseline.series[0].sample_count == 3
        assert value.comparisons[0].deviation_percent == "50.0"
        assert value.current_samples[0].response_sha256 == "d"*64
        assert adapter.exadata_baseline.call_args.args[0].baseline_days == days
        assert service._validate_live_inventory.await_count == 2
        assert service._snapshot.await_count == 2
    asyncio.run(run())


def test_cache_still_rechecks_binding_inventory_current_value_and_scope():
    async def run():
        service, adapter, args = setup_service()
        first = await service.baseline(**args)
        adapter.exadata_latest.return_value = {**adapter.exadata_latest.return_value, "points": ((datetime.now(UTC), "20"),)}
        second = await service.baseline(**args)
        assert second.baseline == first.baseline and second.comparisons[0].deviation_percent == "100"
        assert adapter.exadata_baseline.await_count == 1 and adapter.exadata_latest.await_count == 2
        assert service._validate_live_inventory.await_count == 4 and service._snapshot.await_count == 4
        args["scope"] = NS(domain_id=8, actor_id="tester")
        await service.baseline(**args)
        assert adapter.exadata_baseline.await_count == 2
        frozen = service._snapshot.return_value[0]
        changed = {**frozen, "binding_version": 9}
        service._snapshot.return_value = (changed, adapter)
        await service.baseline(**args)
        assert adapter.exadata_baseline.await_count == 3
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["truncated", "unit", "dimensions", "provider", "no_data", "window_unsupported"])
def test_bad_baseline_never_produces_comparison_or_removes_current_evidence(failure):
    async def run():
        service, adapter, args = setup_service()
        summary = adapter.exadata_baseline.return_value
        if failure == "truncated": summary["truncated"] = True
        if failure == "unit": summary["unit"] = "MB/s"
        if failure == "dimensions": summary["series"][0]["dimensions"] = {"unverified": "a"}
        if failure == "provider": adapter.exadata_baseline.side_effect = DiagnosticSourceAdapterError("SOURCE_PROVIDER_ERROR", "测试")
        if failure == "no_data": summary["series"] = ()
        if failure == "window_unsupported":
            args["query"] = args["query"].model_copy(update={"period": "1m"})
            adapter.exadata_baseline.side_effect = DiagnosticSourceAdapterError(
                "SOURCE_BASELINE_WINDOW_UNSUPPORTED", "原厂历史窗口上限")
        result = await service.baseline(**args)
        assert result.baseline.unavailable_reason and not result.baseline.series and not result.comparisons
        assert result.current_samples[0].value == "15"
        if failure == "window_unsupported":
            assert "最多支持 7 天" in result.baseline.unavailable_reason
            assert "30 天" in result.baseline.unavailable_reason
            assert "可重试" not in result.baseline.unavailable_reason
    asyncio.run(run())


def test_member_and_final_binding_changes_fail_closed_including_cache_hit():
    async def run():
        service, adapter, args = setup_service()
        await service.baseline(**args)
        frozen = service._snapshot.return_value[0]
        service._snapshot.side_effect = [(frozen, adapter), ({**frozen, "source_version": 10}, adapter)]
        with pytest.raises(AIOpsApplicationError): await service.baseline(**args)
        assert adapter.exadata_baseline.await_count == 1
        service._snapshot.side_effect = None
        args["query"] = args["query"].model_copy(update={"component_id": "f"*32})
        with pytest.raises(AIOpsApplicationError): await service.baseline(**args)
        assert adapter.exadata_latest.await_count == 2
    asyncio.run(run())


def test_agent_active_scope_does_not_require_database_target_and_no_database_fallback():
    f = lab()
    planner = TurnPlannerService(uow_factory=None)
    plan = active_plan()
    assert exadata_parameters(plan)["resource_type_key"] == "EXADATA"
    scope, internal, _ = planner._resolve_scope({"systems": [f.row], "question": "分析 Exadata 温度", "policy_snapshot_hash": "a"*64}, plan)
    assert scope.status == "RESOLVED" and not internal["execution_targets"]
    denied, _, _ = planner._resolve_scope({"systems": [], "question": "分析 Exadata 温度", "policy_snapshot_hash": "a"*64}, plan)
    assert denied.status == "DENIED"


def test_exadata_active_dispatch_uses_specialist_and_keeps_ordinary_database_worker_branch():
    async def run():
        adapter = NS(execute=AsyncMock(return_value=CapabilityResult(AIOpsCapability.ACTIVE_DIAGNOSIS, True, "证据分析")))
        planner = TurnPlannerService(uow_factory=None, capability_adapters={AIOpsCapability.ACTIVE_DIAGNOSIS: adapter})
        planner._complete_without_run = AsyncMock(return_value={"status": "COMPLETED"})
        planner._append_event = AsyncMock()
        turn = NS(turn_id=uuid7(), orchestration_internal_json={}, status="ACCEPTED")
        scope = NS(status="RESOLVED")
        result = await planner._route_resolved(None, turn, active_plan(), scope, {"execution_targets": []})
        assert result["status"] == "COMPLETED" and "dispatch" not in result
        adapter.execute.assert_awaited_once()
        normal = active_plan().model_copy(update={"steps": (TurnPlanStepV1(step_id="s1", capability=AIOpsCapability.ACTIVE_DIAGNOSIS,
                objective="普通数据库诊断", scope_type=ScopeType.RESOURCE),)})
        result = await planner._route_resolved(None, turn, normal, scope, {"execution_targets": [{"readiness": "READY"}]})
        assert result["dispatch"] == "ACTIVE_DIAGNOSIS" and adapter.execute.await_count == 1
    asyncio.run(run())


def test_agent_analysis_includes_baseline_calculation_frozen_evidence_and_next_checks():
    async def run():
        f = lab(unit="%", value="15")
        f.adapter.exadata_baseline = AsyncMock(return_value={"series": ({"dimensions": {}, "total": "30", "sample_count": 3},),
            "unit": "%", "truncated": False, "response_sha256": "b"*64})
        result = await f.reader.read(parameters=parameters(metric_codes=[CPU]), plan=active_plan(parameters(metric_codes=[CPU])),
            resources=[resource(f.row)], scope=NS(domain_id=7), analyze=True)
        assert "相对偏离 +50.00%" in result.markdown and "下一步只读排查" in result.markdown
        assert "不是故障阈值" in result.markdown and "[C1]" in result.markdown
        frozen = result.evidence[0]["snapshot"]["analysis_baselines"][0]
        assert frozen["comparisons"][0]["difference"] == "5"
        assert frozen["baseline"]["response_sha256"] == "b"*64
    asyncio.run(run())


def test_agent_state_analysis_interprets_availability_and_links_sample_indexes():
    async def run():
        f = lab(value="0", native_type="NUMBER")
        code = "oracle_exadata.CellSrv_Status_Metric.Status"
        result = await f.reader.read(parameters=parameters(metric_codes=[code]), plan=active_plan(parameters(metric_codes=[code])),
            resources=[resource(f.row)], scope=NS(domain_id=7), analyze=True)
        assert "不可用" in result.markdown and "已定位上述组件状态异常" in result.markdown
        finding = result.evidence[0]["snapshot"]["analysis_findings"][0]
        assert finding["severity"] == "WARNING" and finding["evidence_indexes"] == [0]
        assert not result.evidence[0]["snapshot"]["analysis_baselines"]
    asyncio.run(run())


@pytest.mark.parametrize("days,parts", [(7, 2), (30, 5)])
def test_rest_baseline_fixed_nonoverlapping_parts_merge_counts_not_means(days, parts):
    async def run():
        value, _ = native_adapter()
        value.query_native_metric = AsyncMock(side_effect=[{"series": ({"dimensions": {}, "total": str(index+1), "sample_count": index+1},),
            "unit": "%", "truncated": False, "response_sha256": str(index)*64} for index in range(parts)])
        end = datetime.now(UTC)
        request = ExadataMetricRead(source_locator_key=CELL, window_start=end-timedelta(days=days), window_end=end, baseline_days=days)
        summary = await value.exadata_baseline(request, load_catalog()[2][CPU], "oracle_exadata")
        assert summary["series"][0]["total"] == str(parts*(parts+1)//2)
        assert summary["series"][0]["sample_count"] == parts*(parts+1)//2
        calls = value.query_native_metric.call_args_list
        assert len(calls) == parts
        windows = [(call.args[0].window_start, call.args[0].window_end) for call in calls]
        assert windows[0][0] == request.window_start and windows[-1][1] == request.window_end
        assert all(upper-lower <= timedelta(days=6) for lower, upper in windows)
        assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
        assert all(call.kwargs["_shared_budget"] is True for call in calls)
        assert value._remaining_response_bytes == request.max_response_bytes
    asyncio.run(run())


def test_rest_partition_failure_or_unit_change_cannot_return_partial_baseline():
    async def run():
        for failure in [DiagnosticSourceAdapterError("SOURCE_PROVIDER_ERROR", "测试"),
                        {"series": (), "unit": "MB/s", "truncated": False, "response_sha256": "e"*64}]:
            value, _ = native_adapter()
            value.query_native_metric = AsyncMock(side_effect=[{"series": (), "unit": "%", "truncated": False, "response_sha256": "d"*64}, failure])
            end = datetime.now(UTC)
            request = ExadataMetricRead(source_locator_key=CELL, window_start=end-timedelta(days=7), window_end=end, baseline_days=7)
            with pytest.raises(DiagnosticSourceAdapterError):
                await value.exadata_baseline(request, load_catalog()[2][CPU], "oracle_exadata")
    asyncio.run(run())
