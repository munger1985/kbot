"""具体指标聊天的目录选择、授权交集、专用取数及证据保真回归。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import ValidationError

from aiops_agent.application.exadata import ExadataOEMService
from aiops_agent.application.exadata_chat import (
    ExadataChatQuery, ExadataChatReader, exadata_parameter_contract, select_exadata_systems,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.turn_planner import TurnPlannerService
from platform_core.contracts.aiops import AIOpsCapability, TurnPlanV1, TurnPlanStepV1, ScopeType
from platform_core.contracts.aiops.exadata import ExadataQuery
from platform_core.identity import uuid7


TEMP = "oracle_exadata.Cell_Metric.temperature"
CPU = "oracle_exadata.Cell_Metric.cpu_utilization"
ROOT, CELL = "a" * 32, "b" * 32


def parameters(**changes):
    return {"resource_type_key": "EXADATA", "metric_codes": [TEMP], **changes}


def make_plan(**changes):
    return TurnPlanV1(turn_id=uuid7(), revision_no=1,
        requested_capabilities=(AIOpsCapability.MONITORING_QUERY,),
        policy_snapshot_hash="a" * 64, steps=(TurnPlanStepV1(step_id="s1",
            capability=AIOpsCapability.MONITORING_QUERY, objective="查看指定指标",
            scope_type=ScopeType.RESOURCE, parameters=parameters()),), **changes)


def system(resource_type="EXADATA", name="SIM 整机"):
    sid, rid = uuid7(), uuid7()
    return {"business_system_id": str(sid), "system_code": "sim-system", "display_name": "模拟系统",
            "environment": "TEST", "relations": [], "resources": [{"resource_id": str(rid),
            "system_resource_id": str(uuid7()), "resource_code": "sim-root", "display_name": name,
            "resource_type_key": resource_type, "topology_roles": [], "attributes": {},
            "target_id": str(uuid7()) if resource_type == "DATABASE" else None, "target_ready": False}]}


def resource(system_row):
    return {"business_system_id": system_row["business_system_id"], **system_row["resources"][0]}


def collector(*, cell_count=1, value="32.0", unit=None, native_type="NUMBER", old=False):
    cells = [{"target_id": CELL if index == 0 else f"{index:032x}", "target_type": "oracle_exadata",
              "display_name": f"Cell {index + 1}"} for index in range(cell_count)]
    root = {"target_id": ROOT, "target_type": "oracle_dbmachine", "display_name": "Root"}
    frozen = {"inventory": {"root": root, "members": cells}, "binding_id": uuid7(),
              "binding_version": 1, "source_version": 2, "root_id": ROOT, "inventory_sha256": "c" * 64}
    now = datetime.now(UTC)
    adapter = NS(exadata_latest=AsyncMock(return_value={"points": ((now - timedelta(hours=2) if old else now, value),),
        "dimensions": ({},), "unit": unit, "native_value_type": native_type,
        "captured_at": now, "truncated": False, "response_sha256": "d" * 64}))
    adapter.exadata_history = AsyncMock(return_value=adapter.exadata_latest.return_value)
    service = ExadataOEMService(None)
    service._snapshot = AsyncMock(return_value=(frozen, adapter))
    service._validate_live_inventory = AsyncMock()
    return service, adapter


def lab(**kwargs):
    row = system()
    source = NS(source_id=uuid7(), source_type="OEM", display_name="OEM 测试", monitoring_readiness="READY")
    item = NS(resource_id=UUID(row["resources"][0]["resource_id"]), resource_type_key="EXADATA", sources=(source,))
    monitor = NS(list_business_systems=AsyncMock(return_value=(NS(
        business_system_id=UUID(row["business_system_id"]), resources=(item,)),)))
    service, adapter = collector(**kwargs)
    return NS(row=row, source=source, item=item, monitor=monitor, service=service, adapter=adapter,
              reader=ExadataChatReader(monitoring_service=monitor, exadata_service=service))


async def read(f, *, query=None, plan=None, resources=None):
    return await f.reader.read(parameters=query or parameters(), plan=plan or make_plan(),
        resources=[resource(f.row)] if resources is None else resources, scope=NS(domain_id=7))


def test_router_contract_uses_all_real_definitions_not_keyword_rules():
    contract = exadata_parameter_contract()
    assert len(contract["metric_catalog"]) == 345
    definition = next(item for item in contract["metric_catalog"] if item["code"] == TEMP)
    assert definition["display_name"] == "Exadata Temperature Reading"
    assert "temperature_lower" not in definition["code"]
    normal, exadata, service = contract["parameters_schema"]["oneOf"]
    assert service['properties']['resource_type_key']['enum'] == ['ORACLE_ASM', 'ORACLE_RAC', 'ORACLE_LISTENER']
    assert "resource_type_key" not in normal["properties"]
    assert exadata["properties"]["resource_type_key"]["const"] == "EXADATA"
    assert contract["execution_costs"] == {
        "current_per_component_metric": 1, "baseline_per_component_metric": 2,
        "max_live_queries": 24, "max_total_seconds": 55}


@pytest.mark.parametrize("changes", [
    {"metric_codes": []}, {"metric_codes": ["invented.metric"]}, {"metric_codes": [TEMP, TEMP]},
    {"query_kind": "OVERVIEW"}, {"resource_type_key": "DATABASE"}, {"promql": "arbitrary"},
])
def test_invalid_metric_selection_is_rejected(changes):
    with pytest.raises(ValidationError):
        ExadataChatQuery.model_validate(parameters(**changes))


def test_explicit_selectors_never_fall_back_or_leave_agent_scope():
    row, unrelated = system(), system("DATABASE", "普通数据库")
    assert select_exadata_systems([row, unrelated], parameters())[0]["resources"] == row["resources"]
    assert not select_exadata_systems([unrelated], parameters())
    assert not select_exadata_systems([row], parameters(resource_names=["未授权整机"]))
    assert not select_exadata_systems([row], parameters(system_names=["未授权系统"]))
    assert not select_exadata_systems([row], parameters(resource_names=["SIM 整机", "不存在"]))
    assert select_exadata_systems([row], parameters(resource_names=["SIM-ROOT"]))


def test_planner_does_not_use_trusted_database_hint_for_exadata_request():
    exadata, database = system(), system("DATABASE", "普通数据库")
    planner = TurnPlannerService(uow_factory=None)
    scope, internal, _ = planner._resolve_scope({"systems": [database, exadata],
        "question": "Exadata Cell 温度是多少", "policy_snapshot_hash": "a" * 64,
        "context_hint": {"business_system_id": database["business_system_id"]}}, make_plan())
    assert [item.resource_type_key for item in scope.resources] == ["EXADATA"]
    assert not internal["execution_targets"]
    assert internal["resource_catalog"][0]["resource_id"] == exadata["resources"][0]["resource_id"]
    denied, internal, _ = planner._resolve_scope({"systems": [database], "question": "Exadata Cell 温度是多少",
        "policy_snapshot_hash": "a" * 64}, make_plan())
    assert denied.status == "DENIED"
    assert denied.resolution_basis == ("NO_AUTHORIZED_EXADATA_RESOURCE",)
    assert not internal["execution_targets"]


def test_exact_temperature_query_only_reads_temperature_not_threshold_or_cpu():
    async def run():
        f = lab()
        result = await read(f)
        assert result.gap_code is None
        assert "32.0" in result.markdown
        assert "OEM 未返回单位" in result.markdown
        assert "℃" not in result.markdown
        assert "北京时间" in result.markdown
        assert len(result.evidence) == 1
        snapshot = result.evidence[0]["snapshot"]
        assert [item["metric_code"] for item in snapshot["samples"]] == [TEMP]
        assert snapshot["binding_version"] == 1 and snapshot["source_version"] == 2
        assert snapshot["samples"][0]["response_sha256"] == "d" * 64
        f.adapter.exadata_latest.assert_awaited_once()
        assert f.adapter.exadata_latest.await_args.args[1]["code"] == TEMP
        assert "temperature_lower" not in result.markdown and CPU not in result.markdown
        assert f.service._validate_live_inventory.await_count == 2
    asyncio.run(run())


@pytest.mark.parametrize("value,old,quality", [("0", False, "OBSERVED"), (None, False, "EMPTY"), ("31.9", True, "STALE")])
def test_zero_null_and_stale_are_not_filled_or_declared_healthy(value, old, quality):
    async def run():
        f = lab(value=value, old=old)
        result = await read(f)
        assert result.evidence[0]["snapshot"]["samples"][0]["value"] == value
        assert result.evidence[0]["snapshot"]["samples"][0]["quality"] == quality
        assert bool(result.gap_code) == (quality != "OBSERVED")
        assert "硬件正常" not in result.markdown
    asyncio.run(run())


def test_decimal_precision_and_native_unit_preserved_without_conversion():
    async def run():
        raw = "9007199254740993.123456789"
        result = await read(lab(value=raw, unit="原厂单位"))
        assert raw in result.markdown and "原厂单位" in result.markdown
        assert result.evidence[0]["snapshot"]["samples"][0]["value"] == raw
    asyncio.run(run())


def test_multiple_cells_are_separate_and_named_component_is_strict():
    async def run():
        f = lab(cell_count=2)
        result = await read(f)
        assert "Cell 1" in result.markdown and "Cell 2" in result.markdown
        assert f.adapter.exadata_latest.await_count == 2
        f.adapter.exadata_latest.reset_mock()
        result = await read(f, query=parameters(component_names=["Cell 2"]))
        assert len(result.evidence[0]["snapshot"]["samples"]) == 1
        f.adapter.exadata_latest.assert_awaited_once()
        f.adapter.exadata_latest.reset_mock()
        result = await read(f, query=parameters(component_names=["未授权 Cell"]))
        assert result.gap_code == "EXADATA_QUERY_UNAVAILABLE"
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())


def test_missing_authorized_scope_and_unknown_metric_do_not_touch_monitoring():
    async def run():
        f = lab()
        for query, resources in [(parameters(), []), (parameters(metric_codes=["fake"]), [resource(f.row)]),
            (parameters(query_kind="UNRESOLVED", metric_codes=[]), [resource(f.row)])]:
            result = await read(f, query=query, resources=resources)
            assert result.gap_code
        f.monitor.list_business_systems.assert_not_awaited()
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("source_change,code", [("multiple", "EXADATA_SOURCE_AMBIGUOUS"),
    ("wrong_type", "EXADATA_SOURCE_UNAVAILABLE"), ("not_ready", "EXADATA_SOURCE_UNAVAILABLE"),
    ("wrong_resource", "EXADATA_SOURCE_UNAVAILABLE")])
def test_source_and_catalog_isolation(source_change, code):
    async def run():
        f = lab()
        if source_change == "multiple":
            f.item.sources += (NS(**{**vars(f.source), "source_id": uuid7(), "display_name": "OEM 另一个"}),)
        elif source_change == "wrong_type":
            f.source.source_type = "PROMETHEUS"
        elif source_change == "not_ready":
            f.source.monitoring_readiness = "NOT_CONNECTED"
        else:
            f.item.resource_id = uuid7()
        result = await read(f)
        assert result.gap_code == code
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())


def test_budget_is_for_actual_metric_reads_not_whole_scene_or_all_cells():
    async def run():
        f = lab(cell_count=3)
        result = await read(f, plan=make_plan(budget={"max_live_queries": 1}))
        assert result.gap_code == "EXADATA_METRIC_DATA_PARTIAL"
        assert "QUERY_PAGE_PARTIAL" in [gap["code"] for gap in result.evidence[0]["snapshot"]["gaps"]]
        f.adapter.exadata_latest.assert_awaited_once()
        f = lab()
        result = await read(f, plan=make_plan(budget={"max_live_queries": 0}))
        assert result.gap_code == "EXADATA_QUERY_BUDGET_EXCEEDED"
        f.monitor.list_business_systems.assert_not_awaited()
    asyncio.run(run())


def test_history_keeps_last_null_and_explicit_window_without_snapshot_fallback():
    async def run():
        f = lab()
        raw = f.adapter.exadata_history.return_value
        now = datetime.now(UTC)
        raw.update(points=((now - timedelta(minutes=1), "32.0"), (now, None)), dimensions=({}, {}))
        result = await read(f, query=parameters(mode="HISTORY"),
            plan=make_plan(time_range={"relative_window_seconds": 900}))
        assert "无读数" in result.markdown and "32.0" not in result.markdown
        assert "窗口采样数 2" in result.markdown
        f.adapter.exadata_latest.assert_not_awaited()
        request = f.adapter.exadata_history.await_args.args[0]
        assert (request.window_end - request.window_start).total_seconds() == 900
        f = lab()
        result = await read(f, query=parameters(mode="HISTORY"),
            plan=make_plan(time_range={"relative_window_seconds": 7200}))
        assert result.gap_code == "EXADATA_TIME_RANGE_UNSUPPORTED"
        f.monitor.list_business_systems.assert_not_awaited()
    asyncio.run(run())


def test_collect_internal_selection_cannot_expand_scenes_or_modify_diagnosis():
    async def run():
        f = lab()
        for codes, diagnosis in [(("fake",), False), ((TEMP,), True)]:
            with pytest.raises(AIOpsApplicationError):
                await f.service.collect(scope=NS(), system_id=uuid7(), resource_id=uuid7(), source_id=uuid7(),
                    query=ExadataQuery(scene="cell"), _metric_codes=codes, _diagnosis=diagnosis)
        f.adapter.exadata_latest.assert_not_awaited()
        f.service._snapshot.assert_not_awaited()
    asyncio.run(run())


def test_platform_adapter_uses_only_publicly_resolved_exadata_resources():
    async def run():
        from aiops_agent.application import turn_planner as planner_module
        f = lab()
        ref = NS(business_system_id=UUID(f.row["business_system_id"]),
                 resource_id=UUID(f.row["resources"][0]["resource_id"]))
        values = {"capability": AIOpsCapability.MONITORING_QUERY, "plan": make_plan(),
                  "scope": NS(resources=(ref,)), "internal_scope": {"resource_catalog": [resource(f.row)],
                  "runtime_context": {"domain_id": 7, "actor_id": "tester", "caller_service": "worker",
                                      "request_id": "test", "trace_id": "trace"}}}
        if hasattr(planner_module, "MonitoringQueryAdapter"):
            adapter = planner_module.MonitoringQueryAdapter(f.monitor, exadata_service=f.service)
            values["context"] = NS(configuration_scope=NS(domain_id=uuid7(), tenant_id=uuid7()))
        else:
            from aiops_agent.application.turn_capabilities import MonitoringQueryAdapter
            adapter = MonitoringQueryAdapter(monitoring_service=f.monitor, exadata_service=f.service)
        result = await adapter.execute(**values)
        assert result.supported and result.evidence[0]["snapshot"]["samples"][0]["metric_code"] == TEMP
        f.adapter.exadata_latest.reset_mock()
        f.monitor.list_business_systems.reset_mock()
        values["scope"] = NS(resources=())
        result = await adapter.execute(**values)
        assert not result.supported and result.gap_code == "EXADATA_RESOURCE_SCOPE_EMPTY"
        f.monitor.list_business_systems.assert_not_awaited()
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())


def test_denied_exadata_turn_completes_with_specific_scope_message_without_database_run():
    async def run():
        planner = TurnPlannerService(uow_factory=None)
        scope, internal, _ = planner._resolve_scope({"systems": [system("DATABASE")],
            "question": "Exadata Cell 温度是多少", "policy_snapshot_hash": "a" * 64}, make_plan())
        planner._complete_without_run = AsyncMock(return_value={"status": "PARTIAL"})
        from aiops_agent.application import turn_planner as planner_module
        context = {"context": NS()} if hasattr(planner_module, "SpecialistContext") else {}
        result = await planner._route_resolved(object(), object(), make_plan(), scope, internal, **context)
        assert result["status"] == "PARTIAL"
        args = planner._complete_without_run.await_args.args
        assert "没有匹配的 Exadata 资源" in args[2]
        assert args[4] == ("EXADATA_RESOURCE_SCOPE_EMPTY",)
    asyncio.run(run())


def test_turn_router_receives_real_catalog_and_exact_parameter_contract():
    async def run():
        plan = make_plan()
        model = NS(generate_structured=AsyncMock(return_value=NS(output=plan, receipt=NS())))
        planner = TurnPlannerService(uow_factory=None, structured_model=model,
            prompt_registry=NS(resolve=AsyncMock(return_value=NS(content="测试路由 Prompt", ref=lambda: {}))),
            model_resolver=AsyncMock(return_value={"technical_name": "configured-model"}))
        context = {"turn_id": str(plan.turn_id), "plan_revision": 1,
            "policy_snapshot_hash": plan.policy_snapshot_hash, "question": "Exadata Cell 温度是多少",
            "capabilities": ["MONITORING_QUERY"], "planner_model_id": str(uuid7()),
            "tenant_id": str(uuid7()), "context_hint": {}, "systems": [system("DATABASE")]}
        returned, _ = await planner._plan(context)
        assert returned.steps[0].parameters["metric_codes"] == [TEMP]
        payload = model.generate_structured.await_args.kwargs["input_payload"]
        contract = payload["capability_parameter_contracts"]["MONITORING_QUERY"]
        assert TEMP in {item["code"] for item in contract["metric_catalog"]}
        assert "没有 EXADATA" in contract["policy"]
        assert payload["authorized_business_systems"][0]["resources"][0]["resource_type_key"] == "DATABASE"
    asyncio.run(run())


def test_explicit_source_selection_does_not_query_other_oem_sources():
    async def run():
        f = lab()
        f.item.sources += (NS(**{**vars(f.source), "source_id": uuid7(), "display_name": "OEM 另一个"}),)
        result = await read(f, query=parameters(source_name="OEM 测试"))
        assert result.gap_code is None
        assert result.evidence[0]["source_id"] == str(f.source.source_id)
        f.adapter.exadata_latest.assert_awaited_once()
    asyncio.run(run())


def test_monitoring_timeout_returns_gap_without_other_metric_fallback():
    async def run():
        f = lab()
        f.monitor.list_business_systems.side_effect = TimeoutError
        result = await read(f)
        assert result.gap_code == "EXADATA_QUERY_UNAVAILABLE"
        assert "未回退数据库总览" in result.markdown
        assert not result.evidence
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())


def test_same_timestamp_last_null_is_not_backfilled_and_dimensions_are_separate():
    async def run():
        f = lab()
        now = datetime.now(UTC)
        f.adapter.exadata_latest.return_value.update(points=((now, "32.0"), (now, None), (now, "33.0")),
            dimensions=({"sensor": "A"}, {"sensor": "A"}, {"sensor": "B"}))
        result = await read(f)
        assert "无读数" in result.markdown and "33.0" in result.markdown
        assert "32.0" not in result.markdown
        assert len(result.evidence[0]["snapshot"]["samples"]) == 3
    asyncio.run(run())


def test_multiple_monitoring_steps_do_not_silently_drop_exadata_request():
    from aiops_agent.application.exadata_chat import exadata_parameters
    plan = make_plan()
    extra = TurnPlanStepV1(step_id="s2", capability=AIOpsCapability.MONITORING_QUERY,
                          objective="另一项指标", scope_type=ScopeType.RESOURCE, parameters=parameters(metric_codes=[CPU]))
    plan = plan.model_copy(update={"steps": (*plan.steps, extra)})
    assert exadata_parameters(plan) == {"resource_type_key": "EXADATA", "query_kind": "UNRESOLVED"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
