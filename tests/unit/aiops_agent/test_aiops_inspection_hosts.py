"""巡检主机采集、冻结阈值、逐主机覆盖及执行重授权回归。"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.application.inspections.assessment import (
    assess_inspection, freeze_inspection_policy, monitoring_checks,
)
from aiops_agent.application.inspections.monitoring_hosts import attach_inspection_hosts
from aiops_agent.application.inspections.monitoring_observations import classify_monitoring_facts
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.monitoring_snapshot import MonitoringSnapshotBuilder
from aiops_agent.application.resource_governance import ResourceAdmissionError, ResourceGovernanceService
from aiops_agent.application.source_authorization import source_scope_is_current
from aiops_agent.monitoring.catalog import load_monitoring_profile_catalog
from platform_core.identity import uuid7


NOW = datetime(2026, 11, 11, 8, tzinfo=UTC)
HOST_CODES = ("host.cpu.utilization", "host.memory.utilization", "host.filesystem.utilization")


def fixture(db_type="ORACLE", version="19c"):
    source_id, tenant_id, target_id, system_id = (uuid7() for _ in range(4))
    source = SimpleNamespace(
        diagnostic_source_id=source_id, tenant_id=tenant_id, domain_id=1, status="ENABLED",
        source_type="PROMETHEUS", row_version=1, adapter_id="prometheus",
        adapter_version="1.0.0", connectivity_status="CONNECTED",
        endpoint="http://prometheus.invalid", auth_credential_id=None,
        declared_capabilities_json={"metric.query_range": {}}, config_json={},
    )
    catalog = load_metric_catalog()
    profiles = load_monitoring_profile_catalog(catalog)
    overview = profiles.overview_for_database(db_type, version)
    database_binding = SimpleNamespace(
        resource_source_binding_id=uuid7(), resource_id=uuid7(), diagnostic_source_id=source_id,
        row_version=1, source_locator_key="database-1", source_locator_json={"target_key": "database-1"},
        capability_scope_json={"metric_codes": list(overview.metric_codes)},
        role="PRIMARY", priority=10, mapping_overrides_json={}, query_budget_json={},
    )
    host = SimpleNamespace(resource_id=uuid7(), display_name="数据库主机一", resource_type_key="VIRTUAL_MACHINE")
    binding = SimpleNamespace(
        resource_source_binding_id=uuid7(), diagnostic_source_id=source_id, row_version=1,
        source_locator_key="host-1", source_locator_json={"resource_scope": "HOST", "target_key": "host-1"},
        capability_scope_json={}, mapping_overrides_json={}, query_budget_json={},
    )
    system = SimpleNamespace(business_system_id=system_id)
    relation = SimpleNamespace(relation_id=uuid7(), row_version=1)
    target = SimpleNamespace(target_id=target_id, tenant_id=tenant_id, domain_id=1, db_type=db_type, version_code=version)
    inspection = {"schedule_type": "DAILY", "business_system_id": str(system_id),
                  "report_policy": freeze_inspection_policy(db_type)}
    snapshot = {
        "bindings": [{
            "binding_id": str(database_binding.resource_source_binding_id), "binding_version": 1,
            "source_locator_key": "database-1", "source_locator": database_binding.source_locator_json,
            "source": {"source_id": str(source_id), "config_version": 1,
                       "declared_capabilities": source.declared_capabilities_json},
        }],
        "observation_binding_ids": [], "initial_gaps": [],
        "monitoring_profile": overview.model_dump(mode="json"),
        "window": {"start": (NOW - timedelta(minutes=15)).isoformat(), "end": NOW.isoformat()},
        "current_window": {"start": (NOW - timedelta(minutes=15)).isoformat(), "end": NOW.isoformat()},
        "snapshot_max_age_seconds": 300,
    }
    run = SimpleNamespace(
        ops_run_id=uuid7(), target_id=target_id, tenant_id=tenant_id, domain_id=1,
        agent_version_id=uuid7(), plan_snapshot_json={"monitoring": snapshot, "client_metadata": {"inspection": inspection}},
    )
    uow = SimpleNamespace(
        agents=SimpleNamespace(version_business_system_ids=AsyncMock(return_value=[system_id])),
        business_systems=SimpleNamespace(list_storage_host_bindings=AsyncMock(return_value=[(host, binding, system, relation)])),
        diagnostic_sources=SimpleNamespace(get_scoped=AsyncMock(return_value=source)),
        targets=SimpleNamespace(list_source_bindings=AsyncMock(return_value=[database_binding]), get_scoped=AsyncMock(return_value=target)),
        runs=SimpleNamespace(get_run=AsyncMock(return_value=run), database_now=AsyncMock(return_value=NOW)),
    )
    return SimpleNamespace(
        uow=uow, run=run, snapshot=snapshot, source=source, target=target,
        host=host, binding=binding, system=system, relation=relation, inspection=inspection,
        catalog=catalog, profiles=profiles,
    )


def attach(value, *, allowed=True):
    asyncio.run(attach_inspection_hosts(
        uow=value.uow, snapshot=value.snapshot, target=value.target, run=value.run,
        inspection=value.inspection, allowed_source_ids=(value.source.diagnostic_source_id,) if allowed else (),
        metric_catalog=value.catalog, profile_catalog=value.profiles,
    ))


def facts(value, *, cpu=10):
    result = [{
        "kind": "inspection_monitoring_metric", "metric_code": "db.availability",
        "semantic_role": "AVAILABILITY", "latest": 1, "evidence_ref": "db-evidence",
        "unit": "state", "coverage_ratio": 1,
        "window_start": value.snapshot["current_window"]["start"],
        "window_end": value.snapshot["current_window"]["end"],
        "sample_points": [{"observed_at": NOW.isoformat(), "value": 1}],
    }]
    for expected in value.snapshot.get("required_host_metrics", ()):
        if not expected["binding_ids"]:
            continue
        result.append({
            "kind": "inspection_monitoring_metric", "metric_code": expected["metric_code"],
            "semantic_role": expected["semantic_role"], "binding_id": expected["binding_ids"][0],
            "latest": cpu if expected["semantic_role"] == "CPU_UTILIZATION" else 20,
            "unit": "percent", "coverage_ratio": 1, "evidence_ref": "host-evidence",
            "window_start": value.snapshot["current_window"]["start"],
            "window_end": value.snapshot["current_window"]["end"],
            "sample_points": [{"observed_at": NOW.isoformat(), "value": cpu if expected["semantic_role"] == "CPU_UTILIZATION" else 20}],
        })
    return result


def assessment(value, raw_facts):
    policy = value.inspection["report_policy"]
    checks = [{
        "kind": "inspection_check", "title": spec["title"], "check_ids": [spec["check_id"]],
        "required": spec["required"], "check_status": "CLEAR", "evidence_refs": ["database-evidence"],
    } for spec in policy["checks"]]
    checks.extend(monitoring_checks(
        classify_monitoring_facts(raw_facts, monitoring=value.snapshot),
        profile=value.snapshot["monitoring_profile"], policy=policy, monitoring=value.snapshot,
    ))
    return assess_inspection(checks, policy=policy)


@pytest.mark.parametrize(("db_type", "version"), [("ORACLE", "19c"), ("MYSQL", "8.4"), ("POSTGRESQL", "16")])
def test_inspection_freezes_real_host_metrics_and_catalog_thresholds(db_type, version):
    value = fixture(db_type, version)
    attach(value)
    frozen = value.snapshot["bindings"][-1]
    assert frozen["purpose"] == "INSPECTION_HOST"
    assert frozen["resource_id"] == str(value.host.resource_id)
    assert frozen["source_locator"]["target_key"] == "host-1"
    assert {item["metric_code"] for item in frozen["metrics"]} == set(HOST_CODES)
    assert frozen["binding_id"] in value.snapshot["observation_binding_ids"]
    assert set(value.snapshot["inspection_host_profile"]["metric_thresholds"]) >= set(HOST_CODES)
    assert value.uow.business_systems.list_storage_host_bindings.call_args.kwargs["business_system_ids"] == [value.system.business_system_id]
    assert asyncio.run(source_scope_is_current(uow=value.uow, run=value.run, domain_id=1))
    result = assessment(value, facts(value))
    assert result["completion_status"] == "COMPLETE"
    assert result["health_status"] == "NORMAL"


def test_host_threshold_breach_is_complete_but_not_healthy():
    value = fixture()
    attach(value)
    result = assessment(value, facts(value, cpu=99))
    assert result["completion_status"] == "COMPLETE"
    assert result["health_status"] == "ABNORMAL"


@pytest.mark.parametrize("mutation", [
    "unauthorized_source", "unrelated", "too_many", "inactive", "wrong_scope", "wrong_locator",
    "no_capability", "no_metrics", "only_cpu", "wrong_system",
])
def test_unavailable_host_monitoring_cannot_become_complete(mutation):
    value = fixture()
    if mutation == "unrelated":
        value.uow.business_systems.list_storage_host_bindings.return_value = []
    elif mutation == "too_many":
        value.uow.business_systems.list_storage_host_bindings.return_value *= 17
    elif mutation == "inactive":
        value.source.status = "DISABLED"
    elif mutation == "wrong_scope":
        value.binding.source_locator_json["resource_scope"] = "DATABASE"
    elif mutation == "wrong_locator":
        value.binding.source_locator_json["target_key"] = "unrelated-host"
    elif mutation == "no_capability":
        value.binding.capability_scope_json["capabilities"] = []
    elif mutation == "no_metrics":
        value.binding.capability_scope_json["metric_codes"] = []
    elif mutation == "only_cpu":
        value.binding.capability_scope_json["metric_codes"] = [HOST_CODES[0]]
    elif mutation == "wrong_system":
        value.inspection["business_system_id"] = str(uuid7())
        value.uow.business_systems.list_storage_host_bindings.side_effect = (
            lambda **kwargs: [] if not kwargs["business_system_ids"] else [(value.host, value.binding, value.system, value.relation)]
        )
    attach(value, allowed=mutation != "unauthorized_source")
    result = assessment(value, facts(value))
    assert result["completion_status"] == "PARTIAL"
    if mutation != "only_cpu":
        assert len(value.snapshot["bindings"]) == 1


def test_one_healthy_host_cannot_cover_another_host_without_samples():
    value = fixture()
    second_host = SimpleNamespace(resource_id=uuid7(), display_name="数据库主机二", resource_type_key="PHYSICAL_SERVER")
    second_binding = deepcopy(value.binding)
    second_binding.resource_source_binding_id = uuid7()
    second_binding.source_locator_key = "host-2"
    second_binding.source_locator_json = {"resource_scope": "HOST", "target_key": "host-2"}
    value.uow.business_systems.list_storage_host_bindings.return_value.append((second_host, second_binding, value.system, value.relation))
    attach(value)
    original = facts(value)
    original = [item for item in original if item.get("binding_id") != str(second_binding.resource_source_binding_id)]
    result = assessment(value, original)
    assert result["completion_status"] == "PARTIAL"
    assert any("数据库主机二" in item for item in result["limitations"])


@pytest.mark.parametrize("mutation", ["old_sample", "missing_trace", "empty_series", "missing_metric", "wrong_binding"])
def test_invalid_host_evidence_does_not_complete_the_report(mutation):
    value = fixture()
    attach(value)
    raw = facts(value)
    if mutation == "old_sample":
        raw[1]["sample_points"][0]["observed_at"] = (NOW - timedelta(minutes=10)).isoformat()
    elif mutation == "missing_trace":
        raw[1]["sample_points"] = []
    elif mutation == "empty_series":
        raw[1].update(kind="inspection_monitoring_absence", latest=None)
    elif mutation == "missing_metric":
        raw.pop(1)
    elif mutation == "wrong_binding":
        raw[1]["binding_id"] = str(uuid7())
    assert assessment(value, raw)["completion_status"] == "PARTIAL"


@pytest.mark.parametrize("mutation", ["relation_revoked", "relation_version", "binding_version", "system", "source_version"])
def test_host_collection_and_history_recheck_current_authorization(mutation):
    value = fixture()
    attach(value)
    if mutation == "relation_revoked":
        value.uow.business_systems.list_storage_host_bindings.return_value = []
    elif mutation == "relation_version":
        value.relation.row_version += 1
    elif mutation == "binding_version":
        value.binding.row_version += 1
    elif mutation == "system":
        value.system.business_system_id = uuid7()
    elif mutation == "source_version":
        value.source.row_version += 1
    assert not asyncio.run(source_scope_is_current(uow=value.uow, run=value.run, domain_id=1))


@pytest.mark.parametrize("schedule_type", ["DAILY", "WEEKLY", None])
def test_planning_adds_host_collection_only_to_inspections_without_changing_database_overview(schedule_type):
    value = fixture()
    if schedule_type:
        value.inspection["schedule_type"] = schedule_type
    else:
        value.inspection = {}
    original = value.profiles.overview_for_database("ORACLE", "19c").model_dump(mode="json")

    @asynccontextmanager
    async def factory():
        yield value.uow

    service = TurnPlanningService(
        uow_factory=factory, investigation_reasoner=None, playbook_registry=None,
        task_compiler=None, tool_snapshot_builder=None, agent_catalog=None,
        monitoring_snapshot_builder=MonitoringSnapshotBuilder(
            metric_catalog=value.catalog, default_window_seconds=3600, max_response_bytes=1000000,
        ), monitoring_profile_catalog=value.profiles,
    )
    context = SimpleNamespace(
        target_id=value.target.target_id, domain_id=1, ops_run_id=value.run.ops_run_id,
        source_ids=(value.source.diagnostic_source_id,), inspection=value.inspection,
    )
    snapshot = asyncio.run(service._prepare_monitoring(context, align_with_realtime_profile=True))
    assert any(item.get("purpose") == "INSPECTION_HOST" for item in snapshot["bindings"]) is bool(schedule_type)
    assert snapshot["monitoring_profile"] == original
    if schedule_type == "DAILY":
        assert snapshot["current_window"] == snapshot["window"]
    if schedule_type:
        assert snapshot["inspection_host_profile"]["profile_id"] == "host-overview"
    else:
        value.uow.business_systems.list_storage_host_bindings.assert_not_awaited()


def test_monitor_resource_admission_rejects_revoked_host_relation():
    value = fixture()
    attach(value)
    value.run.status = "RUNNING"
    value.run.cancel_requested_at = None
    value.run.deadline_at = NOW + timedelta(minutes=5)
    task = SimpleNamespace(
        ops_run_id=value.run.ops_run_id, status="RUNNING", lease_token="lease",
        lease_until=NOW + timedelta(minutes=2), started_at=NOW,
    )
    value.uow.runs.get_task = AsyncMock(return_value=task)
    value.uow.resources = SimpleNamespace(database_now=AsyncMock(return_value=NOW))
    value.uow.business_systems.list_storage_host_bindings.return_value = []

    @asynccontextmanager
    async def factory():
        yield value.uow

    context = SimpleNamespace(
        run_id=value.run.ops_run_id, target_id=value.run.target_id, task_id=uuid7(), lease_token="lease",
    )
    with pytest.raises(ResourceAdmissionError, match="RESOURCE_SOURCE_SCOPE_INVALID"):
        asyncio.run(ResourceGovernanceService(uow_factory=factory).acquire_monitor(
            context=context, source_id=value.source.diagnostic_source_id, lease_id=uuid7(),
        ))

def test_related_host_without_any_binding_remains_a_required_missing_check():
    value = fixture()
    second_host = SimpleNamespace(resource_id=uuid7(), display_name="未配置监控主机", resource_type_key="PHYSICAL_SERVER")
    value.uow.business_systems.list_storage_host_bindings.return_value.append(
        (second_host, None, value.system, value.relation),
    )
    attach(value)
    result = assessment(value, facts(value))
    assert result["completion_status"] == "PARTIAL"
    assert sum("未配置监控主机" in item for item in result["limitations"]) == 3
    assert value.uow.business_systems.list_storage_host_bindings.call_args_list[0].kwargs["include_unbound"] is True


def test_repository_preserves_unbound_hosts_only_for_explicit_inspection_discovery():
    import inspect
    from aiops_agent.repositories.business_system import BusinessSystemRepository

    for include_unbound in (False, True):
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [])))
        repository = BusinessSystemRepository(session)
        kwargs = {
            "target_id": uuid7(), "domain_id": 1, "business_system_ids": [uuid7()],
            "include_unbound": include_unbound,
        }
        if "tenant_id" in inspect.signature(repository.list_storage_host_bindings).parameters:
            kwargs["tenant_id"] = uuid7()
        asyncio.run(repository.list_storage_host_bindings(**kwargs))
        sql = str(session.execute.call_args.args[0].compile())
        assert "LEFT OUTER JOIN" in sql
        assert ("resource_source_binding_id IS NOT NULL" in sql) is not include_unbound


def test_real_metric_projection_preserves_host_binding_and_fresh_sample_time():
    from aiops_agent.application.runtime.service import AIOpsRuntimeService
    from aiops_agent.contracts.evidence import MetricObservation, MetricPoint, MetricSeries, ObservationSet
    from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler

    value = fixture()
    attach(value)
    start = datetime.fromisoformat(value.snapshot["window"]["start"])
    observations = []
    for code in HOST_CODES:
        definition = value.catalog.get(code)
        observations.append(MetricObservation(
            metric_code=code, semantic_role=definition.semantic_role,
            semantic_version=definition.semantic_version, unit=definition.unit, value_kind=definition.value_kind,
            window_start=start, window_end=NOW, requested_step_seconds=60, effective_step_seconds=60,
            source_id=str(value.source.diagnostic_source_id), source_type="PROMETHEUS", source_version=1,
            target_id=str(value.target.target_id), binding_id=str(value.binding.resource_source_binding_id),
            external_target_fingerprint="host-1",
            series=(MetricSeries(dimensions={"target_key": "host-1"}, points=(MetricPoint(observed_at=NOW, value=10),)),),
            expected_points=16, actual_points=1, coverage_ratio=1/16,
        ))
    result = ObservationSet(
        target_id=str(value.target.target_id), binding_id=str(value.binding.resource_source_binding_id),
        source_id=str(value.source.diagnostic_source_id), observations=tuple(observations), collected_at=NOW,
    )
    evidence = DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=str(uuid7()), result=result)
    projected = AIOpsRuntimeService._inspection_monitoring_facts(SimpleNamespace(evidence=(evidence,)))
    assert len(projected) == 3
    assert all(item["binding_id"] == str(value.binding.resource_source_binding_id) for item in projected)
    assert assessment(value, [facts(value)[0], *projected])["completion_status"] == "COMPLETE"


@pytest.mark.parametrize("source_type", ["PROMETHEUS", "ZABBIX"])
def test_host_metric_selection_honors_provider_catalog(source_type):
    value = fixture()
    value.source.source_type = source_type
    attach(value)
    assert {item["metric_code"] for item in value.snapshot["bindings"][-1]["metrics"]} == set(HOST_CODES)

@pytest.mark.parametrize("missing_host_sample", [False, True])
def test_report_publication_uses_frozen_host_policy_and_preserves_it(missing_host_sample):
    from aiops_agent.application.runtime.service import AIOpsRuntimeService
    from unittest.mock import Mock

    value = fixture()
    attach(value)
    value.inspection.update(
        template_id="database_daily", template_version="1.0.0", timezone="Asia/Shanghai",
        period_start=(NOW - timedelta(days=1)).isoformat(), period_end=NOW.isoformat(),
    )
    value.run.inspection_fire_id = uuid7()
    value.run.actor_id = "system:inspection-scheduler"
    value.run.plan_snapshot_json["target"] = {"db_type": "ORACLE"}
    raw = facts(value)
    if missing_host_sample:
        raw.pop(1)
    projected = [{
        "kind": "inspection_check", "title": spec["title"], "tool_id": spec["tool_id"],
        "check_ids": [spec["check_id"]], "required": spec["required"],
        "check_status": "CLEAR", "evidence_refs": ["database-evidence"],
    } for spec in value.inspection["report_policy"]["checks"]]
    projected.extend(classify_monitoring_facts(raw, monitoring=value.snapshot))

    async def add_artifact(entity):
        entity.artifact_id = uuid7()
        return entity

    value.uow.inspections = SimpleNamespace(
        publish_report=AsyncMock(side_effect=lambda entity: entity),
    )
    value.uow.runs.add_artifact = AsyncMock(side_effect=add_artifact)
    value.uow.runs.append_event = AsyncMock()
    value.uow.platform_notifications = SimpleNamespace(emit_report_ready=AsyncMock())
    service = AIOpsRuntimeService(
        uow_factory=AsyncMock(), blueprint_registry=AsyncMock(), handler_registry=AsyncMock(),
    )
    service._report_security_level = AsyncMock(return_value=1)
    service._add_outbox = AsyncMock()
    service._inspection_report_projection = Mock(return_value=(tuple(projected), (), {}))
    source = SimpleNamespace(
        artifact_id=uuid7(), content_hash="a"*64, schema_version="AIOPS_TURN_RESULT.v1",
        payload_json={
            "status": "COMPLETED", "sufficiency_status": "ANSWERABLE",
            "blocks": [{"block_type": "MARKDOWN", "schema_version": "AIOPS_MARKDOWN_BLOCK.v1",
                        "payload": {"markdown": "巡检完成。"}, "evidence_refs": []}],
        },
    )
    frozen = deepcopy(value.snapshot)
    artifact = asyncio.run(service._publish_turn_inspection_report(
        uow=value.uow, run=value.run, task=SimpleNamespace(ops_task_id=uuid7()),
        source_artifact=source, now=NOW, trace_id="inspection-host-publication",
    ))
    payload = artifact.payload_json
    assert payload["status"] == ("PARTIAL" if missing_host_sample else "READY")
    assert payload["scope"]["inspection_host_profile"] == frozen["inspection_host_profile"]
    assert payload["scope"]["required_host_metrics"] == frozen["required_host_metrics"]
    assert value.snapshot == frozen
