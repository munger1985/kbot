"""物理容量链路：真实路径、授权主机、同卷预测和只读采集的回归。"""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import importlib.util
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
from aiops_agent.application.diagnosis.storage_capacity import storage_capacity_resolution
from aiops_agent.application.diagnosis.storage_hosts import attach_storage_hosts
from aiops_agent.application.source_authorization import source_scope_is_current
from aiops_agent.contracts.evidence import ObservationSet
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.monitoring.catalog import resolve_metric_definitions
from aiops_agent.monitoring.query_policy import MonitoringQueryRejected, PromQueryPolicy, PromQueryPolicySnapshot
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, MetricsEvidenceRequest
from aiops_agent.repositories.business_system import BusinessSystemRepository
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_turn_answer import _monitoring_artifact
from tests.unit.aiops_agent.test_aiops_observability_stack import _load_stack_script


ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 9, tzinfo=UTC)
SERVER_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
FRAME = {"visualization_profile_id": "storage.trend"}
FILESYSTEM = {"target_key": "host-1", "mountpoint": "/srv/mysql", "device": "/dev/loop3", "fstype": "ext4"}


def storage_fixture(*, growth=1, horizon=30, maximum=100):
    artifact = _monitoring_artifact()
    binding_id = artifact["payload"]["binding_id"]
    original = artifact["payload"]["observations"][0]
    original.update(
        window_start=(NOW - timedelta(days=7)).isoformat(), window_end=NOW.isoformat(),
        expected_points=8, actual_points=8, unit="bytes", value_kind="GAUGE",
    )
    artifact["payload"]["collected_at"] = NOW.isoformat()
    observations = []
    for code, role, values in (
        ("host.filesystem.used_bytes", "STORAGE_USED", [10 + index * growth for index in range(8)]),
        ("host.filesystem.size_bytes", "STORAGE_MAX", [maximum] * 8),
        ("host.filesystem.available_bytes", "STORAGE_FREE", [maximum - 10 - index * growth for index in range(8)]),
        ("host.storage.mount_info", "STORAGE_LOCATION", [int(NOW.timestamp())] * 8),
    ):
        observation = deepcopy(original)
        dims = dict(FILESYSTEM)
        if role == "STORAGE_LOCATION":
            dims.update(database_target_key="mysql-1", server_uuid=SERVER_UUID,
                database_path="/var/lib/mysql", host_path="/srv/mysql")
        observation.update(metric_code=code, semantic_role=role,
            unit="timestamp_seconds" if role == "STORAGE_LOCATION" else "bytes",
            value_kind="STATE" if role == "STORAGE_LOCATION" else "GAUGE",
            series=[{"dimensions": dims, "points": [
                {"observed_at": (NOW - timedelta(days=7-index)).isoformat(), "value": value}
                for index, value in enumerate(values)
            ]}])
        observations.append(observation)
    artifact["payload"]["observations"] = observations
    fact = DbaEvidenceAssessmentHandler._monitoring_fact(
        artifact_id=artifact["artifact_id"], result=ObservationSet.model_validate(artifact["payload"]),
        forecast_horizon_days=horizon,
    )
    paths = TurnEvidenceFact(
        evidence_ref="artifact:paths#rows", artifact_id=str(uuid7()), source_id="db.storage.locations",
        step_id="a1", tool_id="db.storage.locations", measurement_semantics="CURRENT_ACTIVITY",
        presentation_kind="TABLE", captured_at=NOW.isoformat(),
        columns=tuple({"name": name} for name in ("location_kind", "storage_name", "file_path", "server_uuid")),
        rows=(("DATA_DIRECTORY", "datadir", "/var/lib/mysql/", SERVER_UUID),
              ("TABLESPACE", "demo/data", "./demo/data.ibd", SERVER_UUID)),
        row_count=2,
    )
    monitoring = {"bindings": [{
        "binding_id": binding_id, "purpose": "STORAGE_CAPACITY", "resource_name": "DB Host",
        "resource_id": str(uuid7()), "source_locator_key": "host-1",
        "source": {"source_id": artifact["payload"]["source_id"]},
        "storage_context": {"database_locator_keys": ["mysql-1"]},
    }]}
    return artifact, fact, paths, monitoring


def resolve(fact, paths, monitoring):
    return storage_capacity_resolution(task_frame=FRAME, monitoring=monitoring, evidence=(fact, paths))


def remake(artifact, *, horizon=30):
    return DbaEvidenceAssessmentHandler._monitoring_fact(
        artifact_id=artifact["artifact_id"], result=ObservationSet.model_validate(artifact["payload"]),
        forecast_horizon_days=horizon,
    )


def test_logical_database_size_is_not_disk_limit_and_volume_has_own_forecast():
    _, fact, paths, monitoring = storage_fixture()
    result = resolve(fact, paths, monitoring)
    assert result["status"] == "RESOLVED"
    assert len(result["volumes"]) == 1
    volume = result["volumes"][0]
    assert volume["mountpoint"] == "/srv/mysql"
    assert len(volume["paths"]) == 2
    assert volume["metrics"]["host.filesystem.used_bytes"]["estimated_days_to_limit"] == 83
    assert not any("TREND_SEMANTICS_UNSUPPORTED" in warning for warning in fact.warnings)


@pytest.mark.parametrize("offset_minutes,coverage,expected", [
    (30, 1, True), (90, 1, True), (110, 1, False), (30, .99, False),
])
def test_full_window_sampling_endpoint_does_not_permanently_block_forecast(offset_minutes, coverage, expected):
    artifact, _, paths, monitoring = storage_fixture()
    for observation in artifact["payload"]["observations"][:-1]:
        observation["effective_step_seconds"] = 6000
        observation["coverage_ratio"] = coverage
        observation["series"][0]["points"][-1]["observed_at"] = (
            NOW - timedelta(minutes=offset_minutes)).isoformat()
    result = resolve(remake(artifact), paths, monitoring)
    days = result["volumes"][0]["metrics"]["host.filesystem.used_bytes"]["estimated_days_to_limit"]
    assert (days is not None) == expected


def test_sampling_allowance_cannot_turn_a_short_requested_window_into_seven_days():
    artifact, _, paths, monitoring = storage_fixture()
    for observation in artifact["payload"]["observations"][:-1]:
        observation["window_start"] = (NOW - timedelta(days=6)).isoformat()
        observation["effective_step_seconds"] = 86400
        observation["series"][0]["points"] = observation["series"][0]["points"][1:]
        observation["expected_points"] = observation["actual_points"] = 7
    result = resolve(remake(artifact), paths, monitoring)
    assert result["volumes"][0]["metrics"]["host.filesystem.used_bytes"]["estimated_days_to_limit"] is None


def test_prometheus_full_week_grid_preserves_last_sample_and_permits_volume_forecast():
    artifact, _, paths, monitoring = storage_fixture()
    payload = artifact["payload"]
    request = MetricsEvidenceRequest(
        target_id=payload["target_id"], binding_id=payload["binding_id"],
        source_locator_key="host-1", source_locator={"resource_scope": "HOST"},
        metric_definitions=tuple(load_metric_catalog().get(code) for code in (
            "host.filesystem.used_bytes", "host.filesystem.size_bytes")),
        window_start=NOW-timedelta(days=7), window_end=NOW,
        requested_step_seconds=60, max_response_bytes=262144, trace_id="grid-regression",
    )
    adapter = PrometheusAdapter(context=DiagnosticSourceContext(
        source_id=payload["source_id"], source_type="PROMETHEUS", adapter_id="prometheus",
        adapter_version="1.0.0", config_version=1, endpoint="http://prometheus",
        declared_capabilities={"metric.query_range": {}},
    ), session=Mock(), request_timeout_seconds=10, webhook_replay_seconds=300)
    async def query_range(*, request, query, step):
        count = int((request.window_end-request.window_start).total_seconds() // step) + 1
        assert count <= min(item.max_points for item in request.metric_definitions)
        points = [(request.window_start+timedelta(seconds=index*step),
                   10+index*step/86400 if " - " in query else 100) for index in range(count)]
        return [(FILESYSTEM, points)], "a"*64
    adapter._query_range = query_range
    queried = asyncio.run(adapter.query_metrics(request))
    assert not queried.gaps
    payload["observations"] = [item.model_dump(mode="json") for item in queried.observations] + payload["observations"][-1:]
    result = resolve(remake(artifact), paths, monitoring)
    metric = result["volumes"][0]["metrics"]["host.filesystem.used_bytes"]
    assert not metric["quality_flags"]
    assert metric["estimated_days_to_limit"] is not None


@pytest.mark.parametrize("horizon,growth,expected", [(None, 1, None), (30, 0, None), (30, -1, None)])
def test_no_forecast_request_or_non_growing_volume_does_not_fabricate_days(horizon, growth, expected):
    _, fact, paths, monitoring = storage_fixture(horizon=horizon, growth=growth)
    result = resolve(fact, paths, monitoring)
    assert result["volumes"][0]["metrics"]["host.filesystem.used_bytes"]["estimated_days_to_limit"] is expected


@pytest.mark.parametrize("mutation", ["uuid", "target", "host", "expired", "future", "boundary", "source", "untrusted", "missing"])
def test_foreign_stale_ambiguous_or_untrusted_mapping_never_selects_arbitrary_disk(mutation):
    artifact, fact, paths, monitoring = storage_fixture()
    info = artifact["payload"]["observations"][-1]
    dims = info["series"][0]["dimensions"]
    if mutation == "uuid":
        dims["server_uuid"] = "other"
    elif mutation == "target":
        dims["database_target_key"] = "another-database"
    elif mutation == "host":
        dims["target_key"] = "another-host"
    elif mutation in {"expired", "future"}:
        for point in info["series"][0]["points"]:
            point["value"] = int(NOW.timestamp()) + (-301 if mutation == "expired" else 31)
    elif mutation == "boundary":
        dims["database_path"] = "/var/lib/mysqlx"
    elif mutation == "source":
        info["source_id"] = str(uuid7())
    elif mutation == "missing":
        artifact["payload"]["observations"].pop()
    fact = remake(artifact)
    if mutation == "untrusted":
        paths = paths.model_copy(update={"trust_level": "USER_PROVIDED"})
    result = resolve(fact, paths, monitoring)
    assert not result["volumes"]
    assert result["status"] == "UNAVAILABLE"


def test_external_tablespaces_need_their_own_mapping_not_datadir_root_volume():
    artifact, fact, paths, monitoring = storage_fixture()
    paths = paths.model_copy(update={"rows": (*paths.rows, ("TABLESPACE", "external",
        "/external/mysql/data.ibd", SERVER_UUID)), "row_count": 3})
    result = resolve(fact, paths, monitoring)
    assert result["status"] == "PARTIAL"
    assert result["unresolved_paths"][0]["path"] == "/external/mysql/data.ibd"
    assert len(result["volumes"][0]["paths"]) == 2


def test_longest_directory_boundary_mapping_and_multiple_filesystems_are_separate():
    artifact, fact, paths, monitoring = storage_fixture()
    info = artifact["payload"]["observations"][-1]
    nested = deepcopy(info["series"][0])
    nested["dimensions"].update(database_path="/var/lib/mysql/demo", host_path="/fast/mysql",
        mountpoint="/fast", device="/dev/loop4")
    info["series"].append(nested)
    result = resolve(remake(artifact), paths, monitoring)
    assert {volume["mountpoint"] for volume in result["volumes"]} == {"/srv/mysql", "/fast"}
    nested_volume = next(volume for volume in result["volumes"] if volume["mountpoint"] == "/fast")
    assert not nested_volume["metrics"]


def test_equally_specific_mapping_conflict_is_not_arbitrarily_resolved():
    artifact, fact, paths, monitoring = storage_fixture()
    info = artifact["payload"]["observations"][-1]
    duplicate = deepcopy(info["series"][0])
    duplicate["dimensions"]["device"] = "/dev/other"
    info["series"].append(duplicate)
    assert not resolve(remake(artifact), paths, monitoring)["volumes"]


@pytest.mark.parametrize("change", ["resize", "unit", "truncated", "coverage", "time_window"])
def test_incompatible_or_changing_capacity_does_not_predict_disk_exhaustion(change):
    artifact, fact, paths, monitoring = storage_fixture()
    maximum = artifact["payload"]["observations"][1]
    if change == "resize":
        maximum["series"][0]["points"][0]["value"] = 90
    elif change == "unit":
        maximum["unit"] = "kilobytes"
    elif change == "truncated":
        maximum["truncated"] = True
    elif change == "coverage":
        maximum["coverage_ratio"] = .5
    elif change == "time_window":
        maximum["window_start"] = (NOW - timedelta(days=8)).isoformat()
    result = resolve(remake(artifact), paths, monitoring)
    assert not result["volumes"] or result["volumes"][0]["metrics"]["host.filesystem.used_bytes"]["estimated_days_to_limit"] is None


def test_truncated_path_enumeration_and_missing_paths_remain_truthful():
    _, fact, paths, monitoring = storage_fixture()
    result = resolve(fact, paths.model_copy(update={"truncated": True}), monitoring)
    assert result["status"] == "PARTIAL"
    assert result["path_enumeration_truncated"]
    empty = paths.model_copy(update={"rows": (), "row_count": 0})
    assert resolve(fact, empty, monitoring)["status"] == "UNAVAILABLE"


def test_storage_tools_are_versioned_readonly_and_bounded():
    registry = DiagnosticRegistry.load()
    tools = [item for item in registry._tools if item.definition.tool_id == "db.storage.locations"]
    assert len(tools) == 1
    tool = tools[0]
    assert tool.definition.db_type == "MYSQL"
    assert tool.definition.limit_semantics == "BOUNDED_ENUMERATION"
    assert "@@server_uuid" in tool.sql and "information_schema.FILES" in tool.sql
    assert "LIMIT :limit" in tool.sql
    definition = load_metric_catalog().get("host.storage.mount_info")
    assert definition.resource_scope == "HOST"
    assert definition.value_kind == "STATE"
    assert "@ end()" in definition.providers["PROMETHEUS"].query_template


def test_live_storage_host_snapshot_survives_execution_query_validation():
    uow, run, snapshot, source, _ = host_scope()
    attach(uow, run, snapshot, source)
    definitions = resolve_metric_definitions(snapshot["bindings"][-1])
    assert len(definitions) == 5
    mapping = next(item for item in definitions if item.semantic_role == "STORAGE_LOCATION")
    assert "@ end()" in mapping.providers["PROMETHEUS"].query_template


@pytest.mark.parametrize("modifier", ["@ end()", "@ start()", "@ 1234567890", "offset 5m"])
def test_ad_hoc_query_cannot_opt_in_to_mapping_time_policy(modifier):
    query = 'up{target_key="${host_target}"} ' + modifier
    with pytest.raises(MonitoringQueryRejected):
        PromQueryPolicy(PromQueryPolicySnapshot()).validate(query)


@pytest.mark.parametrize("modifier", ["@ start()", "@ 1234567890", "offset 5m", "@ end() offset 5m"])
def test_mapping_policy_does_not_allow_time_escape(modifier):
    definition = load_metric_catalog().get("host.storage.mount_info").model_dump(mode="json")
    definition["providers"]["PROMETHEUS"]["query_template"] = (
        'aiops_database_storage_mount_info{target_key="${host_target}"} ' + modifier)
    with pytest.raises(MonitoringQueryRejected):
        resolve_metric_definitions({"metrics": [definition]})


def test_end_modifier_is_not_enabled_for_regular_catalog_metrics():
    definition = load_metric_catalog().get("host.filesystem.used_bytes").model_dump(mode="json")
    definition["providers"]["PROMETHEUS"]["query_template"] += " @ end()"
    with pytest.raises(MonitoringQueryRejected):
        resolve_metric_definitions({"metrics": [definition]})


def host_scope():
    target, tenant, source_id = uuid7(), uuid7(), uuid7()
    source = SimpleNamespace(diagnostic_source_id=source_id, tenant_id=tenant, domain_id=1,
        status="ENABLED", source_type="PROMETHEUS", row_version=1, adapter_id="prometheus",
        adapter_version="1.0.0", connectivity_status="CONNECTED", endpoint="http://prometheus",
        auth_credential_id=None, declared_capabilities_json={"metric.query_range": {}}, config_json={})
    db_binding = SimpleNamespace(resource_source_binding_id=uuid7(), diagnostic_source_id=source_id,
        source_locator_key="mysql-1", row_version=1)
    host = SimpleNamespace(resource_id=uuid7(), resource_type_key="VIRTUAL_MACHINE", display_name="DB Host")
    binding = SimpleNamespace(resource_source_binding_id=uuid7(), diagnostic_source_id=source_id,
        row_version=1, source_locator_key="host-1", source_locator_json={"resource_scope": "HOST", "target_key": "host-1"},
        capability_scope_json={}, mapping_overrides_json={}, query_budget_json={})
    system = SimpleNamespace(business_system_id=uuid7())
    relation = SimpleNamespace(relation_id=uuid7(), row_version=1)
    snapshot = {"bindings": [{
        "binding_id": str(db_binding.resource_source_binding_id), "binding_version": 1,
        "source_locator_key": "mysql-1", "source_locator": {},
        "source": {"source_id": str(source_id), "config_version": 1, "declared_capabilities": {"metric.query_range": {}}},
    }], "initial_gaps": [], "observation_binding_ids": []}
    run = SimpleNamespace(target_id=target, tenant_id=tenant, domain_id=1, agent_version_id=uuid7(),
        plan_snapshot_json={"monitoring": snapshot})
    rows = [(host, binding, system, relation)]
    uow = SimpleNamespace(
        agents=SimpleNamespace(version_business_system_ids=AsyncMock(return_value=[system.business_system_id])),
        business_systems=SimpleNamespace(list_storage_host_bindings=AsyncMock(return_value=rows)),
        diagnostic_sources=SimpleNamespace(get_scoped=AsyncMock(return_value=source)),
        targets=SimpleNamespace(list_source_bindings=AsyncMock(return_value=[db_binding])),
    )
    return uow, run, snapshot, source, rows


def attach(uow, run, snapshot, source, *, allowed=True):
    return asyncio.run(attach_storage_hosts(uow=uow, snapshot=snapshot,
        target=SimpleNamespace(target_id=run.target_id, domain_id=run.domain_id, db_type="MYSQL"), run=run,
        allowed_source_ids=(source.diagnostic_source_id,) if allowed else (), metric_catalog=load_metric_catalog()))


def test_host_binding_is_authorized_through_agent_business_system_not_hostname_guess():
    uow, run, snapshot, source, rows = host_scope()
    attach(uow, run, snapshot, source)
    assert len(snapshot["bindings"]) == 2
    frozen = snapshot["bindings"][-1]
    assert frozen["source_locator"]["resource_scope"] == "HOST"
    assert frozen["effective_capabilities"] == ["metric.query_range"]
    assert frozen["storage_context"]["database_locator_keys"] == ["mysql-1"]
    assert len(frozen["metrics"]) == 5
    assert asyncio.run(source_scope_is_current(uow=uow, run=run, domain_id=1))


@pytest.mark.parametrize("mutation", ["unauthorized_source", "unrelated", "too_many", "inactive", "scope", "no_capability", "no_metrics"])
def test_host_discovery_never_expands_unauthorized_scope(mutation):
    uow, run, snapshot, source, rows = host_scope()
    binding = rows[0][1]
    if mutation == "unrelated":
        uow.business_systems.list_storage_host_bindings.return_value = []
    elif mutation == "too_many":
        uow.business_systems.list_storage_host_bindings.return_value = rows * 17
    elif mutation == "inactive":
        source.status = "DISABLED"
    elif mutation == "scope":
        binding.source_locator_json["resource_scope"] = "DATABASE"
    elif mutation == "no_capability":
        binding.capability_scope_json["capabilities"] = []
    elif mutation == "no_metrics":
        binding.capability_scope_json["metric_codes"] = []
    attach(uow, run, snapshot, source, allowed=mutation != "unauthorized_source")
    assert len(snapshot["bindings"]) == 1


@pytest.mark.parametrize("mutation", ["relationship_revoked", "relation_version", "binding_version", "system", "locator"])
def test_live_query_and_history_reuse_revalidate_current_relationship_and_permissions(mutation):
    uow, run, snapshot, source, rows = host_scope()
    attach(uow, run, snapshot, source)
    host, binding, system, relation = rows[0]
    if mutation == "relationship_revoked":
        uow.business_systems.list_storage_host_bindings.return_value = []
    elif mutation == "relation_version":
        relation.row_version += 1
    elif mutation == "binding_version":
        binding.row_version += 1
    elif mutation == "system":
        system.business_system_id = uuid7()
    elif mutation == "locator":
        binding.source_locator_json["target_key"] = "other"
    assert not asyncio.run(source_scope_is_current(uow=uow, run=run, domain_id=1))


def test_native_repository_query_requires_system_roles_and_confirmed_relation():
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [])))
    repo = BusinessSystemRepository(session)
    kwargs = {"target_id": uuid7(), "domain_id": 1, "business_system_ids": [uuid7()]}
    if "tenant_id" in inspect.signature(repo.list_storage_host_bindings).parameters:
        kwargs["tenant_id"] = uuid7()
    asyncio.run(repo.list_storage_host_bindings(**kwargs))
    compiled = session.execute.call_args.args[0].compile()
    values = repr(compiled.params)
    assert all(item in values for item in ("RUNS_ON", "CONFIRMED", "WORKLOAD", "HOST", "PHYSICAL_SERVER", "VIRTUAL_MACHINE"))
    assert "JOIN" in str(compiled)


def collector():
    path = ROOT / "scripts/deployment/aiops_observability/storage_path_collector/collector.py"
    spec = importlib.util.spec_from_file_location("storage_collector_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collector_reads_real_mount_devices_and_does_not_require_docker_socket():
    module = collector()
    config = {"mappings": [{"database_target_key": "mysql-1", "server_uuid": SERVER_UUID,
        "database_path": "/var/lib/mysql", "host_path": "/srv/mysql", "container_id": "a"*64}]}
    mounts = "1 0 8:1 / / rw - ext4 /dev/sda1 rw\n2 1 7:3 / /srv/mysql rw - ext4 /dev/loop3 rw"
    stat = lambda path: SimpleNamespace(st_dev=os.makedev(7, 3), st_mode=0o40755)
    namespaces = {"a"*64: module.parse_mounts("1 0 7:3 / /var/lib/mysql rw - ext4 /dev/loop3 rw")}
    payload, errors = module.collect(config, mounts, stamp=NOW.timestamp(), stat=stat, lstat=stat, namespaces=namespaces)
    assert not errors
    assert 'mountpoint="/srv/mysql"' in payload and 'device="/dev/loop3"' in payload
    wrong = lambda path: SimpleNamespace(st_dev=os.makedev(8, 1), st_mode=0o40755)
    payload, errors = module.collect(config, mounts, stamp=NOW.timestamp(), stat=wrong, lstat=wrong, namespaces=namespaces)
    assert errors and "\naiops_database_storage_mount_info{" not in payload


def test_collector_selects_longest_parent_and_escapes_mountinfo_paths():
    module = collector()
    parsed = module.parse_mounts(r"1 0 7:3 / /srv/my\040sql rw - ext4 /dev/loop3 rw")
    assert parsed[0]["mountpoint"] == "/srv/my sql"
    assert not module.contains("/srv/mysql", "/srv/mysql-other")
    with pytest.raises(ValueError):
        module.clean_path("/srv/../secrets")


def test_opt_in_collector_is_networkless_readonly_and_has_no_database_credentials(tmp_path):
    module = _load_stack_script()
    config = tmp_path / "aiops-stack.ini"
    config.write_text(f"""[deployment]
deployment_id = storage-test
role = collector
local_access = false
[host]
enabled = true
target_key = host-1
exporter_bind_address = 10.0.0.88
[storage_path:mysql-data]
enabled = true
database_target_key = mysql-1
server_uuid = {SERVER_UUID}
container_id = {"a"*64}
database_path = /var/lib/mysql
host_path = /srv/mysql
""")
    config.chmod(0o600)
    settings = module._load_settings(config)
    document, _ = module._database_generated_compose(settings)
    service = document["services"]["storage-path-collector"]
    assert service["network_mode"] == "none" and service["read_only"]
    assert service["cap_drop"] == ["ALL"]
    assert not any("docker.sock" in path or "secret" in path for path in service["volumes"])
    assert not service.get("environment")
    assert "storage-path-collector" in module._selected_services(settings)
    assert "--collector.textfile.directory=/textfile" in document["services"]["node-exporter"]["command"]


@pytest.mark.parametrize("db_type", ["MYSQL", "ORACLE", "POSTGRESQL"])
def test_starter_collects_mysql_paths_and_keeps_other_engines_unchanged(db_type):
    from aiops_agent.application.investigation import TurnPlanningService
    service = object.__new__(TurnPlanningService)
    output, tools = service._starter_diagnostic_output(
        context=SimpleNamespace(question="空间变化趋势", target_context={"db_type": db_type, "display_name": "测试库"},
            conversation_starter={"starter_id": "database.storage.trend"}),
        kind="STORAGE_TREND", planning={"visualization_profile_id": "storage.trend",
            "tool_ids": ["db.instance.identity", "db.storage.capacity"]},
        parameters={"days": 7}, title="空间变化趋势",
    )
    assert ("db.storage.locations" in tools) is (db_type == "MYSQL")
    assert any(action.tool_id == "db.storage.locations" for action in output.plan.actions) is (db_type == "MYSQL")
    assert output.task_frame.action_intent == "NONE"


def test_natural_language_storage_plan_adds_authorized_path_tool_before_answer():
    from aiops_agent.application.investigation import TurnPlanningService
    service = object.__new__(TurnPlanningService)
    output, _ = service._starter_diagnostic_output(
        context=SimpleNamespace(question="预测空间增长", target_context={"db_type": "MYSQL", "display_name": "测试库"},
            conversation_starter={"starter_id": "database.storage.trend"}),
        kind="STORAGE_TREND", planning={"visualization_profile_id": "storage.trend",
            "tool_ids": ["db.instance.identity"]}, parameters={"days": 7}, title="空间变化趋势",
    )
    output = output.model_copy(update={"plan": output.plan.model_copy(update={
        "actions": tuple(action for action in output.plan.actions if action.tool_id != "db.storage.locations"),
    })})
    bound = service._bind_target_to_plan(investigation=output,
        target_context={"db_type": "MYSQL", "display_name": "测试库"},
        available_tools=({"tool_id": "db.storage.locations"}, {"tool_id": "db.instance.identity"}))
    assert any(action.tool_id == "db.storage.locations" and not action.deferred for action in bound.plan.actions)
    assert bound.plan.actions[0].tool_id == "db.instance.identity"


@pytest.mark.parametrize("change", ["device", "source_root", "missing_container", "nested_mount"])
def test_current_container_mount_mismatch_does_not_publish_configured_path(change):
    module = collector()
    mapping = {"container_id": "a"*64, "database_path": "/var/lib/mysql", "host_path": "/srv/mysql"}
    host = module.parse_mounts("1 0 7:3 / /srv/mysql rw - ext4 /dev/loop3 rw")[0]
    namespaces = {"a"*64: module.parse_mounts("1 0 7:3 / /var/lib/mysql rw - ext4 /dev/loop3 rw")}
    if change == "device":
        namespaces["a"*64][0]["device_number"] = "7:4"
    elif change == "source_root":
        namespaces["a"*64][0]["root"] = "/another-data"
    elif change == "missing_container":
        namespaces = {}
    else:
        namespaces["a"*64].extend(module.parse_mounts("2 1 7:4 / /var/lib/mysql/external rw - ext4 /dev/loop4 rw"))
    with pytest.raises(ValueError):
        module.verify_container_mount(mapping, host, namespaces)


def test_named_volume_source_root_matches_actual_host_path():
    module = collector()
    mapping = {"container_id": "a"*64, "database_path": "/var/lib/mysql",
        "host_path": "/var/lib/docker/volumes/mysql/_data"}
    host = module.parse_mounts("1 0 8:1 / / rw - ext4 /dev/sda1 rw")[0]
    namespaces = {"a"*64: module.parse_mounts("2 1 8:1 /var/lib/docker/volumes/mysql/_data /var/lib/mysql rw - ext4 /dev/sda1 rw")}
    module.verify_container_mount(mapping, host, namespaces)
