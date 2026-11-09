"""三库通用快捷入口的准入、主体、取证与交付合同回归。"""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.conversation_starters import ConversationStarterCatalog
from aiops_agent.application.conversation_starters.common import COMMON_BASELINES, COMMON_CONSTRAINTS
from aiops_agent.application.conversation_starters.sql_input import sql_starter_subject
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.investigation.discovery import available_tools, available_playbooks, rewrite_incomplete_discovery_actions
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import ToolExecutionSnapshotBuilder
from aiops_agent.tools.capabilities import build_capability_snapshot
from platform_core.contracts.aiops import ConversationStarterSelection
from platform_core.identity import uuid7

ENGINES = ("ORACLE", "POSTGRESQL", "MYSQL")
SUBJECTS = {
    "ORACLE": {"sql_id": "49fbb8nm4fkr2"},
    "POSTGRESQL": {"query_id": "-6589067333046971749"},
    "MYSQL": {"digest": "A" * 64, "schema_name": "test_schema"},
}


def target(engine):
    return SimpleNamespace(target_id=uuid7(), db_type=engine, version_code={"ORACLE": "19c", "POSTGRESQL": "16.15", "MYSQL": "8.4"}[engine],
        display_name="合同测试", status="ENABLED", connectivity_status="CONNECTED", readonly_connection_enabled=True,
        diagnostic_credential_id=uuid7(), endpoint_json={"host": "test"}, capabilities_json={"capabilities": ["DB_READONLY"]})


def freeze(catalog, instance, starter_id, parameters=None):
    options = {"capabilities": {"ACTIVE_DIAGNOSIS"}} if "capabilities" in inspect.signature(catalog.freeze).parameters else {}
    return catalog.freeze(target=instance, selection=ConversationStarterSelection(
        starter_id=starter_id, catalog_version=catalog.version, parameters=parameters or {}), **options)


@pytest.mark.parametrize("engine", ENGINES)
def test_common_entries_exposed_and_sql_identity_frozen(engine):
    catalog, instance = ConversationStarterCatalog(), target(engine)
    items = catalog.list_for_agent(agent_id=uuid7(), capabilities={"ACTIVE_DIAGNOSIS"}, targets=(instance,))["starters"]
    assert len(items) == (26 if engine == "ORACLE" else 10)
    ids = {item["starter_id"] for item in items}
    assert {"database.sql.performance", "database.parameters.review"} <= ids
    assert ("oracle.report.awr" in ids) == (engine == "ORACLE")
    frozen = freeze(catalog, instance, "database.sql.performance", SUBJECTS[engine])
    assert sql_starter_subject(frozen["parameters"], engine)
    assert freeze(catalog, instance, "database.parameters.review")["planning"]["tool_ids"] == ["db.instance.identity", "db.instance.parameters"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("other", ENGINES)
def test_cross_engine_sql_subjects_rejected(engine, other):
    if engine == other:
        return
    with pytest.raises(Exception):
        freeze(ConversationStarterCatalog(), target(engine), "database.sql.performance", SUBJECTS[other])


@pytest.mark.parametrize("parameters", [
    {"query_id": str(2**63)}, {"query_id": "1", "database_oid": True},
    {"query_id": "1", "database_oid": 2**32}, {"query_id": "1", "top_level": "unknown"},
    {"digest": "A" * 63}, {"digest": "A" * 64, "schema_name": "a" * 65},
    {"sql_id": "49fbb8nm4fkr2", "query_id": "1"},
])
def test_invalid_sql_subject_rejected(parameters):
    with pytest.raises(Exception):
        ConversationStarterCatalog().freeze_natural_language(capability_id="database.sql.performance", parameters=parameters)


def test_pg_partial_identity_is_discovered_not_guessed():
    service = object.__new__(TurnPlanningService)
    for raw, expected in (
        ({"query_id": "1", "database_oid": 100}, "db.postgresql.statement.subjects"),
        ({"query_id": "1", "database_oid": 100, "top_level": "false"}, "db.postgresql.statement.subjects"),
        ({"query_id": "1", "database_oid": 100, "user_oid": 10, "top_level": "false"}, "db.postgresql.statement.details"),
    ):
        from platform_core.contracts.aiops import CompactPlanningOutput
        subject = sql_starter_subject(raw, "POSTGRESQL")
        compact = CompactPlanningOutput(problem_statement="单 SQL", subject_ref=subject,
            planning_mode="READ_ONLY_LOOKUP", objectives=("DIAGNOSE",), action_intent="NONE",
            diagnostic_profile="SINGLE_SQL_PERFORMANCE", success_criteria=("回答单 SQL",), public_reasoning_summary="按主体取证")
        output = service._single_sql_investigation_output(question="单 SQL", compact=compact,
            target_context={"db_type": "POSTGRESQL"}, subject=subject)
        assert output.plan.actions[0].tool_id == expected
        assert output.task_frame.subject_ref == subject
        if expected.endswith(".details"):
            assert output.plan.actions[0].input["top_level"] is False


@pytest.mark.parametrize("engine", ENGINES)
def test_all_visible_starters_compile_and_validate_offline(engine):
    import asyncio
    async def run():
        catalog, instance = ConversationStarterCatalog(), target(engine)
        registry, books = DiagnosticRegistry.load(), PlaybookRegistry.load()
        builder = ToolExecutionSnapshotBuilder(playbook_registry=books, diagnostic_registry=registry)
        caps = build_capability_snapshot(agent_id=uuid7(), agent_version=SimpleNamespace(agent_version_id=uuid7()), target=instance, sources=())
        tools, playbooks = available_tools(builder, caps), available_playbooks(books, caps)
        service = object.__new__(TurnPlanningService)
        service._tool_snapshot_builder = builder
        service._record_planning_route = AsyncMock()
        entries = catalog.list_for_agent(agent_id=uuid7(), capabilities={"ACTIVE_DIAGNOSIS"}, targets=(instance,))["starters"]
        for item in entries:
            parameters = {}
            for field in item["input_schema"]:
                if not field.get("required") or "default" in field or engine not in field.get("supported_db_types", (engine,)):
                    continue
                name = field["name"]
                parameters[name] = SUBJECTS[engine].get(name, "2026-11-09T01:00:00+00:00" if "begin" in name else "2026-11-09T02:00:00+00:00")
            frozen = freeze(catalog, instance, item["starter_id"], parameters)
            context = SimpleNamespace(question=frozen["user_message"], conversation_starter=frozen,
                target_context={"db_type": engine, "version_code": instance.version_code, "display_name": instance.display_name, "target_id": str(instance.target_id)},
                capabilities=caps)
            output, chosen, _, _ = await service._plan_conversation_starter(context=context, available_tools=tools, available_playbooks=playbooks)
            output = service._bind_target_to_plan(investigation=service._apply_default_temporal_windows(output),
                target_context=context.target_context, available_tools=chosen)
            output = rewrite_incomplete_discovery_actions(investigation=output, available_tools=chosen) or output
            for action in output.plan.actions:
                if not action.deferred:
                    builder.validate_direct_actions(actions=(action,), capabilities=caps)
            kind = frozen["planning"]["kind"]
            if kind in COMMON_BASELINES:
                assert tuple(action.tool_id for action in output.plan.actions) == COMMON_BASELINES[kind][engine]
                assert all(rule in output.task_frame.constraints for rule in COMMON_CONSTRAINTS)
            if kind == "CURRENT_PERFORMANCE":
                wait = next(action for action in output.plan.actions if action.tool_id in {"db.wait.class_summary", "db.postgresql.wait.current"})
                assert wait.measurement_semantics == ("CURRENT_ACTIVITY" if engine == "POSTGRESQL" else "CUMULATIVE_SINCE_LOAD")
            if kind == "REPLICATION":
                assert output.task_frame.visualization_profile_id == "replication.status"
                assert output.task_frame.visualization_window_seconds == 3600
    asyncio.run(run())


def test_collector_scope_and_replication_semantics():
    import aiops_agent
    root = Path(aiops_agent.__file__).parent / "diagnostics" / "catalog"
    assert "CONNECTION_ID()" in (root / "mysql/sql/session_active.sql").read_text()
    assert "DATABASE()" in (root / "mysql/sql/session_active.sql").read_text()
    assert "SYS_CONTEXT('USERENV', 'SID')" in (root / "oracle/sql/session_active.sql").read_text()
    assert "SYS_CONTEXT('USERENV', 'CON_ID')" in (root / "oracle/sql/session_active.sql").read_text()
    assert "current_database()" in (root / "postgresql/sql/session_active.sql").read_text()
    mysql = (root / "mysql/sql/replication_lag.sql").read_text()
    assert "immediate_commit_timestamp" not in mysql and " AS lag_seconds" not in mysql
    assert "0000-00-00" not in mysql  # 严格日期模式在无通道时也会拒绝零日期常量。
    assert "latest_applied_transaction_latency_seconds" in mysql
    assert "replication_applier_status_by_worker" in mysql
    replay = (root / "postgresql/sql/replication_replay.sql").read_text()
    assert "pg_is_in_recovery()" in replay and "receive_replay_gap_bytes" in replay
    assert "lag_seconds" not in replay
    slots = (root / "postgresql/sql/replication_slot_retention.sql").read_text()
    assert "pg_is_in_recovery()" in slots


def test_common_chart_capacity_slots_remain_distinct():
    import aiops_agent
    path = Path(aiops_agent.__file__).parent / "application/visualization/common_profiles.v1.json"
    profiles = {item["profile_id"]: item for item in json.loads(path.read_text())["profiles"]}
    assert profiles["storage.trend"]["max_charts"] == 4
    assert [slot["slot_id"] for slot in profiles["storage.trend"]["slots"]] == ["storage_used", "storage_utilization", "storage_free", "storage_max"]
