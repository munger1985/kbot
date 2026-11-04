"""Top SQL 多指标、筛选顺序、缺失值与实例隔离回归。"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiops_agent.application.conversation_starters.catalog import ConversationStarterCatalog
from aiops_agent.application.diagnosis.evidence_projection import top_sql_window_evidence_fact
from aiops_agent.application.top_sql_metrics import TopSqlOptions
from aiops_agent.application.top_sql_window import freeze_top_sql_window, validate_top_sql_window
from aiops_agent.application.workload import build_top_sql_window
from platform_core.identity import uuid7


START = datetime(2026, 11, 4, tzinfo=UTC)


def snapshot(minute, database="ORACLE", instance="db:1:pdb"):
    return SimpleNamespace(
        workload_snapshot_id=uuid7(), collected_at=START + timedelta(minutes=minute),
        database_type=database, instance_identity_json={"identity_value": instance, "database_oid": "42"},
        instance_started_at=START - timedelta(days=1), window_seconds=300,
        instance_metrics_json={"uptime_seconds": 86400 + minute * 60},
        database_version="test", capability_probe_version="probe", catalog_version="catalog",
        coverage_json={"DIGEST" if database == "MYSQL" else "STATEMENT": "AVAILABLE"},
        continuity_json={"sql_collection": {"mode": "VISIBLE_SET", "visible_set_read_complete": True}},
        status="READY",
    )


def statement(snap, identity, count, duration, **native):
    return SimpleNamespace(
        workload_snapshot_id=snap.workload_snapshot_id,
        statement_identity_type={"ORACLE": "ORACLE_SQL_ID", "MYSQL": "MYSQL_DIGEST", "POSTGRESQL": "POSTGRESQL_QUERY_ID"}[snap.database_type],
        statement_identity_value=identity,
        database_name="42" if snap.database_type == "POSTGRESQL" else "app",
        user_identifier="user", top_level=1,
        normalized_statement="SELECT * FROM orders WHERE id = 42",
        execution_count=count, total_duration_microseconds=duration,
        rows_processed=count, database_metrics_json=native, quality_status="AVAILABLE",
    )


def query(database="ORACLE", **options):
    left, right = snapshot(0, database), snapshot(5, database)
    before = [
        statement(left, key, 10, 1000000, cpu_time=1000, buffer_gets=10, disk_reads=2,
                  shared_blks_hit=10, shared_blks_read=2, temp_blks_read=0, temp_blks_written=0,
                  rows_examined=10, temp_disk_tables=0, lock_time_picoseconds=0)
        for key in ("slow", "resource")
    ]
    after = [
        statement(right, "slow", 12, 11000000, cpu_time=101000, buffer_gets=20, disk_reads=3,
                  shared_blks_hit=20, shared_blks_read=3, temp_blks_read=1, temp_blks_written=2,
                  rows_examined=20, temp_disk_tables=1, lock_time_picoseconds=1000),
        statement(right, "resource", 30, 2000000, cpu_time=9001000, buffer_gets=1010, disk_reads=102,
                  shared_blks_hit=1010, shared_blks_read=102, temp_blks_read=100, temp_blks_written=200,
                  rows_examined=1010, temp_disk_tables=100, lock_time_picoseconds=1000000000000),
    ]
    return build_top_sql_window(snapshots=[left, right], statements=before + after,
        period_start=left.collected_at, period_end=right.collected_at, limit=options.pop("limit", 10),
        postgresql_database_oid="42", **options)


@pytest.mark.parametrize("database,metric", [
    ("ORACLE", "CPU_TIME"), ("ORACLE", "LOGICAL_READS"), ("ORACLE", "DISK_READS"),
    ("ORACLE", "MEAN_CPU_TIME"), ("ORACLE", "MEAN_LOGICAL_READS"),
    ("POSTGRESQL", "SHARED_BLOCK_ACCESSES"), ("POSTGRESQL", "SHARED_BLOCK_READS"),
    ("POSTGRESQL", "TEMP_BLOCK_READS"), ("POSTGRESQL", "TEMP_BLOCK_WRITES"),
    ("MYSQL", "ROWS_EXAMINED"), ("MYSQL", "TEMP_DISK_TABLES"), ("MYSQL", "LOCK_TIME"),
])
def test_resource_rank_is_not_elapsed_rank(database, metric):
    result = query(database, rank_metric=metric, limit=1)
    assert result["top_statements"][0]["statement_identity_value"] == "resource"
    assert result["rank_metric"] == metric
    assert result["candidate_count"] == 2
    assert result["matched_count"] == 2


def test_filters_run_before_top_n_and_support_combined_thresholds():
    result = query(rank_metric="TOTAL_DURATION", limit=1, min_executions=10,
                   min_cpu_seconds="2", min_logical_reads=500, min_metric_value="0.5")
    assert [row["statement_identity_value"] for row in result["top_statements"]] == ["resource"]
    assert result["matched_count"] == 1
    assert result["query_options"]["min_cpu_seconds"] == "2"
    assert not query(rank_metric="CPU_TIME", min_metric_value="100")["top_statements"]


def test_ascending_and_exact_cpu_units():
    rows = query(rank_metric="CPU_TIME", sort_direction="ASC")["top_statements"]
    assert [row["statement_identity_value"] for row in rows] == ["slow", "resource"]
    assert Decimal(rows[0]["cpu_seconds"]) == Decimal("0.1")
    assert Decimal(rows[1]["mean_cpu_seconds"]) == Decimal("0.45")


def test_missing_and_reset_cpu_do_not_become_zero_or_invalidate_logic_reads():
    left, right = snapshot(0), snapshot(5)
    args = dict(snapshots=[left, right], period_start=left.collected_at, period_end=right.collected_at, limit=10)
    before = statement(left, "a", 1, 100, cpu_time=100, buffer_gets=100)
    after = statement(right, "a", 2, 200, cpu_time=10, buffer_gets=500)
    result = build_top_sql_window(**args, statements=[before, after], rank_metric="LOGICAL_READS")
    row = result["top_statements"][0]
    assert row["cpu_seconds"] is None
    assert Decimal(row["buffer_gets"]) == 400
    assert result["gaps"]["METRIC_COUNTER_RESET"] >= 1
    assert not build_top_sql_window(**args, statements=[before, after], rank_metric="CPU_TIME")["top_statements"]
    after.database_metrics_json.pop("cpu_time")
    missing = build_top_sql_window(**args, statements=[before, after], rank_metric="CPU_TIME")
    assert missing["gaps"]["RANK_METRIC_UNAVAILABLE"] == 1
    assert missing["gaps"]["METRIC_NOT_COLLECTED"] >= 1


def test_resource_delta_without_completed_execution_is_rankable_not_divided_by_zero():
    left, right = snapshot(0), snapshot(5)
    result = build_top_sql_window(snapshots=[left, right], statements=[
        statement(left, "running", 0, 0, cpu_time=0),
        statement(right, "running", 0, 0, cpu_time=2000000),
    ], period_start=left.collected_at, period_end=right.collected_at, limit=10, rank_metric="CPU_TIME")
    row = result["top_statements"][0]
    assert Decimal(row["cpu_seconds"]) == 2
    assert row["mean_cpu_seconds"] is None
    assert row["mean_duration_microseconds"] is None


def test_weighted_per_execution_metrics_and_partial_coverage():
    snaps = [snapshot(0), snapshot(5), snapshot(10)]
    rows = [statement(snaps[0], "a", 0, 0, cpu_time=0),
            statement(snaps[1], "a", 1, 1000000, cpu_time=1000000),
            statement(snaps[2], "a", 10, 20000000, cpu_time=20000000)]
    result = build_top_sql_window(snapshots=snaps, statements=rows,
        period_start=snaps[0].collected_at, period_end=snaps[-1].collected_at, limit=10, rank_metric="MEAN_CPU_TIME")
    assert Decimal(result["top_statements"][0]["mean_cpu_seconds"]) == 2
    rows[-1].database_metrics_json.clear()
    partial = build_top_sql_window(snapshots=snaps, statements=rows,
        period_start=snaps[0].collected_at, period_end=snaps[-1].collected_at, limit=10, rank_metric="MEAN_CPU_TIME")
    assert Decimal(partial["top_statements"][0]["mean_cpu_seconds"]) == 1
    assert partial["top_statements"][0]["metric_quality"]["MEAN_CPU_TIME"] == "PARTIAL"


def test_physical_instances_and_restart_generations_are_not_merged():
    snaps = [snapshot(0, instance="A"), snapshot(5, instance="A"),
             snapshot(10, instance="B"), snapshot(15, instance="B")]
    rows = [statement(snap, "same_sql", index % 2, (index % 2) * 1000000,
                      cpu_time=(index % 2) * 1000000) for index, snap in enumerate(snaps)]
    result = build_top_sql_window(snapshots=snaps, statements=rows,
        period_start=snaps[0].collected_at, period_end=snaps[-1].collected_at, limit=10, rank_metric="CPU_TIME")
    assert len(result["top_statements"]) == 2
    assert {row["instance_identity"]["identity_value"] for row in result["top_statements"]} == {"A", "B"}
    assert result["gaps"]["DISCONTINUITY"] == 1


@pytest.mark.parametrize("database", ["MYSQL", "POSTGRESQL"])
def test_unavailable_metrics_fail_explicitly(database):
    with pytest.raises(ValueError, match="不支持指标"):
        query(database, rank_metric="CPU_TIME")
    with pytest.raises(ValueError, match="不支持指标"):
        query(database, min_logical_reads=1)


@pytest.mark.parametrize("options", [
    {"rank_metric": "cpu;drop table"}, {"sort_direction": "random"},
    {"min_metric_value": "NaN"}, {"min_metric_value": "Infinity"},
    {"min_cpu_seconds": -1}, {"min_cpu_seconds": True},
    {"min_executions": True}, {"min_executions": 1.5}, {"min_logical_reads": True},
])
def test_thresholds_and_sort_are_structured_and_finite(options):
    with pytest.raises(ValueError):
        TopSqlOptions.model_validate(options)


def test_window_freezes_options_and_preserves_timezone():
    frozen = freeze_top_sql_window({"window": "custom", "begin_time": "2026-11-04T08:00:00+08:00",
        "end_time": "2026-11-04T09:00:00+08:00", "rank_metric": "CPU_TIME",
        "min_cpu_seconds": "1.25", "sort_direction": "ASC", "limit": 20}, now=START)
    assert frozen["period_start"] == START.isoformat()
    assert frozen["rank_metric"] == "CPU_TIME"
    assert frozen["min_cpu_seconds"] == "1.25"
    assert validate_top_sql_window(frozen) == frozen
    with pytest.raises(ValueError):
        validate_top_sql_window({**frozen, "unknown": 1})


def test_catalog_exposes_same_metric_contract_to_ui_and_language():
    catalog = ConversationStarterCatalog()
    candidate = next(item for item in catalog.resolver_candidates(supported_db_types={"ORACLE"})
                     if item["capability_id"] == "database.sql.top-current")
    schema = candidate["parameter_schema"]
    metric = next(field for field in schema if field["name"] == "rank_metric")
    cpu = next(option for option in metric["options"] if option["value"] == "CPU_TIME")
    assert cpu["supported_db_types"] == ["ORACLE"]
    assert {"sort_direction", "min_metric_value", "min_cpu_seconds", "min_logical_reads"} <= {field["name"] for field in schema}


def test_evidence_preserves_metrics_query_identity_and_top_n_truncation():
    result = query(rank_metric="CPU_TIME", limit=1, min_cpu_seconds="0.01")
    result["target_id"] = str(uuid7())
    result["collection_enabled"] = True
    fact = top_sql_window_evidence_fact(result=result, artifact_id=uuid7())
    columns = {column["name"]: index for index, column in enumerate(fact.columns)}
    assert Decimal(fact.rows[0][columns["cpu_seconds"]]) == 9
    assert fact.rows[0][columns["buffer_gets"]] == "1000"
    assert "instance_identity" in columns
    assert fact.truncated
    assert "CPU" in " ".join(fact.warnings)
    assert result["target_id"] in " ".join(fact.warnings)


def test_collection_sql_uses_bounded_multi_metric_candidates():
    base = Path(__file__).resolve().parents[3] / "services/aiops_agent/src/aiops_agent/diagnostics/catalog"
    for file in [base / "oracle/sql/workload_statements.sql", base / "mysql/sql/statement_digest_top.sql",
                 *sorted((base / "postgresql/sql").glob("statement_top_*.sql"))]:
        text = file.read_text()
        assert "LEAST(" in text and "ROW_NUMBER()" in text
        assert ":limit" in text
    oracle = (base / "oracle/sql/workload_statements.sql").read_text()
    assert all(field in oracle for field in ("cpu_time", "buffer_gets", "disk_reads"))
    assert "con_id = TO_NUMBER" in oracle

def test_catalog_normalizes_numeric_thresholds_without_accepting_booleans():
    catalog = ConversationStarterCatalog()
    frozen = catalog.freeze_natural_language(capability_id="database.sql.top-current",
        parameters={"rank_metric": "CPU_TIME", "min_executions": "3",
                    "min_logical_reads": "100", "min_cpu_seconds": "1.25"})
    assert frozen["parameters"]["min_executions"] == 3
    assert frozen["parameters"]["min_logical_reads"] == 100
    assert frozen["parameters"]["min_cpu_seconds"] == "1.25"
    precise = catalog.freeze_natural_language(capability_id="database.sql.top-current",
        parameters={"min_logical_reads": "9007199254740993", "min_cpu_seconds": "1.234567890123456789"})
    assert precise["parameters"]["min_logical_reads"] == 9007199254740993
    assert precise["parameters"]["min_cpu_seconds"] == "1.234567890123456789"
    with pytest.raises(Exception, match="必须是整数"):
        catalog.freeze_natural_language(capability_id="database.sql.top-current",
            parameters={"min_executions": True})


def test_resource_query_does_not_fallback_to_elapsed_cumulative_reference():
    from aiops_agent.application.investigation.direct_list import add_top_sql_reference
    from aiops_agent.application.investigation.service import TurnPlanningService

    context = SimpleNamespace(question="CPU Top SQL", target_context={"db_type": "ORACLE"},
        conversation_starter={"starter_id": "database.sql.top-current"})
    investigation, _ = TurnPlanningService._starter_diagnostic_output(
        object.__new__(TurnPlanningService), context=context, kind="TOP_SQL_LIST",
        planning={}, parameters={"rank_metric": "CPU_TIME", "min_cpu_seconds": "1.25"}, title="Top SQL",
    )
    result, reference = add_top_sql_reference(investigation, db_type="ORACLE",
        available_tools=({"tool_id": "db.sql.top_current"},))
    assert reference is None
    assert len(result.plan.actions) == len(investigation.plan.actions)
    assert result.plan.actions[-1].input["rank_metric"] == "CPU_TIME"
    assert result.plan.actions[-1].input["min_cpu_seconds"] == "1.25"


def test_mysql_uptime_estimate_drift_keeps_continuous_statement_baseline():
    left, right = snapshot(0, "MYSQL"), snapshot(5, "MYSQL")
    right.instance_started_at += timedelta(milliseconds=600)
    result = build_top_sql_window(snapshots=[left, right], statements=[
        statement(left, "digest", 1, 100, rows_examined=100),
        statement(right, "digest", 2, 200, rows_examined=500),
    ], period_start=left.collected_at, period_end=right.collected_at, limit=10, rank_metric="ROWS_EXAMINED")
    assert Decimal(result["top_statements"][0]["rows_examined"]) == 400


def test_source_sql_parses_in_each_native_dialect():
    import sqlglot
    base = Path(__file__).resolve().parents[3] / "services/aiops_agent/src/aiops_agent/diagnostics/catalog"
    for dialect, files in (
        ("oracle", [base / "oracle/sql/workload_statements.sql"]),
        ("mysql", [base / "mysql/sql/statement_digest_top.sql"]),
        ("postgres", sorted((base / "postgresql/sql").glob("statement_top_*.sql"))),
    ):
        for file in files:
            parsed = sqlglot.parse_one(file.read_text(), read=dialect)
            assert isinstance(parsed, sqlglot.exp.Select)
