"""三库参数重放的精确类型、词法隔离和安全投影；不连接目标数据库。"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json

import pytest
from pydantic import ValidationError

from aiops_agent.diagnostics.grants import DiagnosticGrantCodec, DiagnosticGrantError
from aiops_agent.domain.evidence.sql_plan_replay import compile_sql_plan_replay, SqlPlanReplayRejected
from aiops_agent.executor.sql_plan_native import check_context, check_identity, project_plan
from platform_core.contracts.aiops.sql_plan_execution import (
    SqlPlanCaptureGrant, SqlPlanCaptureExchange, SqlPlanCaptureResult, SqlPlanEstimateNode,
)
from platform_core.identity import uuid7


PRIVATE = "private-value-must-not-appear-in-plan-results"


def material(*parameters, engine="POSTGRESQL"):
    return {"schema_version": "SQL_PLAN_PARAMETER_MATERIAL.v1",
        "parameters": [{"position": index, "database_type_name": kind, "value": value}
            for index, (kind, value) in enumerate(parameters, start=1)],
        "execution_context": {"database_name": "app", "schema_name": "mes", "role_name": "reader"}}


def compile_one(kind, value, engine="POSTGRESQL"):
    sql = {"POSTGRESQL": "SELECT $1 FROM mes.t", "MYSQL": "SELECT ? FROM mes.t",
        "ORACLE": "SELECT :1 FROM MES.T"}[engine]
    return compile_sql_plan_replay(database_type=engine, sql=sql, document=material((kind, value)),
        allowed_objects=("mes.t",))


@pytest.mark.parametrize("kind,canonical,value,expected", [
    ("pg_catalog.int8", "int8", "9223372036854775807", 9223372036854775807),
    ("INTEGER", "int4", "42", 42), ("smallint", "int2", "-32768", -32768),
    ("bigint", "int8", "42", 42), ("pg_catalog.boolean", "bool", True, True),
    ("character varying", "varchar", PRIVATE, PRIVATE), ("character", "bpchar", PRIVATE, PRIVATE),
    ("decimal", "numeric", "12345678901234567890.123456789", Decimal("12345678901234567890.123456789")),
    ("timestamp with time zone", "timestamptz", "2026-10-28T12:00:00+00:00", datetime(2026, 10, 28, 12, tzinfo=UTC)),
    ("timestamp without time zone", "timestamp", "2026-10-28T12:00:00", datetime(2026, 10, 28, 12)),
    ("integer", "int4", None, None),
])
def test_postgresql_aliases_keep_exact_builtin_types_and_values(kind, canonical, value, expected):
    replay = compile_one(kind, value)
    assert replay.native_types == (canonical,) and replay.parameters == (expected,)
    assert replay.sql == "SELECT $1 FROM mes.t"


@pytest.mark.parametrize("engine,kind,value", [
    ("POSTGRESQL", "custom_domain", None), ("POSTGRESQL", "mes.int4", None),
    ("POSTGRESQL", "int4[]", None), ("POSTGRESQL", "varchar(10)", None),
    ("MYSQL", "custom", None), ("MYSQL", "int unsigned", None), ("MYSQL", "timestamptz", None),
    ("ORACLE", "boolean", None), ("ORACLE", "int8", None), ("ORACLE", "time", None),
    ("POSTGRESQL", "numeric", 12), ("POSTGRESQL", "int4", True),
    ("POSTGRESQL", "int2", "32768"), ("POSTGRESQL", "int4", "2147483648"),
    ("MYSQL", "tinyint", "128"), ("MYSQL", "mediumint", "8388608"),
    ("POSTGRESQL", "numeric", "1e1001"), ("POSTGRESQL", "numeric", "NaN"),
    ("POSTGRESQL", "timestamptz", "2026-10-28T12:00:00"),
    ("POSTGRESQL", "timestamp", "2026-10-28T12:00:00+00:00"),
    ("MYSQL", "time", "12:00:00+00:00"), ("ORACLE", "timestamp", "not-" + PRIVATE),
    ("ORACLE", "date", "2026-10-28T12:00:00+08:00"),
])
def test_unsupported_types_null_types_ranges_and_implicit_conversion_are_rejected_without_values(engine, kind, value):
    with pytest.raises(SqlPlanReplayRejected) as caught:
        compile_one(kind, value, engine)
    assert str(caught.value).startswith("PLAN_") and PRIVATE not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("engine,sql,expected", [
    ("POSTGRESQL", "WITH x AS (SELECT id FROM MES.T WHERE n=$1) SELECT * FROM x", None),
    ("MYSQL", "SELECT '?' FROM mes.t WHERE n=? /* ? */", "SELECT '?' FROM mes.t WHERE n=:plan_bind_1 /* ? */"),
    ("ORACLE", "SELECT ':1' FROM MES.T WHERE N=:1 OR N=:1", "SELECT ':1' FROM MES.T WHERE N=:plan_bind_1 OR N=:plan_bind_1"),
])
def test_native_bindings_are_not_interpolated_and_cte_names_are_not_base_objects(engine, sql, expected):
    kind = "varchar2" if engine == "ORACLE" else "text"
    replay = compile_sql_plan_replay(database_type=engine, sql=sql, document=material((kind, PRIVATE)), allowed_objects=("mes.t",))
    assert replay.sql == (expected or sql) and PRIVATE not in replay.sql
    assert replay.parameters == (PRIVATE,)
    assert PRIVATE not in repr(replay)


@pytest.mark.parametrize("engine,sql", [
    ("POSTGRESQL", "SELECT pg_sleep(1) FROM mes.t"),
    ("POSTGRESQL", "SELECT mes.abs(1) FROM mes.t"),
    ("POSTGRESQL", "SELECT nextval('x') FROM mes.t"),
    ("POSTGRESQL", "SELECT * FROM mes.denied"),
    ("POSTGRESQL", "SELECT * FROM mes.t JOIN mes.denied USING (id)"),
    ("POSTGRESQL", "SELECT * FROM t"),
    ("POSTGRESQL", 'SELECT * FROM "MES"."T"'),
    ("POSTGRESQL", "SELECT 'x'::mes.private_type FROM mes.t"),
    ("ORACLE", 'SELECT * FROM "mes"."t"'),
    ("ORACLE", "SELECT * FROM MES.T@remote"),
    ("MYSQL", "SELECT SLEEP(1) FROM mes.t"),
    ("MYSQL", "SELECT * FROM other.mes.t"),
])
def test_undeclared_objects_custom_functions_types_and_remote_sources_are_not_explained(engine, sql):
    with pytest.raises(SqlPlanReplayRejected):
        compile_sql_plan_replay(database_type=engine, sql=sql, document=material(), allowed_objects=("mes.t",))


def exchange(engine="POSTGRESQL", **updates):
    now = datetime.now(UTC)
    return SqlPlanCaptureExchange(database_type=engine,
        connection_profile={"host": "db.internal", "port": 5432, "database": "app", "service": "app", "tls_enabled": False},
        username="reader", password=PRIVATE, source_sql="SELECT 1 FROM mes.t",
        protected_document=json.dumps(material()), allowed_objects=("mes.t",), statement_timeout_seconds=2,
        source_subject={}, resource_lease={"lease_id": uuid7(), "release_token": "x" * 48,
            "execution_deadline_at": now + timedelta(seconds=20), "held_until": now + timedelta(seconds=30)}, **updates)


def result(**updates):
    payload = dict(capture_run_id=uuid7(), request_id=uuid7(), database_type="POSTGRESQL",
        status="SUCCEEDED", captured_at=datetime.now(UTC), nodes=[{"node_id": 0, "operation": "Seq Scan"}])
    return SqlPlanCaptureResult(**{**payload, **updates})


def test_exchange_only_dedicated_transport_reveals_protected_document():
    item = exchange()
    assert PRIVATE not in repr(item) and PRIVATE not in item.model_dump_json()
    assert item.protected_transport()["password"] == PRIVATE
    assert item.protected_transport()["protected_document"] == json.dumps(material())


@pytest.mark.parametrize("identity", [{"password": PRIVATE}, {"server_uuid": PRIVATE},
    {"database_oid": "0"}, {"database_oid": "1\n"}, {"system_identifier": "-1"},
    {"instance_started_at": "not-a-time"}, {"instance_started_at": "x" * 65}])
def test_unregistered_or_malformed_identity_cannot_reach_plan_evidence(identity):
    with pytest.raises(ValidationError):
        result(physical_identity=identity)


@pytest.mark.parametrize("updates", [
    {"production_execution_verified": True}, {"original_parse_context_verified": True},
    {"representative_parameters": "VERIFIED"}, {"oracle_parameter_peeking": "NOT_PROVEN"},
    {"nodes": []}, {"error_code": "FAILED"},
    {"nodes": [{"node_id": 0, "operation": "Seq Scan"}, {"node_id": 1, "operation": "Seq Scan"}]},
    {"nodes": [{"node_id": 0, "operation": "Seq Scan", "metrics": {"actual_rows": "1"}}]},
    {"nodes": [{"node_id": 0, "operation": "Seq Scan", "metrics": {"estimated_rows": "1e-99999"}}]},
])
def test_estimates_cannot_become_actual_production_or_forged_representative_plans(updates):
    with pytest.raises(ValidationError):
        result(**updates)


def test_context_and_physical_identity_checks_do_not_guess_or_change_role():
    item = material()
    check_context(item, database="app", schema="mes", role="reader")
    with pytest.raises(SqlPlanReplayRejected, match="CONTEXT_MISMATCH"):
        check_context(item, database="app", schema="mes", role="owner")
    item["execution_context"]["search_path"] = ["mes"]
    with pytest.raises(SqlPlanReplayRejected, match="SEARCH_PATH_MISMATCH"):
        check_context(item, database="app", schema="mes", role="reader", search_path=["public"])
    assert check_identity({}, ()) == {}
    with pytest.raises(SqlPlanReplayRejected, match="PHYSICAL_IDENTITY_MISMATCH"):
        check_identity({"database_oid": "123"}, (("database_oid", "456"),))


@pytest.mark.parametrize("engine,raw", [
    ("POSTGRESQL", [{"Plan": {"Node Type": "Seq Scan", "Relation Name": "t", "Schema": "mes",
        "Plan Rows": 2, "Total Cost": 3, "Filter": PRIVATE}}]),
    ("MYSQL", {"query_block": {"table": {"table_name": "t", "access_type": "ALL",
        "rows_examined_per_scan": 2, "attached_condition": PRIVATE}}}),
    ("ORACLE", [{"id": 0, "parent_id": None, "operation": "TABLE ACCESS FULL",
        "object_owner": "MES", "object_name": "T", "cardinality": 2, "filter_predicates": PRIVATE}]),
])
def test_three_engine_native_projection_discards_conditions_bind_values_and_extra_fields(engine, raw):
    nodes = project_plan(raw, engine)
    assert nodes and PRIVATE not in json.dumps(nodes)
    for node in nodes:
        SqlPlanEstimateNode.model_validate(node)


def test_grant_has_separate_signed_type_and_is_rejected_by_both_select_channels():
    now = datetime.now(UTC)
    grant = SqlPlanCaptureGrant(issuer="control", audience="executor", grant_id=uuid7(), issued_at=now,
        expires_at=now + timedelta(seconds=30), run_id=uuid7(), task_id=uuid7(), source_run_id=uuid7(),
        request_id=uuid7(), approval_sha256="a" * 64, lease_token_hash="b" * 64,
        db_type="POSTGRESQL", trace_id="capture-test")
    codec = DiagnosticGrantCodec(secret="test-only-capture-key-" * 3, issuer="control", audience="executor")
    token = codec.issue_plan_capture(grant)
    assert codec.verify_plan_capture(token, now=now) == grant
    for verify in (codec.verify, codec.verify_dynamic):
        with pytest.raises(DiagnosticGrantError):
            verify(token, now=now)
    with pytest.raises(ValidationError):
        SqlPlanCaptureGrant.model_validate({**grant.model_dump(), "expires_at": now + timedelta(seconds=46)})
