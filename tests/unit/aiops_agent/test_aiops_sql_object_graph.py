"""SQL 对象关系必须保留物理范围、语法/运行时差异和可重放证据。"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
from aiops_agent.contracts.measurements import EvidenceProvenance
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.domain.evidence.sql_object_graph import build_sql_object_graph, explicit_relation_names, replay_sql_object_graph
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.executor import DatabaseColumn, DatabaseObservation
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 28, tzinfo=UTC).isoformat()


def fact(engine, tool, row, **updates):
    physical = {"POSTGRESQL": {"system_identifier": "7694909560417873958", "database_oid": "16510"},
        "MYSQL": {"server_uuid": "aa6b4e59-410b-4b92-aa59-b4f07bff7411"},
        "ORACLE": {"dbid": "3017315513", "container_id": "3", "instance_number": "1"}}[engine]
    physical = {**physical, "instance_started_at": NOW}
    artifact = str(uuid7())
    return TurnEvidenceFact(evidence_ref=f"artifact:{artifact}#result", artifact_id=artifact,
        source_id="db.observation", tool_id=tool, step_id="a1", trust_level="SOURCE_VERIFIED",
        measurement_semantics="CUMULATIVE_SINCE_LOAD", captured_at=NOW, presentation_kind="TABLE",
        columns=tuple({"name": key, "logical_type": "STRING"} for key in row), rows=(tuple(row.values()),),
        row_count=1, provenance=EvidenceProvenance(target_id="target", native_origin_ids=("native",),
            physical_identity=tuple(sorted(physical.items())), source_version="catalog.v1", content_hash="a" * 64),
    ).model_copy(update=updates)


def scenario(engine="POSTGRESQL"):
    identity = {"POSTGRESQL": {"query_id": "1285382359565720997", "database_oid": "16510", "user_oid": "42", "top_level": True},
                "MYSQL": {"digest": "a" * 64, "schema_name": "mes"},
                "ORACLE": {"sql_id": "36apy6dncjggq", "inst_id": "1", "child_number": "0", "plan_hash_value": "44"}}[engine]
    schema, table = ("MES", "ORDERS") if engine == "ORACLE" else ("mes", "orders")
    tool, field = {"POSTGRESQL": ("db.postgresql.statement.details", "normalized_statement"),
        "MYSQL": ("db.sql.cursor_details", "digest_text"), "ORACLE": ("db.sql.cursor_details", "sql_text")}[engine]
    text = fact(engine, tool, {**identity, field: f"SELECT * FROM {schema}.{table} WHERE id = 1"})
    indexes = fact(engine, "db.sql.object_indexes", {"schema_name": schema, "table_name": table,
        "relation_name": table, "index_name": "orders_id_idx", "relation_kind": "p"})
    return identity, text, indexes


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_three_engines_connect_explicit_sql_name_and_catalog_index_without_claiming_usage(engine):
    subject, text, indexes = scenario(engine)
    graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=(text, indexes))
    assert {edge.kind for edge in graph.relationships} == {"SQL_TEXT_REFERENCES_OBJECT", "CATALOG_INDEX_ON"}
    assert graph.relationships[0].anchor.evidence_ref == text.evidence_ref
    assert graph.relationships[0].related.evidence_ref == indexes.evidence_ref
    assert "不证明" in graph.relationships[0].limitation
    replay_sql_object_graph(graph, (text, indexes))


@pytest.mark.parametrize("sql,engine,expected", [
    ('SELECT * FROM MES.ORDERS', "POSTGRESQL", (("mes", "orders"),)),
    ('SELECT * FROM "MES"."Orders"', "POSTGRESQL", (("MES", "Orders"),)),
    ('SELECT * FROM mes.orders', "ORACLE", (("MES", "ORDERS"),)),
    ('SELECT * FROM "mes"."Orders"', "ORACLE", (("mes", "Orders"),)),
    ('SELECT * FROM `mes`.`Orders`', "MYSQL", (("mes", "Orders"),)),
    ('SELECT * FROM orders', "POSTGRESQL", ()),
    ('SELECT * FROM mes.orders ...', "POSTGRESQL", ()),
    ('SELECT * FROM mes.orders; SELECT * FROM other.secret', "POSTGRESQL", ()),
    ("SELECT 'mes.secret' FROM mes.orders -- FROM other.secret", "POSTGRESQL", (("mes", "orders"),)),
    ('WITH orders AS (SELECT * FROM mes.orders) SELECT * FROM orders', "POSTGRESQL", (("mes", "orders"),)),
    ('SELECT * FROM db.mes.orders', "POSTGRESQL", ()),
])
def test_schema_qualification_quotes_ctes_comments_and_truncation_are_not_guessed(sql, engine, expected):
    assert explicit_relation_names(sql, engine) == expected


@pytest.mark.parametrize("case", ["unknown_identity", "different_database", "different_cluster", "old_time", "naive_time", "historical", "user_material", "different_boot", "unqualified_sql", "case_mismatch"])
def test_unknown_or_incompatible_scope_does_not_connect_catalog_to_sql(case):
    subject, text, indexes = scenario()
    if case in {"unknown_identity", "different_database", "different_cluster", "different_boot"}:
        physical = dict(indexes.provenance.physical_identity)
        key = {"different_database": "database_oid", "different_cluster": "system_identifier", "different_boot": "instance_started_at"}.get(case)
        if key:
            physical[key] = "other"
        indexes = indexes.model_copy(update={"provenance": None if case == "unknown_identity" else indexes.provenance.model_copy(update={"physical_identity": tuple(physical.items())})})
    elif case in {"old_time", "naive_time"}:
        indexes = indexes.model_copy(update={"captured_at": "2026-10-28T00:06:00+00:00" if case == "old_time" else "2026-10-28T00:00:00"})
    elif case == "historical":
        text = text.model_copy(update={"source_id": "prior_turn.sql_observation"})
    elif case == "user_material":
        indexes = indexes.model_copy(update={"trust_level": "USER_PROVIDED"})
    elif case == "unqualified_sql":
        text = text.model_copy(update={"rows": ((subject["query_id"], "16510", "42", True, "SELECT * FROM orders"),)})
    else:
        indexes = indexes.model_copy(update={"rows": (("MES", "Orders", "Orders", "orders_id_idx", "p"),)})
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text, indexes))
    assert not graph.relationships


def test_other_database_statistics_cannot_borrow_connection_database_identity():
    subject, text, indexes = scenario()
    subject = {**subject, "database_oid": "99999"}
    text = text.model_copy(update={"rows": ((subject["query_id"], "99999", "42", True, "SELECT * FROM mes.orders"),)})
    assert not build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text, indexes)).relationships


def test_partition_descendants_are_not_called_direct_children_and_other_tables_are_excluded():
    subject, text, indexes = scenario()
    row = indexes.rows[0]
    indexes = indexes.model_copy(update={"rows": (row,
        ("archive", "orders", "orders_leaf", "leaf_idx", "r"),
        ("mes", "unrelated", "unrelated", "unrelated_idx", "r")), "row_count": 3})
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text, indexes))
    kinds = [edge.kind for edge in graph.relationships]
    assert "CATALOG_DESCENDANT_OF" in kinds
    assert len([edge for edge in graph.relationships if edge.kind == "CATALOG_INDEX_ON"]) == 2
    assert not any(edge.relation_name == "unrelated" for edge in graph.relationships)
    descendant = next(edge for edge in graph.relationships if edge.kind == "CATALOG_DESCENDANT_OF")
    assert descendant.schema_name == "archive" and "不保证直接父子关系" in descendant.limitation


def test_oracle_matching_child_and_plan_preserved_but_representative_bindings_remain_unknown():
    subject, text, indexes = scenario("ORACLE")
    plan = fact("ORACLE", "db.sql.execution_plan", {**subject,
        "object_owner": "MES", "object_name": "ORDERS", "operation": "TABLE ACCESS"})
    graph = build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=(text, indexes, plan))
    assert "PLAN_ROW_REFERENCES_OBJECT" in {edge.kind for edge in graph.relationships}
    mismatch = plan.model_copy(update={"rows": (tuple({**subject, "plan_hash_value": "55", "object_owner": "MES", "object_name": "ORDERS", "operation": "TABLE ACCESS"}.values()),)})
    graph = build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=(text, indexes, mismatch))
    assert "PLAN_ROW_REFERENCES_OBJECT" not in {edge.kind for edge in graph.relationships}


def test_oracle_root_gvsql_without_container_cannot_be_assumed_to_belong_to_current_container():
    subject, text, indexes = scenario("ORACLE")
    changed = []
    for item in (text, indexes):
        physical = {**dict(item.provenance.physical_identity), "container_id": "1"}
        changed.append(item.model_copy(update={"provenance": item.provenance.model_copy(update={"physical_identity": tuple(physical.items())})}))
    assert not build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=tuple(changed)).relationships


def test_sql_bundle_does_not_choose_one_user_from_ambiguous_statistics():
    subject, text, indexes = scenario()
    text = text.model_copy(update={"rows": (text.rows[0], (subject["query_id"], "16510", "43", True, "SELECT * FROM mes.orders")), "row_count": 2})
    bundle = build_sql_evidence_bundle(task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": {"query_id": subject["query_id"]}},
        database_type="POSTGRESQL", evidence=(text, indexes))
    assert bundle["subject_resolution"]["status"] == "COMPARISON" and bundle["object_graph"] is None
    assert len(bundle["subjects"]) == 2
    assert all(item["object_graph"] is not None for item in bundle["subjects"])


def test_answer_freezes_graph_and_report_replays_original_rows_instead_of_trusting_graph_description():
    subject, text, indexes = scenario()
    evidence = (text, indexes)
    draft = DiagnosisAnswerDraft.model_validate({"claims": [{"claim_id": "c1", "kind": "FACT",
        "subject": "指定 SQL", "statement": "语句文本引用 mes.orders", "level": "OBSERVED", "applicable_time": NOW,
        "evidence_fields": [{"evidence_ref": text.evidence_ref, "row_index": 0, "field": "normalized_statement"}],
        "reasoning": "直接读取规范化文本"}], "goal_satisfied": True})
    context = SimpleNamespace(run_id=uuid7(), task_id=uuid7(), target_id=uuid7(), plan_snapshot={
        "capability_snapshot": {"database_type": "POSTGRESQL", "database_version": "16"},
        "answer_context": {"question": "深入分析", "task_frame": {"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject}}})
    record = freeze_investigation(draft=draft, evidence=evidence, context=context)
    assert record.sql_object_graph and record.sql_object_graph.relationships
    payload = {"status": "COMPLETED", "blocks": [], "evidence": [item.model_dump(mode="json") for item in evidence],
        "investigation_record": record.model_dump(mode="json")}
    source = normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")
    assert source["investigation_record"]["sql_object_graph"] == payload["investigation_record"]["sql_object_graph"]
    payload["investigation_record"]["sql_object_graph"]["relationships"][0]["relation_name"] = "invented"
    with pytest.raises(ValueError, match="对象关系"):
        normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")


def test_graph_limits_rows_and_relationships_without_hiding_truncation():
    subject, text, indexes = scenario()
    rows = tuple(("mes", "orders", "orders", f"idx_{index}", "p") for index in range(1000))
    indexes = indexes.model_copy(update={"rows": rows, "row_count": len(rows)})
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text, indexes))
    assert graph.truncated and len(graph.relationships) <= 128
    replay_sql_object_graph(graph, (text, indexes))


@pytest.mark.parametrize("tamper", ["none", "missing", "hash", "run", "trust", "rows", "identity", "target", "step", "duplicate"])
def test_report_and_case_gate_reopens_original_tool_artifacts_not_just_self_consistent_rows(tamper):
    subject, text, indexes = scenario()
    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
    artifacts, evidence = {}, []
    for item in (text, indexes):
        observation = DatabaseObservation(executor_request_id=uuid7(), target_id=run.target_id,
            tool_id=item.tool_id, tool_version="1.0.0", variant="pg-test", template_sha256="a" * 64,
            db_type="POSTGRESQL", db_version="16", capability_snapshot_hash="b" * 64,
            captured_at=NOW, duration_ms=1,
            columns=tuple(DatabaseColumn(**column, sensitivity="PUBLIC") for column in item.columns),
            rows=item.rows, row_count=item.row_count, truncated=False,
            result_sha256=canonical_hash(item.rows), parameters_sha256="c" * 64,
            provenance={"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": dict(item.provenance.physical_identity)})
        value = item.model_copy(update={"evidence_ref": f"artifact:{item.artifact_id}#a1",
            "columns": tuple(column.model_dump(mode="json") for column in observation.columns),
            "provenance": database_provenance(observation)})
        evidence.append(value)
        body = {"schema_version": "DBA_TOOL_RESULT.v1", "source_type": "TOOL", "source_id": value.source_id,
            "source_version": "1.0.0", "definition_hash": "a" * 64, "output_schema": "DBA_TOOL_RESULT.v1",
            "measurement_semantics": "CUMULATIVE_SINCE_LOAD", "presentation_kind": "TABLE", "status": "SUCCEEDED",
            "tool_outcomes": [{"step_id": "a1", "tool_id": value.tool_id, "tool_version": "1.0.0",
                               "status": "SUCCEEDED", "observation": observation.model_dump(mode="json")}], "sampling": None}
        artifacts[str(value.artifact_id)] = SimpleNamespace(ops_run_id=run.ops_run_id, schema_version="DBA_TOOL_RESULT.v1",
            trust_level="SOURCE_VERIFIED", payload_json=body, content_hash=canonical_hash(body))
    first = evidence[0]
    draft = DiagnosisAnswerDraft.model_validate({"claims": [{"claim_id": "c1", "kind": "FACT", "subject": "SQL",
        "statement": "引用 mes.orders", "level": "OBSERVED", "applicable_time": NOW,
        "evidence_fields": [{"evidence_ref": first.evidence_ref, "row_index": 0, "field": "normalized_statement"}],
        "reasoning": "原始语句文本"}], "goal_satisfied": True})
    context = SimpleNamespace(run_id=run.ops_run_id, target_id=run.target_id, task_id=uuid7(), plan_snapshot={
        "capability_snapshot": {"database_type": "POSTGRESQL"},
        "answer_context": {"task_frame": {"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject}}})
    record = freeze_investigation(draft=draft, evidence=tuple(evidence), context=context)
    assert record.sql_object_graph.relationships
    original = artifacts[str(first.artifact_id)]
    outcome = original.payload_json["tool_outcomes"][0]
    if tamper == "missing":
        artifacts.pop(str(first.artifact_id))
    elif tamper == "hash":
        original.content_hash = "bad"
    elif tamper == "run":
        original.ops_run_id = uuid7()
    elif tamper == "trust":
        original.trust_level = "USER_PROVIDED"
    elif tamper == "rows":
        outcome["observation"]["rows"][0][-1] = "SELECT * FROM mes.unrelated"
    elif tamper == "identity":
        outcome["observation"]["provenance"]["instance_identity"]["system_identifier"] = "other"
    elif tamper == "target":
        outcome["observation"]["target_id"] = str(uuid7())
    elif tamper == "step":
        outcome["step_id"] = "other"
    elif tamper == "duplicate":
        original.payload_json["tool_outcomes"].append(dict(outcome))
    if tamper not in {"none", "hash", "missing"}:
        original.content_hash = canonical_hash(original.payload_json)
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifacts.get(str(artifact_id)))))
    payload = {"investigation_record": record.model_dump(mode="json"), "evidence": [item.model_dump(mode="json") for item in evidence]}
    assert asyncio.run(validate_investigation_sources(uow=uow, run=run, payload=payload)) == (tamper == "none")
