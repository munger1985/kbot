"""开放证据诊断、结构化选行和 MySQL 受控查询的回归；不替代真实数据库验收。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.diagnosis.investigation_record import render_investigation, sql_investigation_draft
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle, exact_wire, validate_sql_diagnosis
from aiops_agent.application.sql_selection import selection_ref, decode_selection, resolve_sql_selection, subject_from_row
from aiops_agent.playbooks import canonical_hash
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.contracts.sql_diagnosis import SqlDiagnosisDraft
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceFact
from aiops_agent.diagnostics.dynamic_query import MySQLDynamicQueryPolicySnapshot, DynamicQueryRejected
from aiops_agent.diagnostics.mysql_dynamic_query import MySQLDynamicQueryPolicy
from aiops_agent.executor.drivers.mysql import _mysql_bound_sql
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.contracts.aiops.conversation import TurnCreate
from platform_core.identity import uuid7
from tests.support.aiops_sql_evidence import backed_sql_fact


def observation(names=("query_id", "database_oid", "execution_count", "pid"), rows=(("1285382359565720997", 16510, 100, 42),), **overrides):
    artifact_id = str(uuid7())
    return TurnEvidenceFact(
        evidence_ref=f"artifact:{artifact_id}#row", artifact_id=artifact_id,
        source_id="db.observation", tool_id="db.postgresql.statement.details", step_id="a1",
        trust_level="SOURCE_VERIFIED", measurement_semantics="CURRENT_ACTIVITY",
        captured_at="2026-10-28T01:00:00+00:00", presentation_kind="TABLE",
        columns=tuple({"name": name, "logical_type": "STRING"} for name in names),
        rows=rows, row_count=len(rows), truncated=False,
    ).model_copy(update=overrides)


def task_frame():
    return {"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "action_intent": "NONE",
            "objectives": ["DIAGNOSE"],
            "subject_ref": {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "1285382359565720997", "database_oid": "16510"}}


def diagnosis(fact, *, field="execution_count", row_index=0, level="SUPPORTED", scope="SUBJECT", goal=True, draft_sql=None):
    return SqlDiagnosisDraft.model_validate({
        "summary": "根据实际观测给出原因方向和有条件措施。",
        "conclusions": [{
            "conclusion_id": "c1", "statement": "该 SQL 的已观测工作量需要结合调用频率治理。",
            "level": level, "scope": scope, "applicable_time": fact.captured_at,
            "evidence_fields": [{"evidence_ref": fact.evidence_ref, "row_index": row_index, "field": field}],
            "reasoning": "使用实际字段而非固定指标模板，其他来源可改变机制判断。",
            "limitations": ["该观测不证明其他时间的负载。"],
        }],
        "recommendations": [{
            "recommendation_id": "r1", "conclusion_ids": ["c1"], "action": "核对调用路径并在同负载下验证调用合并。",
            "intent": "REPAIR", "preconditions": ["业务允许合并重复请求且确认没有事务语义变化"],
            "risks": ["合并可能改变数据新鲜度"], "validation": ["相同业务吞吐下比较请求次数和延迟"],
            "rollback_or_stop": "出现正确性差异即恢复原调用路径",
            "requires_approval": True, "draft_sql": draft_sql,
        }],
        "unresolved_questions": [], "goal_satisfied": goal,
    })


def test_bundle_retains_native_fields_precision_time_and_lineage():
    fact = observation(rows=(("1285382359565720997", 16510, 2**60 + 1, 42),))
    bundle = build_sql_evidence_bundle(task_frame=task_frame(), database_type="POSTGRESQL", evidence=(fact,))
    record = bundle["observations"][0]["records"][0]
    assert record["fields"]["execution_count"] == str(2**60 + 1)
    assert record["role"] == "SUBJECT"
    assert bundle["observations"][0]["lineage"]["artifact_id"] == str(fact.artifact_id)
    assert exact_wire({"value": Decimal("0.000000000000000001"), "flag": False}) == {"value": "1E-18", "flag": False}


@pytest.mark.parametrize("field,index", [("invented_cpu", 0), ("execution_count", 9)])
def test_nonexistent_field_or_row_is_rejected(field, index):
    fact = observation()
    bundle = build_sql_evidence_bundle(task_frame=task_frame(), database_type="POSTGRESQL", evidence=(fact,))
    with pytest.raises(ValueError):
        validate_sql_diagnosis(diagnosis(fact, field=field, row_index=index), bundle)


@pytest.mark.parametrize("kind", ["other_query", "object_only", "model_inference", "user_provided"])
def test_context_cannot_self_certify_sql_causality(kind):
    fact = observation()
    if kind == "other_query":
        fact = fact.model_copy(update={"rows": (("99", 16510, 100, 42),)})
    elif kind == "object_only":
        fact = observation(names=("relation_name", "seq_scan"), rows=(("orders", 9000),))
    else:
        fact = fact.model_copy(update={"trust_level": "USER_PROVIDED" if kind == "user_provided" else "MODEL_INFERENCE"})
    bundle = build_sql_evidence_bundle(task_frame=task_frame(), database_type="POSTGRESQL", evidence=(fact,))
    field = "seq_scan" if kind == "object_only" else "execution_count"
    with pytest.raises(ValueError):
        validate_sql_diagnosis(diagnosis(fact, field=field), bundle)


def render_draft(draft):
    converted = sql_investigation_draft(draft, "SQL 当前语句")
    return render_investigation(SimpleNamespace(
        claims=converted.claims, recommendations=converted.recommendations,
        unresolved_questions=converted.unresolved_questions,
    ))


def test_no_plan_does_not_prevent_session_and_lock_diagnosis():
    sql = observation()
    locks = observation(names=("blocked_pid", "blocking_pid", "wait_event"), rows=((42, 99, "transactionid"),), tool_id="db.postgresql.lock.tree")
    bundle = build_sql_evidence_bundle(task_frame=task_frame(), database_type="POSTGRESQL", evidence=(sql, locks))
    assert bundle["relationships"][0]["kind"] == "SAME_OBSERVED_SESSION"
    assert bundle["observations"][1]["records"][0]["role"] == "RELATED_SESSION"
    draft = diagnosis(sql)
    validate_sql_diagnosis(draft, bundle)
    analysis, solution, refs = render_draft(draft)
    assert "Finding" not in analysis
    assert sql.captured_at in analysis
    assert "适用条件" in solution and "停止或回退" in solution and refs == (sql.evidence_ref,)


def test_context_is_visible_and_user_material_is_not_verified_fact():
    fact = observation()
    bundle = build_sql_evidence_bundle(task_frame=task_frame(), database_type="POSTGRESQL", evidence=(fact,))
    draft = diagnosis(fact, scope="CONTEXT")
    assert "数据库背景" in render_draft(draft)[0]
    bundle["observations"][0]["trust_level"] = "USER_PROVIDED"
    with pytest.raises(ValueError):
        validate_sql_diagnosis(draft, bundle)


def test_far_apart_activity_cannot_create_session_relationship():
    sql = observation()
    locks = observation(names=("blocked_pid", "blocking_pid"), rows=((42, 99),), captured_at="2026-10-28T01:10:00+00:00")
    bundle = build_sql_evidence_bundle(task_frame=task_frame(), database_type="POSTGRESQL", evidence=(sql, locks))
    assert not bundle["relationships"]
    assert bundle["observations"][1]["records"][0]["role"] == "CONTEXT"


def test_hypotheses_cannot_claim_completed_diagnosis():
    fact = observation()
    bundle = build_sql_evidence_bundle(task_frame=task_frame(), database_type="POSTGRESQL", evidence=(fact,))
    with pytest.raises(ValueError):
        validate_sql_diagnosis(diagnosis(fact, level="HYPOTHESIS"), bundle)


def test_repair_and_draft_never_become_execution_authority():
    draft = diagnosis(observation(), draft_sql="CREATE INDEX candidate ON example(id)")
    with pytest.raises(ValidationError):
        draft.model_validate({**draft.model_dump(), "recommendations": [{**draft.recommendations[0].model_dump(), "requires_approval": False}]})
    assert "待验证草案" in render_draft(draft)[1]


def test_single_sql_calls_structured_diagnosis_and_never_redisplays_top_list():
    fact = observation(tool_id="db.postgresql.statement.top")
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,))
    model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(output=diagnosis(fact), receipt=SimpleNamespace(model_dump=lambda **_: {}))))
    prompt = SimpleNamespace(content="诊断", ref=lambda: {})
    prompts = SimpleNamespace(resolve=AsyncMock(return_value=prompt))
    ctx = TaskExecutionContext(
        run_id=uuid7(), task_id=uuid7(), task_key="answer:compose", target_id=uuid7(),
        agent_id=uuid7(), actor_id=uuid7(), trigger_type="MANUAL", trace_id="test", attempt=1, deadline_at=None,
        policy_snapshot={}, plan_snapshot={"capability_snapshot": {"database_type": "POSTGRESQL"}, "answer_context": {"task_frame": task_frame(), "model": {}, "prompts": {}}},
        input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "artifact_id": str(uuid7()), "payload": assessment.model_dump(mode="json")},),
    )
    result = asyncio.run(DbaAnswerComposeHandler(model_client=model, prompts=prompts).execute(ctx))
    assert result.status == "COMPLETED"
    assert model.generate_structured.await_count == 1
    assert model.generate_structured.await_args.kwargs["purpose"] == "aiops.sql-diagnosis"
    assert all(block.payload.get("result_kind") != "TOP_SQL_LIST" for block in result.blocks)
    assert result.blocks[0].payload["sql_diagnosis"]["goal_satisfied"]


def test_selection_fixture_preserves_bigint_and_checks_source_row():
    fact = observation()
    artifact_id = str(uuid7())
    reference = selection_ref(artifact_id=artifact_id, fact=fact, row_index=0)
    data = decode_selection(reference)
    assert data["artifact_id"] == artifact_id
    assert subject_from_row(fact, 0, "POSTGRESQL")["query_id"] == "1285382359565720997"


@pytest.mark.parametrize("reference", ["", "%%%%", "eyJ2IjoxfQ", "[]"])
def test_invalid_selection_encoding_rejected(reference):
    with pytest.raises(ValueError):
        decode_selection(reference)


def selection_scope():
    now = datetime.now(UTC)
    actor, agent, version, target_id, conversation_id, domain_id, tenant_id, run_id, turn_id = (uuid7() for _ in range(9))
    fact = observation()
    artifact_id = uuid7()
    target = SimpleNamespace(target_id=target_id, row_version=2, db_type="POSTGRESQL", status="ENABLED")
    run = SimpleNamespace(ops_run_id=run_id, target_id=target_id, actor_id=actor, agent_id=agent, agent_version_id=version, domain_id=domain_id, tenant_id=tenant_id, plan_snapshot_json={"target": {"row_version": 2, "db_type": "POSTGRESQL"}})
    raw_artifact, fact = backed_sql_fact(fact, run)
    conversation = SimpleNamespace(conversation_id=conversation_id, agent_id=agent, agent_version_id=version, domain_id=domain_id, tenant_id=tenant_id)
    turn = SimpleNamespace(turn_id=turn_id, conversation_id=conversation_id, created_by=actor, sufficiency_artifact_id=artifact_id)
    artifact = SimpleNamespace(artifact_id=artifact_id, ops_run_id=run_id, created_at=now, schema_version="DBA_SUFFICIENCY.v1", payload_json=DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,)).model_dump(mode="json"))
    artifact.trust_level = "SOURCE_VERIFIED"
    artifact.content_hash = canonical_hash(artifact.payload_json)
    uow = SimpleNamespace(
        runs=SimpleNamespace(get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifact if artifact_id == artifact.artifact_id
            else raw_artifact if artifact_id == raw_artifact.artifact_id else None), get_run_scoped=AsyncMock(return_value=run)),
        turns=SimpleNamespace(get_run_link_by_ops_run_id=AsyncMock(return_value=SimpleNamespace(turn_id=turn_id)), get_turn=AsyncMock(return_value=turn)),
        targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target)),
        agents=SimpleNamespace(version_has_target=AsyncMock(return_value=True)),
    )
    return SimpleNamespace(uow=uow, conversation=conversation, actor=actor, artifact=artifact, run=run, target=target, turn=turn, fact=fact, now=now,
                           reference=selection_ref(artifact_id=str(artifact_id), fact=fact, row_index=0))


def test_authorized_historical_selection_does_not_require_immediately_previous_turn():
    fixture = selection_scope()
    result = asyncio.run(resolve_sql_selection(uow=fixture.uow, reference=fixture.reference, conversation=fixture.conversation, actor_id=fixture.actor, now=fixture.now))
    assert result["subject_ref"]["query_id"] == "1285382359565720997"
    assert result["source_turn_id"] == str(fixture.turn.turn_id)
    assert result["captured_at"] == fixture.fact.captured_at


def test_selection_context_never_copies_raw_values_to_unrevokable_plan_or_user_message():
    from aiops_agent.application.sql_selection import sql_selection_context
    fixture = selection_scope()
    selected = asyncio.run(resolve_sql_selection(uow=fixture.uow, reference=fixture.reference,
        conversation=fixture.conversation, actor_id=fixture.actor, now=fixture.now))
    assert selected["facts"] and selected["sources"]
    context = sql_selection_context({**selected, "future_raw_values": {"not_metadata": True}})
    assert "facts" not in context and "sources" not in context and "future_raw_values" not in context
    assert context["selection_ref"] == fixture.reference
    assert context["subject_ref"] == selected["subject_ref"]


@pytest.mark.parametrize("violation", ["actor", "conversation", "agent_version", "target_version", "revoked", "expired", "row_hash", "disabled"])
def test_selection_rechecks_authority_integrity_and_expiry(violation):
    fixture = selection_scope()
    if violation == "actor": fixture.run.actor_id = uuid7()
    if violation == "conversation": fixture.turn.conversation_id = uuid7()
    if violation == "agent_version": fixture.run.agent_version_id = uuid7()
    if violation == "target_version": fixture.target.row_version = 3
    if violation == "revoked": fixture.uow.agents.version_has_target.return_value = False
    if violation == "expired": fixture.artifact.created_at = fixture.now - timedelta(days=2)
    if violation == "disabled": fixture.target.status = "DISABLED"
    if violation == "row_hash":
        fixture.artifact.payload_json["evidence"][0]["rows"][0][2] = 999
    with pytest.raises((ValueError, PermissionError)):
        asyncio.run(resolve_sql_selection(uow=fixture.uow, reference=fixture.reference, conversation=fixture.conversation, actor_id=fixture.actor, now=fixture.now))


def test_mysql_null_schema_is_distinct_from_missing_schema():
    fact = observation(names=("digest", "schema_name"), rows=(("A" * 64, None),))
    subject = subject_from_row(fact, 0, "MYSQL")
    assert "schema_name" in subject and subject["schema_name"] is None


def test_idempotency_digest_includes_selected_row():
    a = TurnCreate(content=({"content_type": "TEXT", "text": "分析"},), idempotency_key="key", sql_diagnosis={"selection_ref": "one"})
    b = a.model_copy(update={"sql_diagnosis": a.sql_diagnosis.model_copy(update={"selection_ref": "two"})})
    assert ConversationTurnService._request_sha256(a) != ConversationTurnService._request_sha256(b)


def test_mysql_readonly_query_has_independent_dialect_and_policy():
    policy = MySQLDynamicQueryPolicy(MySQLDynamicQueryPolicySnapshot(max_rows=12))
    value = policy.validate("SELECT THREAD_ID FROM performance_schema.threads WHERE PROCESSLIST_ID=:pid", {"pid": 42})
    assert value.schema_version == "MYSQL_VALIDATED_DYNAMIC_QUERY.v1"
    assert value.normalized_sql.endswith("LIMIT 12")
    assert value.parameters == {"pid": 42}


@pytest.mark.parametrize("sql", [
    "SELECT SLEEP(10) AS wait",
    "SELECT GET_LOCK('x', 1) AS lock_name",
    "SELECT LOAD_FILE('/etc/passwd') AS data",
    "SELECT THREAD_ID INTO OUTFILE '/tmp/data' FROM performance_schema.threads",
    "SELECT @x := THREAD_ID FROM performance_schema.threads",
    "SELECT @@global.read_only AS value",
    "SELECT * FROM business.orders",
    "SELECT * FROM mysql.user",
    "SELECT * FROM performance_schema.threads FOR UPDATE",
    "SELECT THREAD_ID FROM performance_schema.threads; DELETE FROM business.orders",
    "WITH x AS (SELECT THREAD_ID FROM performance_schema.threads) SELECT * FROM business.x",
    "WITH x AS (DELETE FROM business.orders RETURNING id) SELECT * FROM x",
])
def test_mysql_ast_rejects_mutation_side_effects_and_catalog_bypass(sql):
    with pytest.raises(DynamicQueryRejected):
        MySQLDynamicQueryPolicy(MySQLDynamicQueryPolicySnapshot()).validate(sql)


def test_mysql_lexical_binding_does_not_rewrite_strings_comments_or_percent():
    sql = "SELECT ':secret' AS label, 5 % 2 AS n FROM performance_schema.threads WHERE PROCESSLIST_ID=:pid /* :hidden */"
    bound = _mysql_bound_sql(sql)
    assert "':secret'" in bound and "/* :hidden */" in bound
    assert "5 %% 2" in bound and "PROCESSLIST_ID=%(pid)s" in bound
    assert bound % {"pid": 42} == sql.replace(":pid", "42")
