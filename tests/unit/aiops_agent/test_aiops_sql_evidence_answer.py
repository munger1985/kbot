"""单 SQL 证据摘要的口径、身份隔离与流式输出回归。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.sql_evidence_answer import build_sql_evidence_answer, rebind_prior_sql_facts
from aiops_agent.application.investigation.prior_sql_evidence import load_prior_sql_evidence
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceFact
from aiops_agent.entities import OpsRunEntity
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7


def fact(tool_id, columns, rows, *, semantics="CUMULATIVE_SINCE_LOAD", captured="2026-10-27T10:28:00+00:00", trust="SOURCE_VERIFIED"):
    artifact_id = str(uuid7())
    return TurnEvidenceFact(
        artifact_id=artifact_id, evidence_ref=f"artifact:{artifact_id}#result",
        source_id=tool_id, step_id="result", tool_id=tool_id, trust_level=trust,
        measurement_semantics=semantics, presentation_kind="TABLE", captured_at=captured,
        columns=tuple({"name": name, "logical_type": "STRING"} for name in columns),
        rows=tuple(rows), row_count=len(rows),
    )


def frame(subject):
    return {"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject, "objectives": ["DIAGNOSE"]}


@pytest.mark.parametrize("db_type,subject,tool_id,columns,row,expected", [
    ("POSTGRESQL", {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1, "user_oid": 2, "top_level": True},
     "db.postgresql.statement.details", ("query_id", "database_oid", "user_oid", "top_level", "execution_count", "total_exec_time_ms", "max_exec_time_ms", "shared_blks_hit"),
     ("42", 1, 2, True, 100, 2350, 281, 100000), "总耗时（毫秒） 2,350"),
    ("MYSQL", {"subject_type": "MYSQL_DIGEST", "digest": "ABC", "schema_name": "orders"},
     "db.sql.cursor_details", ("digest", "schema_name", "execution_count", "total_latency_picoseconds"),
     ("abc", "orders", 12, 2_500_000_000), "总耗时（毫秒） 2.5"),
    ("ORACLE", {"subject_type": "ORACLE_SQL_ID", "sql_id": "abc123"},
     "db.sql.cursor_details", ("sql_id", "inst_id", "child_number", "executions", "elapsed_seconds"),
     ("abc123", 1, 2, 20, 1.25), "总耗时（秒） 1.25"),
])
def test_identity_bound_accumulated_evidence(db_type, subject, tool_id, columns, row, expected):
    entry = fact(tool_id, columns, (row,))
    result = build_sql_evidence_answer(task_frame=frame(subject), database_type=db_type, evidence=(entry,))
    assert result.has_statistics
    assert result.evidence_refs == (entry.evidence_ref,)
    assert expected in result.analysis
    assert "累计值，不代表指定时间窗口" in result.analysis
    assert "累计最大耗时不能通过两次快照相减" in result.solution
    assert "共享块命中或对象级扫描统计不能单独证明" in result.analysis
    assert "Finding" not in result.analysis


def test_other_identity_other_trust_and_scope_are_not_used():
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1, "user_oid": 2, "top_level": True}
    columns = ("query_id", "database_oid", "user_oid", "top_level", "execution_count")
    wrong_db = fact("db.postgresql.statement.details", columns, (("42", 3, 2, True, 100),))
    untrusted = fact("db.postgresql.statement.details", columns, (("42", 1, 2, True, 500),), trust="MODEL_INFERENCE")
    wrong_semantics = fact("db.postgresql.statement.details", columns, (("42", 1, 2, True, 999),), semantics="CURRENT_ACTIVITY")
    no_metrics = fact("db.postgresql.statement.details", columns, (("42", 1, 2, True, None),))
    result = build_sql_evidence_answer(task_frame=frame(subject), database_type="POSTGRESQL", evidence=(wrong_db, untrusted, wrong_semantics, no_metrics))
    assert not result.has_statistics
    assert not result.evidence_refs
    assert "不能判断" in result.analysis
    assert build_sql_evidence_answer(task_frame={"diagnostic_profile": "GENERAL", "subject_ref": subject}, database_type="POSTGRESQL", evidence=(wrong_db,)) is None


def test_most_recent_matching_observation_does_not_merge_rows():
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1}
    columns = ("query_id", "database_oid", "user_oid", "execution_count")
    old = fact("db.postgresql.statement.details", columns, (("42", 1, 2, 1),), captured="2026-10-27T10:00:00+00:00")
    new = fact("db.postgresql.statement.details", columns, (("42", 1, 2, 12), ("42", 1, 3, 20)))
    result = build_sql_evidence_answer(task_frame=frame(subject), database_type="POSTGRESQL", evidence=(old, new))
    assert result.evidence_refs == (new.evidence_ref,)
    assert "user_oid=2" in result.analysis and "user_oid=3" in result.analysis
    assert "未合并计算" in result.analysis
    assert "执行次数 32" not in result.analysis


def test_oracle_legacy_subject_and_small_nonzero_latency():
    oracle = frame({"sql_id": "6tjx7su0q5ttj"})
    entry = fact("db.sql.cursor_details", ("sql_id", "executions", "elapsed_seconds"), (("6tjx7su0q5ttj", 1, "0.0000001"),))
    result = build_sql_evidence_answer(task_frame=oracle, database_type="ORACLE", evidence=(entry,))
    assert result.has_statistics and "1e-7" in result.analysis
    assert build_sql_evidence_answer(task_frame=oracle, database_type="POSTGRESQL", evidence=(entry,)) is None
    ambiguous = frame({"sql_id": "6tjx7su0q5ttj", "query_id": "42"})
    assert build_sql_evidence_answer(task_frame=ambiguous, database_type="ORACLE", evidence=(entry,)) is None
    change_request = {**oracle, "action_intent": "EXECUTE"}
    assert build_sql_evidence_answer(task_frame=change_request, database_type="ORACLE", evidence=(entry,)) is None


def test_oracle_starter_builds_a_typed_single_sql_subject():
    context = SimpleNamespace(
        question="分析 SQL 6tjx7su0q5ttj", target_context={"db_type": "ORACLE"},
        conversation_starter={"starter_id": "oracle.sql.healthcheck", "title": "SQL 分析",
                              "planning": {"kind": "SINGLE_SQL"}, "parameters": {"sql_id": "6TJX7SU0Q5TTJ"}},
    )
    planner = object.__new__(TurnPlanningService)
    planner._record_planning_route = AsyncMock()
    available = tuple({"tool_id": item} for item in
                      ("db.sql.cursor_details", "db.oracle.statement.plan", "db.sql.object_statistics", "db.oracle.statement.bind_metadata",
                       "db.sql.object_indexes", "db.sql.object_dependencies", "db.sql.object_candidates"))
    investigation, _, _, _ = asyncio.run(planner._plan_conversation_starter(context=context, available_tools=available, available_playbooks=()))
    assert investigation.task_frame.subject_ref == {"subject_type": "ORACLE_SQL_ID", "sql_id": "6tjx7su0q5ttj"}


def test_single_sql_safe_fallback_stream_and_nonstream_agree():
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1}
    entry = fact("db.postgresql.statement.details", ("query_id", "database_oid", "execution_count"), (("42", 1, 100),))
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(entry,))
    ctx = TaskExecutionContext(
        run_id=uuid7(), task_id=uuid7(), task_key="answer", target_id=uuid7(), agent_id=uuid7(), actor_id=uuid7(),
        trigger_type="MANUAL", trace_id="test", attempt=1, deadline_at=None, policy_snapshot={},
        plan_snapshot={"capability_snapshot": {"database_type": "POSTGRESQL"}, "answer_context": {"task_frame": frame(subject)}},
        input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},),
    )
    handler = DbaAnswerComposeHandler(model_client=None, prompts=None)
    result = asyncio.run(handler.execute(ctx))
    async def collect():
        return [item async for item in handler.execute_stream(ctx)]
    items = asyncio.run(collect())
    assert items[-1].blocks == result.blocks
    assert items[-1].answer_streamed
    text = "".join(item.payload["delta"] for item in items[:-1])
    assert "执行次数 100" in text
    assert "Finding" not in text


def test_prior_sql_is_rebound_to_current_artifact_and_labeled_historical():
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1, "user_oid": 2}
    entry = fact("db.postgresql.statement.top", ("query_id", "database_oid", "user_oid", "execution_count"), (("42", 1, 2, 100), ("42", 1, 3, 999)))
    rebound = rebind_prior_sql_facts(payload={"facts": [entry.model_dump(mode="json")]}, artifact_id=str(uuid7()), task_frame=frame(subject))
    assert len(rebound) == 1 and rebound[0].row_count == 1
    assert rebound[0].evidence_ref != entry.evidence_ref
    result = build_sql_evidence_answer(task_frame=frame(subject), database_type="POSTGRESQL", evidence=rebound)
    assert result.historical_only and "历史观测" in result.analysis
    assert "执行次数 100" in result.analysis and "999" not in result.analysis
    current = fact("db.postgresql.statement.details", ("query_id", "database_oid", "user_oid", "execution_count"), (("42", 1, 2, 200),))
    result = build_sql_evidence_answer(task_frame=frame(subject), database_type="POSTGRESQL", evidence=(*rebound, current))
    assert not result.historical_only and "执行次数 200" in result.analysis
    assert result.evidence_refs == (current.evidence_ref, rebound[0].evidence_ref)
    assert "存在相同标识的历史统计" in result.analysis
    assert "不能单凭此确认实际实例未切换" in result.analysis


def test_historical_citation_has_a_separately_labeled_table():
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1}
    entry = fact("db.postgresql.statement.top", ("query_id", "database_oid", "execution_count"), (("42", 1, 100),))
    prior = rebind_prior_sql_facts(payload={"facts": [entry.model_dump(mode="json")]}, artifact_id=str(uuid7()), task_frame=frame(subject))
    assessment = DbaSufficiencyAssessment(status="NEEDS_EVIDENCE", evidence=prior)
    ctx = TaskExecutionContext(
        run_id=uuid7(), task_id=uuid7(), task_key="answer", target_id=uuid7(), agent_id=uuid7(), actor_id=uuid7(),
        trigger_type="MANUAL", trace_id="test", attempt=1, deadline_at=None, policy_snapshot={},
        plan_snapshot={"capability_snapshot": {"database_type": "POSTGRESQL"}, "answer_context": {"task_frame": frame(subject)}},
        input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},),
    )
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(ctx))
    assert result.status == "PARTIAL"
    tables = [block for block in result.blocks if str(block.block_type) == "TABLE"]
    assert len(tables) == 1
    assert tables[0].payload["title"].startswith("历史观测")
    assert tables[0].payload["captured_at"] == entry.captured_at
    assert tables[0].evidence_refs == (prior[0].evidence_ref,)


def test_prior_evidence_cannot_make_current_assessment_answerable():
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1}
    entry = fact("db.postgresql.statement.top", ("query_id", "database_oid", "execution_count"), (("42", 1, 100),))
    ctx = SimpleNamespace(
        input_artifacts=({"schema_version": "PRIOR_TURN_SQL_EVIDENCE.v1", "artifact_id": str(uuid7()), "payload": {"facts": [entry.model_dump(mode="json")]}},),
        plan_snapshot={"answer_context": {"task_frame": frame(subject)}}, trigger_type="MANUAL",
    )
    result = asyncio.run(DbaEvidenceAssessmentHandler().execute(ctx))
    assert result.status == "NEEDS_EVIDENCE"
    assert len(result.evidence) == 1 and result.evidence[0].source_id == "prior_turn.sql_observation"


@pytest.mark.parametrize("observed_id,missing", [("42", False), ("99", True)])
def test_postgresql_core_evidence_is_database_specific_and_identity_matched(observed_id, missing):
    artifact_id = str(uuid7())
    artifact = {
        "artifact_id": artifact_id, "schema_version": "DBA_TOOL_RESULT.v1",
        "payload": {
            "schema_version": "DBA_TOOL_RESULT.v1", "source_type": "PLAYBOOK",
            "source_id": "db.postgresql.statement.details", "source_version": "1.0.0",
            "definition_hash": "b" * 64, "output_schema": "statement.output.v1",
            "measurement_semantics": "CUMULATIVE_SINCE_LOAD", "presentation_kind": "TABLE", "status": "SUCCEEDED",
            "tool_outcomes": [{
                "step_id": "statement", "tool_id": "db.postgresql.statement.details", "tool_version": "1.0.0",
                "status": "SUCCEEDED", "observation": {
                    "schema_version": "DATABASE_OBSERVATION.v1", "executor_request_id": str(uuid7()),
                    "target_id": str(uuid7()), "tool_id": "db.postgresql.statement.details", "tool_version": "1.0.0",
                    "variant": "pg16", "template_sha256": "c" * 64, "db_type": "POSTGRESQL", "db_version": "16",
                    "capability_snapshot_hash": "d" * 64, "captured_at": "2026-10-27T10:28:00Z", "duration_ms": 2,
                    "columns": [{"name": key, "logical_type": "STRING", "sensitivity": "PUBLIC"} for key in ("query_id", "database_oid", "execution_count")],
                    "rows": [[observed_id, 1, 10]], "row_count": 1, "truncated": False,
                    "result_sha256": "e" * 64, "parameters_sha256": "f" * 64,
                },
            }],
        },
    }
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1}
    ctx = SimpleNamespace(input_artifacts=(artifact,), trigger_type="MANUAL",
                          plan_snapshot={"capability_snapshot": {"database_type": "POSTGRESQL"},
                                         "answer_context": {"task_frame": frame(subject)}})
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(ctx))
    core_gaps = [gap for gap in assessment.gaps if gap.code == "SQL_SUBJECT_EVIDENCE_UNMATCHED"]
    assert bool(core_gaps) == missing
    assert all(gap.step_id == "sql.subject" for gap in core_gaps)


def test_prior_link_through_planning_assessment_and_answer():
    scope_keys = ("domain_id", "target_id", "agent_id", "agent_version_id", "actor_id")
    scope = {key: str(uuid7()) for key in scope_keys if key in OpsRunEntity.__table__.columns}
    prior_run_id, run_id, turn_id, previous_id, conversation_id = (uuid7() for _ in range(5))
    context = SimpleNamespace(**scope, turn_id=turn_id, ops_run_id=run_id, resolved_uploads=(),
                              source_run_evidence=None, recovery_assurance={}, future_observation_evidence=(),
                              database_execution={"target_row_version": 1})
    target_snapshot = {"target": {"row_version": 1, "db_type": "POSTGRESQL"}}
    current_run = SimpleNamespace(**scope, status="RUNNING", ops_run_id=run_id, plan_snapshot_json=target_snapshot)
    old_run = SimpleNamespace(**scope, ops_run_id=prior_run_id, plan_snapshot_json=target_snapshot)
    turn_scope = {"domain_id": scope["domain_id"]}
    current_turn = SimpleNamespace(**turn_scope, turn_no=2, conversation_id=conversation_id, created_by=scope["actor_id"])
    previous = SimpleNamespace(**turn_scope, turn_no=1, conversation_id=conversation_id, created_by=scope["actor_id"],
                               status="COMPLETED", turn_id=previous_id, sufficiency_artifact_id=uuid7())
    entry = fact("db.postgresql.statement.top", ("query_id", "database_oid", "execution_count"), (("42", 1, 100),))
    previous_assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(entry,))
    source_artifact = SimpleNamespace(artifact_id=previous.sufficiency_artifact_id, ops_run_id=prior_run_id,
                                      schema_version="DBA_SUFFICIENCY.v1", payload_json=previous_assessment.model_dump(mode="json"), trust_level="SOURCE_VERIFIED")
    source_artifact.content_hash = canonical_hash(source_artifact.payload_json)

    class Uow:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False

    uow = Uow()
    uow.turns = SimpleNamespace(
        get_turn=AsyncMock(return_value=current_turn), list_prior_turns=AsyncMock(return_value=[previous]),
        get_run_link=AsyncMock(side_effect=lambda *, turn_id, purpose: SimpleNamespace(ops_run_id=run_id if turn_id == context.turn_id else prior_run_id)),
    )
    uow.runs = SimpleNamespace(get_run=AsyncMock(side_effect=lambda *, ops_run_id: current_run if ops_run_id == run_id else old_run),
                               get_artifact=AsyncMock(return_value=source_artifact))
    uow.targets = SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace(status="ENABLED", row_version=1, db_type="POSTGRESQL")))
    uow.agents = SimpleNamespace(version_has_target=AsyncMock(return_value=True))
    planner = object.__new__(TurnPlanningService)
    planner._uow_factory = lambda: uow
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1}
    payload = asyncio.run(planner._load_prior_sql_evidence(context, task_frame=frame(subject)))
    assert payload and payload["source_run_id"] == str(prior_run_id)
    context.prior_sql_evidence = payload
    assert "turn-prior-sql-evidence:1" in planner._initial_user_evidence_keys(context, include_prior_sql=True)
    assert "turn-prior-sql-evidence:1" not in planner._initial_user_evidence_keys(context)
    artifact = planner._artifact(ops_run_id=run_id, artifact_key="turn-prior-sql-evidence:1",
                                 artifact_type="PRIOR_TURN_SQL_EVIDENCE", schema_version="PRIOR_TURN_SQL_EVIDENCE.v1",
                                 payload=payload, producer="test", trust_level="SOURCE_VERIFIED")
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42", "database_oid": 1}
    plan = {"capability_snapshot": {"database_type": "POSTGRESQL"}, "answer_context": {"task_frame": frame(subject)}}
    assess_ctx = SimpleNamespace(input_artifacts=({"schema_version": artifact.schema_version, "artifact_id": str(artifact.artifact_id), "payload": artifact.payload_json},),
                                 plan_snapshot=plan, trigger_type="MANUAL")
    assessment = asyncio.run(DbaEvidenceAssessmentHandler().execute(assess_ctx))
    assert assessment.status == "NEEDS_EVIDENCE"
    answer_ctx = TaskExecutionContext(run_id=run_id, task_id=uuid7(), task_key="answer", target_id=uuid7(), agent_id=uuid7(), actor_id=uuid7(),
                                     trigger_type="MANUAL", trace_id="test", attempt=1, deadline_at=None, policy_snapshot={}, plan_snapshot=plan,
                                     input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
    result = asyncio.run(DbaAnswerComposeHandler(model_client=None, prompts=None).execute(answer_ctx))
    assert result.status == "PARTIAL"
    assert any(block.payload.get("title", "").startswith("历史观测") for block in result.blocks)
    assert str(artifact.artifact_id) in result.evidence[0].evidence_ref
    context.target_id = str(uuid7())
    assert asyncio.run(planner._load_prior_sql_evidence(context, task_frame=frame(subject))) is None
    context.target_id = scope["target_id"]
    old_run.plan_snapshot_json = {"target": {"row_version": 2, "db_type": "POSTGRESQL"}}
    assert asyncio.run(planner._load_prior_sql_evidence(context, task_frame=frame(subject))) is None
    old_run.plan_snapshot_json = target_snapshot
    context.database_execution = {"target_row_version": 2}
    assert asyncio.run(planner._load_prior_sql_evidence(context, task_frame=frame(subject))) is None


@pytest.mark.parametrize("changed", [None, "target_id", "domain_id", "agent_id", "actor_id", "created_by", "conversation_id"])
def test_prior_loader_requires_same_scope_and_trusted_source(changed):
    identifiers = {key: str(uuid7()) for key in ("target_id", "domain_id", "agent_id", "agent_version_id", "actor_id", "created_by", "conversation_id")}
    turn = SimpleNamespace(**{key: identifiers[key] for key in ("domain_id", "created_by", "conversation_id")}, turn_no=2)
    run = SimpleNamespace(**{key: identifiers[key] for key in ("target_id", "domain_id", "agent_id", "agent_version_id", "actor_id")}, plan_snapshot_json={"target": {"row_version": 1, "db_type": "POSTGRESQL"}})
    previous = SimpleNamespace(**{key: identifiers[key] for key in ("domain_id", "created_by", "conversation_id")}, status="COMPLETED", turn_id=uuid7(), sufficiency_artifact_id=uuid7())
    prior_run = SimpleNamespace(**{key: identifiers[key] for key in ("target_id", "domain_id", "agent_id", "agent_version_id", "actor_id")}, ops_run_id=uuid7(), plan_snapshot_json={"target": {"row_version": 1, "db_type": "POSTGRESQL"}})
    if changed:
        setattr(previous if changed in {"created_by", "conversation_id"} else prior_run, changed, str(uuid7()))
    entry = fact("db.postgresql.statement.top", ("query_id", "database_oid"), (("42", 1),))
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(entry,))
    artifact = SimpleNamespace(artifact_id=previous.sufficiency_artifact_id, ops_run_id=prior_run.ops_run_id, schema_version="DBA_SUFFICIENCY.v1", payload_json=assessment.model_dump(mode="json"), trust_level="SOURCE_VERIFIED")
    artifact.content_hash = canonical_hash(artifact.payload_json)
    uow = SimpleNamespace(turns=SimpleNamespace(list_prior_turns=AsyncMock(return_value=[previous]), get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=prior_run.ops_run_id))), runs=SimpleNamespace(get_run=AsyncMock(return_value=prior_run), get_artifact=AsyncMock(return_value=artifact)),
        targets=SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace(status="ENABLED", row_version=1, db_type="POSTGRESQL"))), agents=SimpleNamespace(version_has_target=AsyncMock(return_value=True)))
    loaded = asyncio.run(load_prior_sql_evidence(uow=uow, turn=turn, run=run, task_frame=frame({"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "42"})))
    assert (loaded is not None) == (changed is None)
