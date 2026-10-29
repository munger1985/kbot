"""不同 SQL 标识的逐条诊断、比较、历史搜索及报告隔离，不替代真实模型和三库验收。"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.investigation_record import investigation_report_source
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle, validate_sql_diagnosis
from aiops_agent.application.diagnosis.sql_evidence_answer import build_sql_evidence_answer, rebind_prior_sql_facts
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.sql_selection import decode_selection, subject_from_row
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.domain.sql_subjects import normalize_sql_comparison
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from platform_core.contracts.aiops.conversation import MeasurementSemantics
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import diagnosis, observation
from tests.unit.aiops_agent.test_aiops_sql_subject_resolution import scenario
from tests.unit.aiops_agent.test_aiops_sql_subject_search import load, search_fixture
from tests.support.aiops_sql_review import diagnostic_response


def comparison_fixture(engine="POSTGRESQL"):
    single, fact = scenario(engine)
    identity = {"POSTGRESQL": "query_id", "MYSQL": "digest", "ORACLE": "sql_id"}[engine]
    second_id = {"POSTGRESQL": "-42", "MYSQL": "b" * 64, "ORACLE": "3j2dsgrcqr6m4"}[engine]
    other_id = {"POSTGRESQL": "777", "MYSQL": "c" * 64, "ORACLE": "4j2dsgrcqr6m4"}[engine]
    names = [item["name"] for item in fact.columns]
    left = observation(names=names, rows=(fact.rows[0],), tool_id=fact.tool_id, measurement_semantics=MeasurementSemantics.CUMULATIVE_SINCE_LOAD)
    mapped = {**dict(zip(names, fact.rows[1], strict=True)), identity: second_id}
    right = observation(names=names[::-1],
        rows=(tuple(({**mapped, identity: other_id})[name] for name in names[::-1]),
              tuple(mapped[name] for name in names[::-1])),
        tool_id=fact.tool_id, measurement_semantics=MeasurementSemantics.CUMULATIVE_SINCE_LOAD,
        captured_at="2026-10-28T01:00:01+00:00")
    subjects = [subject_from_row(left, 0, engine), subject_from_row(right, 1, engine)]
    frame = {**single, "diagnostic_profile": "SQL_COMPARISON", "objectives": ["DIAGNOSE", "COMPARE"],
             "subject_ref": {"subjects": subjects}}
    return frame, (left, right)


def comparison_draft(bundle, facts):
    field = "executions" if bundle["database_type"] == "ORACLE" else "execution_count"
    base = diagnosis(facts[0], field=field, level="OBSERVED")
    conclusions = []
    for index, item in enumerate(bundle["subjects"], start=1):
        conclusions.append(base.conclusions[0].model_copy(update={
            "conclusion_id": f"c{index}", "subject_ids": (item["subject_id"],),
            "evidence_fields": (base.conclusions[0].evidence_fields[0].model_copy(update={
                "evidence_ref": item["evidence_ref"], "row_index": item["row_index"],
            }),),
        }))
    comparison = conclusions[0].model_copy(update={
        "conclusion_id": f"c{len(conclusions) + 1}", "scope": "COMPARISON",
        "subject_ids": tuple(item["subject_id"] for item in bundle["subjects"]),
        "evidence_fields": tuple(item.evidence_fields[0] for item in conclusions),
    })
    return base.model_copy(update={"conclusions": (*conclusions, comparison)})


def answer_context(frame, facts, engine):
    target_id = next((fact.provenance.target_id for fact in facts
        if fact.provenance is not None and fact.provenance.collected_end is not None), str(uuid7()))
    return TaskExecutionContext(run_id=uuid7(), task_id=uuid7(), task_key="answer:compose", target_id=target_id,
        agent_id=uuid7(), actor_id=uuid7(), trigger_type="MANUAL", trace_id="comparison", attempt=1,
        deadline_at=None, policy_snapshot={}, plan_snapshot={"capability_snapshot": {"database_type": engine},
        "answer_context": {"question": "比较这两条 SQL 的成本与措施", "task_frame": frame, "model": {}, "prompts": {}}},
        input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "artifact_id": str(uuid7()),
                          "payload": DbaSufficiencyAssessment(status="ANSWERABLE", evidence=facts).model_dump(mode="json")},))


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_type_safe_comparison_is_same_target_same_engine_and_disjoint(engine):
    frame, _ = comparison_fixture(engine)
    compact = SimpleNamespace(diagnostic_profile="SQL_COMPARISON", subject_ref=frame["subject_ref"])
    subjects = TurnPlanningService._comparison_sql_subjects(compact, {"db_type": engine})
    assert len(subjects) == 2
    assert TurnPlanningService._single_sql_subject(compact, {"db_type": engine}) is None
    other = "MYSQL" if engine == "POSTGRESQL" else "POSTGRESQL"
    assert TurnPlanningService._comparison_sql_subjects(compact, {"db_type": other}) is None


@pytest.mark.parametrize("problem", ["duplicate", "overlap", "one", "five", "mixed", "null_oid", "bool_oid", "too_big", "extra_identity", "unknown_dimension"])
def test_invalid_comparison_does_not_silently_drop_or_expand_request(problem):
    frame, _ = comparison_fixture()
    raw = deepcopy(frame["subject_ref"])
    if problem == "duplicate":
        raw["subjects"][1] = deepcopy(raw["subjects"][0])
    elif problem == "overlap":
        raw["subjects"][1] = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": raw["subjects"][0]["query_id"]}
    elif problem == "one":
        raw["subjects"] = raw["subjects"][:1]
    elif problem == "five":
        raw["subjects"] *= 3
    elif problem == "mixed":
        raw["subjects"][1] = {"subject_type": "MYSQL_DIGEST", "digest": "a" * 64}
    elif problem == "extra_identity":
        raw["subjects"][0]["digest"] = "a" * 64
    elif problem == "unknown_dimension":
        raw["subjects"][0]["target_id"] = "other"
    else:
        raw["subjects"][0]["database_oid"] = {"null_oid": None, "bool_oid": True, "too_big": 4294967296}[problem]
    assert normalize_sql_comparison(raw) is None


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_distinct_identities_project_exact_original_rows_and_never_merge_counts(engine):
    frame, facts = comparison_fixture(engine)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type=engine, evidence=facts)
    assert bundle["subject_resolution"]["requests_complete"]
    assert len(bundle["requests"]) == len(bundle["subjects"]) == 2
    assert bundle["object_graph"] is None
    assert {item["row_index"] for item in bundle["subjects"]} == {0, 1}
    for item in bundle["subjects"]:
        assert item["object_graph"] is not None
        assert item["object_graph"]["subject"] == item["subject_ref"]
    validate_sql_diagnosis(comparison_draft(bundle, facts), bundle)
    assert not bundle["comparison_policy"]["implicit_aggregation"]


@pytest.mark.parametrize("problem", ["missing", "historical_only", "conflict", "truncated", "incomplete_identity"])
def test_one_missing_or_ambiguous_request_keeps_other_sql_but_blocks_complete_comparison(problem):
    frame, facts = comparison_fixture()
    left, right = facts
    if problem == "missing":
        facts = (left,)
    elif problem == "historical_only":
        facts = (left, right.model_copy(update={"source_id": "prior_turn.sql_observation"}))
    elif problem == "truncated":
        facts = (left, right.model_copy(update={"truncated": True}))
    elif problem == "incomplete_identity":
        row_index = next(index for index, column in enumerate(right.columns) if column["name"] == "user_oid")
        frame["subject_ref"]["subjects"][1].pop("user_oid")
        facts = (left, right.model_copy(update={"columns": tuple(column for index, column in enumerate(right.columns) if index != row_index),
            "rows": tuple(tuple(value for index, value in enumerate(row) if index != row_index) for row in right.rows)}))
    else:
        row = right.rows[1]
        facts = (left, right.model_copy(update={"columns": (*right.columns, {"name": "normalized_statement"}),
            "rows": ((*row, "SELECT 1"), (*row, "SELECT 2"))}))
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=facts)
    assert bundle["subject_resolution"]["status"] == "COMPARISON"
    assert not bundle["subject_resolution"]["requests_complete"]
    assert bundle["requests"][0]["status"] == "RESOLVED"
    assert any(row["role"] == "SUBJECT" for observation in bundle["observations"] for row in observation["records"])
    draft = comparison_draft(bundle, facts)
    with pytest.raises(ValueError):
        validate_sql_diagnosis(draft, bundle)
    partial = draft.model_copy(update={"goal_satisfied": False, "conclusions": draft.conclusions[:1]})
    validate_sql_diagnosis(partial, bundle)


@pytest.mark.parametrize("problem", ["no_comparison", "missing_individual", "wrong_subject", "other_row"])
def test_model_cannot_call_a_partial_or_cross_wired_answer_complete(problem):
    frame, facts = comparison_fixture()
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=facts)
    draft = comparison_draft(bundle, facts)
    first, second, compare = draft.conclusions
    if problem == "no_comparison":
        draft = draft.model_copy(update={"conclusions": (first, second)})
    elif problem == "missing_individual":
        draft = draft.model_copy(update={"conclusions": (first, compare)})
    elif problem == "wrong_subject":
        draft = draft.model_copy(update={"conclusions": (first.model_copy(update={"subject_ids": second.subject_ids}), second, compare)})
    else:
        bad = second.evidence_fields[0].model_copy(update={"row_index": 0})
        draft = draft.model_copy(update={"conclusions": (first, second.model_copy(update={"evidence_fields": (bad,)}), compare)})
    with pytest.raises(ValueError):
        validate_sql_diagnosis(draft, bundle)


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_comparison_table_unions_columns_but_selections_use_each_original_fact_and_row(engine):
    frame, facts = comparison_fixture(engine)
    context = answer_context(frame, facts, engine)
    assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=facts)
    table, = DbaAnswerComposeHandler._sql_candidate_blocks(assessment, context, comparison=True)
    names = [column["name"] for column in table.payload["columns"]]
    key = {"POSTGRESQL": "query_id", "MYSQL": "digest", "ORACLE": "sql_id"}[engine]
    for index, request in enumerate(frame["subject_ref"]["subjects"]):
        assert table.payload["rows"][index][names.index(key)].casefold() == str(request[key]).casefold()
        assert table.payload["rows"][index][names.index("observation_captured_at")] == facts[index].captured_at
        selection = decode_selection(table.payload["row_selections"][index])
        assert selection["evidence_ref"] == facts[index].evidence_ref
        assert selection["row_index"] == index
    assert table.evidence_refs == tuple(fact.evidence_ref for fact in facts)


def test_comparison_search_inherits_both_requested_rows_once_with_shared_budget():
    fixture = search_fixture()
    fixture.frame.update(diagnostic_profile="SQL_COMPARISON", subject_ref={"subjects": [
        {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": key} for key in ("42", "99")
    ]})
    payload = load(fixture)
    assert len(payload["facts"]) == 1
    assert payload["facts"][0]["rows"] == [list(row) for row in fixture.entry.rows]
    assert payload["search"]["row_limit"] == 512 and payload["search"]["fact_limit"] == 8
    inherited = rebind_prior_sql_facts(payload=payload, artifact_id=str(uuid7()), task_frame=fixture.frame)
    assert len(inherited) == 1 and len(inherited[0].rows) == 2
    bundle = build_sql_evidence_bundle(task_frame=fixture.frame, database_type="POSTGRESQL", evidence=inherited)
    assert not bundle["subject_resolution"]["requests_complete"]
    assert all(request["status"] == "UNOBSERVED" for request in bundle["requests"])


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_each_independent_graph_and_diagnosis_survives_answer_and_report_freeze(engine):
    frame, facts = comparison_fixture(engine)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type=engine, evidence=facts)
    draft = comparison_draft(bundle, facts)
    model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(
        output=diagnostic_response(draft), receipt=SimpleNamespace(model_dump=lambda **_: {}))))
    context = answer_context(frame, facts, engine)
    result = asyncio.run(DbaAnswerComposeHandler(model_client=model,
        prompts=SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="诊断", ref=lambda: {})))).execute(context))
    assert result.status == "COMPLETED" and result.investigation_record is not None
    record = result.investigation_record
    assert record.sql_object_graph is None and len(record.sql_subject_graphs) == 2
    assert len(record.claims) == 3
    source = investigation_report_source(record)
    assert source["investigation_record"]["sql_subject_graphs"] == [graph.model_dump(mode="json") for graph in record.sql_subject_graphs]
    for request in frame["subject_ref"]["subjects"]:
        identity = next(request[key] for key in ("query_id", "sql_id", "digest") if key in request)
        assert str(identity).casefold() in record.claims[-1].subject.casefold()
    assert any(block.payload.get("result_kind") == "SQL_SUBJECT_COMPARISON" for block in result.blocks)
    assert not any(block.payload.get("result_kind") == "TOP_SQL_LIST" for block in result.blocks)
    assert model.generate_structured.await_count == 1


def test_missing_request_does_not_discard_partial_diagnosis_or_enter_global_clarification():
    frame, facts = comparison_fixture()
    from tests.unit.aiops_agent.test_aiops_turn_answer import _tool_artifact
    artifact = _tool_artifact(semantics="CUMULATIVE_SINCE_LOAD", row_count=1, tool_id=facts[0].tool_id)
    raw = artifact["payload"]["tool_outcomes"][0]["observation"]
    raw.update(db_type="POSTGRESQL", db_version="16",
        columns=[{**column, "sensitivity": "PUBLIC"} for column in facts[0].columns],
        rows=facts[0].rows, captured_at=facts[0].captured_at)
    context = SimpleNamespace(input_artifacts=(artifact,), trigger_type="MANUAL",
        plan_snapshot={"answer_context": {"task_frame": frame}, "capability_snapshot": {"database_type": "POSTGRESQL"}})
    result = asyncio.run(DbaEvidenceAssessmentHandler(model_client=None, prompts=None).execute(context))
    assert result.status == "PARTIAL" and any(gap.code == "SQL_COMPARISON_INCOMPLETE" for gap in result.gaps)
    assert any(fact.rows for fact in result.evidence)


def test_no_model_returns_honest_per_sql_summary_for_stream_and_nonstream():
    frame, facts = comparison_fixture()
    context = answer_context(frame, facts, "POSTGRESQL")
    handler = DbaAnswerComposeHandler(model_client=None, prompts=None)
    result = asyncio.run(handler.execute(context))
    async def collect():
        return [item async for item in handler.execute_stream(context)]
    streamed = asyncio.run(collect())[-1]
    assert result.status == streamed.status == "PARTIAL"
    assert result.blocks == streamed.blocks
    text = result.blocks[0].payload["markdown"]
    assert "执行次数 100" in text and "执行次数 200" in text
    assert "\\n" not in text
    assert result.investigation_record is None


def test_shared_bundle_limits_context_projection_without_multiplying_by_four_requests():
    frame, facts = comparison_fixture()
    names = ("query_id", "database_oid", "user_oid", "top_level", "execution_count")
    rows = tuple((str(index), 16510, 10, True, index) for index in range(100))
    context = observation(names=names, rows=rows, tool_id="db.postgresql.activity.sample")
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=(*facts, context))
    assert len(bundle["observations"]) == 3
    assert all(len(item["records"]) <= 40 for item in bundle["observations"])



def test_comparison_plans_both_sqls_through_existing_worker_pipeline_and_public_view():
    from aiops_agent.diagnostics.registry import DiagnosticRegistry
    from aiops_agent.playbooks import PlaybookRegistry
    from aiops_agent.ports.model import StructuredModelResult
    from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
    from platform_core.contracts.aiops.conversation import TurnTaskFrameView
    from platform_core.contracts.aiops.investigation import CompactPlanningOutput
    from tests.unit.aiops_agent.test_aiops_dba_playbook_planning import _PlanningUow, _CompactSingleSqlReasoner, _AgentCatalog

    class Reasoner(_CompactSingleSqlReasoner):
        async def plan_compact(self, **kwargs):
            routed = await super().plan_compact(**kwargs)
            raw = routed.output.model_dump(mode="python")
            raw.update(diagnostic_profile="SQL_COMPARISON", objectives=["DIAGNOSE", "COMPARE"],
                subject_ref={"subjects": [{"subject_type": "ORACLE_SQL_ID", "sql_id": value}
                    for value in ("6TJX7SU0Q5TTJ", "3J2DSGRCQR6M4")]},
                completion_requirements=[{"requirement_id": "r1", "description": "逐条读取两条 SQL",
                    "accepted_tool_ids": ["db.sql.cursor_details"], "minimum_successful_results": 2}])
            return StructuredModelResult(output=CompactPlanningOutput.model_validate(raw), receipt=routed.receipt)

        async def plan(self, **kwargs):
            planned = await super().plan(**kwargs)
            left = planned.output.plan.actions[0]
            right = left.model_copy(update={"action_id": "a2", "input": {"sql_id": "3j2dsgrcqr6m4"}})
            return StructuredModelResult(output=planned.output.model_copy(update={
                "plan": planned.output.plan.model_copy(update={"actions": (left, right)})}), receipt=planned.receipt)

    uow = _PlanningUow()
    uow.target.capabilities_json["capabilities"] = ["DB_READONLY", "dynamic_performance_views", "oracle_dynamic_readonly"]
    uow.target.capabilities_json["privileges"] = ["V_$INSTANCE", "V_$DATABASE", "GV_$SQL"]
    question = "比较 SQL 6tjx7su0q5ttj 和 3j2dsgrcqr6m4 的资源消耗并提出措施"
    uow.message.payload_json = {"text": question, "content": [{"content_type": "TEXT", "text": question}]}
    diagnostics = DiagnosticRegistry.load()
    playbooks = PlaybookRegistry.load(allowed_tools=frozenset(
        (item.definition.tool_id, item.definition.version) for item in diagnostics.tools))
    service = TurnPlanningService(uow_factory=lambda: uow, investigation_reasoner=Reasoner(),
        playbook_registry=playbooks, task_compiler=InvestigationTaskCompiler(playbooks),
        tool_snapshot_builder=ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=diagnostics),
        agent_catalog=_AgentCatalog())
    result = asyncio.run(service.execute({"domain_id": str(uow.turn.domain_id), "turn_id": str(uow.turn.turn_id)}))
    assert result["status"] == "COLLECTING"
    snapshot = uow.run.plan_snapshot_json
    assert snapshot["planning_route"]["mode"] == "FULL_INVESTIGATION"
    frame = snapshot["answer_context"]["task_frame"]
    assert frame["diagnostic_profile"] == "SQL_COMPARISON"
    assert [item["sql_id"] for item in frame["subject_ref"]["subjects"]] == ["6tjx7su0q5ttj", "3j2dsgrcqr6m4"]
    assert TurnTaskFrameView.model_validate({key: frame[key] for key in ("diagnostic_profile", "subject_ref")})
    assert [item.tool_id for item in uow.tool_invocations] == ["db.instance.identity", "db.sql.cursor_details", "db.sql.cursor_details"]


@pytest.mark.parametrize("problem", ["missing_compare", "execute", "one_request"])
def test_public_task_frame_cannot_redefine_comparison_as_change_or_single_lookup(problem):
    from pydantic import ValidationError
    from platform_core.contracts.aiops.investigation import TaskFrame
    frame, _ = comparison_fixture()
    frame.update(problem_statement="比较两条 SQL", success_criteria=["比较并给出措施"])
    if problem == "missing_compare":
        frame["objectives"] = ["DIAGNOSE"]
    elif problem == "execute":
        frame["action_intent"] = "EXECUTE"
    else:
        frame["subject_ref"]["subjects"] = frame["subject_ref"]["subjects"][:1]
    with pytest.raises(ValidationError):
        TaskFrame.model_validate(frame)


def test_resolution_collects_subjects_and_text_conflicts_across_separate_results():
    frame, combined = scenario("POSTGRESQL")
    left = observation(names=tuple(column["name"] for column in combined.columns),
        rows=(combined.rows[0],), tool_id=combined.tool_id)
    right = observation(names=tuple(column["name"] for column in combined.columns),
        rows=(combined.rows[1],), tool_id=combined.tool_id)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=(left, right))
    assert bundle["subject_resolution"]["status"] == "COMPARISON"
    assert len(bundle["subjects"]) == 2
    assert {item["evidence_ref"] for item in bundle["subjects"]} == {left.evidence_ref, right.evidence_ref}
    left = left.model_copy(update={"columns": (*left.columns, {"name": "normalized_statement"}), "rows": ((*left.rows[0], "SELECT 1"),)})
    collision = observation(names=tuple(column["name"] for column in left.columns),
        rows=((*combined.rows[0], "SELECT 2"),), tool_id=left.tool_id)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=(left, collision))
    assert bundle["subject_resolution"]["identity_conflict"]
    assert bundle["subject_resolution"]["status"] == "AMBIGUOUS"


def test_comparison_subject_limit_is_shared_and_removed_subjects_cannot_supply_direct_evidence():
    frame, _ = comparison_fixture()
    for subject in frame["subject_ref"]["subjects"]:
        subject.pop("user_oid")
    names = ("query_id", "database_oid", "user_oid", "top_level", "execution_count")
    rows = tuple((subject["query_id"], 16510, user, True, user) for subject in frame["subject_ref"]["subjects"] for user in range(1, 21))
    fact = observation(names=names, rows=rows)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=(fact,))
    assert len(bundle["subjects"]) == 20
    assert bundle["subject_resolution"]["observed_candidate_count"] == 40
    assert bundle["subject_resolution"]["truncated"] and not bundle["subject_resolution"]["requests_complete"]
    assert all(row["subject_ids"] for item in bundle["observations"] for row in item["records"] if row["role"] == "SUBJECT")


def test_comparing_only_each_identifiers_internal_scopes_is_not_cross_sql_comparison():
    frame, facts = comparison_fixture()
    for subject in frame["subject_ref"]["subjects"]:
        subject.pop("user_oid")
    left, right = facts
    left = left.model_copy(update={"rows": (left.rows[0], (*left.rows[0][:2], 30, *left.rows[0][3:])), "row_count": 2})
    names = [column["name"] for column in right.columns]
    right_row = dict(zip(names, right.rows[1], strict=True))
    right = right.model_copy(update={"rows": (right.rows[1], tuple(({**right_row, "user_oid": 40})[name] for name in names)), "row_count": 2})
    facts = (left, right)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=facts)
    draft = comparison_draft(bundle, facts)
    individual = draft.conclusions[:-1]
    internal = []
    for index, request in enumerate(bundle["requests"], start=5):
        members = [item for item in individual if item.subject_ids[0] in request["subject_ids"]]
        internal.append(members[0].model_copy(update={"conclusion_id": f"c{index}", "scope": "COMPARISON",
            "subject_ids": tuple(item.subject_ids[0] for item in members),
            "evidence_fields": tuple(item.evidence_fields[0] for item in members)}))
    invalid = draft.model_copy(update={"conclusions": (*individual, *internal)})
    with pytest.raises(ValueError, match="跨 SQL"):
        validate_sql_diagnosis(invalid, bundle)


def test_unknown_target_engine_cannot_infer_execution_scope_from_sql_identifiers():
    frame, facts = comparison_fixture()
    assert build_sql_evidence_bundle(task_frame=frame, database_type="", evidence=facts) is None
    assert build_sql_evidence_answer(task_frame=frame, database_type="", evidence=facts) is None


def test_zero_query_id_is_not_dropped_and_negative_zero_cannot_bypass_overlap_guard():
    subject = {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": 0, "database_oid": 1, "user_oid": 2, "top_level": True}
    other = {**subject, "query_id": "1"}
    normalized = normalize_sql_comparison({"subjects": [subject, other]}, "POSTGRESQL")
    assert normalized[0]["query_id"] == "0"
    assert normalize_sql_comparison({"subjects": [subject, {**subject, "query_id": "-0"}]}, "POSTGRESQL") is None
    from aiops_agent.application.diagnosis.sql_evidence_answer import _matches
    assert _matches({"query_id": 0, "database_oid": 1, "user_oid": 2, "top_level": True}, normalized[0], "query_id")


def test_four_requested_sqls_can_all_be_diagnosed_and_compared_with_one_shared_bundle():
    frame, facts = comparison_fixture()
    entries = list(facts)
    left = facts[0]
    for key in ("101", "102"):
        frame["subject_ref"]["subjects"].append({**frame["subject_ref"]["subjects"][0], "query_id": key})
        entries.append(observation(names=tuple(column["name"] for column in left.columns),
            rows=((key, *left.rows[0][1:]),), tool_id=left.tool_id, measurement_semantics=MeasurementSemantics.CUMULATIVE_SINCE_LOAD))
    facts = tuple(entries)
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="POSTGRESQL", evidence=facts)
    assert len(bundle["requests"]) == len(bundle["subjects"]) == 4
    assert bundle["subject_resolution"]["requests_complete"]
    validate_sql_diagnosis(comparison_draft(bundle, facts), bundle)
