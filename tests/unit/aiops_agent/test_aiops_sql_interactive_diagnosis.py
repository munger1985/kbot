"""三库交互诊断要求真实计划解释，而非完整报告或节点串联。"""

import asyncio
from copy import deepcopy
import hashlib
import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from aiops_agent.adapters.model_serving import AIOpsStructuredModelClient
from platform_core.identity import uuid7

from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.diagnosis.sql_review import (
    _reference_projection, diagnostic_response_model, expand_diagnostic_response, plan_review_tasks,
)
from tests.support.aiops_sql_review import diagnostic_response
from tests.unit.aiops_agent.test_aiops_oracle_cursor_plan import direct, fact, observation
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import diagnosis
from tests.unit.aiops_agent.test_aiops_sql_plan_import import SUBJECTS, anchor, upload
from tests.unit.aiops_agent.test_aiops_sql_plan_priority import frame


def native_scenario():
    entry, plan = direct(), fact(observation())
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE", evidence=(entry, plan))
    response = diagnostic_response(diagnosis(entry), plan)
    return entry, plan, bundle, response


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
def test_three_engines_keep_plan_mechanism_in_claims_for_answer_report_and_case(engine):
    entry, (_, plan) = anchor(engine), upload(engine)
    subject = {"subject_type": {"ORACLE": "ORACLE_SQL_ID", "POSTGRESQL": "POSTGRESQL_QUERY_ID", "MYSQL": "MYSQL_DIGEST"}[engine], **SUBJECTS[engine]}
    bundle = build_sql_evidence_bundle(task_frame=frame(engine, subject), database_type=engine, evidence=(entry, plan))
    original = diagnosis(entry)
    hypothesis = original.conclusions[0].model_copy(update={"conclusion_id": "c2", "level": "HYPOTHESIS",
        "statement": "人工计划提示该访问机制需要结合生产来源继续核验。"})
    response = diagnostic_response(original.model_copy(update={"conclusions": (*original.conclusions, hypothesis)}), plan)
    data = response.model_dump(mode="json")
    data["plan_reviews"][0]["conclusion_id"] = "c2"
    restored = expand_diagnostic_response(diagnostic_response_model(bundle).model_validate(data), bundle, {})
    assert "计划节点 1" in restored.conclusions[1].reasoning
    assert restored.conclusions[1].level == "HYPOTHESIS"
    assert any(ref.evidence_ref == plan.evidence_ref and ref.field == "operation" for ref in restored.conclusions[1].evidence_fields)
    assert plan.captured_at in restored.conclusions[1].applicable_time


def test_matched_plan_cannot_be_ignored_by_successful_diagnostic_response():
    _, _, bundle, response = native_scenario()
    data = response.model_dump(mode="json")
    data["plan_reviews"] = []
    with pytest.raises(ValidationError):
        diagnostic_response_model(bundle).model_validate(data)
    with pytest.raises(ValueError, match="必须交付"):
        expand_diagnostic_response(response.model_copy(update={"plan_reviews": ()}), bundle, {})


@pytest.mark.parametrize("tamper", ["other_plan", "unknown_node", "duplicate_node", "unknown_conclusion"])
def test_plan_mechanism_cannot_bind_other_source_or_nonexistent_node(tamper):
    _, _, bundle, response = native_scenario()
    data = response.model_dump(mode="json")
    review = data["plan_reviews"][0]
    if tamper == "other_plan":
        review["evidence_ref"] = "artifact:another#plan"
    elif tamper == "unknown_node":
        review["node_ids"] = [99]
    elif tamper == "duplicate_node":
        review["node_ids"] = [1, 1]
    else:
        review["conclusion_id"] = "c9"
    with pytest.raises(ValueError):
        expand_diagnostic_response(diagnostic_response_model(bundle).model_validate(data), bundle, {})


def test_short_reference_expands_only_into_current_authorized_evidence():
    entry, plan, bundle, response = native_scenario()
    aliases = {entry.evidence_ref: "e1", plan.evidence_ref: "e2"}
    data = _reference_projection(response.model_dump(mode="json"), aliases)
    assert data["findings"][0]["evidence_fields"][0]["evidence_ref"] == "e1"
    restored = expand_diagnostic_response(diagnostic_response_model(bundle).model_validate(data), bundle, aliases)
    assert restored.conclusions[0].evidence_fields[0].evidence_ref == entry.evidence_ref
    data["findings"][0]["evidence_fields"][0]["evidence_ref"] = "e999"
    with pytest.raises(ValueError, match="证据字段"):
        expand_diagnostic_response(diagnostic_response_model(bundle).model_validate(data), bundle, aliases)


def test_reference_aliases_never_modify_sql_values_or_object_names():
    data = {"evidence_ref": "original", "fields": {"sql_text": "original", "object": "original", "evidence_ref": "original"}}
    original = deepcopy(data)
    projected = _reference_projection(data, {"original": "e1"})
    assert projected["fields"] == data["fields"] and data == original
    assert projected["evidence_ref"] == "e1"


def test_plan_review_is_bound_to_raw_operation_field_not_only_summary_text():
    _, plan, bundle, response = native_scenario()
    restored = expand_diagnostic_response(response, bundle, {})
    plan_refs = [ref for ref in restored.conclusions[0].evidence_fields if ref.evidence_ref == plan.evidence_ref]
    assert {ref.field for ref in plan_refs} == {"operation", "options"}
    assert all(ref.row_index == 1 for ref in plan_refs)
    assert len(plan_review_tasks(bundle)[0]["nodes"]) == 2


def test_missing_plan_does_not_block_other_direct_sql_diagnosis():
    entry = direct()
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE", evidence=(entry,))
    response = diagnostic_response(diagnosis(entry))
    restored = expand_diagnostic_response(diagnostic_response_model(bundle).model_validate(response.model_dump()), bundle, {})
    assert restored.verdict == "PROBLEM_FOUND" and not plan_review_tasks(bundle)


def test_single_subject_rejects_report_length_instead_of_requesting_more_output_tokens():
    _, _, bundle, response = native_scenario()
    data = response.model_dump(mode="json")
    data["findings"] = [dict(data["findings"][0], conclusion_id=f"c{index}") for index in range(1, 6)]
    with pytest.raises(ValidationError):
        diagnostic_response_model(bundle).model_validate(data)


def test_related_nodes_must_be_combined_in_one_review_per_finding():
    _, _, bundle, response = native_scenario()
    data = response.model_dump(mode="json")
    data["plan_reviews"].append(dict(data["plan_reviews"][0], node_ids=[0]))
    with pytest.raises(ValidationError, match="每条 finding"):
        diagnostic_response_model(bundle).model_validate(data)


def test_review_bounds_key_nodes_without_truncating_displayed_plan():
    _, _, bundle, response = native_scenario()
    data = response.model_dump(mode="json")
    data["plan_reviews"][0]["node_ids"] = [0, 1, 2, 3, 4]
    with pytest.raises(ValidationError):
        diagnostic_response_model(bundle).model_validate(data)
    assert len(plan_review_tasks(bundle)[0]["nodes"]) == 2


@pytest.mark.parametrize("tamper", ["unknown_field", "goal_conflict", "unapproved_repair"])
def test_formal_evidence_and_goal_errors_are_caught_inside_model_contract(tamper):
    _, _, bundle, response = native_scenario()
    data = response.model_dump(mode="json")
    if tamper == "unknown_field":
        data["findings"][0]["evidence_fields"][0]["field"] = "not_returned"
    elif tamper == "goal_conflict":
        data["goal_satisfied"] = True
        data["goal_assessments"] = [{"goal_id": "answer", "status": "PARTIAL"}]
    else:
        data["recommendations"][0]["requires_approval"] = False
        data["recommendations"][0]["intent"] = "REPAIR"
    with pytest.raises(ValidationError):
        diagnostic_response_model(bundle, {}).model_validate(data)


def test_sparse_out_of_order_native_plan_ids_reference_original_rows():
    entry = direct()
    raw = observation()
    rows = [list(row) for row in raw.rows]
    names = [column.name for column in raw.columns]
    rows[0][names.index("plan_line_id")] = 10
    rows[1][names.index("plan_line_id")] = 30
    rows[1][names.index("parent_id")] = 10
    plan = fact(raw.model_copy(update={"rows": tuple(tuple(row) for row in reversed(rows))}))
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE", evidence=(entry, plan))
    restored = expand_diagnostic_response(diagnostic_response(diagnosis(entry), plan), bundle, {})
    refs = [ref for ref in restored.conclusions[0].evidence_fields if ref.evidence_ref == plan.evidence_ref]
    assert refs and all(ref.row_index == 0 for ref in refs)


def test_invalid_plan_reference_is_repaired_inside_the_same_bounded_model_request():
    from tests.unit.aiops_agent.test_aiops_model_serving import _SequencedSession
    _, _, bundle, response = native_scenario()
    valid = response.model_dump(mode="json")
    invalid = deepcopy(valid)
    invalid["plan_reviews"][0]["node_ids"] = [99]
    session = _SequencedSession(tuple(json.dumps(data) for data in (invalid, valid)))
    client = AIOpsStructuredModelClient(base_url="http://model.invalid", audience="model-serving",
        caller_service="aiops-agent", timeout_seconds=30, session=session)
    prompt = "仅依据现场证据解释计划"
    async def exercise():
        with patch("aiops_agent.adapters.model_serving.build_internal_auth_headers", return_value={}):
            return await client.generate_structured(purpose="diagnosis.test", output_model=diagnostic_response_model(bundle, {}),
                model_snapshot={"technical_name": "test", "revision": "test", "model_id": str(uuid7()), "tenant_id": str(uuid7())},
                prompt_ref={"content": prompt, "prompt_id": "test", "prompt_version": "1.0.0",
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
                input_payload={}, deadline=None, idempotency_key="test-repair")
    result = asyncio.run(exercise())
    assert len(session.payloads) == 2
    assert result.output.plan_reviews[0].node_ids == (1,)
    assert "input_value" not in session.payloads[1]["messages"][-1]["content"]
