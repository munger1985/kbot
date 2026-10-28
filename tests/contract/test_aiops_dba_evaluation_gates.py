"""评测器必须按真实公共合同运行，关键失败不允许用平均分掩盖。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.evaluation import evaluate_aiops_dba_chat as evaluation
from platform_core.identity import uuid7
from platform_core.contracts.aiops.conversation import ConversationStart


def test_deterministic_only_is_not_business_release_approval():
    assert evaluation.release_blockers([{"name": "terminal_status", "passed": True}], None) == ["SEMANTIC_REVIEW_NOT_RUN"]


def test_high_judge_score_cannot_mask_forbidden_or_scope_failures():
    semantic = {"score": 100, "passed_behaviors": [], "failed_behaviors": [],
        "violated_forbidden_behaviors": ["CUMULATIVE_AS_WINDOW"], "reasons": []}
    blockers = evaluation.release_blockers([{"name": "resource_scope", "passed": False}], semantic)
    assert "resource_scope" in blockers
    assert "FORBIDDEN:CUMULATIVE_AS_WINDOW" in blockers


def test_missing_semantic_fields_cannot_silently_pass():
    assert "SEMANTIC_REVIEW_INVALID" in evaluation.release_blockers([], {"score": 100})


def test_each_database_has_at_least_eight_explicit_fixture_scenarios():
    dataset = evaluation.load_dataset()
    assert evaluation.validate_dataset(dataset) == []
    for kind in evaluation.DATABASE_TYPES:
        assert len([case for case in dataset["cases"] if kind in case["database_types"]]) >= 8
    assert all(case["fixture_requirements"] for case in dataset["cases"])


@pytest.mark.parametrize("prefix", ["/api/v1/apps/aiops", "/api/v1/apps/aiops/domains/demo/ops"])
def test_current_create_payload_uses_resource_hint_not_target_and_runtime_filters_database(monkeypatch, prefix):
    calls = []
    async def request(_session, method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST":
            return {"conversation_id": "conversation", "turn_id": "turn"}, {}
        return {"status": "COMPLETED", "answer_blocks": []}, {}
    monkeypatch.setattr(evaluation, "_request_json", request)
    monkeypatch.setattr(evaluation, "_stream_events", AsyncMock(return_value=[]))
    case = {"case_id": "postgresql-top", "question": "Top SQL", "materials": [],
        "expected_behaviors": ["QUERY_TOP_SQL"], "forbidden_behaviors": ["INVENT_SQL_TEXT"]}
    config = evaluation.RuntimeConfig(base_url="https://example.invalid", api_key="test-only",
        agent_id=str(uuid7()), resource_id=str(uuid7()), database_type="POSTGRESQL",
        timeout_seconds=1, source_run_id=None, operations_logs_url=None, api_path_prefix=prefix)
    result = asyncio.run(evaluation._execute_case(object(), config, case, None))
    payload = calls[0][2]["payload"]
    assert "target_id" not in payload
    validated = ConversationStart.model_validate(payload)
    assert calls[0][1] == "https://example.invalid" + prefix + "/conversations"
    assert str(validated.conversation.agent_id) == config.agent_id
    assert payload["conversation"]["context_hint"] == {"source_kind": "RESOURCE_PAGE", "resource_ids": [config.resource_id]}
    assert "SEMANTIC_REVIEW_NOT_RUN" in result["release_blockers"]
    assert "INDEPENDENT_FIXTURE_TRUTH_MISSING" in result["release_blockers"]


def test_source_run_is_in_first_turn_not_conversation_envelope(monkeypatch):
    calls = []
    async def request(_session, method, url, **kwargs):
        calls.append(kwargs)
        if method == "POST":
            return {"conversation_id": "conversation", "turn_id": "turn"}, {}
        return {"status": "COMPLETED", "answer_blocks": []}, {}
    monkeypatch.setattr(evaluation, "_request_json", request)
    monkeypatch.setattr(evaluation, "_stream_events", AsyncMock(return_value=[]))
    run_id = str(uuid7())
    config = evaluation.RuntimeConfig(base_url="https://example.invalid", api_key="test-only",
        agent_id=str(uuid7()), resource_id=str(uuid7()), database_type="POSTGRESQL",
        timeout_seconds=1, source_run_id=run_id, operations_logs_url=None)
    asyncio.run(evaluation._execute_case(object(), config, {
        "case_id": "follow-up", "question": "继续调查", "requires_source_run": True,
        "expected_behaviors": ["INHERIT_SOURCE_RUN_EVIDENCE"], "forbidden_behaviors": [],
    }, None))
    validated = ConversationStart.model_validate(calls[0]["payload"])
    assert str(validated.first_turn.source_run_id) == run_id
    assert "source_run_id" not in calls[0]["payload"]["conversation"]

def test_every_readonly_scenario_blocks_unrequested_execution_even_without_specific_forbidden_label():
    case = {"expected_behaviors": ["QUERY_TOP_SQL"], "forbidden_behaviors": ["INVENT_SQL_TEXT"]}
    turn = {"status": "COMPLETED", "scope_resolution": {"status": "RESOLVED", "resources": [{"resource_id": "resource"}]}}
    checks = evaluation._deterministic_checks(case, turn,
        [{"event_type": "execution.started", "payload": {}}], resource_id="resource")
    assert next(value for value in checks if value["name"] == "no_unapproved_execution")["passed"] is False
