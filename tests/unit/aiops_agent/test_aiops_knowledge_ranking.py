"""有界知识业务重排；确定性合同不等于真实 KC／模型检索质量。"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from aiops_agent.application.knowledge_ranking import (
    FACTOR_WEIGHTS, RANKING_POLICY, knowledge_rank_key, knowledge_rank_scorecard,
)
from tests.unit.aiops_agent.test_aiops_case_applicability import (
    case_profile, context, judgments, search_fixture,
)
from aiops_agent.application.case_applicability import case_reuse_decision
from tests.unit.aiops_agent.test_aiops_knowledge_candidate_scope import setup_search


NOW = datetime(2026, 11, 3, tzinfo=UTC)


def manual():
    return {
        "source_evidence_ids": ["source-1"],
        "procedures": [{"intent": "CHANGE", "steps": [{"source_evidence_ids": ["source-1"]}],
            "validations": [{"description": "按原文核对结果"}]}],
    }


def card(*, kind="MANUAL", profile=None, relevance=0.03, age=0, purpose="CHANGE", decision=None):
    created = NOW - timedelta(days=age) if age is not None else None
    return knowledge_rank_scorecard(candidate={"asset_kind": kind, "profile": profile or {},
        "created_at": created}, result={"applicability_decision": decision or {}},
        kc_relevance=relevance, purpose=purpose, now=NOW)


def case_card(*, kind="VERIFIED_RESOLUTION", relevance=0.03, age=0, purpose="CHANGE", changes=None):
    value = case_profile(kind, **(changes or {})).model_dump(mode="json")
    decision = case_reuse_decision(profile=value, context=context(), judgments=judgments(value))
    return card(kind="DIAGNOSIS_CASE", profile=value, relevance=relevance, age=age,
        purpose=purpose, decision=decision)


def test_near_relevance_can_be_reordered_by_grounded_structure_without_new_hits():
    rich = card(profile=manual(), relevance=0.03)
    weak = card(relevance=0.031)
    assert rich["score"] > weak["score"]
    assert rich["factors"]["evidence_structure"] == 1
    assert rich["factors"]["verification_linkage"] == 1
    assert rich["factors"]["purpose_fit"] == 1
    assert rich["factors"]["current_applicability"] == 0
    assert rich["applicability_state"] == "NOT_ASSESSED"


@pytest.mark.parametrize("kind", ["DIAGNOSTIC_REFERENCE", "VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION", "NEGATIVE_EXPERIENCE"])
@pytest.mark.parametrize("purpose", ["DIAGNOSE", "EXPLAIN", "PLAN", "CHANGE", "VERIFY"])
def test_every_case_purpose_has_bounded_explainable_scores_not_execution_authority(kind, purpose):
    value = case_card(kind=kind, purpose=purpose)
    assert value["schema_version"] == RANKING_POLICY
    assert value["kc_relevance"] <= value["score"] <= value["kc_relevance"] * 1.2
    assert 0 <= value["bonus"] <= 0.2
    assert all(0 <= factor <= 1 for factor in value["factors"].values())
    assert sum(FACTOR_WEIGHTS.values()) == pytest.approx(0.2)
    assert value["weights"] == FACTOR_WEIGHTS
    assert not value["execution_authorized"]
    assert not value["historical_knowledge_is_current_evidence"]
    assert not value["quality_empirically_validated"]


def test_older_relevant_case_cannot_be_displaced_by_fresher_irrelevant_material():
    old = case_card(relevance=0.03, age=3650)
    new = card(profile=manual(), relevance=0.02)
    assert old["score"] > new["score"]
    assert old["factors"]["version_freshness"] == pytest.approx(1 / 11)


def test_freshness_is_bounded_version_age_not_publication_or_current_fact():
    now = card(profile=manual())
    older = card(profile=manual(), age=365)
    missing = card(profile=manual(), age=None)
    assert now["score"] > older["score"] > missing["score"]
    assert now["factors"]["version_freshness"] == 1
    assert older["factors"]["version_freshness"] == 0.5
    assert missing["factors"]["version_freshness"] == 0
    assert now["freshness_basis"] == "ASSET_VERSION_CREATED_AT"


@pytest.mark.parametrize("created", [None, "2026-11-03", NOW.replace(tzinfo=None), NOW + timedelta(days=1)])
def test_missing_invalid_naive_or_future_time_gets_no_freshness_credit(created):
    value = knowledge_rank_scorecard(candidate={"asset_kind": "MANUAL", "profile": {},
        "created_at": created}, result={}, kc_relevance=0.03, purpose="DIAGNOSE", now=NOW)
    assert value["factors"]["version_freshness"] == 0
    assert value["version_age_days"] is None


@pytest.mark.parametrize("score", [None, -1, float("nan"), float("inf"), "invalid", 0])
def test_invalid_or_zero_relevance_cannot_gain_relevance_from_business_metadata(score):
    assert card(profile=manual(), relevance=score)["score"] == 0


@pytest.mark.parametrize("status", ["UNKNOWN", "CONFLICT", "PROHIBITED"])
def test_only_actual_current_decision_not_request_matching_gives_applicability_bonus(status):
    value = case_profile().model_dump(mode="json")
    score = card(kind="DIAGNOSIS_CASE", profile=value, decision={"status": status})
    assert score["factors"]["current_applicability"] == 0
    assert score["applicability_state"] == status


def test_absent_target_decision_does_not_infer_match_from_profile_or_empty_filters():
    value = case_profile().model_dump(mode="json")
    score = knowledge_rank_scorecard(candidate={"asset_kind": "DIAGNOSIS_CASE", "profile": value},
        result={"applicability": "MATCHED", "matched_dimensions": ["DATABASE_TYPE"]},
        kc_relevance=0.03, purpose="DIAGNOSE", now=NOW)
    assert score["factors"]["current_applicability"] == 0


def test_negative_measured_lesson_is_not_penalized_for_its_outcome():
    positive = case_card(purpose="VERIFY")
    negative = case_card(kind="NEGATIVE_EXPERIENCE", purpose="VERIFY")
    assert positive["factors"] == negative["factors"]
    assert positive["score"] == negative["score"]
    assert not negative["execution_authorized"]


@pytest.mark.parametrize("change", ["no_controlled", "unlinked", "incomparable", "not_verified", "limited"])
def test_verification_bonus_requires_real_linked_per_action_method_not_success_label(change):
    value = case_profile().model_dump(mode="json")
    if change == "no_controlled":
        value["verification"].pop("controlled_verification_ref")
    elif change == "unlinked":
        value["action_effects"][0]["evidence_refs"] = ["unrelated"]
    elif change == "incomparable":
        value["verification"]["comparability"] = "UNKNOWN"
    elif change == "not_verified":
        value["action_effects"][0]["result"] = "NOT_VERIFIED"
    else:
        value["action_effects"][0]["limitations"] = ["缺少对照"]
    assert card(kind="DIAGNOSIS_CASE", profile=value)["factors"]["verification_linkage"] == 0


def test_manual_source_id_not_in_profile_does_not_fabricate_grounded_steps():
    value = manual()
    value["source_evidence_ids"] = ["another-source"]
    score = card(profile=value)
    assert score["factors"]["evidence_structure"] == 0
    assert score["factors"]["verification_linkage"] == 0
    assert score["factors"]["purpose_fit"] == 0


def test_manual_and_case_have_no_blanket_source_kind_priority():
    historical = case_card(relevance=0.03, purpose="VERIFY")
    specification = card(relevance=0.03, purpose="VERIFY")
    assert historical["score"] > specification["score"]
    assert card(profile=manual(), relevance=0.04)["score"] > historical["score"]


def test_adding_duplicate_steps_does_not_give_a_quantity_bonus():
    value = manual()
    before = card(profile=value)
    value["procedures"][0]["steps"] *= 100
    assert card(profile=value) == before


def test_equal_scores_use_stable_version_not_type_or_citation_count():
    score = card()
    rows = [{"ranking": score, "asset_version_id": key, "asset_kind": kind,
        "citation_pack": list(range(count))} for key, kind, count in [
            ("b", "MANUAL", 100), ("a", "DIAGNOSIS_CASE", 1)]]
    assert [row["asset_version_id"] for row in sorted(rows, key=knowledge_rank_key)] == ["a", "b"]


def test_actual_search_reranks_grounded_body_matches_and_returns_scorecard():
    async def run():
        f = setup_search(size=2, max_results=1)
        weak, rich = f.rows
        rich[1].profile_json = manual()
        for row in f.rows:
            row[1].created_at = NOW

        async def discover(**kwargs):
            assert not f.state.transaction_open
            return {"candidates": [{"collection_id": str(row[2].collection_id),
                "bundle_revision_id": str(row[2].bundle_revision_id), "rrf_score": score}
                for row, score in [(weak, 0.031), (rich, 0.03)]]}

        async def retrieve(**kwargs):
            assert not f.state.transaction_open
            return {"citations": [{"collection_id": str(row[2].collection_id),
                "bundle_revision_id": str(row[2].bundle_revision_id),
                "citation_id": str(row[1].asset_version_id), "quote": "可引用的真实正文"}
                for row in f.rows]}

        f.service._kc.discover.side_effect = discover
        f.service._kc.retrieve_evidence.side_effect = retrieve
        f.kwargs["request"] = f.request.model_copy(update={"purpose": "CHANGE"})
        result = await f.service.search(**f.kwargs)
        assert result["coverage"]["ranking_policy"] == RANKING_POLICY
        assert result["results"][0]["asset_id"] == str(rich[0].asset_id)
        score = result["results"][0]["ranking"]
        assert score["kc_relevance"] == 0.03
        assert score["score"] > 0.031
        assert f.state.rechecks == 1
    asyncio.run(run())


def test_case_search_scorecard_keeps_exact_current_applicability_and_citations(monkeypatch):
    async def run():
        f = search_fixture(monkeypatch)
        response = await f.service.search(**f.kwargs)
        row = response["results"][0]
        assert row["ranking"]["factors"]["current_applicability"] == 1
        assert row["ranking"]["applicability_state"] == row["applicability_decision"]["status"]
        assert row["citation_pack"][0]["locator"]["case_applicability"] == row["applicability_decision"]
    asyncio.run(run())


def test_hard_action_gate_runs_before_any_business_ranking(monkeypatch):
    async def run():
        f = search_fixture(monkeypatch)
        f.target.db_type = "ORACLE"
        f.kwargs["request"] = f.request.model_copy(update={"purpose": "CHANGE"})
        response = await f.service.search(**f.kwargs)
        assert response["results"] == []
        assert any("不满足当前处置推荐条件" in value for value in response["warnings"])
    asyncio.run(run())
