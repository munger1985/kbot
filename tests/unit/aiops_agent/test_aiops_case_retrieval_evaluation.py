"""案例检索评分器的离线合同；所有合成响应均明确标记为 OFFLINE_CONTRACT。"""

import asyncio
import hashlib
import json
from unittest.mock import patch
from uuid import UUID

import pytest

from tests.evaluation.evaluate_aiops_case_retrieval import (
    DATASET, _NoRedirect, evaluate, run_live, validate_dataset, validate_manifest,
)


def identity(number):
    return str(UUID(int=number))


def fixtures():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    assets = {}
    for index, (key, spec) in enumerate(dataset["asset_specs"].items(), 1):
        assets[key] = {"asset_version_id": identity(index * 10), "bundle_revision_id": identity(index * 10 + 1),
            "document_version_id": identity(index * 10 + 2), "lineage_id": "source:" + key,
            "body_sha256": hashlib.sha256(key.encode()).hexdigest()}
    assets["pg_lock_copy"]["lineage_id"] = assets["pg_lock"]["lineage_id"]
    bindings = {case["id"]: {"agent_id": identity(1000), "target_id": identity(1001),
        "context_fingerprint": "a" * 64} for case in dataset["cases"]}
    return dataset, {"schema_version": "OPS_CASE_RETRIEVAL_FIXTURES.v1", "assets": assets, "case_bindings": bindings}


def row_for(key, case, dataset, manifest):
    spec, asset = dataset["asset_specs"][key], manifest["assets"][key]
    citation = {"quote": "来自固定测试原文的引用", "bundle_revision_id": asset["bundle_revision_id"],
        "document_version_id": asset["document_version_id"], "locator": {}}
    row = {"asset_version_id": asset["asset_version_id"], "asset_kind": spec["kind"],
        "authority": spec["case_kind"], "citation_pack": [citation]}
    if spec["kind"] == "DIAGNOSIS_CASE":
        units, offset = [], 0
        for kind in ["PROBLEM", "CONCLUSION", "BOUNDARIES", "VERIFICATION", "UNRESOLVED", "ACTION_EFFECT"]:
            quote = "合成知识单元：" + kind
            units.append({"unit_id": kind.lower(), "kind": kind, "start": offset, "end": offset + len(quote),
                "quote": quote, "content_hash": hashlib.sha256(quote.encode()).hexdigest(), "matched": kind == "ACTION_EFFECT"})
            offset += len(quote) + 2
        primary = case.get("equivalent_assets", {}).get(key, key)
        allowed = primary in case["action_allowed"]
        decision = {"schema_version": "OPS_CASE_REUSE_DECISION.v1",
            "context_fingerprint": manifest["case_bindings"][case["id"]]["context_fingerprint"],
            "status": case.get("expected_decisions", {}).get(primary, case["expected_case_decision"]),
            "execution_authorized": False, "historical_knowledge_is_current_evidence": False,
            "requires_current_verification": True, "action_recommendation_allowed": allowed,
            "action_use": "WARNING_ONLY" if spec["case_kind"] == "NEGATIVE_EXPERIENCE" else (
                "REFERENCE_REQUIRES_LOCAL_VALIDATION" if allowed else "REFERENCE_ONLY_NO_ACTION_RECOMMENDATION"),
            "actions": [{"action_ordinal": 1, "action_recommendation_allowed": allowed, "execution_authorized": False,
                "controlled_effect_linked": allowed, "effect_result": "IMPROVED" if allowed else "NOT_VERIFIED"}]}
        citation["locator"].update(case_context={"schema_version": "OPS_CASE_CONTEXT.v1",
            "role": "HISTORICAL_KNOWLEDGE_NOT_CURRENT_EVIDENCE", "body_sha256": asset["body_sha256"],
            "bundle_revision_id": asset["bundle_revision_id"], "document_version_id": asset["document_version_id"],
            "units": units}, case_applicability=decision)
        row.update(applicability_decision=decision, reuse_context={"applicability_decision": decision})
    return row


def records_for(dataset, manifest):
    return [{"case_id": case["id"], "evidence_kind": "OFFLINE_CONTRACT",
        "response": {"status": "READY", "coverage": {"registry_scope_complete": True},
            "results": [row_for(key, case, dataset, manifest) for key, grade in
                sorted(case["relevance"].items(), key=lambda item: -item[1]) if grade > 0]}}
        for case in dataset["cases"]]


def result_for(result, case_id):
    return next(case for case in result["cases"] if case["case_id"] == case_id)


def test_full_dataset_and_synthetic_contract_never_claim_real_dependency_acceptance():
    dataset, manifest = fixtures()
    assert validate_dataset(dataset) == 20
    result = evaluate(dataset, records_for(dataset, manifest), manifest)
    assert result["passed"] and result["case_count"] == 20
    assert result["evidence_kind"] == "OFFLINE_OR_MIXED_NOT_LIVE_ACCEPTANCE"
    assert not result["live_requests_performed"] and not result["production_acceptance_proven"]


def test_a_source_label_in_export_cannot_prove_live_requests_were_performed():
    dataset, manifest = fixtures()
    records = records_for(dataset, manifest)
    for record in records:
        record["evidence_kind"] = "LIVE_DEPENDENCIES"
    result = evaluate(dataset, records, manifest)
    assert result["evidence_kind"] == "DECLARED_LIVE_EXPORT_NOT_INDEPENDENTLY_VERIFIED"
    assert not result["live_requests_performed"] and not result["production_acceptance_proven"]


def test_scores_actual_application_response_with_model_and_kc_substitutes(monkeypatch):
    from tests.unit.aiops_agent.test_aiops_case_applicability import search_fixture

    async def run():
        fixture = search_fixture(monkeypatch)
        response = await fixture.service.search(**fixture.kwargs)
        dataset, manifest = fixtures()
        case = next(case for case in dataset["cases"] if case["id"] == "postgres_context")
        row = response["results"][0]
        citation = row["citation_pack"][0]
        expected = manifest["assets"]["pg_lock"]
        expected.update(asset_version_id=row["asset_version_id"],
            bundle_revision_id=citation["bundle_revision_id"],
            document_version_id=citation["document_version_id"],
            body_sha256=citation["locator"]["case_context"]["body_sha256"])
        manifest["case_bindings"][case["id"]]["context_fingerprint"] = row["applicability_decision"]["context_fingerprint"]
        records = records_for(dataset, manifest)
        next(record for record in records if record["case_id"] == case["id"])["response"] = response
        assert evaluate(dataset, records, manifest)["passed"]

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["missing", "extra", "duplicate", "no_provenance"])
def test_exact_response_coverage_is_required_and_cannot_skip_missing_cases(failure):
    dataset, manifest = fixtures()
    records = records_for(dataset, manifest)
    if failure == "missing":
        records.pop()
    elif failure == "extra":
        records.append({**records[0], "case_id": "unexpected"})
    elif failure == "duplicate":
        records.append(records[0])
    else:
        records[0].pop("evidence_kind")
    with pytest.raises(ValueError):
        evaluate(dataset, records, manifest)


@pytest.mark.parametrize("failure", ["asset_missing", "duplicate_version", "revision_missing",
    "body_hash", "lineage", "binding_missing", "context", "agent", "alias_lineage"])
def test_incomplete_or_ambiguous_manifest_is_rejected_before_requests(failure):
    dataset, manifest = fixtures()
    if failure == "asset_missing":
        manifest["assets"].pop("draft")
    elif failure == "duplicate_version":
        manifest["assets"]["draft"]["asset_version_id"] = manifest["assets"]["pg_lock"]["asset_version_id"]
    elif failure == "revision_missing":
        manifest["assets"]["draft"].pop("bundle_revision_id")
    elif failure == "body_hash":
        manifest["assets"]["pg_lock"]["body_sha256"] = "invalid"
    elif failure == "lineage":
        manifest["assets"]["pg_lock"].pop("lineage_id")
    elif failure == "binding_missing":
        manifest["case_bindings"].pop("oracle_context")
    elif failure == "context":
        manifest["case_bindings"]["oracle_context"]["context_fingerprint"] = "invalid"
    elif failure == "agent":
        manifest["case_bindings"]["oracle_context"]["agent_id"] = "unknown"
    else:
        manifest["assets"]["pg_lock_copy"]["lineage_id"] = "different-source"
    with pytest.raises(ValueError):
        validate_manifest(dataset, manifest)


@pytest.mark.parametrize("failure,expected", [
    ("old_revision", "WRONG_EXACT_REVISION"), ("locator_conflict", "WRONG_EXACT_REVISION"),
    ("old_document", "WRONG_CONTENT_DOCUMENT"), ("body_hash", "CASE_BODY_HASH_OR_CLOSURE_MISSING"),
    ("closure", "CASE_CONDITION_EFFECT_CLOSURE_MISSING"), ("unit_hash", "CASE_UNIT_QUOTE_OR_HASH_INVALID"),
    ("context", "WRONG_CURRENT_CONTEXT"), ("current_evidence", "UNSAFE_EXECUTION_OR_CURRENT_FACT"),
    ("execute", "UNSAFE_EXECUTION_OR_CURRENT_FACT"), ("per_action_execute", "INVALID_ACTION_GATE"),
    ("effect_unlinked", "UNVERIFIED_ACTION_PROMOTED"), ("scope", "INCOMPLETE_REGISTRY_SCOPE"),
    ("source_kind", "SOURCE_KIND_OR_AUTHORITY_CHANGED"), ("quote", "MISSING_QUOTE"),
    ("decision_missing", "MISSING_VERSIONED_DECISION"), ("diverged", "CITATION_REUSE_CONTEXT_DIVERGED"),
])
def test_unsafe_or_inexact_response_fails_even_with_perfect_recall(failure, expected):
    dataset, manifest = fixtures()
    records = records_for(dataset, manifest)
    record = records[0]
    row = record["response"]["results"][0]
    citation = row["citation_pack"][0]
    context = citation["locator"]["case_context"]
    decision = citation["locator"]["case_applicability"]
    if failure == "old_revision":
        citation["bundle_revision_id"] = identity(999)
    elif failure == "locator_conflict":
        citation["locator"]["bundle_revision_id"] = identity(999)
    elif failure == "old_document":
        citation["document_version_id"] = identity(999)
    elif failure == "body_hash":
        context["body_sha256"] = "b" * 64
    elif failure == "closure":
        context["units"] = []
    elif failure == "unit_hash":
        context["units"][0]["quote"] = "被改写的原文"
    elif failure == "context":
        decision["context_fingerprint"] = "b" * 64
    elif failure == "current_evidence":
        decision["historical_knowledge_is_current_evidence"] = True
    elif failure == "execute":
        decision["execution_authorized"] = True
    elif failure == "per_action_execute":
        decision["actions"][0]["execution_authorized"] = True
    elif failure == "effect_unlinked":
        decision["actions"][0]["controlled_effect_linked"] = False
    elif failure == "scope":
        record["response"]["coverage"]["registry_scope_complete"] = False
    elif failure == "source_kind":
        row["authority"] = "DIAGNOSTIC_REFERENCE"
    elif failure == "quote":
        citation["quote"] = ""
    elif failure == "decision_missing":
        decision.pop("schema_version")
    else:
        row["applicability_decision"] = {}
    result = result_for(evaluate(dataset, records, manifest), record["case_id"])
    assert not result["passed"] and expected in result["failures"]
    assert result["recall"] == 1


def test_equivalent_single_copy_can_match_gold_but_two_same_lineage_results_fail():
    dataset, manifest = fixtures()
    records = records_for(dataset, manifest)
    record = next(row for row in records if row["case_id"] == "shared_lineage")
    case = next(case for case in dataset["cases"] if case["id"] == "shared_lineage")
    copy = row_for("pg_lock_copy", case, dataset, manifest)
    record["response"]["results"] = [copy]
    assert evaluate(dataset, records, manifest)["passed"]
    record["response"]["results"].append(row_for("pg_lock", case, dataset, manifest))
    result = result_for(evaluate(dataset, records, manifest), "shared_lineage")
    assert "DUPLICATE_SOURCE_LINEAGE" in result["failures"] and result["ndcg"] <= 1


@pytest.mark.parametrize("asset", ["draft", "retired", "foreign_domain", "source_revoked", "old_revision"])
def test_any_unpublished_or_out_of_scope_hit_fails(asset):
    dataset, manifest = fixtures()
    records = records_for(dataset, manifest)
    records[0]["response"]["results"].append(row_for(asset, dataset["cases"][0], dataset, manifest))
    result = result_for(evaluate(dataset, records, manifest), records[0]["case_id"])
    assert "FORBIDDEN_ASSET" in result["failures"]


def test_unknown_hits_keep_their_rank_instead_of_inflating_ndcg():
    dataset, manifest = fixtures()
    records = records_for(dataset, manifest)
    records[0]["response"]["results"].insert(0, {"asset_version_id": identity(999)})
    result = result_for(evaluate(dataset, records, manifest), records[0]["case_id"])
    assert result["recall"] == 1 and result["ndcg"] < 0.8
    assert "RANKING_BELOW_GATE" in result["failures"]


@pytest.mark.parametrize("endpoint", ["https://user:secret@example.invalid/operations-knowledge/search-preview",
    "https://example.invalid/operations-knowledge/search-preview?key=secret",
    "https://example.invalid/operations-knowledge/search-preview#fragment", "file:///tmp/fixture.json",
    "https://example.invalid/initialize"])
def test_live_runner_rejects_credential_urls_or_non_search_endpoints(endpoint):
    dataset, manifest = fixtures()
    with patch("tests.evaluation.evaluate_aiops_case_retrieval.build_opener") as opener:
        with pytest.raises(ValueError):
            run_live(dataset, manifest, endpoint=endpoint, headers={"Authorization": "test-not-a-secret"})
        opener.assert_not_called()


def test_redirect_is_never_followed_with_authentication_headers():
    with pytest.raises(ValueError):
        _NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.invalid/")
