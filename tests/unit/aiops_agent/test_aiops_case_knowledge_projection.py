"""案例知识投影的语义、血缘、脱敏与定位专项；不代表真实 KC 处理验收。"""

import hashlib
from copy import deepcopy

import pytest
from pydantic import ValidationError

from aiops_agent.application.case_knowledge_projection import (
    case_knowledge_markdown, compile_case_knowledge, investigation_knowledge, case_projection_is_current,
    case_context_for_citation,
)
from aiops_agent.application.case_semantic_redaction import (
    SemanticRedactionOutput, apply_semantic_redaction, shared_content_hash,
)
from aiops_agent.application.operations_knowledge import _prepare_case_revision
from aiops_agent.contracts.case_knowledge import CaseActionEffect, CaseKnowledgeManifest
from aiops_agent.contracts.knowledge import DiagnosisCaseDraft, DiagnosisCaseProfile
from aiops_agent.playbooks import canonical_hash
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_report_lifecycle import fixture


def profile(**changes):
    values = {
        "case_kind": "DIAGNOSTIC_REFERENCE",
        "problem_signature": {"normalized_symptoms": ["连接等待，需要区分原因"]},
        "environment": {"database_type": "POSTGRESQL", "database_version": "16"},
        "root_cause": {"summary": "尚未确认根因", "effective_level": "INCONCLUSIVE", "trust_level": "USER_PROVIDED"},
        "actions": ({"summary": "调整调用频率", "source_kind": "USER_PROVIDED", "evidence_refs": ["receipt-1"]},),
        "verification": {"method": "USER_REPORTED", "result": "IMPROVED"},
        "applicability": ("仅限同等负载",),
        "contraindications": ("不适用于主备切换",),
        "source": {"conversation_id": "internal-conversation", "run_ids": ["internal-run"]},
    }
    return DiagnosisCaseProfile.model_validate({**values, **changes})


def source():
    artifact = fixture().artifact
    record = deepcopy(artifact.payload_json["investigation_record"])
    record["claims"][0]["reasoning"] = "隐藏推理文本不进入知识正文"
    record["claims"].append({
        **record["claims"][0], "claim_id": "c2", "kind": "CAUSE", "level": "SUPPORTED",
        "statement": "当前证据支持一种原因，但仍有反证",
        "counterevidence_fields": record["claims"][0]["evidence_fields"],
        "limitations": ["未完成完整业务窗口比较"],
    })
    record["unresolved_questions"] = ["是否存在其他竞争负载"]
    artifact.payload_json["investigation_record"] = record
    artifact.content_hash = canonical_hash(artifact.payload_json)
    return artifact


def test_checks_keep_selected_findings_support_counterevidence_and_limits_without_hidden_reasoning():
    original = source()
    steps, questions = investigation_knowledge(artifact=original, claim_ids={"c1", "c2"})
    assert len(steps) == 2
    assert steps[1].source.counterevidence_refs == steps[0].source.evidence_refs
    assert steps[1].source.artifact_id == str(original.artifact_id)
    assert steps[1].source.content_hash == original.content_hash
    assert steps[1].level == "SUPPORTED" and steps[1].limitations
    assert questions == ("是否存在其他竞争负载",)
    wire = str([step.model_dump() for step in steps])
    assert "隐藏推理" not in wire and "'rows'" not in wire
    assert investigation_knowledge(artifact=original, claim_ids={"c1"})[0] == (steps[0],)


@pytest.mark.parametrize("selection", [{"c3"}, {"c1", "c3"}])
def test_nonexistent_selected_claim_is_not_invented(selection):
    with pytest.raises(ValueError, match="未选择或已失效"):
        investigation_knowledge(artifact=source(), claim_ids=selection)


def test_projection_is_deterministic_and_each_locator_hashes_exact_unicode_slice():
    value = profile(investigation_steps=investigation_knowledge(artifact=source(), claim_ids={"c1", "c2"})[0])
    first = compile_case_knowledge(title="连接等待复盘", profile=value)
    assert first == compile_case_knowledge(title="连接等待复盘", profile=value)
    assert hashlib.sha256(first.body).hexdigest() == first.content_hash
    manifest = CaseKnowledgeManifest.model_validate(first.manifest())
    text = first.body.decode()
    for unit in manifest.units:
        assert hashlib.sha256(text[unit.start:unit.end].encode()).hexdigest() == unit.content_hash
    assert len({unit.unit_id for unit in manifest.units}) == len(manifest.units)
    assert "internal-run" not in text and "internal-conversation" not in text


def test_action_unit_is_self_contained_and_does_not_attribute_overall_improvement_to_action():
    value = profile()
    projection = compile_case_knowledge(title="复盘", profile=value)
    action = next(unit for unit in projection.units if unit.unit_id == "case-action-1")
    body = projection.body.decode()[action.start:action.end]
    for expected in ("人工登记", "整体结果", "逐项效果：未建立", "仅限同等负载", "不适用于主备切换", "不代替本次现场证据"):
        assert expected in body
    assert "不能将整体改善归因于本项" in body


def test_exact_action_effect_retains_result_method_and_limits():
    value = profile(action_effects=(CaseActionEffect(
        action_ordinal=1, result="UNCHANGED", method="MATCHED_CONTROL",
        summary="原测量未达到改善条件", evidence_refs=("effect-1",),
        limitations=("只测量了一次窗口",),
    ),))
    text = compile_case_knowledge(title="复盘", profile=value).body.decode()
    assert "逐项结果：无明显变化" in text
    assert "只测量了一次窗口" in text
    assert "原测量未达到改善条件" in text


@pytest.mark.parametrize("effect", [
    {"action_ordinal": 2, "result": "IMPROVED", "method": "OBSERVATIONAL", "summary": "改善", "evidence_refs": ["effect"]},
    {"action_ordinal": 1, "result": "IMPROVED", "method": "OBSERVATIONAL", "summary": "改善", "evidence_refs": []},
])
def test_effect_requires_existing_action_and_evidence(effect):
    with pytest.raises(ValidationError):
        profile(action_effects=(effect,))


def test_negative_actions_are_warnings_not_recommendations():
    value = profile(case_kind="NEGATIVE_EXPERIENCE", verification={"result": "ADVERSE", "method": "USER_REPORTED"})
    projection = compile_case_knowledge(title="失败复盘", profile=value)
    action = next(unit for unit in projection.units if unit.unit_id == "case-action-1")
    assert "不作为推荐措施" in projection.body.decode()[action.start:action.end]
    assert value.case_kind == "NEGATIVE_EXPERIENCE"


def test_markdown_compiler_preserves_source_and_seal_and_registers_manifest():
    value = profile()
    source_before, seal_before = deepcopy(value.source), shared_content_hash("复盘", value)
    first = case_knowledge_markdown(title="复盘", profile=value)
    assert value.source == source_before
    assert value.knowledge_projection.body_sha256 == hashlib.sha256(first).hexdigest()
    assert shared_content_hash("复盘", value) == seal_before
    assert case_knowledge_markdown(title="复盘", profile=value) == first


def test_redaction_precedes_projection_and_does_not_publish_raw_ids_or_markdown():
    value = profile(root_cause={"summary": "<script>secret</script>\n## forged [link](https://private.example/a) password=topsecret"},
        actions=({"summary": "检查 10.0.0.88 /home/customer/private", "evidence_refs": []},))
    text = case_knowledge_markdown(title="## injected 10.0.0.88", profile=value).decode()
    for secret in ("10.0.0.88", "/home/customer/private", "private.example", "topsecret", "<script>"):
        assert secret not in text
    assert "\n## forged" not in text
    assert "未登记实际执行证据" in text


def test_semantic_redaction_does_not_scan_source_locators_or_projection_metadata():
    value = profile(investigation_steps=investigation_knowledge(artifact=source(), claim_ids={"c2"})[0])
    case_knowledge_markdown(title="复盘", profile=value)
    from aiops_agent.application.case_semantic_redaction import _fields
    _, fields, _ = _fields("复盘", value)
    text = str(fields)
    assert value.investigation_steps[0].source.artifact_id not in text
    assert value.investigation_steps[0].step_id not in text
    assert value.knowledge_projection.body_sha256 not in text
    title, revised = apply_semantic_redaction(title="复盘", profile=value,
        output=SemanticRedactionOutput(spans=()), model_ref={}, prompt_ref={})
    assert revised.investigation_steps[0].source == value.investigation_steps[0].source
    assert title == "复盘"


def test_revision_inherits_read_only_checks_but_discards_effect_of_changed_action():
    steps = investigation_knowledge(artifact=source(), claim_ids={"c2"})[0]
    base = profile(investigation_steps=steps, unresolved_questions=("需要更多时间窗口",),
        action_effects=(CaseActionEffect(action_ordinal=1, result="UNCHANGED", method="OBSERVATIONAL",
            summary="未达改善", evidence_refs=("effect-1",)),))
    edited = profile(actions=({"summary": "全新建议，不是已执行动作", "evidence_refs": ["forged"]},),
        unresolved_questions=("伪造已解决",), investigation_steps=())
    result = _prepare_case_revision(base_profile=base, command=DiagnosisCaseDraft(
        expected_asset_row_version=1, base_version_id=uuid7(), display_name="修订", profile=edited,
    ))
    assert result.investigation_steps == steps
    assert result.unresolved_questions == base.unresolved_questions
    assert result.action_effects == ()
    assert result.actions[0]["evidence_refs"] == []
    assert result.knowledge_projection is None


def test_manifest_rejects_missing_relationship_target():
    projection = compile_case_knowledge(title="复盘", profile=profile())
    manifest = projection.manifest()
    manifest["units"][0]["related_unit_ids"] = ["invented-unit"]
    with pytest.raises(ValidationError):
        CaseKnowledgeManifest.model_validate(manifest)


def test_publish_projection_fence_rejects_missing_stale_title_body_or_manifest():
    value = profile()
    assert not case_projection_is_current(title="复盘", profile=value)
    case_knowledge_markdown(title="复盘", profile=value)
    assert case_projection_is_current(title="复盘", profile=value)
    assert not case_projection_is_current(title="改名后的复盘", profile=value)
    value.root_cause = {"summary": "已修改但未重新入库"}
    assert not case_projection_is_current(title="复盘", profile=value)


def test_more_than_one_hundred_unrelated_successors_do_not_invalidate_frozen_claims():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from aiops_agent.application.case_source_validation import current_case_claim_ids
    from aiops_agent.contracts.investigation import InvestigationRecord

    original = source()
    record = InvestigationRecord.model_validate(original.payload_json["investigation_record"])
    turn = SimpleNamespace(conversation_id=uuid7(), turn_id=uuid7(), turn_no=1)
    successors = [SimpleNamespace(turn_id=uuid7(), turn_no=index) for index in range(2, 253)]
    async def page(**kwargs):
        return [value for value in successors if value.turn_no > kwargs["after_turn_no"]][:kwargs["limit"]]
    turns = SimpleNamespace(
        get_run_link_by_ops_run_id=AsyncMock(return_value=SimpleNamespace(turn_id=turn.turn_id)),
        get_turn=AsyncMock(return_value=turn), list_turns=AsyncMock(side_effect=page),
        get_run_link=AsyncMock(return_value=None),
    )
    current = asyncio.run(current_case_claim_ids(
        uow=SimpleNamespace(turns=turns), domain_id=uuid7(), run=SimpleNamespace(ops_run_id=uuid7()),
        artifact=original, record=record,
    ))
    assert current == {"c1", "c2"}
    assert turns.list_turns.await_count == 3
    assert turns.list_turns.await_args.kwargs["after_turn_no"] == 201


def test_action_hit_expands_exact_body_context_without_pretending_to_be_new_kc_evidence():
    value = profile(case_kind="NEGATIVE_EXPERIENCE", verification={"result": "ADVERSE", "method": "USER_REPORTED"})
    body = case_knowledge_markdown(title="失败复盘", profile=value).decode()
    revision_id, document_id = uuid7(), uuid7()
    context = case_context_for_citation(profile=value, document_version_id=document_id, citation={
        "bundle_revision_id": str(revision_id),
        "items": [{"evidence": {"heading_path": ["失败复盘", "[case-action-1] 登记措施与效果 1"]}}],
    })
    units = {unit["unit_id"]: unit for unit in context["units"]}
    assert {"case-action-1", "case-problem", "case-conclusion", "case-verification", "case-boundaries", "case-unresolved"} <= set(units)
    assert context["document_version_id"] == str(document_id)
    assert context["bundle_revision_id"] == str(revision_id)
    assert context["action_use"] == "WARNING_ONLY"
    assert all(unit["quote"] == body[unit["start"]:unit["end"]] for unit in context["units"])
    assert all("evidence_id" not in unit for unit in context["units"])
    assert all(hashlib.sha256(unit["quote"].encode()).hexdigest() == unit["content_hash"] for unit in context["units"])
    with pytest.raises(ValueError, match="预算"):
        case_context_for_citation(profile=value, document_version_id=document_id, max_characters=10, citation={
            "bundle_revision_id": str(revision_id), "items": [{"evidence": {"heading_path": ["[case-action-1]"]}}],
        })
    with pytest.raises(ValueError, match="语义单元"):
        case_context_for_citation(profile=value, document_version_id=document_id, citation={"bundle_revision_id": str(revision_id)})
