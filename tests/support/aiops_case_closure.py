"""真实收口合同的测试来源夹具，不以缺少血缘的旧 JSON 代替有效来源。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.case_closure import closure_reference, closure_claim_refs
from aiops_agent.contracts.knowledge import DiagnosisClosure, DiagnosisCaseProfile
from aiops_agent.playbooks import canonical_hash
from platform_core.identity import uuid7


def install_closure(f, *, result="RESOLVED", action_refs=True):
    f.turn.conversation_id = f.conversation_id
    f.artifact.created_at = f.now
    f.uow.targets.get_scoped.return_value.status = "ENABLED"
    f.uow.targets.get_scoped.return_value.row_version = 1
    link = SimpleNamespace(ops_run_id=f.run_id, turn_id=f.turn_id, purpose="PRIMARY")
    f.uow.turns.get_turn = AsyncMock(return_value=f.turn)
    f.uow.turns.list_run_links = AsyncMock(return_value=[link])
    f.uow.turns.list_turns = AsyncMock(return_value=[])
    f.uow.runs.list_artifacts = AsyncMock(side_effect=lambda **kw: [
        row for row in f.state.artifacts.values() if row.ops_run_id == kw["ops_run_id"]])
    refs = [closure_reference(turn=f.turn, run=f.run, artifact=f.artifact,
        roles=("INVESTIGATION",), claim_ids=("c1",))]
    actions = []
    if action_refs:
        from aiops_agent.contracts.change import AdvisoryActionResult
        payload = AdvisoryActionResult(proposal_id=str(uuid7()), status="EXECUTED", occurred_at=f.now,
            submitted_at=f.now, submitted_by=str(f.actor), note="人工登记操作", result_hash="a" * 64).model_dump(mode="json")
        f.action_artifact = SimpleNamespace(artifact_id=uuid7(), ops_run_id=f.run_id, ops_task_id=f.task_id,
            schema_version="USER_PROVIDED_ACTION_RESULT.v1", payload_json=payload, content_hash=canonical_hash(payload),
            trust_level="USER_PROVIDED", security_level=3, created_at=f.now)
        f.state.artifacts[f.action_artifact.artifact_id] = f.action_artifact
        refs.append(closure_reference(turn=f.turn, run=f.run, artifact=f.action_artifact, roles=("ACTION",)))
        actions = [str(f.action_artifact.artifact_id)]
    profile = DiagnosisCaseProfile(
        environment={"database_type": "POSTGRESQL"},
        case_kind="NEGATIVE_EXPERIENCE" if action_refs and result in {"ADVERSE", "NOT_ACHIEVED"} else "DIAGNOSTIC_REFERENCE",
        problem_signature={"normalized_symptoms": ["业务曾有等待"]},
        root_cause={"summary": "人工复盘，不代替数据库证据", "evidence_refs": [str(f.artifact.artifact_id)]},
        actions=({"summary": "调整调用频率", "source_kind": "USER_PROVIDED", "evidence_refs": actions},),
        verification={"method": "USER_REPORTED", "result": result, "before_refs": [], "after_refs": [],
            "direct_effect": "UNKNOWN", "measured_effect": "INCONCLUSIVE", "comparability": "UNKNOWN",
            "symptom_recovered": True, "guardrails_passed": None, "attribution_strength": "UNKNOWN",
            "causal_limitations": ["人工反馈，未经受控对照验证"]},
        source={"conversation_id": str(f.conversation_id), "turn_ids": [str(f.turn_id)],
            "run_ids": [str(f.run_id)], "claim_refs": closure_claim_refs(f.artifact, {"c1"}),
            "source_fingerprint": "a" * 64},
    )
    payload = DiagnosisClosure(conversation_id=f.conversation_id, source_turn_ids=(f.turn_id,),
        source_fingerprint="a" * 64, source_refs=tuple(refs), target_id=f.target, target_version=1,
        closed_by=str(f.actor), closed_at=f.now, profile=profile).model_dump(mode="json")
    f.state.closure = SimpleNamespace(artifact_id=uuid7(), ops_run_id=f.run_id,
        ops_task_id=f.task_id, schema_version="DIAGNOSIS_CLOSURE.v1", payload_json=payload,
        content_hash=canonical_hash(payload), security_level=max(ref.security_level for ref in refs),
        provenance_json={}, created_at=f.now)
    f.state.artifacts[f.state.closure.artifact_id] = f.state.closure
    return f
