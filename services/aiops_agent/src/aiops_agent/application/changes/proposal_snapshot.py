"""从冻结 Action Plan 确定性生成单条 Proposal 快照。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from aiops_agent.contracts.change import (
    ActionPlan,
    ActionPlanItem,
    ChangeProposalSnapshot,
)
from platform_core.identity import uuid7


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def build_proposal_snapshot(
    *,
    plan: ActionPlan,
    action: ActionPlanItem,
    run_id: str,
    task_id: str,
    target_id: str,
    target_version: int,
    now: datetime,
) -> ChangeProposalSnapshot:
    """一次只固化一个 ordinal，后续动作必须等待上一条验证成功。"""
    rendered = action.rendered_action
    proposal_id = uuid7()
    body = {
        "proposal_id": str(proposal_id),
        "run_id": run_id,
        "task_id": task_id,
        "target_id": target_id,
        "target_version": target_version,
        "solution_group_key": plan.solution_group_key,
        "command_ordinal": action.ordinal,
        "proposal_version": 1,
        "mode": action.mode,
        "action_family": action.action_family,
        "effect_class": action.effect_class,
        "execution_mode": action.execution_mode,
        "executor_kind": action.executor_kind,
        "canonical_object_ref": action.canonical_object_ref,
        "action_template_id": action.action_template_id,
        "action_template_version": action.action_template_version,
        "action_template_variant": action.variant,
        "action_template_hash": rendered["template_hash"],
        "renderer_version": rendered["renderer_version"],
        "canonical_parameters": action.canonical_parameters,
        "parameter_fact_refs": action.parameter_fact_refs,
        "parameters_hash": rendered["parameters_hash"],
        "rendered_command": rendered["command_text"],
        "command_hash": rendered["command_hash"],
        "risk_level": rendered["risk_level"],
        "lock_impact": action.lock_impact,
        "estimated_duration_seconds": action.estimated_duration_seconds,
        "impact": "；".join(action.expected_effects),
        "rationale": action.rationale,
        "preconditions": action.precondition_tool_refs,
        "rollback_plan": (
            action.rollback_description or "该动作不可逆，需重新建立会话"
        ),
        "verification_plan": action.verification_tool_refs,
        "evidence_refs": tuple(
            sorted(set(action.parameter_fact_refs.values()))
        ),
        "policy_decision_hash": plan.policy_decision_hash,
        "expires_at": now + timedelta(minutes=15),
    }
    return ChangeProposalSnapshot(**body, proposal_hash=_hash(body))


def proposal_summary_payload(snapshot: ChangeProposalSnapshot) -> dict[str, Any]:
    """生成聊天与后续串行动作共用的权威展示负载。"""
    pending = (
        snapshot.mode == "AGENT_EXECUTE"
        and snapshot.execution_mode == "EXECUTABLE_AFTER_APPROVAL"
    )
    return {
        "proposal_id": snapshot.proposal_id,
        "proposal_hash": snapshot.proposal_hash,
        "row_version": 1,
        "status": "PENDING_APPROVAL" if pending else "ADVISORY_READY",
        "action_template_id": snapshot.action_template_id,
        "action_family": snapshot.action_family,
        "effect_class": snapshot.effect_class,
        "execution_mode": snapshot.execution_mode,
        "executor_kind": snapshot.executor_kind,
        "canonical_object_ref": snapshot.canonical_object_ref,
        "risk_level": snapshot.risk_level,
        "rationale": snapshot.rationale,
        "impact": snapshot.impact,
        "parameters": snapshot.canonical_parameters,
        "command_preview": snapshot.rendered_command,
        "lock_impact": snapshot.lock_impact,
        "estimated_duration_seconds": snapshot.estimated_duration_seconds,
        "preconditions": snapshot.preconditions,
        "rollback_plan": snapshot.rollback_plan,
        "verification_plan": snapshot.verification_plan,
        "expires_at": snapshot.expires_at.isoformat(),
    }
