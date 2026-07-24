"""步骤 9 变更建议链路的严格 Artifact Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from platform_core.contracts.aiops.types import UtcDatetime


class _ChangeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionPlanItem(_ChangeContract):
    ordinal: int = Field(ge=1, le=16)
    action_template_id: str
    action_template_version: str
    variant: str
    mode: Literal["ADVISORY", "AGENT_EXECUTE"]
    canonical_parameters: dict[str, int | str]
    parameter_fact_refs: dict[str, str]
    rationale: str = Field(min_length=1, max_length=2000)
    expected_effects: tuple[str, ...]
    precondition_tool_refs: tuple[str, ...]
    verification_tool_refs: tuple[str, ...]
    rollback_description: str | None = None
    rendered_action: dict[str, Any]


class ActionPlan(_ChangeContract):
    schema_version: Literal["ACTION_PLAN.v1"] = "ACTION_PLAN.v1"
    solution_group_key: str
    target_id: str
    root_cause_level: str
    actions: tuple[ActionPlanItem, ...] = ()
    decision: Literal["NO_ACTION", "ADVISORY", "AGENT_EXECUTE"]
    decision_reasons: tuple[str, ...] = ()
    policy_decision_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    action_catalog_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class ChangeProposalSnapshot(_ChangeContract):
    schema_version: Literal["CHANGE_PROPOSAL_SNAPSHOT.v1"] = (
        "CHANGE_PROPOSAL_SNAPSHOT.v1"
    )
    proposal_id: str
    run_id: str
    task_id: str
    target_id: str
    target_version: int = Field(ge=1)
    solution_group_key: str
    command_ordinal: int = Field(ge=1)
    proposal_version: int = Field(ge=1)
    mode: Literal["ADVISORY", "AGENT_EXECUTE"]
    action_template_id: str
    action_template_version: str
    action_template_variant: str
    action_template_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    renderer_version: str
    canonical_parameters: dict[str, int | str]
    parameter_fact_refs: dict[str, str]
    parameters_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    rendered_command: str
    command_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    risk_level: str
    impact: str
    rationale: str
    preconditions: tuple[str, ...]
    rollback_plan: str
    verification_plan: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    policy_decision_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    expires_at: UtcDatetime
    proposal_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class ProposalOutcome(_ChangeContract):
    schema_version: Literal["PROPOSAL_OUTCOME.v1"] = (
        "PROPOSAL_OUTCOME.v1"
    )
    status: Literal["CREATED", "NOT_REQUIRED"]
    proposal: ChangeProposalSnapshot | None = None
    reason: str | None = None


class AdvisoryActionResult(_ChangeContract):
    schema_version: Literal["USER_PROVIDED_ACTION_RESULT.v1"] = (
        "USER_PROVIDED_ACTION_RESULT.v1"
    )
    proposal_id: str
    status: Literal["EXECUTED", "FAILED", "CANCELLED"]
    occurred_at: UtcDatetime
    submitted_at: UtcDatetime
    submitted_by: str
    note: str | None = Field(default=None, max_length=4000)
    bounded_output: str | None = Field(default=None, max_length=16000)
    result_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class AdvisoryVerificationScope(_ChangeContract):
    schema_version: Literal["ADVISORY_VERIFICATION_SCOPE.v1"] = (
        "ADVISORY_VERIFICATION_SCOPE.v1"
    )
    proposal_id: str
    source_run_id: str
    result_artifact_id: str
    action_template_id: str
    canonical_parameters: dict[str, int | str]
    verification_tool_refs: tuple[str, ...]
    manual_result_status: Literal["EXECUTED"]
    initial_gap_codes: tuple[str, ...] = ()


class ActionVerification(_ChangeContract):
    schema_version: Literal["ACTION_VERIFICATION.v1"] = (
        "ACTION_VERIFICATION.v1"
    )
    proposal_id: str
    source_run_id: str
    result_artifact_id: str
    status: Literal[
        "VERIFIED", "NOT_ACHIEVED", "ADVERSE", "INCONCLUSIVE"
    ]
    summary: str
    target_still_present: bool | None = None
    blocking_still_present: bool | None = None
    checked_tool_refs: tuple[str, ...] = ()
    gap_codes: tuple[str, ...] = ()
    evidence_hashes: tuple[str, ...] = ()
