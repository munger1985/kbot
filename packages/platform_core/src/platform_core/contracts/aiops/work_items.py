"""DBA 工作项的公开合同。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .findings import FindingCard, FindingSeverity
from .types import (
    AIOpsContract,
    CursorPage,
    JsonObject,
    PUBLIC_SCHEMA_VERSION,
    UUIDv7,
    UtcDatetime,
)


class WorkItemType(StrEnum):
    INCIDENT_RESPONSE = "INCIDENT_RESPONSE"
    RISK_REMEDIATION = "RISK_REMEDIATION"
    PROBLEM_INVESTIGATION = "PROBLEM_INVESTIGATION"
    OPTIMIZATION = "OPTIMIZATION"
    OBSERVABILITY_GAP = "OBSERVABILITY_GAP"
    RECOVERY_OBSERVATION = "RECOVERY_OBSERVATION"


class WorkItemSourceKind(StrEnum):
    ALERT = "ALERT"
    INSPECTION = "INSPECTION"
    CHAT = "CHAT"
    MANUAL = "MANUAL"


class WorkItemPriority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class WorkItemStatus(StrEnum):
    PENDING_TRIAGE = "PENDING_TRIAGE"
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING = "WAITING"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class WorkItemPhase(StrEnum):
    TRIAGE = "TRIAGE"
    DIAGNOSIS = "DIAGNOSIS"
    REMEDIATION = "REMEDIATION"
    APPROVAL = "APPROVAL"
    EXECUTION = "EXECUTION"
    VERIFICATION = "VERIFICATION"


class WorkItemResolutionCode(StrEnum):
    FIXED = "FIXED"
    MITIGATED = "MITIGATED"
    OBSERVED_STABLE = "OBSERVED_STABLE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ACCEPTED_RISK = "ACCEPTED_RISK"
    DUPLICATE = "DUPLICATE"
    NO_ACTION = "NO_ACTION"
    CANNOT_REPRODUCE = "CANNOT_REPRODUCE"


class WorkItemAssignmentSource(StrEnum):
    ROUTING_RULE = "ROUTING_RULE"
    TARGET_DEFAULT = "TARGET_DEFAULT"
    AGENT_DEFAULT = "AGENT_DEFAULT"
    MANUAL = "MANUAL"
    CLAIM = "CLAIM"


class ResponsibilityGroupStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class ResponsibilityGroupMemberRole(StrEnum):
    LEAD = "LEAD"
    MEMBER = "MEMBER"


class WorkItemDecision(StrEnum):
    ACTION_REQUIRED = "ACTION_REQUIRED"
    MANUAL_INVESTIGATION = "MANUAL_INVESTIGATION"
    OBSERVE = "OBSERVE"
    NO_WORK_ITEM = "NO_WORK_ITEM"


class WorkItemRoutingDecision(AIOpsContract):
    decision: WorkItemDecision
    recommended_priority: WorkItemPriority
    reason_codes: tuple[str, ...] = ()
    action_summary: str
    impact: str | None = None
    confirmation: str | None = None
    evidence_refs: tuple[str, ...] = ()
    recommended_playbook_id: UUIDv7 | None = None
    observation_window: str | None = None


class WorkItemResourceKind(StrEnum):
    RUN = "RUN"
    SITUATION = "SITUATION"
    REPORT = "REPORT"
    PROPOSAL = "PROPOSAL"
    VERIFICATION = "VERIFICATION"


class WorkItemQueueSummary(AIOpsContract):
    my_open: int = Field(ge=0)
    unassigned: int = Field(ge=0)
    urgent: int = Field(ge=0)
    overdue: int = Field(ge=0)
    pending_verification: int = Field(ge=0)


class WorkItemSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    work_item_id: UUIDv7
    item_key: str
    target_id: UUIDv7
    target_name: str
    work_type: WorkItemType
    source_kind: WorkItemSourceKind
    title: str
    summary: str
    severity: FindingSeverity
    priority: WorkItemPriority
    status: WorkItemStatus
    phase: WorkItemPhase
    wait_reason: str | None = None
    responsibility_group_id: UUIDv7 | None = None
    responsibility_group_name: str | None = None
    assignee_user_id: str | None = None
    assignment_source: WorkItemAssignmentSource | None = None
    assigned_by: str | None = None
    assigned_at: UtcDatetime | None = None
    acknowledgement_due_at: UtcDatetime | None = None
    resolution_due_at: UtcDatetime | None = None
    verification_due_at: UtcDatetime | None = None
    first_observed_at: UtcDatetime
    last_observed_at: UtcDatetime
    occurrence_count: int = Field(ge=1)
    reopen_count: int = Field(ge=0)
    resolution_code: WorkItemResolutionCode | None = None
    completion_note: str | None = None
    completed_by: str | None = None
    completed_at: UtcDatetime | None = None
    verification_result: str | None = None
    verified_by: str | None = None
    verified_at: UtcDatetime | None = None
    resolved_at: UtcDatetime | None = None
    closed_at: UtcDatetime | None = None
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class WorkItemOccurrenceView(AIOpsContract):
    occurrence_id: UUIDv7
    ops_run_id: UUIDv7
    situation_id: UUIDv7 | None = None
    finding_id: str
    finding_type: str
    severity: FindingSeverity
    confirmation: str
    finding: FindingCard | None = None
    evidence_refs: tuple[str, ...] = ()
    observed_at: UtcDatetime
    created_at: UtcDatetime


class WorkItemResourceLinkView(AIOpsContract):
    resource_kind: WorkItemResourceKind
    resource_id: UUIDv7
    role: str
    label: str | None = None
    status: str | None = None
    created_at: UtcDatetime | None = None


class WorkItemActivityView(AIOpsContract):
    activity_id: UUIDv7
    activity_type: str
    actor_id: str
    from_status: WorkItemStatus | None = None
    to_status: WorkItemStatus | None = None
    from_phase: WorkItemPhase | None = None
    to_phase: WorkItemPhase | None = None
    detail: JsonObject = Field(default_factory=dict)
    created_at: UtcDatetime


class WorkItemView(WorkItemSummary):
    occurrences: tuple[WorkItemOccurrenceView, ...] = ()
    links: tuple[WorkItemResourceLinkView, ...] = ()
    activities: tuple[WorkItemActivityView, ...] = ()
    resolution_note: str | None = None
    routing_decision: WorkItemRoutingDecision | None = None


class WorkItemPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[WorkItemSummary, ...] = ()
    queue_summary: WorkItemQueueSummary


class WorkItemCreate(AIOpsContract):
    target_id: UUIDv7
    work_type: WorkItemType
    title: str = Field(min_length=1, max_length=512)
    summary: str = Field(min_length=1, max_length=4000)
    severity: FindingSeverity
    priority: WorkItemPriority
    responsibility_group_id: UUIDv7 | None = None
    assignee_user_id: str | None = Field(default=None, max_length=256)


class WorkItemRouteRun(AIOpsContract):
    ops_run_id: UUIDv7
    finding_ids: tuple[str, ...] = Field(default=(), max_length=100)
    include_gaps: bool = True


class WorkItemAssignment(AIOpsContract):
    expected_row_version: int = Field(ge=1)
    responsibility_group_id: UUIDv7 | None = None
    assignee_user_id: str | None = Field(default=None, max_length=256)


class WorkItemVersionCommand(AIOpsContract):
    expected_row_version: int = Field(ge=1)


class WorkItemCompletion(WorkItemVersionCommand):
    resolution_code: WorkItemResolutionCode
    completion_note: str = Field(min_length=1, max_length=4000)
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=100)


class WorkItemVerification(WorkItemVersionCommand):
    passed: bool
    note: str = Field(min_length=1, max_length=4000)
    verification_id: UUIDv7 | None = None


class ResponsibilityGroupMemberView(AIOpsContract):
    user_id: str
    member_role: ResponsibilityGroupMemberRole
    status: ResponsibilityGroupStatus
    active_work_item_count: int = Field(ge=0)
    created_at: UtcDatetime


class ResponsibilityGroupSummary(AIOpsContract):
    responsibility_group_id: UUIDv7
    name: str
    description: str | None = None
    status: ResponsibilityGroupStatus
    lead_user_id: str | None = None
    active_member_count: int = Field(ge=0)
    agent_binding_count: int = Field(ge=0)
    target_binding_count: int = Field(ge=0)
    unassigned_work_item_count: int = Field(ge=0)
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class ResponsibilityGroupView(ResponsibilityGroupSummary):
    members: tuple[ResponsibilityGroupMemberView, ...] = ()


class ResponsibilityGroupPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[ResponsibilityGroupSummary, ...] = ()


class ResponsibilityGroupCreate(AIOpsContract):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1000)
    lead_user_id: str | None = Field(default=None, max_length=256)


class ResponsibilityGroupPatch(AIOpsContract):
    expected_row_version: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1000)
    status: ResponsibilityGroupStatus | None = None
    lead_user_id: str | None = Field(default=None, max_length=256)


class ResponsibilityGroupMemberUpsert(AIOpsContract):
    member_role: ResponsibilityGroupMemberRole = ResponsibilityGroupMemberRole.MEMBER


class WorkItemTransition(AIOpsContract):
    expected_row_version: int = Field(ge=1)
    status: WorkItemStatus
    phase: WorkItemPhase
    wait_reason: str | None = Field(default=None, max_length=256)
    resolution_code: WorkItemResolutionCode | None = None
    resolution_note: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_transition_details(self) -> "WorkItemTransition":
        if self.status == WorkItemStatus.WAITING and not self.wait_reason:
            raise ValueError("等待状态必须填写等待原因")
        if self.status in {WorkItemStatus.RESOLVED, WorkItemStatus.CLOSED}:
            if self.resolution_code is None or not self.resolution_note:
                raise ValueError("解决或关闭事项必须填写解决代码和说明")
        elif self.resolution_code is not None or self.resolution_note is not None:
            raise ValueError("非解决状态不能填写解决信息")
        return self
