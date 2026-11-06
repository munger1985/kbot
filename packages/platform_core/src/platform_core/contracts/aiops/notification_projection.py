"""通知待办对账的最小服务合同，不包含业务正文或用户凭据。"""

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .types import AIOpsContract, HitlStatus, OpsRunStatus, ProposalStatus, UUIDv7


class WorkflowProjectionRequest(AIOpsContract):
    resource_type: Literal["aiops_hitl", "aiops_proposal", "aiops_run"]
    resource_id: UUIDv7
    domain_id: str = Field(min_length=1, max_length=128)
    tenant_id: UUID | None = None



class NotificationResourceRequest(AIOpsContract):
    """只接受已有任务的白名单定位，不接受 URL、正文或执行命令。"""
    event_type: str = Field(min_length=3, max_length=160)
    domain_id: str = Field(min_length=1, max_length=128)
    tenant_id: UUID | None = None
    route_key: Literal["aiops.run.detail", "aiops.situation.detail", "aiops.report.detail"]
    route_params: dict[str, str] = Field(max_length=12)

    @model_validator(mode="after")
    def locator(self):
        from platform_core.notifications.delivery import EXTERNAL_EVENTS
        if self.event_type not in EXTERNAL_EVENTS:
            raise ValueError("NOTIFICATION_EXTERNAL_EVENT_INVALID")
        allowed = {"domain_id", "run_id", "proposal_id", "hitl_id", "execution_id",
                   "situation_id", "report_id", "job_id", "report_version"}
        params = self.route_params
        if set(params) - allowed or params.get("domain_id") != self.domain_id:
            raise ValueError("NOTIFICATION_ROUTE_SCOPE_INVALID")
        for key, value in params.items():
            if key.endswith("_id") and key != "domain_id":
                UUID(value)
        if params.get("report_version") is not None and not params["report_version"].isdigit():
            raise ValueError("NOTIFICATION_REPORT_VERSION_INVALID")
        required = {"aiops.run.detail": "run_id", "aiops.situation.detail": "situation_id",
                    "aiops.report.detail": "report_id"}[self.route_key]
        if required not in params:
            raise ValueError("NOTIFICATION_RESOURCE_REQUIRED")
        route_fields = {
            "aiops.run.detail": {"domain_id", "run_id", "proposal_id", "hitl_id", "execution_id"},
            "aiops.situation.detail": {"domain_id", "situation_id"},
            "aiops.report.detail": {"domain_id", "report_id", "report_version", "job_id"},
        }
        if set(params) - route_fields[self.route_key]:
            raise ValueError("NOTIFICATION_ROUTE_FIELDS_INVALID")
        if self.event_type.startswith("aiops.proposal.") or self.event_type.startswith("aiops.execution."):
            if self.route_key != "aiops.run.detail" or "proposal_id" not in params:
                raise ValueError("NOTIFICATION_PROPOSAL_REQUIRED")
        if self.event_type.startswith("aiops.execution.") and "execution_id" not in params:
            raise ValueError("NOTIFICATION_EXECUTION_REQUIRED")
        if self.event_type.startswith("aiops.diagnostic.input_") and ("hitl_id" not in params or self.route_key != "aiops.run.detail"):
            raise ValueError("NOTIFICATION_HITL_REQUIRED")
        if self.event_type.startswith("aiops.situation.") and self.route_key != "aiops.situation.detail":
            raise ValueError("NOTIFICATION_SITUATION_REQUIRED")
        if self.event_type == "aiops.report.ready" and self.route_key != "aiops.report.detail":
            raise ValueError("NOTIFICATION_REPORT_REQUIRED")
        return self


class NotificationResourceSnapshot(AIOpsContract):
    request: NotificationResourceRequest
    actionable: bool
    source_version: int = Field(ge=1)
    initiator_actor_id: str | None = None
    run_actor_id: str | None = None
    assignee_actor_id: str | None = None

class WorkflowProjectionSnapshot(AIOpsContract):
    request: WorkflowProjectionRequest
    source_status: str
    source_version: int = Field(ge=1, strict=True)
    work_item_status: Literal["COMPLETED", "CANCELLED"] | None = None

    @model_validator(mode="after")
    def validate_status(self):
        kind = self.request.resource_type
        if kind == "aiops_hitl":
            HitlStatus(self.source_status)
            expected = (None if self.source_status == "PENDING" else
                "CANCELLED" if self.source_status in {"EXPIRED", "CANCELLED"} else "COMPLETED")
        elif kind == "aiops_proposal":
            ProposalStatus(self.source_status)
            expected = (None if self.source_status == "PENDING_APPROVAL" else
                "CANCELLED" if self.source_status in {"EXPIRED", "SUPERSEDED"} else "COMPLETED")
        else:
            OpsRunStatus(self.source_status)
            expected = self.work_item_status
        if self.work_item_status != expected:
            raise ValueError("待办状态与业务状态不一致")
        return self
