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
