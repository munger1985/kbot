"""后台输出执行的当前授权查询；不是登录身份或可持久化的授权令牌。"""

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .output_jobs import OutputJobSource, OutputJobType
from .types import AIOpsContract, UtcDatetime, UUIDv7


class OutputAuthorityRequest(AIOpsContract):
    job_id: UUIDv7
    job_type: OutputJobType
    source: OutputJobSource
    origin: Literal["HUMAN"]
    actor_id: str = Field(min_length=1, max_length=256)
    domain_id: str = Field(min_length=1, max_length=128)
    tenant_id: UUID | None = None
    source_agent_ids: tuple[UUIDv7, ...] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_source(self):
        allowed = {"CONVERSATION", "RUN"} if self.job_type == "REPORT_GENERATION" else {"CONVERSATION", "REPORT"}
        if self.source.kind not in allowed or len(set(self.source_agent_ids)) != len(self.source_agent_ids):
            raise ValueError("任务类型、来源或 Agent 清单不一致")
        if not self.actor_id.strip() or not self.domain_id.strip():
            raise ValueError("执行主体和业务空间不能为空")
        return self


class OutputAuthoritySnapshot(AIOpsContract):
    """仅在本次阶段使用；下一阶段、完成和重试均须重新查询。"""

    request: OutputAuthorityRequest
    permissions: tuple[str, ...]
    max_security_level: int | None = Field(default=None, ge=0, le=5)
    domain_code: str | None = None
    checked_at: UtcDatetime


def output_required_permissions(request: OutputAuthorityRequest) -> frozenset[str]:
    required = {"aiops:use"}
    if request.job_type == "CASE_EXTRACTION" and request.source.kind == "REPORT":
        required.add("aiops:knowledge_manage")
    return frozenset(required)
