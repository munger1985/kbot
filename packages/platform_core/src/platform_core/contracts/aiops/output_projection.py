"""通知对账专用的最小服务合同；不返回冻结正文、凭据或结果内容。"""

from uuid import UUID

from pydantic import Field, model_validator

from .output_authority import OutputAuthorityRequest
from .output_jobs import OutputJobStage, OutputJobStatus, OutputJobType
from .types import AIOpsContract, UtcDatetime, UUIDv7


class OutputProjectionRequest(AIOpsContract):
    job_id: UUIDv7
    job_type: OutputJobType
    actor_id: str = Field(min_length=1, max_length=256)
    domain_id: str = Field(min_length=1, max_length=128)
    tenant_id: UUID | None = None


class OutputProjectionSnapshot(AIOpsContract):
    request: OutputProjectionRequest
    authority: OutputAuthorityRequest
    status: OutputJobStatus
    stage: OutputJobStage
    source_version: int = Field(ge=1, strict=True)
    source_updated_at: UtcDatetime
    source_security_level: int = Field(ge=1, le=5)
    error_code: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_identity(self):
        for field in ("job_id", "job_type", "actor_id", "domain_id", "tenant_id"):
            if getattr(self.request, field) != getattr(self.authority, field):
                raise ValueError("任务投影与授权主体不一致")
        if (self.status == "SUCCEEDED") != (self.stage == "COMPLETED"):
            raise ValueError("任务投影成功状态与阶段不一致")
        if (self.status == "FAILED") != (self.stage == "FAILED"):
            raise ValueError("任务投影失败状态与阶段不一致")
        return self
