"""报告与案例独立输出任务的共享合同。"""

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .types import AIOpsContract, Sha256Digest, UtcDatetime, UUIDv7


OutputJobType = Literal["REPORT_GENERATION", "CASE_EXTRACTION"]
OutputJobStatus = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]
OutputJobStage = Literal[
    "ACCEPTED", "VALIDATING", "ASSEMBLING", "REDACTING",
    "INGESTING", "INDEXING", "FINALIZING", "COMPLETED", "FAILED",
]


class OutputJobSource(AIOpsContract):
    kind: Literal["CONVERSATION", "RUN", "REPORT"]
    source_id: UUIDv7
    conversation_id: UUIDv7 | None = None
    run_id: UUIDv7 | None = None
    report_id: UUIDv7 | None = None


class OutputJobResult(AIOpsContract):
    report_id: UUIDv7 | None = None
    report_version: str | None = None
    asset_id: UUIDv7 | None = None
    asset_version_id: UUIDv7 | None = None

    @model_validator(mode="after")
    def validate_locator(self):
        report = self.report_id is not None
        case = self.asset_id is not None and self.asset_version_id is not None
        if report == case or ((self.asset_id is None) != (self.asset_version_id is None)):
            raise ValueError("结果必须定位一份报告或一个确切案例版本")
        if not report and self.report_version is not None:
            raise ValueError("案例结果不能携带报告版本")
        return self


class OutputJobReceipt(AIOpsContract):
    job_id: UUIDv7
    job_type: OutputJobType
    status: OutputJobStatus
    stage: OutputJobStage
    source: OutputJobSource
    submitted_at: UtcDatetime
    started_at: UtcDatetime | None = None
    completed_at: UtcDatetime | None = None
    attempt_count: int = Field(ge=0)
    result: OutputJobResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    row_version: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_state(self):
        if self.status == "SUCCEEDED":
            if self.result is None or self.stage != "COMPLETED" or self.completed_at is None:
                raise ValueError("成功任务必须有终态时间和确切结果")
        elif self.result is not None:
            raise ValueError("未成功任务不能公开结果定位")
        if self.result is not None:
            if (self.job_type == "REPORT_GENERATION") != (self.result.report_id is not None):
                raise ValueError("任务类型和结果类型不一致")
        if self.retryable and self.status != "FAILED":
            raise ValueError("只有失败任务可以手工重试")
        if self.status == "FAILED" and (not self.error_code or self.completed_at is None):
            raise ValueError("失败任务必须有错误码和终态时间")
        return self


class OutputJobList(AIOpsContract):
    items: tuple[OutputJobReceipt, ...]
    next_cursor: UUID | None = None


class OutputJobRetry(AIOpsContract):
    expected_row_version: int = Field(ge=1)


class OutputSourceArtifact(AIOpsContract):
    run_id: UUIDv7
    artifact_id: UUIDv7
    schema_version: str = Field(min_length=1, max_length=128)
    content_hash: Sha256Digest
    security_level: int = Field(ge=1, le=5)
    turn_id: UUIDv7 | None = None


class FrozenOutputInput(AIOpsContract):
    """仅由服务端冻结，不接受浏览器提交原件、模型凭据或授权快照。"""

    schema_version: Literal["OPS_OUTPUT_INPUT.v1"] = "OPS_OUTPUT_INPUT.v1"
    source: OutputJobSource
    artifacts: tuple[OutputSourceArtifact, ...] = Field(min_length=1, max_length=512)
    source_fingerprint: Sha256Digest
    template: dict | None = None
    parameters: dict = Field(default_factory=dict)
    frozen_at: UtcDatetime
