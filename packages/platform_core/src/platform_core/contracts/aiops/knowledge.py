"""运维知识复用反馈；人工反馈不改变验证等级和发布状态。"""

from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from .types import AIOpsContract


class KnowledgeReuseFeedback(AIOpsContract):
    feedback_id: UUID
    asset_version_id: UUID
    source_run_id: UUID | None = None
    applicability: Literal["APPLICABLE", "NOT_APPLICABLE", "UNCERTAIN"]
    check_value: Literal["USEFUL", "NOT_USEFUL", "UNKNOWN"] = "UNKNOWN"
    action_effect: Literal["IMPROVED", "UNCHANGED", "DEGRADED", "INCONCLUSIVE", "NOT_ATTEMPTED"] = "NOT_ATTEMPTED"
    notes: str = Field(min_length=1, max_length=1000)

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value):
        if not value.strip():
            raise ValueError("请说明适用差异、检查结果或失败原因")
        return value.strip()
