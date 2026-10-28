"""人工澄清等待不消耗机器预算，也不能重置已消耗的预算。"""

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PausedTurnBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["PAUSED_TURN_BUDGET.v1"] = "PAUSED_TURN_BUDGET.v1"
    remaining_milliseconds: int = Field(ge=0, le=86_400_000, strict=True)
    paused_at: datetime
    resume_until: datetime

    @model_validator(mode="after")
    def validate_times(self):
        if self.paused_at.tzinfo is None or self.resume_until.tzinfo is None or not (
            timedelta(0) < self.resume_until - self.paused_at <= timedelta(days=1)
        ):
            raise ValueError("澄清续跑期限必须带时区且不超过一天")
        return self
