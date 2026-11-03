"""跨数据库共享的日常巡检区域与版式合同，不承载引擎判断规则。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DAILY_SECTION_IDS = (
    "OVERVIEW", "AVAILABILITY", "RESOURCES", "CAPACITY", "SESSIONS",
    "PERFORMANCE", "SQL", "REPLICATION", "BACKUP", "MAINTENANCE", "BASELINE", "ACTIONS",
)


class DailySectionLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str
    visibility: Literal["ALWAYS", "WHEN_RELEVANT"] = "ALWAYS"
    normal_detail: Literal["FULL", "SUMMARY"] = "FULL"
    show_statistics: bool = True
    show_trend: bool = True
    expanded: bool = True


class DailyReportLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["DAILY_REPORT_LAYOUT.v1"] = "DAILY_REPORT_LAYOUT.v1"
    title: str = Field(default="数据库日常巡检报告", min_length=1, max_length=128)
    sections: tuple[DailySectionLayout, ...] = Field(default_factory=lambda: tuple(
        DailySectionLayout(section_id=key) for key in DAILY_SECTION_IDS
    ))

    @field_validator("title")
    @classmethod
    def validate_title(cls, value):
        if not value.strip():
            raise ValueError("日报标题不能为空白")
        return value.strip()

    @model_validator(mode="after")
    def validate_sections(self):
        ids = tuple(section.section_id for section in self.sections)
        if len(ids) != len(DAILY_SECTION_IDS) or set(ids) != set(DAILY_SECTION_IDS):
            raise ValueError("日报版式必须完整包含十二个共享区域，不能重复、遗漏或添加未知区域")
        if ids[0] != "OVERVIEW" or ids[-1] != "ACTIONS":
            raise ValueError("巡检总览必须在最前，处置与复查清单必须在最后")
        for section in self.sections:
            if section.section_id in {"OVERVIEW", "ACTIONS"} and section.visibility != "ALWAYS":
                raise ValueError("总览和处置区域必须始终展示")
        return self
