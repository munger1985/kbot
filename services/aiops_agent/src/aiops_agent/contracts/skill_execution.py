"""DBA Skill 执行结果 Artifact 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from aiops_agent.contracts.artifacts.database import EvidenceGap
from platform_core.contracts.aiops.conversation import MeasurementSemantics
from platform_core.contracts.aiops.executor import DatabaseObservation


class SkillToolOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    tool_id: str
    tool_version: str
    status: Literal["SUCCEEDED", "GAP", "SKIPPED"]
    observation: DatabaseObservation | None = None
    gap: EvidenceGap | None = None


class DbaSkillResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DBA_SKILL_RESULT.v1"] = "DBA_SKILL_RESULT.v1"
    skill_id: str
    skill_version: str
    manifest_hash: str
    output_schema: str
    measurement_semantics: MeasurementSemantics
    status: Literal["SUCCEEDED", "PARTIAL", "FAILED"]
    tool_outcomes: tuple[SkillToolOutcome, ...]
