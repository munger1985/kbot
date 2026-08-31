"""DBA 调查 Tool 执行结果 Artifact 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from aiops_agent.contracts.artifacts.database import EvidenceGap
from platform_core.contracts.aiops.playbooks import PresentationPreference
from platform_core.contracts.aiops.types import MeasurementSemantics
from platform_core.contracts.aiops.executor import DatabaseObservation


class ToolOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    tool_id: str
    tool_version: str
    status: Literal["SUCCEEDED", "GAP", "SKIPPED"]
    observation: DatabaseObservation | None = None
    gap: EvidenceGap | None = None


class DbaToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DBA_TOOL_RESULT.v1"] = "DBA_TOOL_RESULT.v1"
    source_type: Literal["PLAYBOOK", "TOOL"]
    source_id: str
    source_version: str
    definition_hash: str
    output_schema: str
    measurement_semantics: MeasurementSemantics
    presentation_kind: PresentationPreference
    status: Literal["SUCCEEDED", "PARTIAL", "FAILED"]
    tool_outcomes: tuple[ToolOutcome, ...]


def is_turn_evidence_outcome(
    result: DbaToolResult,
    outcome: ToolOutcome,
) -> bool:
    """判断工具结果是否应作为当前 Turn 的可引用证据。"""
    return (
        outcome.observation is not None
        and (
            outcome.tool_id != "db.instance.identity"
            or result.source_type == "TOOL"
        )
    )
