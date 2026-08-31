"""调查规划所需的冻结上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from aiops_agent.application.conversation_inputs import (
    ConversationUploadSource,
    ResolvedConversationUpload,
)
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot


@dataclass(frozen=True, slots=True)
class TurnPlanningContext:
    domain_id: int
    turn_id: UUID
    conversation_id: UUID
    ops_run_id: UUID
    agent_id: UUID
    target_id: UUID
    source_ids: tuple[UUID, ...]
    actor_id: str
    question: str
    content: tuple[dict, ...]
    image_capabilities: dict
    recent_context: tuple[str, ...]
    trace_id: str
    deadline: datetime | None
    target_context: dict
    capabilities: DbaCapabilitySnapshot
    database_execution: dict
    change_context: dict
    source_run_evidence: dict | None = None
    raw_uploads: tuple[ConversationUploadSource, ...] = ()
    resolved_uploads: tuple[ResolvedConversationUpload, ...] = ()
    input_artifact_id: UUID | None = None
    upload_artifact_ids: tuple[tuple[int, UUID, UUID | None], ...] = ()


class PlanningAlreadyApplied(Exception):
    def __init__(self, result: dict) -> None:
        super().__init__("Turn 计划已经持久化")
        self.result = result
