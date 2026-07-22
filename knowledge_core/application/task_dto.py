"""Versioned Agent↔Skill task DTOs shared by 3.5 and future 4.0 routing."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeTask:
    task_id: str
    parent_run_id: str
    domain_id: int
    agent_id: int
    original_query: str
    standalone_query: str
    response_goal: str = "ANSWER"
    collection_ids: tuple[int, ...] = ()
    security_level: int = 3
    prior_citations: tuple[dict[str, Any], ...] = ()
    deadline_ms: int = 120000
    # Optional per-request override.  In normal operation the V2 route resolves
    # the model from the Agent configuration, so this is primarily useful for
    # tests and controlled rollout.
    answer_model: str | None = None


@dataclass
class KnowledgeTaskResult:
    task_id: str
    status: str
    citation_pack: dict[str, Any] | None = None
    grounded_findings: list[dict[str, Any]] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    clarification: str | None = None
    warnings: list[str] = field(default_factory=list)
    retrieval_run_id: str | None = None
