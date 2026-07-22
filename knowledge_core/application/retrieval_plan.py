"""Versioned retrieval planning and object-level selection contracts.

The planner is deterministic at the boundary. An optional general LLM adapter
may enrich the plan/selection, but it can never alter scope, status or safety
filters and may only return pre-allocated labels.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Protocol, Sequence


TASK_MODES = frozenset({"DISCOVER", "ANSWER", "SUMMARIZE", "COMPARE"})
TARGET_LEVELS = frozenset({"BUNDLE", "DOCUMENT", "EVIDENCE"})
COVERAGE_MODES = frozenset({"BREADTH", "DEPTH", "BALANCED", "STRUCTURAL"})
EVIDENCE_PREFERENCES = frozenset({"TEXT", "TABLE", "TABLE_ROW", "IMAGE", "SHEET", "CELL_RANGE"})
RELEVANCE_LEVELS = ("DIRECT", "STRONG", "POSSIBLE", "SEMANTIC_ONLY", "IRRELEVANT")
SUPPORT_LEVELS = ("DIRECT_SUPPORT", "PARTIAL_SUPPORT", "CONTEXT_ONLY", "CONTRADICTS", "NO_SUPPORT")


@dataclass(frozen=True)
class RetrievalQueryPlan:
    query_plan_id: str
    version: str
    task_mode: str
    target_level: str
    coverage_mode: str
    evidence_preferences: tuple[str, ...]
    semantic_query: str
    exact_phrases: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    hard_filters: dict[str, Any] = field(default_factory=dict)
    soft_facets: dict[str, Any] = field(default_factory=dict)
    resolved_references: tuple[dict[str, Any], ...] = ()
    plan_status: str = "READY"
    warnings: tuple[str, ...] = ()


class RetrievalQueryPlanner:
    def plan(self, *, query: str, task_mode: str = "ANSWER", target_level: str = "EVIDENCE", coverage_mode: str = "BALANCED", evidence_preferences: Sequence[str] = (), hard_filters: dict[str, Any] | None = None) -> RetrievalQueryPlan:
        task_mode, target_level, coverage_mode = task_mode.upper(), target_level.upper(), coverage_mode.upper()
        warnings: list[str] = []
        if task_mode not in TASK_MODES or target_level not in TARGET_LEVELS or coverage_mode not in COVERAGE_MODES:
            task_mode, target_level, coverage_mode = "ANSWER", "EVIDENCE", "BALANCED"
            warnings.append("invalid planning dimensions; defaulted")
        preferences = tuple(value.upper() for value in evidence_preferences if value.upper() in EVIDENCE_PREFERENCES)
        phrases = tuple(re.findall(r'"([^"\n]+)"', query))
        identifiers = tuple(sorted(set(re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b|\b\d{4,}\b", query))))
        normalized = " ".join(query.split())
        fingerprint = sha256((normalized + task_mode + target_level + coverage_mode).encode()).hexdigest()[:24]
        return RetrievalQueryPlan(
            query_plan_id=f"qplan:{fingerprint}", version="retrieval-plan/v1",
            task_mode=task_mode, target_level=target_level, coverage_mode=coverage_mode,
            evidence_preferences=preferences, semantic_query=normalized,
            exact_phrases=phrases, identifiers=identifiers,
            hard_filters=dict(hard_filters or {}), plan_status="DEGRADED_DEFAULT" if warnings else "READY",
            warnings=tuple(warnings),
        )


@dataclass(frozen=True)
class CandidateDecision:
    candidate_label: str
    relevance: str
    matched_requirements: tuple[str, ...] = ()
    reason_refs: tuple[str, ...] = ()


class CandidateSelector(Protocol):
    async def select(self, *, query: str, candidates: Sequence[dict[str, Any]]) -> Sequence[CandidateDecision]: ...


class DeterministicCandidateSelector:
    """Safe degraded selector used when the general LLM is unavailable."""
    async def select(self, *, query: str, candidates: Sequence[dict[str, Any]]) -> Sequence[CandidateDecision]:
        terms = {term.lower() for term in re.findall(r"\w+", query) if len(term) > 2}
        decisions = []
        for index, candidate in enumerate(candidates):
            text = " ".join(str(candidate.get(key, "")) for key in ("display_title", "profile_text", "matched_snippet")).lower()
            overlap = len(terms.intersection(set(re.findall(r"\w+", text))))
            relevance = "DIRECT" if overlap >= 3 else "STRONG" if overlap else "SEMANTIC_ONLY"
            decisions.append(CandidateDecision(str(candidate.get("candidate_label", index)), relevance, (), ("lexical_overlap",)))
        return tuple(decisions)


@dataclass(frozen=True)
class EvidenceSupportDecision:
    group_label: str
    support: str
    primary_item_labels: tuple[str, ...] = ()
    structural_context_labels: tuple[str, ...] = ()
    neighbor_labels: tuple[str, ...] = ()
    answerable_aspects: tuple[str, ...] = ()
    unsupported_aspects: tuple[str, ...] = ()


class EvidenceSupportJudge(Protocol):
    async def judge(self, *, query: str, groups: Sequence[dict[str, Any]]) -> Sequence[EvidenceSupportDecision]: ...


class DeterministicEvidenceSupportJudge:
    async def judge(self, *, query: str, groups: Sequence[dict[str, Any]]) -> Sequence[EvidenceSupportDecision]:
        terms = {term.lower() for term in re.findall(r"\w+", query) if len(term) > 2}
        result = []
        for index, group in enumerate(groups):
            text = str(group.get("primary_text", group.get("content_text", ""))).lower()
            overlap = len(terms.intersection(set(re.findall(r"\w+", text))))
            support = "DIRECT_SUPPORT" if overlap >= 2 else "PARTIAL_SUPPORT" if overlap else "NO_SUPPORT"
            primary = tuple(str(value) for value in group.get("primary_item_labels", ()))
            result.append(EvidenceSupportDecision(str(group.get("group_label", index)), support, primary_item_labels=primary))
        return tuple(result)
