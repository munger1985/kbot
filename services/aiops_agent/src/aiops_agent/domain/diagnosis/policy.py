"""Evidence Request 校验和根因等级确定性上限。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from aiops_agent.contracts.diagnosis import (
    DiagnosisRoundAssessment,
    DiagnosisRoundDraft,
    EvidenceIndex,
    RootCauseAssessment,
    ValidatedEvidencePlan,
    ValidatedEvidenceRequest,
)
from aiops_agent.contracts.diagnosis.models import RejectedEvidenceRequest
from aiops_agent.diagnostics.registry import DiagnosticRegistry


_GRADE_ORDER = {
    "INCONCLUSIVE": 0,
    "POSSIBLE": 1,
    "PROBABLE": 2,
    "CONFIRMED": 3,
}


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class EvidenceRequestBudget:
    remaining_tool_calls: int
    prior_fingerprints: frozenset[str] = frozenset()


def validate_evidence_requests(
    draft: DiagnosisRoundDraft,
    *,
    database_snapshot: dict[str, Any],
    registry: DiagnosticRegistry,
    budget: EvidenceRequestBudget,
) -> ValidatedEvidencePlan:
    selected_tools = {
        item["tool_id"]: item
        for item in database_snapshot.get("tools", [])
    }
    known_hypotheses = {
        item.hypothesis_key for item in draft.hypotheses
    }
    accepted: list[ValidatedEvidenceRequest] = []
    rejected: list[RejectedEvidenceRequest] = []
    seen = set(budget.prior_fingerprints)
    for request in draft.evidence_requests:
        reason = None
        tool = selected_tools.get(request.tool_id)
        if tool is None:
            reason = "TOOL_NOT_AVAILABLE"
        elif not request.hypothesis_keys or not set(
            request.hypothesis_keys
        ) <= known_hypotheses:
            reason = "HYPOTHESIS_REFERENCE_INVALID"
        else:
            try:
                exact = registry.resolve_exact(
                    tool_id=tool["tool_id"],
                    tool_version=tool["version"],
                    db_type=database_snapshot["db_type"],
                    variant=tool["variant"],
                    template_sha256=tool["template_sha256"],
                )
                parameters = registry.validate_parameters(
                    exact, request.parameters
                )
            except (LookupError, ValueError):
                reason = "PARAMETER_INVALID"
                parameters = {}
            fingerprint = _hash(
                {
                    "target_row_version": database_snapshot[
                        "target_row_version"
                    ],
                    "tool_id": request.tool_id,
                    "parameters": parameters,
                }
            )
            if reason is None and fingerprint in seen:
                reason = "DUPLICATE_REQUEST"
            if reason is None and len(accepted) >= budget.remaining_tool_calls:
                reason = "BUDGET_EXCEEDED"
        if reason is not None:
            rejected.append(
                RejectedEvidenceRequest(
                    request_key=request.request_key,
                    tool_id=request.tool_id,
                    reason_code=reason,
                )
            )
            continue
        seen.add(fingerprint)
        accepted.append(
            ValidatedEvidenceRequest(
                request_key=request.request_key,
                tool_id=request.tool_id,
                tool_version=tool["version"],
                variant=tool["variant"],
                parameters=parameters,
                request_fingerprint=fingerprint,
                hypothesis_keys=request.hypothesis_keys,
            )
        )
    if accepted:
        decision = "COLLECT"
    elif draft.stop_recommendation == "FINALIZE":
        decision = "FINALIZE"
    else:
        decision = "STOP_INCONCLUSIVE"
    return ValidatedEvidencePlan(
        round_no=draft.round_no,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        remaining_tool_budget=max(
            0, budget.remaining_tool_calls - len(accepted)
        ),
        decision=decision,
    )


def assess_root_cause(
    *,
    target_id: str,
    evidence: EvidenceIndex,
    assessment: DiagnosisRoundAssessment,
) -> RootCauseAssessment:
    supported = [
        item
        for item in assessment.hypothesis_assessments
        if item.status == "SUPPORTED" and item.causal_role == "ROOT"
    ]
    primary = supported[0] if supported else None
    all_refs = {item.fact_id: item for item in evidence.facts}
    supporting_refs = tuple(
        dict.fromkeys(
            ref
            for item in supported
            for ref in item.supporting_fact_refs
            if ref in all_refs
        )
    )
    counter_refs = tuple(
        dict.fromkeys(
            ref
            for item in assessment.hypothesis_assessments
            for ref in item.counter_fact_refs
            if ref in all_refs
        )
    )
    verified_supporting_refs = tuple(
        ref
        for ref in supporting_refs
        if all_refs[ref].trust_level == "SOURCE_VERIFIED"
    )
    groups = {
        all_refs[ref].source_group_id for ref in verified_supporting_refs
    }
    direct_tests = sum(
        1
        for item in supported
        for result in item.test_results
        if result.outcome == "SUPPORTS" and result.strength == "DIRECT"
    )
    quality_flags = {
        flag
        for ref in supporting_refs
        for flag in all_refs[ref].quality_flags
    }
    reasons: list[str] = []
    if primary is None or not verified_supporting_refs:
        ceiling = "INCONCLUSIVE"
        reasons.append("没有由当前状态事实支持的根因假设")
    elif quality_flags & {
        "TRUNCATED",
        "LOW_COVERAGE",
        "WINDOW_MISMATCH",
        "UNIT_INCOMPATIBLE",
    }:
        ceiling = "POSSIBLE"
        reasons.append("关键证据存在质量缺口")
    elif (
        len(groups) >= 2
        and direct_tests >= 2
        and not counter_refs
        and not any(
            item.remaining_gaps
            for item in assessment.hypothesis_assessments
            if item.status == "SUPPORTED"
        )
    ):
        ceiling = "CONFIRMED"
    elif len(groups) >= 2 and direct_tests >= 1 and not counter_refs:
        ceiling = "PROBABLE"
        reasons.append("尚缺完整的直接致因闭环验证")
    else:
        ceiling = "POSSIBLE"
        reasons.append("证据独立性或区分性测试不足")
    suggested = assessment.suggested_root_cause_level
    if primary is None:
        suggested = "INCONCLUSIVE"
    effective = min(
        (suggested, ceiling), key=lambda item: _GRADE_ORDER[item]
    )
    return RootCauseAssessment(
        target_id=target_id,
        suggested_level=suggested,
        eligible_ceiling=ceiling,
        effective_level=effective,
        primary_hypothesis_key=(
            primary.hypothesis_key if primary else None
        ),
        contributing_hypothesis_keys=tuple(
            item.hypothesis_key
            for item in assessment.hypothesis_assessments
            if item.status == "SUPPORTED"
            and item.causal_role == "CONTRIBUTOR"
        ),
        supporting_fact_refs=supporting_refs,
        counter_fact_refs=counter_refs,
        unresolved_gaps=tuple(
            gap
            for item in assessment.hypothesis_assessments
            for gap in item.remaining_gaps
        ),
        downgrade_reasons=tuple(reasons),
    )
