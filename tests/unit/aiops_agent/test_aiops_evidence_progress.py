"""证据去重以语义与对象为准，不能以新 artifact 引用当作进展。"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from aiops_agent.application.runtime.service import _assessment_progress, _replan_budget_exhausted
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceFact, TurnEvidenceGap
from platform_core.identity import uuid7


def fact(*, reference="one", query_id="42", captured_at="2026-01-01T00:00:00Z", semantics="CUMULATIVE_SINCE_LOAD"):
    return TurnEvidenceFact(
        evidence_ref=reference, artifact_id=str(uuid7()), source_id="statement", step_id="a1",
        tool_id="db.postgresql.statement.details", measurement_semantics=semantics,
        presentation_kind="TABLE", captured_at=captured_at,
        columns=({"name": "query_id"}, {"name": "execution_count"}),
        rows=((query_id, 10),), row_count=1,
    )


def assessment(evidence=(), gaps=()):
    return DbaSufficiencyAssessment(status="PARTIAL", evidence=evidence, gaps=gaps)


def test_new_reference_is_not_new_evidence_but_distinct_object_is():
    previous = assessment((fact(),)).model_dump(mode="json")
    assert not _assessment_progress(previous, assessment((fact(reference="two"),)))
    assert _assessment_progress(previous, assessment((fact(reference="two", query_id="43"),)))


def test_repeated_activity_samples_and_resolved_gaps_are_progress():
    before = fact(semantics="CURRENT_ACTIVITY")
    later = fact(semantics="CURRENT_ACTIVITY", captured_at="2026-01-01T00:00:10Z")
    assert _assessment_progress(assessment((before,)).model_dump(mode="json"), assessment((later,)))
    gap = TurnEvidenceGap(source_id="db", step_id="a1", code="MISSING", detail="missing")
    assert _assessment_progress(assessment((before,), (gap,)).model_dump(mode="json"), assessment((before,)))


def test_replan_budget_stops_new_replans_by_count_or_elapsed_time():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    turn = SimpleNamespace(current_plan_revision=2, started_at=start)
    assert not _replan_budget_exhausted(turn, now=start + timedelta(seconds=179), max_replans=2, max_seconds=180)
    assert _replan_budget_exhausted(turn, now=start + timedelta(seconds=180), max_replans=2, max_seconds=180)
    turn.current_plan_revision = 3
    assert _replan_budget_exhausted(turn, now=start, max_replans=2, max_seconds=180)
