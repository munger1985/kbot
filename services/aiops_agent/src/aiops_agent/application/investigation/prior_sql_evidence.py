"""读取同一会话、主体与 Target 的上一轮 SQL 证据，不继承模型结论。"""

from __future__ import annotations

from pydantic import ValidationError

from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment


_SQL_TOOLS = frozenset({
    "db.postgresql.statement.details", "db.postgresql.statement.top",
    "db.mysql.statement.digest.top", "db.sql.cursor_details", "db.sql.top_current",
})


async def load_prior_sql_evidence(*, uow, turn, run) -> dict | None:
    if int(turn.turn_no) <= 1:
        return None
    previous = await uow.turns.get_previous_turn(
        conversation_id=turn.conversation_id, before_turn_no=int(turn.turn_no),
    )
    if previous is None or str(previous.status) not in {"COMPLETED", "PARTIAL"}:
        return None
    if any(str(getattr(previous, key)) != str(getattr(turn, key)) for key in ("domain_id", "conversation_id", "created_by")):
        return None
    if previous.sufficiency_artifact_id is None:
        return None
    link = await uow.turns.get_run_link(turn_id=previous.turn_id, purpose="PRIMARY")
    previous_run = await uow.runs.get_run(ops_run_id=link.ops_run_id) if link else None
    if previous_run is None or any(
        str(getattr(previous_run, key)) != str(getattr(run, key))
        for key in ("domain_id", "target_id", "agent_id", "actor_id")
    ):
        return None
    previous_target = dict(dict(previous_run.plan_snapshot_json or {}).get("target") or {})
    current_target = dict(dict(run.plan_snapshot_json or {}).get("target") or {})
    if (
        previous_target.get("row_version") is None
        or current_target.get("row_version") is None
        or not previous_target.get("db_type")
        or not current_target.get("db_type")
        or str(previous_target["row_version"]) != str(current_target["row_version"])
        or str(previous_target.get("db_type") or "").upper() != str(current_target.get("db_type") or "").upper()
    ):
        return None
    artifact = await uow.runs.get_artifact(artifact_id=previous.sufficiency_artifact_id)
    if artifact is None or str(artifact.ops_run_id) != str(previous_run.ops_run_id) or artifact.schema_version != "DBA_SUFFICIENCY.v1":
        return None
    try:
        assessment = DbaSufficiencyAssessment.model_validate(artifact.payload_json)
    except (ValidationError, TypeError, ValueError):
        return None
    facts = [
        fact.model_dump(mode="json") for fact in assessment.evidence
        if fact.trust_level == "SOURCE_VERIFIED" and fact.tool_id in _SQL_TOOLS
        and fact.source_id != "prior_turn.sql_observation"
        and fact.presentation_kind in {"TABLE", "TABLE_AND_CHART"}
    ][:8]
    if not facts:
        return None
    return {
        "source_turn_id": str(previous.turn_id), "source_run_id": str(previous_run.ops_run_id),
        "source_artifact_id": str(artifact.artifact_id), "facts": facts,
    }
