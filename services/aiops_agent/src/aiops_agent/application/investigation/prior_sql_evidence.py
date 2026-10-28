"""有界搜索同会话的原始 SQL 观察；历史结论、跨主体值和旧授权不能被继承。"""

from __future__ import annotations

import json

from aiops_agent.application.diagnosis.sql_evidence_answer import _matches, sql_subject_specs
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.playbooks import canonical_hash


_SQL_TOOLS = frozenset({
    "db.postgresql.statement.details", "db.postgresql.statement.subjects", "db.postgresql.statement.top",
    "db.mysql.statement.digest.top", "db.sql.cursor_details", "db.sql.top_current",
    "workload.sql.top_window",
})
MAX_TURNS = 16
MAX_FACTS = 8
MAX_ROWS = 512
MAX_BYTES = 262144


async def load_prior_sql_evidence(*, uow, turn, run, task_frame: dict) -> dict | None:
    """每次重验当前授权和来源哈希；仅原样继承匹配主体的行，不生成窗口差分。"""
    snapshot = dict(run.plan_snapshot_json or {})
    frozen = dict(snapshot.get("target") or {})
    matches = sql_subject_specs(task_frame, str(frozen.get("db_type") or "").upper())
    if not matches:
        return None
    target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=run.domain_id)
    if (target is None or str(target.status) != "ENABLED"
        or str(target.row_version) != str(frozen.get("row_version"))
        or str(target.db_type).upper() != str(frozen.get("db_type")).upper()
        or not await uow.agents.version_has_target(agent_version_id=run.agent_version_id, target_id=run.target_id)):
        return None
    selection = dict(dict(snapshot.get("client_metadata") or {}).get("sql_selection") or {})
    # 显式选择同样重新读取原 Artifact，不能将接收 Turn 时的可信标签永久沿用。
    if selection:
        from aiops_agent.application.sql_selection import resolve_sql_selection
        reference = selection.get("selection_ref")
        conversation = await uow.conversations.get_conversation(
            conversation_id=turn.conversation_id, domain_id=run.domain_id,
        )
        if not reference or conversation is None:
            return None
        try:
            resolved = await resolve_sql_selection(
                uow=uow, reference=reference, conversation=conversation, actor_id=run.actor_id,
                ttl_seconds=int(selection["selection_ttl_seconds"]),
            )
        except (ValueError, PermissionError, KeyError, TypeError):
            return None
        if resolved["subject_ref"] != selection.get("subject_ref") or resolved["target_id"] != str(run.target_id):
            return None
        return resolved
    if int(turn.turn_no) <= 1:
        return None
    previous_turns = await uow.turns.list_prior_turns(
        conversation_id=turn.conversation_id, before_turn_no=int(turn.turn_no), limit=MAX_TURNS + 1,
    )
    facts, sources = [], []
    row_count, byte_count, scanned = 0, 0, 0
    truncated = len(previous_turns) > MAX_TURNS
    seen = set()
    for previous in previous_turns[:MAX_TURNS]:
        scanned += 1
        if str(previous.status) not in {"COMPLETED", "PARTIAL"} or previous.sufficiency_artifact_id is None:
            continue
        if any(str(getattr(previous, key)) != str(getattr(turn, key))
               for key in ("domain_id", "conversation_id", "created_by")):
            continue
        link = await uow.turns.get_run_link(turn_id=previous.turn_id, purpose="PRIMARY")
        prior_run = await uow.runs.get_run(ops_run_id=link.ops_run_id) if link else None
        if prior_run is None or any(str(getattr(prior_run, key)) != str(getattr(run, key))
                                   for key in ("domain_id", "target_id", "agent_id", "agent_version_id", "actor_id")):
            continue
        prior_target = dict(dict(prior_run.plan_snapshot_json or {}).get("target") or {})
        if (str(prior_target.get("row_version")) != str(frozen.get("row_version"))
            or str(prior_target.get("db_type") or "").upper() != str(frozen.get("db_type") or "").upper()):
            continue
        artifact = await uow.runs.get_artifact(artifact_id=previous.sufficiency_artifact_id)
        if (artifact is None or str(artifact.ops_run_id) != str(prior_run.ops_run_id)
            or artifact.schema_version != "DBA_SUFFICIENCY.v1"
            or artifact.trust_level != "SOURCE_VERIFIED"
            or canonical_hash(artifact.payload_json) != artifact.content_hash):
            continue
        try:
            assessment = DbaSufficiencyAssessment.model_validate(artifact.payload_json)
        except (ValueError, TypeError):
            continue
        for fact in assessment.evidence:
            if (fact.trust_level != "SOURCE_VERIFIED" or fact.tool_id not in _SQL_TOOLS
                or fact.source_id == "prior_turn.sql_observation"
                or fact.presentation_kind not in {"TABLE", "TABLE_AND_CHART"}):
                continue
            names = [str(column.get("name") or "").lower() for column in fact.columns]
            if len(names) != len(set(names)):
                continue
            indices = [index for index, row in enumerate(fact.rows[:MAX_ROWS])
                       if len(row) == len(names) and any(_matches(dict(zip(names, row, strict=True)), subject, spec[1]) for subject, spec in matches)]
            if not indices:
                continue
            key = (str(fact.artifact_id), fact.step_id)
            if key in seen:
                continue
            seen.add(key)
            rows = tuple(fact.rows[index] for index in indices)
            projected = fact.model_copy(update={
                "rows": rows, "row_count": len(rows), "truncated": fact.truncated or len(fact.rows) > MAX_ROWS,
                "derivations": (), "calculation_graphs": (), "series_computations": (),
            }).model_dump(mode="json")
            size = len(json.dumps(projected, ensure_ascii=False, separators=(",", ":")).encode())
            if len(facts) >= MAX_FACTS or row_count + len(rows) > MAX_ROWS or byte_count + size > MAX_BYTES:
                truncated = True
                continue
            facts.append(projected)
            sources.append({
                "fact_index": len(facts) - 1, "source_turn_id": str(previous.turn_id),
                "source_run_id": str(prior_run.ops_run_id), "source_artifact_id": str(artifact.artifact_id),
                "source_content_hash": artifact.content_hash, "source_row_indices": indices,
            })
            row_count += len(rows)
            byte_count += size
    if not facts and not truncated:
        return None
    return {
        **({key: sources[0][key] for key in ("source_turn_id", "source_run_id", "source_artifact_id")} if sources else {}),
        "facts": facts, "sources": sources,
        "search": {"scanned_turns": scanned, "turn_limit": MAX_TURNS, "fact_limit": MAX_FACTS,
                   "row_limit": MAX_ROWS, "byte_limit": MAX_BYTES, "truncated": truncated,
                   "limitation": "仅搜索当前会话内同用户、Agent 版本和未变更 Target 的可核验证据；保留原时间，不代表当前值。"},
    }
