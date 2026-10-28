"""历史 SQL 在发现、提交、领取、报告和案例中的原始来源及当前授权重放。"""

import json
from uuid import UUID

from aiops_agent.application.diagnosis.evidence_projection import (
    database_evidence_fact, project_evidence_rows, top_sql_window_evidence_fact,
)
from aiops_agent.application.investigation.evidence_scope import RUN_EVIDENCE_SCOPE_FIELDS, TURN_EVIDENCE_SCOPE_FIELDS
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceFact
from aiops_agent.playbooks import canonical_hash


SCHEMA = "PRIOR_TURN_SQL_EVIDENCE.v1"
ARTIFACT_TYPE = "PRIOR_TURN_SQL_EVIDENCE"
KEY = "turn-prior-sql-evidence:1"
_COMPUTED = {"derivations", "calculation_graphs", "series_computations", "visualizations"}


async def validate_original_sql_fact(*, uow, run, fact):
    """充分性快照只作定位；必须回查原始工具或原始窗口工件，不能循环自证。"""
    try:
        if fact.source_id == "prior_turn.sql_observation" or fact.followup_run_id is not None:
            return False
        artifact = await uow.runs.get_artifact(artifact_id=UUID(str(fact.artifact_id)))
        if (artifact is None or str(artifact.ops_run_id) != str(run.ops_run_id)
            or artifact.trust_level != "SOURCE_VERIFIED"
            or canonical_hash(artifact.payload_json) != artifact.content_hash):
            return False
        if artifact.schema_version == "DBA_TOOL_RESULT.v1":
            result = DbaToolResult.model_validate(artifact.payload_json)
            outcomes = [item for item in result.tool_outcomes if item.step_id == fact.step_id
                and item.tool_id == fact.tool_id and item.status == "SUCCEEDED" and item.observation is not None]
            if len(outcomes) != 1 or str(outcomes[0].observation.target_id) != str(run.target_id):
                return False
            frozen_type = str(dict(run.plan_snapshot_json or {}).get("target", {}).get("db_type") or "").upper()
            if outcomes[0].observation.db_type != frozen_type:
                return False
            expected = database_evidence_fact(result=result, outcome=outcomes[0], artifact_id=artifact.artifact_id)
        elif artifact.schema_version == "TOP_SQL_WINDOW_RESULT.v1":
            if str(artifact.payload_json.get("target_id")) != str(run.target_id):
                return False
            expected = top_sql_window_evidence_fact(result=artifact.payload_json["result"], artifact_id=artifact.artifact_id)
            if expected is None:
                return False
        else:
            return False
        return expected.model_dump(mode="json", exclude=_COMPUTED) == fact.model_dump(mode="json", exclude=_COMPUTED)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def _same_scope(left, right, fields):
    return all(str(getattr(left, key)) == str(getattr(right, key)) for key in fields)


async def validate_prior_sql_payload(*, uow, run, payload, turn=None):
    """独立重建每份筛行结果及源行号，复用不延续旧 Target 或 Agent 授权。"""
    try:
        from aiops_agent.application.investigation.prior_sql_evidence import MAX_FACTS, MAX_ROWS, MAX_BYTES, _SQL_TOOLS
        facts, sources = payload["facts"], payload["sources"]
        if (not isinstance(facts, list) or not isinstance(sources, list) or len(facts) != len(sources)
            or len(facts) > MAX_FACTS or sum(len(item.get("rows") or ()) for item in facts) > MAX_ROWS
            or sum(len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode()) for item in facts) > MAX_BYTES):
            return False
        if not facts:
            return not sources and bool(dict(payload.get("search") or {}).get("truncated"))
        if turn is None:
            link = await uow.turns.get_run_link_by_ops_run_id(ops_run_id=run.ops_run_id)
            turn = await uow.turns.get_turn(domain_id=run.domain_id, turn_id=link.turn_id) if link else None
        if turn is None or str(turn.created_by) != str(run.actor_id):
            return False
        frozen = dict(dict(run.plan_snapshot_json or {}).get("target") or {})
        target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=run.domain_id)
        if (target is None or str(target.status) != "ENABLED"
            or str(target.row_version) != str(frozen.get("row_version"))
            or str(target.db_type).upper() != str(frozen.get("db_type")).upper()
            or not await uow.agents.version_has_target(agent_version_id=run.agent_version_id, target_id=run.target_id)):
            return False
        seen = set()
        for index, (raw, proof) in enumerate(zip(facts, sources, strict=True)):
            if type(proof["fact_index"]) is not int or proof["fact_index"] != index:
                return False
            previous = await uow.turns.get_turn(domain_id=run.domain_id, turn_id=UUID(proof["source_turn_id"]))
            if (previous is None or previous.status not in {"COMPLETED", "PARTIAL"}
                or not _same_scope(previous, turn, TURN_EVIDENCE_SCOPE_FIELDS)
                or int(previous.turn_no) >= int(turn.turn_no)
                or str(previous.sufficiency_artifact_id) != proof["source_artifact_id"]):
                return False
            link = await uow.turns.get_run_link(turn_id=previous.turn_id, purpose="PRIMARY")
            source_run = await uow.runs.get_run(ops_run_id=UUID(proof["source_run_id"]))
            if (link is None or str(link.ops_run_id) != proof["source_run_id"] or source_run is None
                or not _same_scope(source_run, run, RUN_EVIDENCE_SCOPE_FIELDS)):
                return False
            old_target = dict(dict(source_run.plan_snapshot_json or {}).get("target") or {})
            if any(str(old_target.get(key)) != str(frozen.get(key)) for key in ("row_version", "db_type")):
                return False
            artifact = await uow.runs.get_artifact(artifact_id=UUID(proof["source_artifact_id"]))
            if (artifact is None or str(artifact.ops_run_id) != str(source_run.ops_run_id)
                or artifact.schema_version != "DBA_SUFFICIENCY.v1" or artifact.trust_level != "SOURCE_VERIFIED"
                or artifact.content_hash != proof["source_content_hash"]
                or canonical_hash(artifact.payload_json) != artifact.content_hash):
                return False
            assessment = DbaSufficiencyAssessment.model_validate(artifact.payload_json)
            originals = [item for item in assessment.evidence if item.evidence_ref == proof["source_evidence_ref"]]
            if len(originals) != 1 or originals[0].tool_id not in _SQL_TOOLS:
                return False
            original = originals[0]
            identity = (str(original.artifact_id), original.step_id)
            if identity in seen or not await validate_original_sql_fact(uow=uow, run=source_run, fact=original):
                return False
            seen.add(identity)
            indices = proof["source_row_indices"]
            if not isinstance(indices, list) or not indices:
                return False
            projected = project_evidence_rows(original, tuple(indices), truncated=len(original.rows) > MAX_ROWS)
            if projected.model_dump(mode="json") != TurnEvidenceFact.model_validate(raw).model_dump(mode="json"):
                return False
        return True
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


async def prior_sql_security_level(*, uow, payload):
    """筛行和跨轮不能降低原工具或充分性工件的密级。"""
    level = 1
    for raw, proof in zip(payload["facts"], payload["sources"], strict=True):
        for identity in (proof["source_artifact_id"], raw["artifact_id"]):
            source = await uow.runs.get_artifact(artifact_id=UUID(str(identity)))
            if source is None or type(source.security_level) is not int:
                raise ValueError("历史 SQL 来源密级不可核验")
            level = max(level, source.security_level)
    return level


async def validate_reused_prior_sql(*, uow, run, artifact, turn=None):
    try:
        return (artifact.artifact_type == ARTIFACT_TYPE and artifact.artifact_key == KEY
            and artifact.schema_version == SCHEMA and artifact.trust_level == "SOURCE_VERIFIED"
            and str(artifact.ops_run_id) == str(run.ops_run_id)
            and canonical_hash(artifact.payload_json) == artifact.content_hash
            and await validate_prior_sql_payload(uow=uow, run=run, payload=artifact.payload_json, turn=turn)
            and int(artifact.security_level) >= await prior_sql_security_level(uow=uow, payload=artifact.payload_json))
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


async def validate_prior_sql_facts(*, uow, run, payload):
    """无卡片、无原生映射或无对象图也必须校验历史 SQL 来源。"""
    from aiops_agent.application.diagnosis.sql_evidence_answer import rebind_prior_sql_facts
    sources = {}
    try:
        for raw in payload.get("evidence", ()):
            if raw.get("source_id") != "prior_turn.sql_observation":
                continue
            fact = TurnEvidenceFact.model_validate(raw)
            identity = str(fact.artifact_id)
            if identity not in sources:
                artifact = await uow.runs.get_artifact(artifact_id=UUID(str(fact.artifact_id)))
                if artifact is None or not await validate_reused_prior_sql(uow=uow, run=run, artifact=artifact):
                    return False
                frame = dict(dict(run.plan_snapshot_json or {}).get("answer_context") or {}).get("task_frame") or {}
                engine = str(dict(run.plan_snapshot_json or {}).get("target", {}).get("db_type") or "").upper()
                sources[identity] = rebind_prior_sql_facts(payload=artifact.payload_json, artifact_id=identity,
                    task_frame=frame, database_type=engine)
            expected = [item for item in sources[identity] if item.evidence_ref == fact.evidence_ref]
            if len(expected) != 1 or expected[0].model_dump(mode="json") != fact.model_dump(mode="json"):
                return False
        return True
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


async def revalidate_prior_sql_inputs(*, uow, run_id, artifacts, selected):
    """每次领取撤销失效历史观测及其旧判断，不把删数值后的旧解释继续送入模型。"""
    imported = [item for item in artifacts if item.artifact_type == ARTIFACT_TYPE]
    if not imported:
        return selected
    from aiops_agent.contracts.turn_answer import TurnEvidenceGap
    from platform_core.contracts.aiops import SufficiencyStatus
    run = await uow.runs.get_run(ops_run_id=run_id)
    invalid = {str(item.artifact_id) for item in imported
        if run is None or not await validate_reused_prior_sql(uow=uow, run=run, artifact=item)}
    if not invalid:
        return selected
    gap = TurnEvidenceGap(source_id="prior_turn.sql_observation", step_id="prior.sql", code="PRIOR_SQL_SOURCE_UNAVAILABLE",
        detail="历史 SQL 的当前授权或原始来源已变化，旧观测与旧判断未沿用；请在当前权限下重新取证。", retryable=False)
    result = []
    for item in selected:
        raw = dict(item.payload or {})
        if str(item.artifact_id) in invalid or item.schema_version == "AIOPS_TURN_RESULT.v1" and any(
            str(fact.get("artifact_id")) in invalid for fact in raw.get("evidence", ())):
            safe = {"code": gap.code}
            item = item.model_copy(update={"schema_version": "PRIOR_SQL_UNAVAILABLE.v1", "payload": safe, "content_hash": canonical_hash(safe)})
        elif item.schema_version == "DBA_SUFFICIENCY.v1" and any(str(fact.get("artifact_id")) in invalid for fact in raw.get("evidence", ())):
            assessment = DbaSufficiencyAssessment.model_validate(raw)
            safe = assessment.model_copy(update={
                "status": SufficiencyStatus.UNSAFE if assessment.status == SufficiencyStatus.UNSAFE else SufficiencyStatus.PARTIAL,
                "evidence": tuple(fact for fact in assessment.evidence if str(fact.artifact_id) not in invalid),
                "gaps": (*assessment.gaps, gap), "investigation": None, "clarification_question": None,
                "reasons": ("历史 SQL 来源已失效，旧调查判断未沿用。",),
            }).model_dump(mode="json")
            item = item.model_copy(update={"payload": safe, "content_hash": canonical_hash(safe)})
        result.append(item)
    return tuple(result)
