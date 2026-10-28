"""有界复用同会话后台观察；报告和案例再次重放真实来源，不让导入副本自证。"""

from __future__ import annotations

import json
from datetime import timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.application.investigation.evidence_scope import (
    RUN_EVIDENCE_SCOPE_FIELDS as _SCOPE, TURN_EVIDENCE_SCOPE_FIELDS as _TURN_SCOPE,
)
from aiops_agent.application.investigation.future_observation import (
    FOLLOWUP_SCHEMA, FOLLOWUP_WORKFLOW, future_observation_payload,
)
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.playbooks import canonical_hash


MAX_TURNS = 16
MAX_GROUPS = 2
MAX_FACTS = 8
MAX_ROWS = 512
MAX_BYTES = 262144
ARTIFACT_TYPE = "FUTURE_OBSERVATION_EVIDENCE"
KEY_PREFIX = "turn-future-observation:"
_TERMINAL = frozenset({"COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED"})


class FutureObservationOrigin(BaseModel):
    """导入只冻结原始终态来源的引用，不冻结永久授权。"""

    model_config = ConfigDict(extra="forbid")

    source_turn_id: UUID
    background_run_id: UUID
    source_artifact_id: UUID
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_truncated: bool = False


def _same_scope(left, right, fields):
    return all(str(getattr(left, field)) == str(getattr(right, field)) for field in fields)


async def _current_turn(*, uow, run):
    link = await uow.turns.get_run_link_by_ops_run_id(ops_run_id=run.ops_run_id)
    return await uow.turns.get_turn(domain_id=run.domain_id, turn_id=link.turn_id) if link else None


async def _source_series(*, uow, run, turn, source_turn_id, background_run_id, now):
    """重新核验同会话来源、当前 Target/Agent 授权及全部原始时点。"""
    previous = await uow.turns.get_turn(domain_id=run.domain_id, turn_id=source_turn_id)
    if (previous is None or turn is None or not _same_scope(previous, turn, _TURN_SCOPE)
        or str(turn.created_by) != str(run.actor_id)
        or int(previous.turn_no) >= int(turn.turn_no) or previous.status not in {"COMPLETED", "PARTIAL"}):
        return None
    link = await uow.turns.get_run_link(turn_id=previous.turn_id, purpose="PRIMARY")
    background = await uow.runs.get_run(ops_run_id=background_run_id)
    if (link is None or background is None or background.workflow_kind != FOLLOWUP_WORKFLOW
        or background.status not in _TERMINAL or background.final_artifact_id is None
        or not _same_scope(background, run, _SCOPE)):
        return None
    info = dict(dict(background.plan_snapshot_json or {}).get("future_observation") or {})
    frozen = dict(dict(run.plan_snapshot_json or {}).get("target") or {})
    database = dict(dict(dict(background.plan_snapshot_json or {}).get("investigation_execution") or {}).get("database") or {})
    if (info.get("source_run_id") != str(link.ops_run_id)
        or str(frozen.get("row_version")) != str(database.get("target_row_version"))
        or str(frozen.get("db_type")) != str(database.get("db_type"))):
        return None
    final = await uow.runs.get_artifact(artifact_id=background.final_artifact_id)
    if (final is None or str(final.ops_run_id) != str(background.ops_run_id)):
        return None
    payload = await future_observation_payload(uow=uow, run=background, now=now, artifact=final)
    if payload is None:
        return None
    summary = DbaToolResult.model_validate(payload).observation_series
    if summary is None or summary.finished_at > now + timedelta(seconds=1):
        return None
    return background, final, payload


async def load_future_observation_evidence(*, uow, turn, run, prior_sql=None):
    """在共享事实/行数/字节预算内定位至多两组终态观察，不全库搜索。"""
    if int(turn.turn_no) <= 1:
        return ()
    previous_turns = await uow.turns.list_prior_turns(
        conversation_id=turn.conversation_id, before_turn_no=int(turn.turn_no), limit=MAX_TURNS + 1,
    )
    prior_facts = tuple(dict(prior_sql or {}).get("facts") or ())
    facts = len(prior_facts)
    rows = sum(len(fact.get("rows") or ()) for fact in prior_facts)
    used_bytes = sum(len(json.dumps(fact, ensure_ascii=False, separators=(",", ":")).encode()) for fact in prior_facts)
    now = await uow.runs.database_now()
    entries, seen = [], set()
    bounded = len(previous_turns) > MAX_TURNS or bool(dict(dict(prior_sql or {}).get("search") or {}).get("truncated"))
    for previous in previous_turns[:MAX_TURNS]:
        if previous.status not in {"COMPLETED", "PARTIAL"} or not _same_scope(previous, turn, _TURN_SCOPE):
            continue
        link = await uow.turns.get_run_link(turn_id=previous.turn_id, purpose="PRIMARY")
        if link is None:
            continue
        request = await uow.runs.get_artifact_by_key(ops_run_id=link.ops_run_id, artifact_key="future-observation:request")
        if (request is None or str(request.ops_run_id) != str(link.ops_run_id)
            or request.schema_version != FOLLOWUP_SCHEMA or request.trust_level != "SOURCE_VERIFIED"
            or canonical_hash(request.payload_json) != request.content_hash):
            continue
        try:
            background_id = UUID(request.payload_json["background_run_id"])
            if background_id in seen:
                continue
            seen.add(background_id)
            resolved = await _source_series(uow=uow, run=run, turn=turn, source_turn_id=previous.turn_id,
                background_run_id=background_id, now=now)
            if resolved is None:
                continue
            background, final, payload = resolved
            result = DbaToolResult.model_validate(payload)
            observations = [item.observation for item in result.tool_outcomes if item.observation is not None]
            count, row_count = len(observations), sum(item.row_count for item in observations)
            size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
            if not count:
                continue
            if len(entries) >= MAX_GROUPS or facts + count > MAX_FACTS or rows + row_count > MAX_ROWS or used_bytes + size > MAX_BYTES:
                bounded = True
                continue
            origin = FutureObservationOrigin(
                source_turn_id=previous.turn_id, background_run_id=background.ops_run_id,
                source_artifact_id=final.artifact_id, source_content_hash=final.content_hash,
            )
            entries.append({
                "artifact_key": f"{KEY_PREFIX}{len(entries) + 1}", "payload": payload,
                "provenance": {"producer": "aiops.future-observation-linker",
                    "future_observation_source": origin.model_dump(mode="json")},
                "security_level": int(final.security_level),
            })
            facts += count
            rows += row_count
            used_bytes += size
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    for entry in entries:
        entry["provenance"]["future_observation_source"]["search_truncated"] = bounded
    return tuple(entries)


async def validate_reused_future_observation(*, uow, run, artifact, now=None, turn=None):
    """领取、报告和案例使用时都重验；汇总副本的哈希一致并不足够。"""
    try:
        if (artifact.artifact_type != ARTIFACT_TYPE or not artifact.artifact_key.startswith(KEY_PREFIX)
            or str(artifact.ops_run_id) != str(run.ops_run_id)
            or artifact.schema_version != "DBA_TOOL_RESULT.v1" or artifact.trust_level != "SOURCE_VERIFIED"
            or canonical_hash(artifact.payload_json) != artifact.content_hash):
            return False
        origin = FutureObservationOrigin.model_validate(dict(artifact.provenance_json or {})["future_observation_source"])
        turn = turn or await _current_turn(uow=uow, run=run)
        now = now or await uow.runs.database_now()
        resolved = await _source_series(uow=uow, run=run, turn=turn,
            source_turn_id=origin.source_turn_id, background_run_id=origin.background_run_id, now=now)
        if resolved is None:
            return False
        _, final, payload = resolved
        return (str(final.artifact_id) == str(origin.source_artifact_id)
            and final.content_hash == origin.source_content_hash == artifact.content_hash
            and canonical_hash(payload) == artifact.content_hash
            and int(artifact.security_level) >= int(final.security_level))
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


async def revalidate_future_observation_inputs(*, uow, run_id, artifacts, selected):
    """Worker 领取时重新授权；撤权不让旧充分性副本把值再次送入模型。"""
    imported = [item for item in artifacts if item.artifact_type == ARTIFACT_TYPE]
    if not imported:
        return selected
    from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceGap
    from platform_core.contracts.aiops import SufficiencyStatus

    run = await uow.runs.get_run(ops_run_id=run_id)
    now = await uow.runs.database_now()
    invalid = {str(item.artifact_id) for item in imported
        if run is None or not await validate_reused_future_observation(uow=uow, run=run, artifact=item, now=now)}
    if not invalid:
        return selected
    gap = TurnEvidenceGap(source_id="evidence.future_observation", step_id="future.observation", code="FOLLOWUP_SOURCE_UNAVAILABLE",
        detail="后台观察的当前授权或原始来源已变化，未沿用旧结果；需要在当前权限下重新取证。", retryable=False)
    result = []
    for item in selected:
        raw = dict(item.payload or {})
        if str(item.artifact_id) in invalid:
            payload = {"code": gap.code}
            item = item.model_copy(update={"schema_version": "FUTURE_OBSERVATION_UNAVAILABLE.v1",
                "payload": payload, "content_hash": canonical_hash(payload)})
        elif item.schema_version == "DBA_SUFFICIENCY.v1" and any(str(fact.get("artifact_id")) in invalid for fact in raw.get("evidence", ())):
            assessment = DbaSufficiencyAssessment.model_validate(raw)
            # 模型旧判断可能引用撤销来源；整份调查解释不沿用，而不是只删数值留下结论。
            safe = assessment.model_copy(update={
                "status": SufficiencyStatus.UNSAFE if assessment.status == SufficiencyStatus.UNSAFE else SufficiencyStatus.PARTIAL,
                "evidence": tuple(fact for fact in assessment.evidence if str(fact.artifact_id) not in invalid),
                "gaps": (*assessment.gaps, gap), "investigation": None,
                "clarification_question": None,
                "reasons": ("后台观察已失效，旧调查判断未沿用。",),
            })
            payload = safe.model_dump(mode="json")
            item = item.model_copy(update={"payload": payload, "content_hash": canonical_hash(payload)})
        elif item.schema_version == "AIOPS_TURN_RESULT.v1" and any(str(fact.get("artifact_id")) in invalid for fact in raw.get("evidence", ())):
            payload = {"code": gap.code}
            item = item.model_copy(update={"schema_version": "FUTURE_OBSERVATION_UNAVAILABLE.v1",
                "payload": payload, "content_hash": canonical_hash(payload)})
        result.append(item)
    return tuple(result)


async def validate_future_observation_facts(*, uow, run, payload):
    """即使没有 Finding/对象图，报告或案例引用后台时点也必须重放原始来源。"""
    from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
    from aiops_agent.contracts.turn_answer import TurnEvidenceFact

    sources = {}
    try:
        for raw in payload.get("evidence", ()):
            if not raw.get("followup_run_id"):
                continue
            fact = TurnEvidenceFact.model_validate(raw)
            identity = str(fact.artifact_id)
            if identity not in sources:
                artifact = await uow.runs.get_artifact(artifact_id=UUID(str(fact.artifact_id)))
                if artifact is None or not await validate_reused_future_observation(uow=uow, run=run, artifact=artifact):
                    return False
                origin = FutureObservationOrigin.model_validate(artifact.provenance_json["future_observation_source"])
                sources[identity] = (origin.background_run_id, DbaToolResult.model_validate(artifact.payload_json), artifact.provenance_json)
            background_id, source, provenance = sources[identity]
            outcomes = [item for item in source.tool_outcomes if item.step_id == fact.step_id
                and item.tool_id == fact.tool_id and item.status == "SUCCEEDED" and item.observation is not None]
            if len(outcomes) != 1 or fact.source_id != source.source_id or str(fact.followup_run_id) != str(background_id):
                return False
            expected = database_evidence_fact(result=source, outcome=outcomes[0], artifact_id=identity, provenance=provenance)
            if (str(outcomes[0].observation.target_id) != str(run.target_id)
                or expected.model_dump(mode="json") != fact.model_dump(mode="json")):
                return False
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
    return True
