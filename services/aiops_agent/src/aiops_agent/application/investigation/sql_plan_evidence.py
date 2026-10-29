"""历史已存计划的有界发现、来源复核与模型输入围栏；不重启采集。"""

import json
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.diagnosis.sql_evidence_answer import _matches, sql_subject_specs
from aiops_agent.application.diagnosis.sql_plans import controlled_estimate_document
from aiops_agent.application.investigation.evidence_scope import RUN_EVIDENCE_SCOPE_FIELDS, TURN_EVIDENCE_SCOPE_FIELDS
from aiops_agent.application.sql_plan_results import owner_scope
from aiops_agent.application.sql_plan_results import RESULT_KEY, RESULT_SCHEMA
from aiops_agent.playbooks import canonical_hash


SCHEMA = "SQL_PLAN_CAPTURE_EVIDENCE.v1"
ARTIFACT_TYPE = "SQL_PLAN_CAPTURE_EVIDENCE"
KEY_PREFIX = "turn-sql-plan-capture:"
MAX_TURNS, MAX_PLANS, MAX_FACTS, MAX_ROWS, MAX_BYTES = 16, 4, 8, 512, 262144


class SqlPlanCaptureOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_turn_id: UUID
    source_run_id: UUID
    request_id: UUID
    capture_run_id: UUID
    result_artifact_id: UUID
    result_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_truncated: bool = False


def _same_scope(left, right, fields):
    return all(str(getattr(left, key)) == str(getattr(right, key)) for key in fields)


async def _resolve(*, uow, run, turn, origin, reader):
    """不打开嵌套UoW；不兑换参数，不借审批过期扩大执行权，也不把副本当来源。"""
    if reader is None or turn is None or str(turn.created_by) != str(run.actor_id):
        return None
    previous = await uow.turns.get_turn(domain_id=run.domain_id, turn_id=origin.source_turn_id)
    if (previous is None or previous.status not in {"COMPLETED", "PARTIAL"}
        or int(previous.turn_no) >= int(turn.turn_no)
        or not _same_scope(previous, turn, TURN_EVIDENCE_SCOPE_FIELDS)):
        return None
    link = await uow.turns.get_run_link(turn_id=previous.turn_id, purpose="PRIMARY")
    if link is None or str(link.ops_run_id) != str(origin.source_run_id):
        return None
    source = await uow.runs.get_run(ops_run_id=origin.source_run_id)
    if source is None or not _same_scope(source, run, RUN_EVIDENCE_SCOPE_FIELDS):
        return None
    await reader._materials._access(uow, run)
    verified = await reader.read(uow=uow, scope=owner_scope(source), source=source, request_id=origin.request_id)
    result, artifact = verified.result, verified.artifact
    now = await uow.runs.database_now()
    if (result is None or result.status != "SUCCEEDED" or artifact is None
        or result.captured_at > now or result.capture_run_id != origin.capture_run_id
        or str(artifact.artifact_id) != str(origin.result_artifact_id)
        or artifact.content_hash != origin.result_content_hash):
        return None
    return verified


async def load_sql_plan_capture_evidence(*, uow, turn, run, task_frame, reader, prior_sql=None, future=()):
    """最多16轮/4份计划，与历史SQL、后台观察共享8事实/512行/256KiB预算。"""
    matches = sql_subject_specs(task_frame, str(dict(run.plan_snapshot_json or {}).get("target", {}).get("db_type") or "").upper())
    if reader is None or not matches or int(turn.turn_no) <= 1:
        return ()
    previous_turns = await uow.turns.list_prior_turns(conversation_id=turn.conversation_id,
        before_turn_no=int(turn.turn_no), limit=MAX_TURNS + 1)
    prior = tuple(dict(prior_sql or {}).get("facts") or ())
    used_facts = len(prior)
    used_rows = sum(len(fact.get("rows") or ()) for fact in prior)
    used_bytes = sum(len(json.dumps(fact, ensure_ascii=False, separators=(",", ":")).encode()) for fact in prior)
    for entry in future:
        from aiops_agent.contracts.tool_execution import DbaToolResult
        result = DbaToolResult.model_validate(entry["payload"])
        observations = [item.observation for item in result.tool_outcomes if item.observation is not None]
        used_facts += len(observations)
        used_rows += sum(item.row_count for item in observations)
        used_bytes += len(json.dumps(entry["payload"], ensure_ascii=False, separators=(",", ":")).encode())
    bounded = len(previous_turns) > MAX_TURNS or bool(dict(dict(prior_sql or {}).get("search") or {}).get("truncated"))
    bounded = bounded or any(dict(entry.get("provenance") or {}).get("future_observation_source", {}).get("search_truncated") for entry in future)
    entries, seen = [], set()
    for previous in previous_turns[:MAX_TURNS]:
        if previous.status not in {"COMPLETED", "PARTIAL"} or not _same_scope(previous, turn, TURN_EVIDENCE_SCOPE_FIELDS):
            continue
        link = await uow.turns.get_run_link(turn_id=previous.turn_id, purpose="PRIMARY")
        if link is None:
            continue
        source = await uow.runs.get_run(ops_run_id=link.ops_run_id)
        if source is None or not _same_scope(source, run, RUN_EVIDENCE_SCOPE_FIELDS):
            continue
        # 仓储查询本身有界且仅查当前来源Run，不读取全域工件。
        originals = await uow.runs.list_plan_capture_results(ops_run_id=link.ops_run_id, limit=17)
        bounded = bounded or len(originals) > 16
        for artifact in originals[:16]:
            try:
                if (artifact.schema_version != RESULT_SCHEMA or str(artifact.ops_run_id) != str(link.ops_run_id)
                    or artifact.trust_level != "SOURCE_VERIFIED" or canonical_hash(artifact.payload_json) != artifact.content_hash
                    or artifact.artifact_key != RESULT_KEY + ":" + artifact.payload_json["request_id"]):
                    continue
                request_id = UUID(artifact.payload_json["request_id"])
                if request_id in seen:
                    continue
                seen.add(request_id)
                background = await uow.runs.get_artifact_by_key(ops_run_id=UUID(artifact.payload_json["capture_run_id"]), artifact_key=RESULT_KEY)
                if background is None or background.payload_json != artifact.payload_json:
                    continue
                origin = SqlPlanCaptureOrigin(source_turn_id=previous.turn_id, source_run_id=link.ops_run_id,
                    request_id=request_id, capture_run_id=UUID(artifact.payload_json["capture_run_id"]),
                    result_artifact_id=background.artifact_id, result_content_hash=background.content_hash)
                verified = await _resolve(uow=uow, run=run, turn=turn, origin=origin, reader=reader)
                if verified is None:
                    continue
                document = controlled_estimate_document(verified)
                if document is None or not any(_matches(document.subject, subject, spec[1]) for subject, spec in matches):
                    continue
                payload = {"schema_version": SCHEMA, "target_id": str(run.target_id),
                    "origin": origin.model_dump(mode="json"), "document": document.model_dump(mode="json")}
                size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
                if (len(entries) >= MAX_PLANS or used_facts >= MAX_FACTS
                    or used_rows + len(document.nodes) > MAX_ROWS or used_bytes + size > MAX_BYTES):
                    bounded = True
                    continue
                entries.append({"artifact_key": KEY_PREFIX + str(len(entries) + 1), "payload": payload,
                    "security_level": max(int(artifact.security_level), int(verified.artifact.security_level))})
                used_facts += 1
                used_rows += len(document.nodes)
                used_bytes += size
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, AIOpsApplicationError):
                continue
    for entry in entries:
        entry["payload"]["origin"]["search_truncated"] = bool(bounded)
    if bounded and not entries:
        return ({"artifact_key": KEY_PREFIX + "search", "schema_version": "SQL_PLAN_CAPTURE_SEARCH.v1",
            "payload": {"code": "PLAN_CAPTURE_SEARCH_INCOMPLETE"}, "security_level": 1},)
    return tuple(entries)


async def validate_reused_plan_capture(*, uow, run, artifact, reader, turn=None):
    try:
        if (artifact.artifact_type != ARTIFACT_TYPE or not artifact.artifact_key.startswith(KEY_PREFIX)
            or artifact.schema_version != SCHEMA or artifact.trust_level != "SOURCE_VERIFIED"
            or str(artifact.ops_run_id) != str(run.ops_run_id)
            or canonical_hash(artifact.payload_json) != artifact.content_hash
            or artifact.payload_json.get("schema_version") != SCHEMA
            or artifact.payload_json.get("target_id") != str(run.target_id)):
            return False
        if turn is None:
            link = await uow.turns.get_run_link_by_ops_run_id(ops_run_id=run.ops_run_id)
            turn = await uow.turns.get_turn(domain_id=run.domain_id, turn_id=link.turn_id) if link else None
        origin = SqlPlanCaptureOrigin.model_validate(artifact.payload_json["origin"])
        verified = await _resolve(uow=uow, run=run, turn=turn, origin=origin, reader=reader)
        if verified is None or int(artifact.security_level) < int(verified.artifact.security_level):
            return False
        document = controlled_estimate_document(verified)
        return document is not None and artifact.payload_json["document"] == document.model_dump(mode="json")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, AIOpsApplicationError):
        return False


async def revalidate_plan_capture_inputs(*, uow, run_id, artifacts, selected, reader):
    """每次Worker领取重新复核；旧充分性与旧结论不能继续向模型泄露失效计划。"""
    imported = [item for item in artifacts if item.artifact_type == ARTIFACT_TYPE]
    if not imported:
        return selected
    from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceGap
    from platform_core.contracts.aiops import SufficiencyStatus

    run = await uow.runs.get_run(ops_run_id=run_id)
    invalid = {str(item.artifact_id) for item in imported if run is None
        or not await validate_reused_plan_capture(uow=uow, run=run, artifact=item, reader=reader)}
    if not invalid:
        return selected
    gap = TurnEvidenceGap(source_id="sql.plan_capture", step_id="controlled-estimate", code="PLAN_CAPTURE_SOURCE_UNAVAILABLE",
        detail="估算计划的当前访问权或独立来源已变化，未沿用旧计划和旧判断；可在当前权限下继续其他取证。", retryable=False)
    result = []
    for item in selected:
        raw = dict(item.payload or {})
        if str(item.artifact_id) in invalid or item.schema_version == "AIOPS_TURN_RESULT.v1" and any(
            str(fact.get("artifact_id")) in invalid for fact in raw.get("evidence", ())):
            payload = {"code": gap.code}
            item = item.model_copy(update={"schema_version": "SQL_PLAN_CAPTURE_UNAVAILABLE.v1",
                "payload": payload, "content_hash": canonical_hash(payload)})
        elif item.schema_version == "DBA_SUFFICIENCY.v1" and any(str(fact.get("artifact_id")) in invalid for fact in raw.get("evidence", ())):
            assessment = DbaSufficiencyAssessment.model_validate(raw)
            safe = assessment.model_copy(update={
                "status": SufficiencyStatus.UNSAFE if assessment.status == SufficiencyStatus.UNSAFE else SufficiencyStatus.PARTIAL,
                "evidence": tuple(fact for fact in assessment.evidence if str(fact.artifact_id) not in invalid),
                "gaps": (*assessment.gaps, gap), "investigation": None, "clarification_question": None,
                "reasons": ("估算计划来源已失效，旧调查判断未沿用。",)})
            payload = safe.model_dump(mode="json")
            item = item.model_copy(update={"payload": payload, "content_hash": canonical_hash(payload)})
        result.append(item)
    return tuple(result)
