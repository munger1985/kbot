"""报告和案例共用原始来源复核，不把自述一致的派生计算或对象关系当作源证据。"""

from uuid import UUID

from aiops_agent.application.diagnosis.monitoring_calculations import validate_monitoring_trace_sources
from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.application.diagnosis.native_measurements import native_mapping_tool_ids
from aiops_agent.contracts.investigation import InvestigationRecord
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import TurnEvidenceFact, TurnEvidenceGap
from aiops_agent.playbooks import canonical_hash
from aiops_agent.application.investigation.future_observation_evidence import validate_future_observation_facts
from aiops_agent.application.diagnosis.source_history_validation import validate_source_history_facts
from aiops_agent.application.diagnosis.sql_plans import (
    sql_plan_upload_fact, database_cursor_plan, controlled_estimate_fact,
    SQL_PLAN_TOOL_ID, CURSOR_PLAN_TOOL_ID, CURSOR_BINDS_TOOL_ID, CONTROLLED_PLAN_TOOL_ID,
)


async def validate_investigation_sources(*, uow, run, payload, plan_capture_reader=None):
    if not await validate_source_history_facts(uow=uow, run=run, payload=payload):
        return False
    if not await validate_future_observation_facts(uow=uow, run=run, payload=payload):
        return False
    if not await validate_monitoring_trace_sources(uow=uow, run=run, payload=payload):
        return False
    record_payload = payload.get("investigation_record") or {}
    if not isinstance(record_payload, dict):
        return False
    if (record_payload.get("sql_object_graph") is None and not record_payload.get("sql_subject_graphs")
            and not record_payload.get("sql_plan_assessments")
            and not any(item.get("sql_plan") or item.get("tool_id") in {SQL_PLAN_TOOL_ID, CURSOR_PLAN_TOOL_ID, CURSOR_BINDS_TOOL_ID, CONTROLLED_PLAN_TOOL_ID} for item in payload.get("evidence", ()))
            and not record_payload.get("access_gaps")
            and not any(item.get("pagination") for item in payload.get("evidence", ()))
            and not any(item.get("source_pagination") for item in payload.get("evidence", ()))
            and not any(item.get("source_pagination") for item in record_payload.get("observations", ()))
            and not any(item.get("pagination") for item in record_payload.get("observations", ()))
            and not any(item.get("object_access") for item in record_payload.get("observations", ()))
            and not any((item.get("provenance") or {}).get("native_lineage") for item in record_payload.get("observations", ()))
            and not any((item.get("provenance") or {}).get("native_lineage") for item in payload.get("evidence", ()))
            and not any(item.get("tool_id") in native_mapping_tool_ids() and item.get("source_id") != "prior_turn.sql_observation" for item in payload.get("evidence", ()))
            and not any(item.get("object_access") for item in payload.get("evidence", ()))
            and not any(item.get("object_access") for item in payload.get("evidence_gaps", ()))):
        return True
    try:
        record = InvestigationRecord.model_validate(record_payload) if record_payload else None
        facts = tuple(TurnEvidenceFact.model_validate(item) for item in payload.get("evidence", ()))
        if record is not None and (record.target_id != str(run.target_id) or record.run_id != str(run.ops_run_id)):
            return False
        if record is not None:
            record.validate_observations(facts)
            record.validate_access_gaps(tuple(TurnEvidenceGap.model_validate(item) for item in payload.get("evidence_gaps", ())))
        graphs = record.sql_graphs if record is not None else ()
        used = {ref.evidence_ref for graph in graphs for edge in graph.relationships for ref in (edge.anchor, edge.related)}
        used.update(ref.evidence_ref for graph in graphs for reference in graph.references for ref in reference.evidence_refs)
        used.update(fact.evidence_ref for fact in facts if fact.object_access or fact.pagination is not None)
        used.update(fact.evidence_ref for fact in facts if fact.provenance is not None and fact.provenance.native_lineage is not None)
        used.update(fact.evidence_ref for fact in facts if fact.tool_id in native_mapping_tool_ids() and fact.source_id != "prior_turn.sql_observation")
        used.update(fact.evidence_ref for fact in facts if fact.sql_plan is not None or fact.tool_id in {SQL_PLAN_TOOL_ID, CURSOR_PLAN_TOOL_ID, CURSOR_BINDS_TOOL_ID, CONTROLLED_PLAN_TOOL_ID})
        used.update(plan.anchor_ref for plan in (record.sql_plan_assessments if record is not None else ()) if plan.anchor_ref is not None)
        used.update(ref for plan in (record.sql_plan_assessments if record is not None else ()) for ref in plan.bind_metadata_refs)
        # 多主体图可能没有可建立的对象边；其已确认语句判断仍须回查原工具，而非仅信任冻结副本。
        used.update(ref.evidence_ref for claim in (record.claims if record is not None else ())
                    for ref in (*claim.evidence_fields, *claim.counterevidence_fields)
                    if any(fact.evidence_ref == ref.evidence_ref and fact.tool_id.startswith("db.")
                           and fact.trust_level == "SOURCE_VERIFIED" and fact.source_id != "prior_turn.sql_observation"
                           for fact in facts))
        sources = {}
        for fact in facts:
            if fact.evidence_ref not in used:
                continue
            identity = str(fact.artifact_id)
            if fact.tool_id == CONTROLLED_PLAN_TOOL_ID or fact.sql_plan is not None and fact.sql_plan.origin == "CONTROLLED_ESTIMATE":
                from aiops_agent.application.investigation.sql_plan_evidence import validate_reused_plan_capture
                artifact = await uow.runs.get_artifact(artifact_id=UUID(identity))
                if artifact is None or not await validate_reused_plan_capture(
                    uow=uow, run=run, artifact=artifact, reader=plan_capture_reader):
                    return False
                expected = controlled_estimate_fact({"artifact_id": identity, "payload": artifact.payload_json})
                if expected.model_dump(mode="json") != fact.model_dump(mode="json"):
                    return False
                continue
            if (fact.sql_plan is not None and fact.sql_plan.origin == "USER_UPLOAD") or fact.tool_id == SQL_PLAN_TOOL_ID:
                artifact = await uow.runs.get_artifact(artifact_id=UUID(identity))
                if (artifact is None or str(artifact.ops_run_id) != str(run.ops_run_id)
                    or artifact.schema_version != "USER_UPLOAD_EXTRACT.v1"
                    or artifact.trust_level != "USER_PROVIDED"
                    or canonical_hash(artifact.payload_json) != artifact.content_hash):
                    return False
                expected = sql_plan_upload_fact({"artifact_id": identity, "payload": artifact.payload_json})
                if expected is None or expected.model_dump(mode="json") != fact.model_dump(mode="json"):
                    return False
                continue
            if identity not in sources:
                artifact = await uow.runs.get_artifact(artifact_id=UUID(identity))
                if (artifact is None or str(artifact.ops_run_id) != str(run.ops_run_id)
                    or artifact.schema_version != "DBA_TOOL_RESULT.v1"
                    or artifact.trust_level != "SOURCE_VERIFIED"
                    or canonical_hash(artifact.payload_json) != artifact.content_hash):
                    return False
                sources[identity] = DbaToolResult.model_validate(artifact.payload_json)
            source = sources[identity]
            outcomes = [item for item in source.tool_outcomes if item.step_id == fact.step_id
                and item.tool_id == fact.tool_id and item.status == "SUCCEEDED" and item.observation is not None]
            if len(outcomes) != 1 or fact.source_id != source.source_id:
                return False
            observation = outcomes[0].observation
            semantics = "SNAPSHOT_DELTA" if observation.variant == "mysql_8_digest_window" and fact.tool_id == "db.sql.top_current" else source.measurement_semantics
            if fact.measurement_semantics != semantics or fact.presentation_kind != source.presentation_kind:
                return False
            if (fact.evidence_ref != f"artifact:{identity}#{fact.step_id}"
                or fact.trust_level != "SOURCE_VERIFIED"
                or str(observation.target_id) != str(run.target_id)
                or (graphs and observation.db_type not in {graph.database_type for graph in graphs})
                or fact.captured_at != observation.captured_at.isoformat()
                or fact.row_count != observation.row_count or fact.truncated != observation.truncated
                or fact.columns != tuple(column.model_dump(mode="json") for column in observation.columns)
                or fact.provenance != database_provenance(observation)
                or fact.pagination != observation.pagination
                or fact.object_access != observation.object_access):
                return False
            if fact.sql_plan != database_cursor_plan(observation)[0]:
                return False
            # 与生产投影相同的精确序列化，不把大整数的字符串形式误当作篡改。
            expected_rows = fact.model_copy(update={"rows": observation.rows}).model_dump(mode="json")["rows"]
            if fact.model_dump(mode="json")["rows"] != expected_rows:
                return False
        for gap in (record.access_gaps if record is not None else ()):
            prefix, step_id = gap.evidence_ref.rsplit("#", 1)
            identity = str(UUID(prefix.removeprefix("artifact:")))
            if prefix != f"artifact:{identity}" or step_id != gap.step_id:
                return False
            if identity not in sources:
                artifact = await uow.runs.get_artifact(artifact_id=UUID(identity))
                if (artifact is None or str(artifact.ops_run_id) != str(run.ops_run_id)
                        or artifact.schema_version != "DBA_TOOL_RESULT.v1"
                        or artifact.trust_level != "SOURCE_VERIFIED"
                        or canonical_hash(artifact.payload_json) != artifact.content_hash):
                    return False
                sources[identity] = DbaToolResult.model_validate(artifact.payload_json)
            source = sources[identity]
            outcomes = [item for item in source.tool_outcomes if item.step_id == gap.step_id
                        and item.status == "GAP" and item.gap is not None]
            if (source.source_id != gap.source_id or len(outcomes) != 1
                    or outcomes[0].gap.code != gap.code or outcomes[0].gap.object_access != gap.object_access):
                return False
    except (ValueError, TypeError, IndexError, KeyError, AttributeError):
        return False
    return True
