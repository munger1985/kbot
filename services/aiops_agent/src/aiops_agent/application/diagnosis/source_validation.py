"""报告和案例共用原始来源复核，不把自述一致的派生计算或对象关系当作源证据。"""

from uuid import UUID

from aiops_agent.application.diagnosis.monitoring_calculations import validate_monitoring_trace_sources
from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.contracts.investigation import InvestigationRecord
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import TurnEvidenceFact, TurnEvidenceGap
from aiops_agent.playbooks import canonical_hash
from aiops_agent.application.investigation.future_observation_evidence import validate_future_observation_facts


async def validate_investigation_sources(*, uow, run, payload):
    if not await validate_future_observation_facts(uow=uow, run=run, payload=payload):
        return False
    if not await validate_monitoring_trace_sources(uow=uow, run=run, payload=payload):
        return False
    record_payload = payload.get("investigation_record") or {}
    if not isinstance(record_payload, dict):
        return False
    if (record_payload.get("sql_object_graph") is None and not record_payload.get("sql_subject_graphs")
            and not record_payload.get("access_gaps")
            and not any(item.get("pagination") for item in payload.get("evidence", ()))
            and not any(item.get("pagination") for item in record_payload.get("observations", ()))
            and not any(item.get("object_access") for item in record_payload.get("observations", ()))
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
        used.update(fact.evidence_ref for fact in facts if fact.object_access or fact.pagination is not None)
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
            if fact.measurement_semantics != source.measurement_semantics or fact.presentation_kind != source.presentation_kind:
                return False
            observation = outcomes[0].observation
            if (fact.evidence_ref != f"artifact:{identity}#{fact.step_id}"
                or str(observation.target_id) != str(run.target_id)
                or (graphs and observation.db_type not in {graph.database_type for graph in graphs})
                or fact.captured_at != observation.captured_at.isoformat()
                or fact.row_count != observation.row_count or fact.truncated != observation.truncated
                or fact.columns != tuple(column.model_dump(mode="json") for column in observation.columns)
                or fact.provenance != database_provenance(observation)
                or fact.pagination != observation.pagination
                or fact.object_access != observation.object_access):
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
