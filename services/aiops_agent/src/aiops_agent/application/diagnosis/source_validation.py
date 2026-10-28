"""报告和案例共用原始来源复核，不把自述一致的派生计算或对象关系当作源证据。"""

from uuid import UUID

from aiops_agent.application.diagnosis.monitoring_calculations import validate_monitoring_trace_sources
from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.contracts.investigation import InvestigationRecord
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.playbooks import canonical_hash


async def validate_investigation_sources(*, uow, run, payload):
    if not await validate_monitoring_trace_sources(uow=uow, run=run, payload=payload):
        return False
    record_payload = payload.get("investigation_record") or {}
    if not isinstance(record_payload, dict):
        return False
    if record_payload.get("sql_object_graph") is None:
        return True
    try:
        record = InvestigationRecord.model_validate(record_payload)
        facts = tuple(TurnEvidenceFact.model_validate(item) for item in payload.get("evidence", ()))
        if record.target_id != str(run.target_id) or record.run_id != str(run.ops_run_id):
            return False
        record.validate_observations(facts)
        used = {ref.evidence_ref for edge in record.sql_object_graph.relationships for ref in (edge.anchor, edge.related)}
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
                or observation.db_type != record.sql_object_graph.database_type
                or fact.captured_at != observation.captured_at.isoformat()
                or fact.row_count != observation.row_count or fact.truncated != observation.truncated
                or fact.columns != tuple(column.model_dump(mode="json") for column in observation.columns)
                or fact.provenance != database_provenance(observation)):
                return False
            # 与生产投影相同的精确序列化，不把大整数的字符串形式误当作篡改。
            expected_rows = fact.model_copy(update={"rows": observation.rows}).model_dump(mode="json")["rows"]
            if fact.model_dump(mode="json")["rows"] != expected_rows:
                return False
    except (ValueError, TypeError, IndexError, KeyError, AttributeError):
        return False
    return True
