"""原生历史/日志事实必须回放原始来源，报告与案例不得改写分页边界。"""

from uuid import UUID

from aiops_agent.application.diagnosis.provenance import native_metric_provenance
from aiops_agent.contracts.evidence import LogEvidenceSet, ObservationSet
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.domain.evidence.source_history import NATIVE_COLUMNS, native_history_rows, log_pagination_warnings
from aiops_agent.playbooks import canonical_hash


async def validate_source_history_facts(*, uow, run, payload):
    sources = {}
    try:
        for raw in payload.get("evidence", ()):
            if raw.get("tool_id") not in {"monitor.query_native_metric", "loki.query_range"}:
                if raw.get("source_pagination"):
                    return False
                continue
            fact = TurnEvidenceFact.model_validate(raw)
            identity = str(fact.artifact_id)
            if identity not in sources:
                artifact = await uow.runs.get_artifact(artifact_id=UUID(identity))
                if (artifact is None or str(artifact.ops_run_id) != str(run.ops_run_id)
                    or artifact.trust_level != "SOURCE_VERIFIED"
                    or canonical_hash(artifact.payload_json) != artifact.content_hash):
                    return False
                sources[identity] = artifact
            artifact = sources[identity]
            if fact.trust_level != "SOURCE_VERIFIED" or fact.measurement_semantics != "HISTORICAL_SAMPLES" or fact.presentation_kind != "TABLE":
                return False
            if fact.tool_id == "monitor.query_native_metric":
                if artifact.schema_version != "OBSERVATION_SET.v1":
                    return False
                source = ObservationSet.model_validate(artifact.payload_json)
                prefix, index = fact.evidence_ref.rsplit("#native-metric/", 1)
                if prefix != f"artifact:{identity}" or not index.isdecimal():
                    return False
                samples = source.native_metrics[int(index)]
                if (str(source.target_id) != str(run.target_id) or samples.target_id != source.target_id
                    or samples.binding_id != source.binding_id or samples.source_id != source.source_id
                    or fact.source_id != str(samples.source_id) or fact.step_id != samples.action_id
                    or fact.captured_at != samples.captured_at.isoformat()
                    or fact.source_pagination != samples.pagination
                    or fact.provenance != native_metric_provenance(samples)
                    or fact.columns != tuple({"name": name} for name in NATIVE_COLUMNS)):
                    return False
                rows = native_history_rows(samples)
                truncated = samples.truncated
            else:
                if artifact.schema_version != "LOG_EVIDENCE_SET.v1":
                    return False
                source = LogEvidenceSet.model_validate(artifact.payload_json)
                if (str(source.target_id) != str(run.target_id)
                    or fact.evidence_ref != f"artifact:{identity}#loki"
                    or fact.source_id != "logging.range" or fact.step_id != "loki"
                    or fact.captured_at != source.collected_at.isoformat()
                    or fact.source_pagination is not None or fact.warnings != log_pagination_warnings(source)
                    or fact.columns != ({"name": "observed_at", "logical_type": "DATETIME"},
                        {"name": "line", "logical_type": "STRING"}, {"name": "labels", "logical_type": "JSON"})):
                    return False
                rows = tuple((entry.observed_at.isoformat(), entry.line, dict(entry.labels)) for entry in source.entries)
                truncated = source.truncated
            if fact.pagination is not None or fact.row_count != len(rows) or fact.truncated != truncated:
                return False
            if fact.model_dump(mode="json")["rows"] != fact.model_copy(update={"rows": rows}).model_dump(mode="json")["rows"]:
                return False
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False
    return True
