"""从受控监控原始序列提取身份见证，并在报告/案例重新核验来源。"""

from datetime import datetime, timezone, timedelta
import hashlib
import re

from aiops_agent.contracts.physical_identity import SourceIdentityWitness
from aiops_agent.domain.evidence.physical_identity import (
    COLLECTOR_LABELS, IDENTITY_FIELDS, IDENTITY_METRIC_CODE, IDENTITY_TEMPLATE_ID,
    IDENTITY_TEMPLATE_VERSION, canonical_native_identity,
)
from aiops_agent.domain.evidence.observations import parse_numeric_sample


def source_identity_witnesses(*, artifact_id, result):
    witnesses = []
    for metric_index, observation in enumerate(result.observations):
        same_row = observation.metric_code == "db.native.active_sessions"
        if observation.metric_code != IDENTITY_METRIC_CODE and not same_row:
            continue
        template_id = "prom.database.native.active_sessions" if same_row else IDENTITY_TEMPLATE_ID
        protocol_ok = (
            observation.source_type == "PROMETHEUS"
            and observation.semantic_role == ("ACTIVE_SESSIONS" if same_row else "DATABASE_IDENTITY")
            and observation.semantic_version == "1.0.0"
            and observation.unit == ("sessions" if same_row else "unix_seconds")
            and observation.value_kind == ("GAUGE" if same_row else "STATE")
            and observation.provenance.get("template_id") == template_id
            and observation.provenance.get("template_version") == IDENTITY_TEMPLATE_VERSION
            and re.fullmatch(r"[a-f0-9]{64}", str(observation.provenance.get("provider_response_hash") or "")) is not None
            and str(observation.target_id) == str(result.target_id)
            and str(observation.source_id) == str(result.source_id)
            and str(observation.binding_id) == str(result.binding_id)
        )
        for series_index, series in enumerate(observation.series[:16]):
            labels = series.dimensions
            engine = labels.get("database_type")
            if engine not in IDENTITY_FIELDS:
                continue
            identity, flags = canonical_native_identity(engine, labels)
            flags = list(flags)
            if not protocol_ok or labels.get("metric_identity_version" if same_row else "identity_version") != "1":
                flags.append("IDENTITY_PROTOCOL_UNVERIFIED")
            if same_row:
                flags.append("SAME_QUERY_METRIC_IDENTITY_REPORTED")
            scope = []
            for key in COLLECTOR_LABELS:
                value = labels.get(key)
                if not value:
                    continue
                if len(value) > 256 or any(ord(char) < 32 for char in value):
                    flags.append("IDENTITY_SCOPE_MISMATCH")
                    continue
                scope.append((key, value))
            scope = tuple(sorted(scope))
            if not labels.get("job") or not labels.get("instance"):
                flags.append("IDENTITY_SCOPE_MISSING")
            if hashlib.sha256(str(labels.get("target_key") or "").encode("utf-8")).hexdigest() != observation.external_target_fingerprint:
                flags.append("IDENTITY_SCOPE_MISMATCH")
            if observation.truncated or len(observation.series) > 16:
                flags.append("TRUNCATED")
            times = []
            for point_index, point in enumerate(series.points[:120]):
                number = parse_numeric_sample(point.value)
                if same_row:
                    try:
                        clocks = observation.provenance["metric_row_scrape_times"]
                        clock = datetime.fromisoformat(clocks[series_index][point_index])
                        if observation.provenance["metric_row_protocol"] != "DATABASE_METRIC_ROW.v1" or clock.tzinfo is None:
                            raise ValueError
                        number = parse_numeric_sample(str(clock.timestamp()))
                    except (ValueError, TypeError, KeyError, IndexError):
                        number = None
                if (point.quality != "GOOD" or number is None or not 0 <= number <= 10000000000
                    or not observation.window_start <= point.observed_at <= observation.window_end):
                    flags.append("IDENTITY_SAMPLE_INVALID")
                    continue
                # timestamp(原始info序列)返回抓取时点；不把求值时点当作来源采样时点。
                sampled_at = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(number * 1000000))
                if not observation.window_start - timedelta(seconds=120) <= sampled_at <= point.observed_at or (point.observed_at - sampled_at).total_seconds() > 120:
                    flags.append("IDENTITY_SAMPLE_INVALID")
                    continue
                times.append(sampled_at)
            if len(series.points) > 120:
                flags.append("IDENTITY_SAMPLE_BUDGET_EXCEEDED")
            if not times:
                flags.append("IDENTITY_SAMPLE_INVALID")
            if observation.coverage_ratio < 1:
                flags.append("PARTIAL_COVERAGE")
            if len(witnesses) >= 16:
                return tuple(item.model_copy(update={"quality_flags": tuple(dict.fromkeys((*item.quality_flags, "TRUNCATED")))}) for item in witnesses)
            witnesses.append(SourceIdentityWitness(
                input_ref=f"artifact:{artifact_id}#/observations/{metric_index}/series/{series_index}",
                database_type=engine, target_id=observation.target_id, source_id=observation.source_id,
                binding_id=observation.binding_id, source_version=observation.source_version,
                collector_scope=scope, native_identity=identity,
                sample_times=tuple(sorted(set(times))), quality_flags=tuple(dict.fromkeys(flags)),
                protocol_version="DATABASE_METRIC_ROW.v1" if same_row else "DATABASE_IDENTITY_INFO.v1",
            ))
    return tuple(witnesses[:16])


async def validate_identity_witness_sources(*, uow, run, payload):
    """删除见证或图标记不能把原始身份观测变成未经复核的已认证关系。"""
    from aiops_agent.contracts.evidence import ObservationSet
    from aiops_agent.contracts.turn_answer import TurnEvidenceFact
    from aiops_agent.playbooks import canonical_hash
    from aiops_agent.application.diagnosis.provenance import monitoring_provenance

    try:
        for raw in payload.get("evidence", ()):
            if raw.get("tool_id") != "metric.query_range" and not raw.get("source_identity_witnesses"):
                continue
            fact = TurnEvidenceFact.model_validate(raw)
            artifact = await uow.runs.get_artifact(artifact_id=fact.artifact_id)
            if (artifact is None or str(artifact.ops_run_id) != str(run.ops_run_id)
                or artifact.schema_version != "OBSERVATION_SET.v1" or artifact.trust_level != "SOURCE_VERIFIED"
                or canonical_hash(artifact.payload_json) != artifact.content_hash
                or fact.tool_id != "metric.query_range" or fact.source_id != "monitoring.overview"
                or fact.trust_level != "SOURCE_VERIFIED"):
                return False
            source = ObservationSet.model_validate(artifact.payload_json)
            if str(source.target_id) != str(run.target_id) or fact.source_identity_witnesses != source_identity_witnesses(artifact_id=fact.artifact_id, result=source):
                return False
            if (fact.evidence_ref != f"artifact:{fact.artifact_id}#prometheus"
                or fact.provenance != monitoring_provenance(source)
                or fact.captured_at != source.collected_at.isoformat()):
                return False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError):
        return False
    return True
