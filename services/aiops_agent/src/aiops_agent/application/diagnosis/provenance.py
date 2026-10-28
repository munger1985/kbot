"""只从受控生产者构建观测来源；未知物理身份不以逻辑 Target 冒充。"""

from aiops_agent.contracts.measurements import EvidenceProvenance
from aiops_agent.playbooks import canonical_hash
from datetime import timedelta


def database_provenance(observation):
    native = dict(observation.provenance or {})
    physical = native.get("instance_identity") or {}
    verified_identity = tuple(sorted((str(key), str(value)) for key, value in physical.items())) if isinstance(physical, dict) else ()
    return EvidenceProvenance(
        target_id=str(observation.target_id),
        native_origin_ids=(f"database:{observation.target_id}",),
        physical_identity=verified_identity,
        source_version=f"{observation.db_version}:{observation.variant}:{observation.template_sha256}",
        content_hash=observation.result_sha256,
        collected_start=observation.captured_at - timedelta(milliseconds=observation.duration_ms),
        collected_end=observation.captured_at,
        quality_flags=(
            *(("TRUNCATED",) if observation.truncated else ()),
            *(("PHYSICAL_IDENTITY_UNKNOWN",) if not verified_identity else ()),
            "NATIVE_ORIGIN_COARSE",
        ),
    )


def native_metric_provenance(samples):
    return EvidenceProvenance(
        target_id=str(samples.target_id), native_origin_ids=(f"database:{samples.target_id}",),
        source_version=samples.source_version, content_hash=samples.response_sha256,
        window_start=samples.window_start, window_end=samples.window_end,
        quality_flags=("PHYSICAL_IDENTITY_UNKNOWN", "NATIVE_ORIGIN_COARSE", "COUNTER_SEMANTICS_UNKNOWN",
                       *(("TRUNCATED",) if samples.truncated else ())),
    )


def monitoring_provenance(result):
    observations = result.observations
    return EvidenceProvenance(
        target_id=str(result.target_id),
        native_origin_ids=(f"database:{result.target_id}",),
        source_version=canonical_hash(sorted({
            f"{item.source_id}:{item.source_version}" for item in observations
        })),
        content_hash=canonical_hash(result.model_dump(mode="json")),
        window_start=min((item.window_start for item in observations), default=None),
        window_end=max((item.window_end for item in observations), default=None),
        quality_flags=tuple(dict.fromkeys((
            "PHYSICAL_IDENTITY_UNKNOWN", "NATIVE_ORIGIN_COARSE",
            *(flag for item in observations for flag in (
                *(("PARTIAL_COVERAGE",) if item.coverage_ratio < 1 else ()),
                *(("TRUNCATED",) if item.truncated else ()),
            )),
        ))),
    )
