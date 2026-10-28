"""未来观察的原始来源重放与有界汇总；不计算未经核验的历史差分。"""

from datetime import datetime, timedelta

from aiops_agent.contracts.artifacts.database import EvidenceGap
from aiops_agent.contracts.tool_execution import DbaToolResult, DiagnosticObservationSeriesSummary, ToolOutcome
from aiops_agent.playbooks import canonical_hash


def observation_gap(base, code, detail):
    payload = base.model_dump(mode="json")
    payload.update(
        sampling=None, observation_series=None, status="FAILED",
        tool_outcomes=[ToolOutcome(
            step_id="future.gap", tool_id=base.source_id, tool_version=base.source_version,
            status="GAP", gap=EvidenceGap(code=code, tool_id=base.source_id, detail=detail, retryable=False),
        ).model_dump(mode="json")],
    )
    return DbaToolResult.model_validate(payload)


def replay_point(artifact):
    """只采纳完整且哈希未变化的已核验原始产物。"""
    if (artifact.get("schema_version") != "DBA_TOOL_RESULT.v1"
        or artifact.get("trust_level") != "SOURCE_VERIFIED"
        or canonical_hash(artifact.get("payload")) != artifact.get("content_hash")):
        raise ValueError("未来观察原始产物来源不可重放")
    return DbaToolResult.model_validate(artifact["payload"])


def replay_baseline(artifacts, info):
    rows = [item for item in artifacts if item.get("artifact_key") == "future:baseline"]
    if len(rows) != 1 or rows[0].get("content_hash") != info["source_hash"]:
        raise ValueError("未来观察缺少唯一冻结基线")
    base = replay_point(rows[0])
    if (base.status != "SUCCEEDED" or len(base.tool_outcomes) != 1
        or base.tool_outcomes[0].status != "SUCCEEDED" or base.tool_outcomes[0].observation is None
        or base.tool_outcomes[0].observation.truncated or base.sampling is not None or base.observation_series is not None):
        raise ValueError("未来观察基线不是完整的单次成功结果")
    return base


def point_error(result, *, base, target_id, previous, now, interval, earliest=None):
    """每次执行后立即校验，防止坏时点继续释放后续查询。"""
    if (result.source_id != base.source_id or result.source_version != base.source_version
        or result.measurement_semantics != base.measurement_semantics or result.definition_hash != base.definition_hash
        or result.source_type != base.source_type or result.output_schema != base.output_schema
        or result.sampling is not None or result.observation_series is not None
        or len(result.tool_outcomes) != 1):
        return "FOLLOWUP_IDENTITY_MISMATCH"
    outcome = result.tool_outcomes[0]
    observation = outcome.observation
    if outcome.tool_id != base.source_id or outcome.tool_version != base.source_version:
        return "FOLLOWUP_IDENTITY_MISMATCH"
    if observation is None:
        return None
    original = base.tool_outcomes[0].observation
    if (str(observation.target_id) != str(target_id) or observation.tool_id != base.source_id
        or observation.tool_version != base.source_version or observation.db_type != original.db_type
        or observation.db_version != original.db_version
        or observation.template_sha256 != original.template_sha256
        or observation.parameters_sha256 != original.parameters_sha256 or observation.variant != original.variant
        or observation.capability_snapshot_hash != original.capability_snapshot_hash):
        return "FOLLOWUP_IDENTITY_MISMATCH"
    original_native, current_native = dict(original.provenance or {}), dict(observation.provenance or {})
    if original_native.get("identity_scope") == "SAME_CONNECTION_CURRENT" and original_native.get("instance_identity"):
        if (current_native.get("identity_scope") != "SAME_CONNECTION_CURRENT"
            or current_native.get("instance_identity") != original_native["instance_identity"]):
            return "FOLLOWUP_PHYSICAL_IDENTITY_CHANGED"
    captured = observation.captured_at
    if (captured.tzinfo is None or captured > now + timedelta(seconds=1)
        or (earliest is not None and captured < earliest)
        or any(observation.executor_request_id == item.executor_request_id for item in previous)
        or (previous and (captured - previous[-1].captured_at).total_seconds() < interval)):
        return "FOLLOWUP_TIME_NOT_COMPARABLE"
    return None


def assemble_observation_series(*, artifacts, info, target_id, now, missing_code=None):
    base = replay_baseline(artifacts, info)
    expected = ("future:baseline", *(info["point_artifact_keys"][key] for key in info["point_keys"]))
    by_key = {}
    for item in artifacts:
        key = item.get("artifact_key")
        if key in expected:
            if key in by_key:
                raise ValueError("未来观察存在重复时点产物")
            by_key[key] = item
    if missing_code is None and set(by_key) != set(expected):
        raise ValueError("未来观察汇总缺少原始时点产物")
    outcomes, previous, completed = [], [], 0
    stop = "COMPLETE"
    for index, key in enumerate(expected, start=1):
        if key not in by_key:
            source = observation_gap(base, missing_code, "未来观察未执行完毕，保留已完成时点；没有补造缺失观察")
        else:
            source = replay_point(by_key[key])
        error = point_error(source, base=base, target_id=target_id, previous=previous,
            now=now, interval=info["requested"]["interval_seconds"])
        if error:
            source = observation_gap(base, error, "未来观察的身份或时间不可比较，未纳入该结果")
        outcome = source.tool_outcomes[0]
        outcomes.append(outcome.model_copy(update={"step_id": f"future.observation.{index}"}))
        if source.status != "SUCCEEDED" or outcome.status != "SUCCEEDED" or outcome.observation is None:
            stop = "INVALID_SAMPLE_TIME" if error == "FOLLOWUP_TIME_NOT_COMPARABLE" else "QUERY_GAP"
            break
        previous.append(outcome.observation)
        completed += 1
        if outcome.observation.truncated:
            stop = "RESULT_LIMIT"
            break
    return DbaToolResult(
        source_type=base.source_type, source_id=base.source_id, source_version=base.source_version,
        definition_hash=base.definition_hash, output_schema=base.output_schema,
        measurement_semantics=base.measurement_semantics, presentation_kind=base.presentation_kind,
        status="SUCCEEDED" if stop == "COMPLETE" else "PARTIAL" if completed else "FAILED",
        tool_outcomes=tuple(outcomes),
        observation_series=DiagnosticObservationSeriesSummary(
            requested=info["requested"], completed_count=completed,
            started_at=datetime.fromisoformat(info["started_at"]), finished_at=now, stop_reason=stop,
        ),
    )
