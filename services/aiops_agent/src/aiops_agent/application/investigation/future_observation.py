"""按既有 Run/Task 持久化未来观察；不持有连接等待，不改变目标采集策略。"""

from __future__ import annotations

from copy import deepcopy
import json
from datetime import datetime, timedelta, timezone
from uuid import UUID

from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.entities import OpsArtifactEntity, OpsRunEntity, OpsTaskEntity
from aiops_agent.playbooks import canonical_hash
from aiops_agent.tools.observation_series import assemble_observation_series
from aiops_agent.domain.operations import normalize_task_type
from platform_core.contracts.aiops.investigation import DiagnosticObservationSchedule
from platform_core.identity import uuid7


FOLLOWUP_WORKFLOW = "EVIDENCE_FOLLOWUP"
FOLLOWUP_SCHEMA = "DBA_FUTURE_OBSERVATION_REQUEST.v1"


def _schedule_info(run):
    return dict(dict(run.plan_snapshot_json or {}).get("future_observation") or {})


async def followup_access_error(*, uow, run, now, for_execution=True):
    """每次领取和兑换凭据前重验来源、当前版本授权及只读策略。"""
    try:
        info = _schedule_info(run)
        source = await uow.runs.get_run(ops_run_id=UUID(info["source_run_id"]))
        if (source is None or str(source.domain_id) != str(run.domain_id)
            or str(source.target_id) != str(run.target_id) or str(source.agent_id) != str(run.agent_id)
            or str(source.agent_version_id) != str(run.agent_version_id)
            or str(source.actor_id) != str(run.actor_id)
            or source.cancel_requested_at is not None or source.status in {"CANCELLED", "CANCELLING", "FAILED", "EXPIRED"}):
            return "FOLLOWUP_SOURCE_UNAVAILABLE"
        artifact = await uow.runs.get_artifact(artifact_id=UUID(info["source_artifact_id"]))
        if (artifact is None or str(artifact.ops_run_id) != str(source.ops_run_id)
            or artifact.schema_version != "DBA_TOOL_RESULT.v1" or artifact.trust_level != "SOURCE_VERIFIED"
            or canonical_hash(artifact.payload_json) != artifact.content_hash or artifact.content_hash != info["source_hash"]):
            return "FOLLOWUP_SOURCE_UNAVAILABLE"
        request = await uow.runs.get_artifact_by_key(ops_run_id=source.ops_run_id, artifact_key="future-observation:request")
        if (request is None or request.schema_version != FOLLOWUP_SCHEMA
            or canonical_hash(request.payload_json) != request.content_hash
            or request.payload_json != info or info["background_run_id"] != str(run.ops_run_id)
            or str(run.source_result_artifact_id) != str(artifact.artifact_id)):
            return "FOLLOWUP_SOURCE_UNAVAILABLE"
        if for_execution and (now >= run.deadline_at or run.cancel_requested_at is not None):
            return "FOLLOWUP_DEADLINE_OR_CANCELLED"
        target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=run.domain_id)
        frozen = dict(dict(run.plan_snapshot_json or {}).get("investigation_execution") or {})["database"]
        if (target is None or int(target.row_version) != int(frozen["target_row_version"])
            or str(target.status) != "ENABLED" or not target.readonly_connection_enabled
            or str(target.diagnostic_credential_id) != str(frozen["diagnostic_credential_id"])
            or str(target.db_type) != str(frozen["db_type"])
            or dict(target.endpoint_json or {}) != dict(frozen["connection_profile"])):
            return "FOLLOWUP_TARGET_CHANGED"
        binding = await uow.agents.get_active(domain_id=run.domain_id, agent_id=run.agent_id, target_id=run.target_id)
        if binding is None or str(binding.binding_id) != str(run.agent_version_id):
            return "FOLLOWUP_ACCESS_CHANGED"
        policy_id = dict(source.plan_snapshot_json or {}).get("binding", {}).get("policy_id")
        if str(binding.policy_id or "") != str(policy_id or ""):
            return "FOLLOWUP_ACCESS_CHANGED"
        if policy_id:
            policy = await uow.policies.get_scoped(policy_id=UUID(str(policy_id)), domain_id=run.domain_id)
            frozen_policy = dict(run.policy_snapshot_json or {})
            if (policy is None or policy.status != "ACTIVE"
                or not dict(policy.rules_json or {}).get("readonly_database_enabled", True)
                or policy.policy_hash != frozen_policy.get("policy_hash")):
                return "FOLLOWUP_ACCESS_CHANGED"
    except (KeyError, ValueError, TypeError, AttributeError):
        return "FOLLOWUP_SOURCE_UNAVAILABLE"
    return None


async def schedule_future_observation(*, uow, run, task, artifact, now):
    """与基线完成同事务创建持久化任务，整 Run 最多一个未来观察组。"""
    if run.workflow_kind == FOLLOWUP_WORKFLOW or artifact.schema_version != "DBA_TOOL_RESULT.v1":
        return None
    execution = dict(dict(run.plan_snapshot_json or {}).get("investigation_execution") or {})
    direct = dict(execution.get("direct_invocations") or {})
    dynamic = dict(execution.get("dynamic_invocations") or {})
    invocation = direct.get(task.task_key) or dynamic.get(task.task_key)
    if not invocation or invocation.get("observation_schedule") is None:
        return None
    requested = DiagnosticObservationSchedule.model_validate(invocation["observation_schedule"])
    result = DbaToolResult.model_validate(artifact.payload_json)
    observations = [item for item in result.tool_outcomes if item.status == "SUCCEEDED" and item.observation is not None]
    if (artifact.trust_level != "SOURCE_VERIFIED" or result.status != "SUCCEEDED"
        or len(result.tool_outcomes) != 1 or len(observations) != 1 or observations[0].observation.truncated
        or result.sampling is not None or result.observation_series is not None):
        artifact.provenance_json = {**dict(artifact.provenance_json or {}), "future_observation": {
            "status": "NOT_SCHEDULED", "reason": "BASELINE_UNAVAILABLE",
        }}
        return None
    baseline = observations[0].observation
    if (baseline.captured_at.tzinfo is None or baseline.captured_at > now + timedelta(seconds=1)
        or (now - baseline.captured_at).total_seconds() > 30 or str(baseline.target_id) != str(run.target_id)):
        artifact.provenance_json = {**dict(artifact.provenance_json or {}), "future_observation": {
            "status": "NOT_SCHEDULED", "reason": "BASELINE_TIME_OR_IDENTITY_INVALID",
        }}
        return None
    existing = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key="future-observation:request")
    if existing is not None:
        if (canonical_hash(existing.payload_json) != existing.content_hash
            or existing.schema_version != FOLLOWUP_SCHEMA):
            raise ValueError("未来观察调度请求来源不可重放")
        if existing.payload_json.get("source_artifact_id") == str(artifact.artifact_id):
            return UUID(existing.payload_json["background_run_id"])
        artifact.provenance_json = {**dict(artifact.provenance_json or {}), "future_observation": {
            "status": "NOT_SCHEDULED", "reason": "RUN_GROUP_ALREADY_SCHEDULED",
        }}
        return None
    background_id = uuid7()
    point_keys = [f"future:{index}" for index in range(2, requested.count + 1)]
    point_task_ids = {key: uuid7() for key in point_keys}
    info = {
        "source_run_id": str(run.ops_run_id), "source_artifact_id": str(artifact.artifact_id),
        "source_hash": artifact.content_hash, "background_run_id": str(background_id),
        "requested": requested.model_dump(mode="json"), "started_at": baseline.captured_at.isoformat(),
        "kind": "DIRECT" if task.task_key in direct else "DYNAMIC",
        "point_keys": point_keys,
        "point_artifact_keys": {
            key: f"task:{point_task_ids[key]}:output:default:vDBA_TOOL_RESULT.v1" for key in point_keys
        },
    }
    frozen = deepcopy(invocation)
    frozen["sampling"] = None
    frozen["observation_schedule"] = None
    limits = frozen["tool"]["limits"] if info["kind"] == "DIRECT" else frozen["limits"]
    if (not 1 <= int(limits["statement_timeout_seconds"]) <= 5 or not 1 <= int(limits["max_result_rows"]) <= 200
        or not 1 <= int(limits["max_result_bytes"]) <= 131072):
        raise ValueError("持久化未来观察必须冻结低成本查询和结果上限")
    frozen_execution = {
        **{key: deepcopy(execution[key]) for key in ("database", "diagnostic_catalog_hash", "capability_snapshot_hash")},
        "direct_invocations": {key: deepcopy(frozen) for key in point_keys} if info["kind"] == "DIRECT" else {},
        "dynamic_invocations": {key: deepcopy(frozen) for key in point_keys} if info["kind"] == "DYNAMIC" else {},
    }
    await uow.runs.add_run(OpsRunEntity(
        ops_run_id=background_id,
        domain_id=run.domain_id, target_id=run.target_id, agent_id=run.agent_id,
        agent_version_id=run.agent_version_id, actor_id=run.actor_id, trigger_type="API",
        interaction_mode="AUTONOMOUS", workflow_kind=FOLLOWUP_WORKFLOW,
        source_result_artifact_id=artifact.artifact_id, original_request="按冻结的只读查询补采未来观察时点",
        idempotency_key=f"future-observation:{artifact.artifact_id}", status="RUNNING", started_at=now,
        plan_snapshot_json={
            "blueprint": {"id": "evidence.future-observation", "version": "1", "final_task_key": "future:series"},
            "future_observation": info, "investigation_execution": frozen_execution,
            "target": deepcopy(dict(run.plan_snapshot_json or {}).get("target") or {}),
            "binding": deepcopy(dict(run.plan_snapshot_json or {}).get("binding") or {}),
        },
        policy_snapshot_json=deepcopy(run.policy_snapshot_json or {}),
        deadline_at=baseline.captured_at + timedelta(seconds=(requested.count - 1) * requested.interval_seconds + requested.count * 60),
        trace_id=run.trace_id, created_at=now, updated_at=now,
    ))
    tasks = []
    for index, key in enumerate(point_keys, start=2):
        dependencies = [] if index == 2 else [point_keys[index - 3]]
        tasks.append(OpsTaskEntity(
            ops_task_id=point_task_ids[key], ops_run_id=background_id, task_key=key, task_type=normalize_task_type("TOOL_INVOKE"),
            handler_id="dba.future-point.invoke", handler_version="1", input_schema_version="FUTURE_POINT_INPUT.v1",
            output_schema_version="DBA_TOOL_RESULT.v1", depends_on_json=dependencies,
            input_artifacts_json=["future:baseline", *point_keys[:index - 2]], status="READY" if not dependencies else "PENDING",
            priority=80, available_at=baseline.captured_at + timedelta(seconds=(index - 1) * requested.interval_seconds),
            max_attempts=1, timeout_seconds=15,
        ))
    tasks.append(OpsTaskEntity(
        ops_task_id=uuid7(), ops_run_id=background_id, task_key="future:series", task_type="REPORT",
        handler_id="dba.future-series.assemble", handler_version="1", input_schema_version="FUTURE_SERIES_INPUT.v1",
        output_schema_version="DBA_TOOL_RESULT.v1", depends_on_json=point_keys,
        input_artifacts_json=["future:baseline", *point_keys], status="PENDING", priority=90,
        available_at=now, max_attempts=1, timeout_seconds=10,
    ))
    await uow.runs.add_tasks(tasks)
    for owner, key, schema, payload, provenance in (
        (background_id, "future:baseline", artifact.schema_version, artifact.payload_json,
            {"producer": "aiops.future-observation", "source_artifact_id": str(artifact.artifact_id), "source_hash": artifact.content_hash}),
        (run.ops_run_id, "future-observation:request", FOLLOWUP_SCHEMA, info, {"producer": "aiops.future-observation"}),
    ):
        await uow.runs.add_artifact(OpsArtifactEntity(
            artifact_id=uuid7(), ops_run_id=owner, ops_task_id=None, artifact_key=key,
            artifact_type="EVIDENCE", schema_version=schema, payload_json=deepcopy(payload),
            content_hash=canonical_hash(payload), byte_size=len(json.dumps(payload, ensure_ascii=False).encode()),
            provenance_json=provenance, trust_level="SOURCE_VERIFIED", security_level=artifact.security_level,
        ))
    artifact.provenance_json = {**dict(artifact.provenance_json or {}), "future_observation": {"status": "SCHEDULED", **info}}
    await uow.runs.append_event(
        ops_run_id=run.ops_run_id, ops_task_id=task.ops_task_id,
        event_type="evidence.future_observation.scheduled", event_key=f"future-observation:{artifact.artifact_id}",
        visibility="USER", payload_json={"background_run_id": str(background_id),
            "requested": requested.model_dump(mode="json"), "public_summary": "已安排后台未来观察；不会补造过去窗口"},
    )
    return background_id


async def preserve_future_observation(*, uow, run, now, reason):
    """异常终态也产生唯一可展示序列，原始证据仍独立保留。"""
    if run.workflow_kind != FOLLOWUP_WORKFLOW:
        return False
    if run.final_artifact_id is not None:
        return True
    existing = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key="future:partial-series")
    if existing is not None:
        run.final_artifact_id = existing.artifact_id
        return True
    artifacts = await uow.runs.list_artifacts(ops_run_id=run.ops_run_id)
    try:
        result = assemble_observation_series(
            artifacts=tuple({
                "artifact_key": item.artifact_key, "payload": item.payload_json,
                "content_hash": item.content_hash, "schema_version": item.schema_version,
                "trust_level": item.trust_level,
            } for item in artifacts),
            info=_schedule_info(run), target_id=run.target_id, now=now, missing_code=reason,
        )
    except (ValueError, TypeError, KeyError):
        await uow.runs.append_event(
            ops_run_id=run.ops_run_id, event_type="evidence.future_observation.unavailable",
            event_key=f"future-observation:{run.ops_run_id}:source-invalid", visibility="USER",
            payload_json={"code": "FOLLOWUP_SOURCE_UNAVAILABLE",
                "public_summary": "原始观察来源无法重放，未生成汇总；停止操作仍有效"},
        )
        return False
    payload = result.model_dump(mode="json")
    artifact = await uow.runs.add_artifact(OpsArtifactEntity(
        artifact_id=uuid7(), ops_run_id=run.ops_run_id, ops_task_id=None,
        artifact_key="future:partial-series", artifact_type="EVIDENCE",
        schema_version="DBA_TOOL_RESULT.v1", payload_json=payload, content_hash=canonical_hash(payload),
        byte_size=len(json.dumps(payload, ensure_ascii=False).encode()),
        provenance_json={"producer": "aiops.future-observation", "reason": reason,
            "source_artifact_ids": [str(item.artifact_id) for item in artifacts
                if item.schema_version == "DBA_TOOL_RESULT.v1"]},
        trust_level="SOURCE_VERIFIED", security_level=max(int(item.security_level) for item in artifacts),
    ))
    run.final_artifact_id = artifact.artifact_id
    return True


async def future_observation_payload(*, uow, run, now, artifact=None):
    """运行中可查看已落盘时点；终态重放原始来源，不信任汇总副本自证。"""
    if await followup_access_error(uow=uow, run=run, now=now, for_execution=False):
        return None
    artifacts = await uow.runs.list_artifacts(ops_run_id=run.ops_run_id)
    raw = tuple({
        "artifact_key": item.artifact_key, "payload": item.payload_json, "content_hash": item.content_hash,
        "schema_version": item.schema_version, "trust_level": item.trust_level,
    } for item in artifacts)
    try:
        info = _schedule_info(run)
        if artifact is None:
            return assemble_observation_series(
                artifacts=raw, info=info, target_id=run.target_id, now=now, missing_code="FOLLOWUP_PENDING",
            ).model_dump(mode="json")
        if (artifact.schema_version != "DBA_TOOL_RESULT.v1" or artifact.trust_level != "SOURCE_VERIFIED"
            or canonical_hash(artifact.payload_json) != artifact.content_hash):
            return None
        final = DbaToolResult.model_validate(artifact.payload_json)
        if final.observation_series is None or final.observation_series.finished_at > now + timedelta(seconds=1):
            return None
        replay = assemble_observation_series(
            artifacts=raw, info=info, target_id=run.target_id, now=final.observation_series.finished_at,
            missing_code=dict(artifact.provenance_json or {}).get("reason"),
        )
        if canonical_hash(replay.model_dump(mode="json")) != artifact.content_hash:
            return None
        return final.model_dump(mode="json")
    except (ValueError, TypeError, KeyError):
        return None
