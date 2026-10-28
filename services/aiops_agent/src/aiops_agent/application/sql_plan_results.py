"""在既有事务内复核采集结果；读取审批安全元数据，不兑换代表参数或延长执行权。"""

from dataclasses import dataclass
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict
from aiops_agent.application.sql_plan_materials import MATERIAL_PREFIX, MATERIAL_SCHEMA, _source_rows
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureResult


WORKFLOW = "SQL_PLAN_CAPTURE"
RESULT_SCHEMA = "SQL_PLAN_CAPTURE_RESULT.v1"
RESULT_KEY = "sql-plan-capture:result"
HANDLER = "sql.plan-capture"


def task_binding(task):
    return {name: getattr(task, name) for name in ("task_key", "task_type", "handler_id", "handler_version",
        "input_schema_version", "output_schema_version", "depends_on_json", "input_artifacts_json",
        "max_attempts", "timeout_seconds")}


@dataclass(frozen=True)
class VerifiedSqlPlanCapture:
    run: object
    task: object
    request: dict
    original: object
    artifact: object | None
    result: SqlPlanCaptureResult | None


class SqlPlanCaptureResultReader:
    def __init__(self, *, materials, vault):
        self._materials = materials
        self._vault = vault

    async def scheduled(self, *, uow, scope, source, request_id, lock=True):
        _, protected = await self._vault.get(uow=uow, scope=scope, request_id=request_id, lock=lock)
        request = dict((protected or {}).get("request") or {})
        execution = dict((protected or {}).get("execution") or {})
        if (request.get("ops_run_id") != str(source.ops_run_id)
            or request.get("requested_by") != str(source.actor_id)
            or request.get("request_id") != str(request_id)
            or execution.get("source_run_id") != str(source.ops_run_id)
            or execution.get("request_id") != str(request_id)):
            raise resource_not_found("计划采集后台运行")
        try:
            capture_id, task_id = UUID(execution["capture_run_id"]), UUID(execution["capture_task_id"])
        except (KeyError, ValueError, TypeError):
            raise resource_not_found("计划采集后台运行") from None
        run = await uow.runs.get_run(ops_run_id=capture_id, lock=lock)
        task = await uow.runs.get_task(ops_task_id=task_id, lock=lock)
        info = dict(getattr(run, "plan_snapshot_json", None) or {}).get("sql_plan_capture") or {}
        if (run is None or task is None or run.workflow_kind != WORKFLOW
            or self._materials._vault._scope(source) != self._materials._vault._scope(run)
            or any(str(getattr(run, key)) != str(getattr(source, key)) for key in
                ("target_id", "actor_id", "agent_id", "agent_version_id"))
            or info != {key: execution.get(key) for key in info}
            or execution.get("capture_run_id") != str(run.ops_run_id)
            or execution.get("plan_sha256") != canonical_hash(run.plan_snapshot_json)
            or execution.get("task_sha256") != canonical_hash(task_binding(task))
            or task.ops_run_id != run.ops_run_id or task.handler_id != HANDLER
            or execution.get("capture_task_id") != str(task.ops_task_id)
            or run.deadline_at is None or execution.get("deadline_at") != run.deadline_at.isoformat()
            or execution.get("approval_sha256") != canonical_hash({"request": request, "decision": protected["decision"]})
            or str(run.source_result_artifact_id) != request.get("source", {}).get("source_artifact_id")):
            raise state_conflict("计划采集后台引用无法复核")
        return run, task, protected

    async def read(self, *, uow, scope, source, request_id, lock=False):
        # 过期审批/材料不抹除已采集结果，也不成为再次执行或永久读取的凭证。
        target, access = await self._materials._access(uow, source)
        run, task, protected = await self.scheduled(uow=uow, scope=scope, source=source,
            request_id=request_id, lock=lock)
        frozen = protected["request"]
        if frozen["frozen"]["target_state_hash"] != access["target_state_hash"]:
            raise state_conflict("计划采集时的目标身份或凭据引用已变化")
        material = await uow.runs.get_artifact_by_key(ops_run_id=source.ops_run_id,
            artifact_key=MATERIAL_PREFIX + frozen["material_id"])
        original = await uow.runs.get_artifact(artifact_id=UUID(frozen["source"]["source_artifact_id"]))
        if (material is None or str(material.ops_run_id) != str(source.ops_run_id)
            or material.schema_version != MATERIAL_SCHEMA
            or material.content_hash != canonical_hash(material.payload_json)
            or material.content_hash != frozen["material_receipt_hash"]
            or material.payload_json.get("source") != frozen["source"]
            or original is None or original.content_hash != material.payload_json.get("source_hash")
            or original.security_level != frozen["security_level"]
            or frozen["source"] not in [item.model_dump(mode="json") for item in _source_rows(original, source, target.db_type)]):
            raise state_conflict("计划采集原始SQL来源无法复核")
        artifact = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=RESULT_KEY)
        result = None
        if artifact is not None:
            execution = protected["execution"]
            provenance = dict(artifact.provenance_json or {})
            if (artifact.schema_version != RESULT_SCHEMA or artifact.trust_level != "SOURCE_VERIFIED"
                or artifact.ops_run_id != run.ops_run_id or artifact.security_level < frozen["security_level"]
                or artifact.content_hash != canonical_hash(artifact.payload_json)
                or artifact.content_hash != execution.get("result_sha256")
                or provenance.get("producer") != "aiops-db-executor"
                or provenance.get("verification") != "EXECUTOR_REPORTED"
                or provenance.get("source_run_id") != str(source.ops_run_id)
                or provenance.get("approval_sha256") != execution["approval_sha256"]):
                raise state_conflict("计划采集结果无法独立复核")
            result = SqlPlanCaptureResult.model_validate(artifact.payload_json)
            if (result.database_type != frozen["source"]["database_type"]
                or result.request_id != request_id or result.capture_run_id != run.ops_run_id):
                raise state_conflict("计划采集结果引擎或引用不一致")
        return VerifiedSqlPlanCapture(run=run, task=task, request=frozen, original=original, artifact=artifact, result=result)
