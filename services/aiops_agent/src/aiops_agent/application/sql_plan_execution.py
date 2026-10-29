"""受控估算的后台调度、当前围栏、一次性保护材料兑换及权威结果接纳。"""

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import secrets
from uuid import UUID

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.errors import AIOpsApplicationError, state_conflict
from aiops_agent.application.execution_fence import ExecutionFenceError
from aiops_agent.application.resource_governance import TERMINAL_RUN_STATES, reserve_readonly_resource, token_hash
from aiops_agent.application.sql_plan_captures import PREFIX
from aiops_agent.application.sql_plan_results import (
    HANDLER, RESULT_KEY, RESULT_SCHEMA, WORKFLOW, SqlPlanCaptureResultReader, task_binding as _task_binding,
    native_comparison_result_authorized,
)
from aiops_agent.domain.evidence.sql_plan_capture import source_sql_text
from aiops_agent.domain.operations import normalize_task_type
from aiops_agent.domain.operations.budget import task_deadline
from aiops_agent.entities import OpsArtifactEntity, OpsRunEntity, OpsTaskEntity
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_execution import (
    SqlPlanCaptureExchange, SqlPlanCaptureResult, SqlPlanCaptureExecutionView,
)
from platform_core.identity import uuid7


def owner_scope(run):
    # 内部复核仍使用平台的不可变Scope；不制造可变替身，也不把它当外部认证。
    return ConfigurationScope(actor_id=run.actor_id, principal_id=run.actor_id,
        request_id="sql-plan-capture:" + str(run.ops_run_id),
        trace_id=getattr(run, "trace_id", None) or str(run.ops_run_id), **{
        key: getattr(run, key) for key in ("tenant_id", "domain_id") if hasattr(run, key)
    })


class SqlPlanCaptureExecutionService:
    def __init__(self, *, captures, codec, credentials, audience, clock=None):
        self._captures = captures
        self.results = SqlPlanCaptureResultReader(materials=captures._materials, vault=captures._vault)
        self._uow_factory = captures._uow_factory
        self._codec = codec
        self._credentials = credentials
        self._audience = audience
        self._clock = clock or (lambda: datetime.now(UTC))

    async def start(self, *, scope, run_id, request_id, command, agent_ids=()):
        now = self._clock()
        async with self._uow_factory() as uow:
            source = await self._captures._run(uow, scope, run_id, owner=True, agent_ids=agent_ids)
            _, current, availability = await self._captures._policy(uow, source)
            row, protected, view = await self._captures._load(uow=uow, scope=scope, run=source,
                request_id=request_id, now=now, current=current, availability=availability)
            if command.request_sha256 != view.request_sha256:
                raise state_conflict("计划采集必须持有当前有效、所见哈希一致的独立审批")
            if protected.get("execution") is not None:
                # 已调度请求只返回原引用，过期不创建第二次执行，也不伪造仍有效的审批。
                await self._scheduled(uow=uow, scope=scope, source=source, request_id=request_id)
                await uow.commit()
                return view
            if view.state != "APPROVED":
                raise state_conflict("计划采集必须持有当前有效、所见哈希一致的独立审批")
            capture_id, task_id = uuid7(), uuid7()
            approval_hash = canonical_hash({"request": protected["request"], "decision": protected["decision"]})
            info = {"source_run_id": str(source.ops_run_id), "request_id": str(request_id),
                "approval_sha256": approval_hash, "capture_run_id": str(capture_id), "capture_task_id": str(task_id),
                "database_type": view.source.database_type, "security_level": protected["request"]["security_level"]}
            plan = {"blueprint": {"id": "sql.plan-capture", "version": "1", "final_task_key": "plan:capture"},
                "sql_plan_capture": info,
                "investigation_execution": {"database": deepcopy(source.plan_snapshot_json["investigation_execution"]["database"])}}
            await uow.runs.add_run(OpsRunEntity(
                ops_run_id=capture_id, **self._captures._materials._vault._scope(scope),
                target_id=source.target_id, agent_id=source.agent_id, agent_version_id=source.agent_version_id,
                actor_id=source.actor_id, trigger_type="API", interaction_mode="AUTONOMOUS", workflow_kind=WORKFLOW,
                source_result_artifact_id=view.source.source_artifact_id,
                original_request="按独立审批采集估算计划；不执行原业务SQL或ANALYZE",
                idempotency_key="sql-plan-capture:" + str(request_id), status="RUNNING", started_at=now,
                plan_snapshot_json=plan, policy_snapshot_json=deepcopy(getattr(source, "policy_snapshot_json", None) or {}),
                deadline_at=view.expires_at, trace_id=getattr(source, "trace_id", None) or str(uuid7()), created_at=now, updated_at=now))
            capture_task = OpsTaskEntity(ops_task_id=task_id, ops_run_id=capture_id,
                task_key="plan:capture", task_type=normalize_task_type("TOOL_INVOKE"),
                handler_id=HANDLER, handler_version="1", input_schema_version="SQL_PLAN_CAPTURE_INPUT.v1",
                output_schema_version=RESULT_SCHEMA, depends_on_json=[], input_artifacts_json=[],
                status="READY", priority=80, available_at=now, max_attempts=1, timeout_seconds=30)
            await uow.runs.add_tasks([capture_task])
            execution = {**info, "plan_sha256": canonical_hash(plan),
                "task_sha256": canonical_hash(_task_binding(capture_task)), "deadline_at": view.expires_at.isoformat()}
            protected = {**protected, "execution": execution}
            self._captures._vault.update(row=row, scope=scope, payload=protected)
            await uow.runs.append_event(ops_run_id=source.ops_run_id, event_type="sql.plan_capture.scheduled",
                event_key=PREFIX + str(request_id) + ":scheduled", visibility="USER",
                payload_json={"capture_run_id": str(capture_id), "request_id": str(request_id),
                    "public_summary": "已创建独立后台估算任务；执行前仍复核审批、材料、授权和租约"})
            await uow.commit()
            return self._captures._view(protected["request"], "APPROVED", protected["decision"], execution)

    async def _scheduled(self, *, uow, scope, source, request_id):
        return await self.results.scheduled(uow=uow, scope=scope, source=source, request_id=request_id)

    async def status(self, *, scope, run_id, request_id, agent_ids=()):
        async with self._uow_factory() as uow:
            source = await self._captures._run(uow, scope, run_id, owner=True, agent_ids=agent_ids)
            verified = await self.results.read(uow=uow, scope=scope, source=source, request_id=request_id, lock=True)
            run, task = verified.run, verified.task
            view = SqlPlanCaptureExecutionView(request_id=request_id, capture_run_id=run.ops_run_id,
                phase=self._phase(run, task),
                run_status=run.status, cancel_requested=run.cancel_requested_at is not None,
                can_cancel=run.status not in TERMINAL_RUN_STATES and run.cancel_requested_at is None,
                result=verified.result)
            await uow.commit()
            return view

    async def cancel(self, *, scope, run_id, request_id, runtime, agent_ids=()):
        async with self._uow_factory() as uow:
            source = await self._captures._run(uow, scope, run_id, owner=True, agent_ids=agent_ids)
            # 即使执行策略或目标已撤回，所有者仍可请求停止；此处不读取安全结果或参数。
            run, task, _ = await self._scheduled(uow=uow, scope=scope, source=source, request_id=request_id)
            capture_id, version = run.ops_run_id, int(run.row_version)
            terminal = run.status in TERMINAL_RUN_STATES or run.cancel_requested_at is not None
            view = SqlPlanCaptureExecutionView(request_id=request_id, capture_run_id=capture_id,
                phase=self._phase(run, task),
                run_status=run.status, cancel_requested=run.cancel_requested_at is not None,
                can_cancel=False)
            await uow.commit()
        if not terminal:
            # 复用持久化Runtime终止协议；版本变化由调用者刷新，不能重试成新的执行权限。
            receipt = await runtime.request_cancel(ops_run_id=capture_id, domain_id=scope.domain_id,
                actor_id=scope.actor_id, expected_row_version=version,
                idempotency_key="sql-plan-capture:" + str(request_id), trace_id=scope.trace_id)
            view = SqlPlanCaptureExecutionView.model_validate({**view.model_dump(),
                "run_status": receipt.status, "cancel_requested": True,
                "phase": "TERMINAL" if receipt.status in TERMINAL_RUN_STATES else "CANCEL_REQUESTED"})
        return view

    @staticmethod
    def _phase(run, task):
        if run.status in TERMINAL_RUN_STATES:
            return "TERMINAL"
        if run.cancel_requested_at is not None:
            return "CANCEL_REQUESTED"
        return "RUNNING" if task.status == "RUNNING" else "QUEUED"

    async def validate(self, *, uow, grant, now):
        """固定先锁源Run，再锁后台Run/Task及授权记录；不读取参数原值。"""
        source = await uow.runs.get_run(ops_run_id=grant.source_run_id, lock=True)
        run = await uow.runs.get_run(ops_run_id=grant.run_id, lock=True)
        if source is None or run is None:
            raise ExecutionFenceError("EXECUTION_AUTHORIZATION_REVOKED")
        scope = owner_scope(source)
        if (self._captures._materials._vault._scope(scope) != self._captures._materials._vault._scope(run)
            or run.workflow_kind != WORKFLOW or str(run.target_id) != str(source.target_id)
            or str(run.agent_id) != str(source.agent_id) or str(run.agent_version_id) != str(source.agent_version_id)
            or str(run.actor_id) != str(source.actor_id)):
            raise ExecutionFenceError("EXECUTION_AUTHORIZATION_REVOKED")
        if run.cancel_requested_at is not None or run.status in TERMINAL_RUN_STATES:
            raise ExecutionFenceError("EXECUTION_CANCELLED")
        if run.deadline_at is None or now >= min(run.deadline_at, grant.expires_at):
            raise ExecutionFenceError("EXECUTION_DEADLINE_EXCEEDED")
        info = dict(run.plan_snapshot_json or {}).get("sql_plan_capture") or {}
        if (info.get("source_run_id") != str(source.ops_run_id) or info.get("request_id") != str(grant.request_id)
            or info.get("capture_run_id") != str(run.ops_run_id) or info.get("capture_task_id") != str(grant.task_id)
            or info.get("database_type") != grant.db_type or info.get("approval_sha256") != grant.approval_sha256):
            raise ExecutionFenceError("EXECUTION_AUTHORIZATION_REVOKED")
        task = await uow.runs.get_task(ops_task_id=grant.task_id, lock=True)
        if (task is None or task.ops_run_id != run.ops_run_id or task.handler_id != "sql.plan-capture"
            or task.status != "RUNNING" or task.lease_token is None or task.lease_until is None
            or task.started_at is None or now >= task.lease_until
            or not secrets.compare_digest(hashlib.sha256(str(task.lease_token).encode()).hexdigest(), grant.lease_token_hash)):
            raise ExecutionFenceError("EXECUTION_LEASE_INVALID")
        if now >= task_deadline(task):
            raise ExecutionFenceError("EXECUTION_DEADLINE_EXCEEDED")
        await uow.targets.get_scoped(target_id=run.target_id, domain_id=run.domain_id, lock=True)
        try:
            _, current, availability = await self._captures._policy(uow, source)
            row, protected, view = await self._captures._load(uow=uow, scope=scope, run=source,
                request_id=grant.request_id, now=now, current=current, availability=availability)
            execution = dict(protected.get("execution") or {})
            if (view.state != "APPROVED"
                or canonical_hash({"request": protected["request"], "decision": protected["decision"]}) != grant.approval_sha256
                or any(execution.get(key) != value for key, value in info.items())
                or execution.get("plan_sha256") != canonical_hash(run.plan_snapshot_json)
                or execution.get("task_sha256") != canonical_hash(_task_binding(task))
                or run.deadline_at is None or execution.get("deadline_at") != run.deadline_at.isoformat()
                or str(run.source_result_artifact_id) != str(view.source.source_artifact_id)):
                raise state_conflict("采集审批已失效")
        except AIOpsApplicationError:
            raise ExecutionFenceError("EXECUTION_AUTHORIZATION_REVOKED") from None
        return source, run, task, row, protected

    async def exchange(self, token):
        grant = self._codec.verify_plan_capture(token, now=self._clock())
        if grant.audience != self._audience:
            raise ExecutionFenceError("EXECUTION_GRANT_INVALID")
        async with self._uow_factory() as uow:
            now = self._clock()
            source, run, task, row, protected = await self.validate(uow=uow, grant=grant, now=now)
            execution = dict(protected["execution"])
            if execution.get("consumed_grant_id") is not None:
                raise ExecutionFenceError("PLAN_CAPTURE_ALREADY_CONSUMED")
            request = protected["request"]
            scope = owner_scope(source)
            receipt, view = await self._captures._material(uow, scope, source, UUID(request["material_id"]), now, request["frozen"])
            lease = await reserve_readonly_resource(uow=uow, lease_id=grant.grant_id, run_id=run.ops_run_id,
                target_id=run.target_id, source_kind="DATABASE", source_id=run.target_id,
                deadline_at=min(grant.expires_at, run.deadline_at, task.lease_until, task_deadline(task)))
            document = await self._captures._materials._vault.open_for_capture(
                uow=uow, scope=scope, material_id=view.material_id, binding=receipt.payload_json, now=now)
            original = await uow.runs.get_artifact(artifact_id=view.source.source_artifact_id)
            observation = next(item["observation"] for item in original.payload_json["tool_outcomes"]
                if item["step_id"] == view.source.step_id)
            fields = dict(zip((column["name"].lower() for column in observation["columns"]),
                observation["rows"][view.source.row_index], strict=True))
            sql = source_sql_text(fields, view.source.database_type)
            if sql is None:
                raise ExecutionFenceError("PLAN_SOURCE_UNAVAILABLE")
            target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=run.domain_id)
            credentials = await self._credentials.read(uow=uow, **self._captures._materials._vault._scope(scope),
                credential_id=target.diagnostic_credential_id, credential_kind="target_diagnostic",
                external_key=target.target_id, lock=True)
            # 返回前再次收紧时效；兑换标记与租约同事务提交，任何Grant都不可二次兑换。
            if self._clock() >= min(grant.expires_at, lease.execution_deadline_at):
                raise ExecutionFenceError("EXECUTION_DEADLINE_EXCEEDED")
            result = SqlPlanCaptureExchange(database_type=grant.db_type, connection_profile=target.endpoint_json,
                username=credentials["username"], password=credentials["password"], source_sql=sql,
                protected_document=json.dumps(document, ensure_ascii=False),
                allowed_objects=request["frozen"]["capture_policy"]["allowed_objects"],
                statement_timeout_seconds=request["statement_timeout_seconds"], source_subject=view.source.subject,
                native_bind_comparison=grant.db_type == "ORACLE"
                    and request["frozen"]["capture_policy"].get("native_bind_comparison_enabled", False)
                    and protected["decision"].get("acknowledge_native_bind_comparison", False),
                source_identity=dict(dict(observation.get("provenance") or {}).get("instance_identity") or {}),
                resource_lease=lease)
            protected = {**protected, "execution": {**execution, "consumed_grant_id": str(grant.grant_id)}}
            self._captures._vault.update(row=row, scope=scope, payload=protected)
            await uow.commit()
            return result

    async def report(self, command):
        result, receipt = command.result, command.resource_lease
        async with self._uow_factory() as uow:
            # 结果接纳只凭原资源租约证明，不重新授予已过期的执行权。
            run = await uow.runs.get_run(ops_run_id=result.capture_run_id)
            info = dict(getattr(run, "plan_snapshot_json", None) or {}).get("sql_plan_capture") or {}
            if run is None or run.workflow_kind != WORKFLOW or str(result.request_id) != info.get("request_id"):
                raise ExecutionFenceError("PLAN_CAPTURE_RESULT_INVALID")
            try:
                source_id = UUID(info["source_run_id"])
            except (ValueError, TypeError, KeyError):
                raise ExecutionFenceError("PLAN_CAPTURE_RESULT_INVALID") from None
            source = await uow.runs.get_run(ops_run_id=source_id, lock=True)
            run = await uow.runs.get_run(ops_run_id=result.capture_run_id, lock=True)
            if (source is None or run is None
                or self._captures._materials._vault._scope(source) != self._captures._materials._vault._scope(run)
                or any(str(getattr(source, key)) != str(getattr(run, key)) for key in
                    ("target_id", "actor_id", "agent_id", "agent_version_id"))):
                raise ExecutionFenceError("PLAN_CAPTURE_RESULT_INVALID")
            scope = owner_scope(source)
            protected_row, protected = await self._captures._vault.get(uow=uow, scope=scope, request_id=result.request_id)
            execution = dict((protected or {}).get("execution") or {})
            lease = await uow.resources.get_lease(receipt.lease_id, lock=True)
            now = self._clock()
            if (lease is None or str(lease.run_id) != str(run.ops_run_id) or str(lease.target_id) != str(run.target_id)
                or lease.source_kind != "DATABASE" or str(lease.source_id) != str(run.target_id)
                or execution.get("consumed_grant_id") != str(lease.lease_id)
                or execution.get("capture_run_id") != str(run.ops_run_id)
                or any(execution.get(key) != value for key, value in info.items())
                or execution.get("plan_sha256") != canonical_hash(run.plan_snapshot_json)
                or run.deadline_at is None or execution.get("deadline_at") != run.deadline_at.isoformat()
                or protected.get("request", {}).get("ops_run_id") != str(source.ops_run_id)
                or canonical_hash({"request": protected["request"], "decision": protected["decision"]}) != info.get("approval_sha256")
                or result.database_type != info.get("database_type")
                or not native_comparison_result_authorized(result, protected["request"], protected["decision"])
                or not secrets.compare_digest(lease.release_token_hash, token_hash(receipt.release_token))
                or lease.execution_deadline_at != receipt.execution_deadline_at or lease.held_until != receipt.held_until
                or result.captured_at < lease.acquired_at or result.captured_at > now
                or (result.status == "SUCCEEDED" and result.captured_at > lease.execution_deadline_at)):
                raise ExecutionFenceError("PLAN_CAPTURE_RESULT_INVALID")
            payload = result.model_dump(mode="json")
            digest = canonical_hash(payload)
            if execution.get("result_sha256") not in (None, digest):
                raise ExecutionFenceError("PLAN_CAPTURE_RESULT_CONFLICT")
            for owner, key in ((run.ops_run_id, RESULT_KEY), (source.ops_run_id, RESULT_KEY + ":" + str(result.request_id))):
                prior = await uow.runs.get_artifact_by_key(ops_run_id=owner, artifact_key=key)
                if prior is not None:
                    if prior.content_hash != canonical_hash(payload) or prior.payload_json != payload:
                        raise ExecutionFenceError("PLAN_CAPTURE_RESULT_CONFLICT")
                    if owner == run.ops_run_id:
                        run.final_artifact_id = prior.artifact_id
                    continue
                saved = await uow.runs.add_artifact(OpsArtifactEntity(artifact_id=uuid7(), ops_run_id=owner,
                    ops_task_id=None, artifact_key=key, artifact_type="EVIDENCE", schema_version=RESULT_SCHEMA,
                    payload_json=payload, content_hash=canonical_hash(payload),
                    byte_size=len(json.dumps(payload, ensure_ascii=False).encode()),
                    provenance_json={"producer": "aiops-db-executor", "verification": "EXECUTOR_REPORTED",
                        "source_run_id": str(source.ops_run_id), "approval_sha256": info["approval_sha256"]},
                    trust_level="SOURCE_VERIFIED", security_level=info["security_level"]))
                if owner == run.ops_run_id:
                    # 独立回执即使遇到Worker失联/取消也保留为交付物；绝不重开终态或自报整个Run成功。
                    run.final_artifact_id = saved.artifact_id
            self._captures._vault.update(row=protected_row, scope=scope,
                payload={**protected, "execution": {**execution, "result_sha256": digest}})
            await uow.commit()
            return result


async def validate_capture_completion(*, uow, run, task, artifact):
    """Worker返回不能替代隔离Executor的权威回传；缺失/不一致不可完成为成功。"""
    if run.workflow_kind != WORKFLOW:
        return
    if task.handler_id != "sql.plan-capture" or artifact.schema_version != RESULT_SCHEMA:
        raise state_conflict("计划采集任务或结果合同不一致")
    result = SqlPlanCaptureResult.model_validate(artifact.payload_json)
    info = dict(run.plan_snapshot_json or {}).get("sql_plan_capture") or {}
    if (str(result.capture_run_id) != str(run.ops_run_id) or str(result.request_id) != info.get("request_id")
        or result.database_type != info.get("database_type")):
        raise state_conflict("计划采集结果不属于本运行")
    recorded = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=RESULT_KEY)
    minimum_level = info.get("security_level")
    if type(minimum_level) is not int or not 0 <= minimum_level <= 999:
        raise state_conflict("计划采集缺少冻结的源密级")
    # 通用Worker默认密级为1；任何回显/失败副本都不能低于原始SQL证据。
    artifact.security_level = max(getattr(artifact, "security_level", 0), minimum_level)
    valid = (recorded is not None and recorded.schema_version == RESULT_SCHEMA and recorded.trust_level == "SOURCE_VERIFIED"
        and recorded.security_level >= minimum_level
        and recorded.content_hash == canonical_hash(recorded.payload_json)
        and dict(recorded.provenance_json or {}).get("verification") == "EXECUTOR_REPORTED"
        and dict(recorded.provenance_json or {}).get("approval_sha256") == info.get("approval_sha256")
        and dict(recorded.payload_json or {}).get("capture_run_id") == str(run.ops_run_id)
        and dict(recorded.payload_json or {}).get("request_id") == info.get("request_id"))
    if result.status == "SUCCEEDED" and (not valid or recorded.payload_json != artifact.payload_json):
        raise state_conflict("计划采集缺少可复核的隔离执行器回传")
    return recorded if valid else None
