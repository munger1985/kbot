"""独立计划采集审批；每次操作复核材料、源证据、当前目标与专用策略。"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from uuid import UUID

from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found, state_conflict
from aiops_agent.application.sql_plan_capture_vault import SqlPlanCaptureVault
from aiops_agent.application.sql_plan_materials import MATERIAL_PREFIX, MATERIAL_SCHEMA
from aiops_agent.entities import OpsArtifactEntity
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_capture import SqlPlanCapturePage, SqlPlanCapturePolicy, SqlPlanCaptureView
from platform_core.identity import uuid7


REQUEST_SCHEMA = "SQL_PLAN_CAPTURE_REQUEST.v1"
DECISION_SCHEMA = "SQL_PLAN_CAPTURE_DECISION.v1"
PREFIX = "sql-plan-capture:"


class SqlPlanCaptureService:
    def __init__(self, *, uow_factory, materials, approval_enabled=False, clock=None):
        self._uow_factory = uow_factory
        self._materials = materials
        self._vault = SqlPlanCaptureVault(material_vault=materials._vault)
        self._approval_enabled = approval_enabled
        self._clock = clock or (lambda: datetime.now(UTC))

    async def _run(self, uow, scope, run_id, *, owner, agent_ids):
        run = await uow.runs.get_run_scoped(ops_run_id=run_id, domain_id=scope.domain_id, lock=True)
        if (run is None or any(str(getattr(run, key)) != str(value) for key, value in self._materials._vault._scope(scope).items())
            or owner and str(run.actor_id) != str(scope.actor_id)
            or agent_ids and str(run.agent_id) not in {str(item) for item in agent_ids}):
            raise resource_not_found("计划采集审批")
        return run

    @staticmethod
    def _owner_scope(scope, run):
        # 仅用可信 Run 的所有者复核材料状态；审批身份和审计仍使用真实调用者，绝不返回参数。
        return replace(scope, actor_id=run.actor_id)

    async def _policy(self, uow, run):
        target, access = await self._materials._access(uow, run)
        binding = await uow.agents.get_active(domain_id=run.domain_id, agent_id=run.agent_id, target_id=run.target_id)
        policy = await uow.policies.get_scoped(policy_id=binding.policy_id, domain_id=run.domain_id) if binding.policy_id else None
        raw = dict(policy.rules_json or {}).get("sql_plan_capture") if policy is not None else None
        try:
            capture_policy = SqlPlanCapturePolicy.model_validate(raw) if isinstance(raw, dict) else SqlPlanCapturePolicy()
        except ValueError:
            raise state_conflict("计划采集专用策略无效") from None
        if any(not isinstance(name, str) or not name.strip() or len(name) > 256 or any(ord(c) < 32 for c in name)
            for name in capture_policy.allowed_objects):
            raise state_conflict("计划采集对象范围无效")
        availability = "AVAILABLE" if capture_policy.enabled and capture_policy.allowed_objects else "POLICY_REQUIRED"
        if not self._approval_enabled:
            availability = "DISABLED"
        frozen = {**access, "policy_rules_hash": canonical_hash(dict(policy.rules_json or {})) if policy else None,
            "capture_policy": capture_policy.model_dump(mode="json"), "database_type": target.db_type}
        return capture_policy, frozen, availability

    async def _material(self, uow, scope, run, material_id, now, frozen):
        receipt = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=MATERIAL_PREFIX + str(material_id))
        if receipt is None or receipt.schema_version != MATERIAL_SCHEMA:
            raise state_conflict("计划采集所需材料不可用")
        view = await self._materials._view(uow=uow, scope=self._owner_scope(scope, run), run=run,
            artifact=receipt, now=now, access={key: frozen[key] for key in ("target_state_hash", "binding_state_hash")})
        if view.state != "STORED" or view.capture_preflight is None or view.capture_preflight.status != "AWAITING_DATABASE_VALIDATION":
            raise state_conflict("计划材料已失效或静态预检未通过")
        return receipt, view

    async def _add(self, uow, run, key, schema, payload, *, security_level):
        return await uow.runs.add_artifact(OpsArtifactEntity(artifact_id=uuid7(), ops_run_id=run.ops_run_id,
            ops_task_id=None, artifact_key=key, artifact_type="APPROVAL", schema_version=schema,
            payload_json=payload, content_hash=canonical_hash(payload),
            byte_size=len(json.dumps(payload, ensure_ascii=False).encode()),
            provenance_json={"producer": "aiops.sql-plan-capture", "contains_parameter_values": False},
            trust_level="SYSTEM", security_level=security_level))

    def _view(self, request, state, decision=None):
        return SqlPlanCaptureView(request_id=request["request_id"], ops_run_id=request["ops_run_id"],
            material_id=request["material_id"], source=request["source"], request_sha256=canonical_hash(request),
            state=state, allowed_objects=request["frozen"]["capture_policy"]["allowed_objects"],
            statement_timeout_seconds=request["statement_timeout_seconds"],
            require_distinct_approver=request["frozen"]["capture_policy"]["require_distinct_approver"],
            oracle_plan_table_write=request["oracle_plan_table_write"], requested_by=request["requested_by"],
            decided_by=decision["decided_by"] if decision else None,
            created_at=request["created_at"], expires_at=request["expires_at"])

    @staticmethod
    def _request_id(artifact):
        # 索引从工件键读取；不能让可篡改正文决定去哪里寻找独立密文。
        key = getattr(artifact, "artifact_key", "") or ""
        prefix = PREFIX + "request:"
        try:
            if not key.startswith(prefix):
                raise ValueError
            return UUID(key.removeprefix(prefix))
        except (ValueError, TypeError, AttributeError):
            raise state_conflict("计划采集请求索引无法复核") from None

    async def _load(self, *, uow, scope, run, request_id, now, current=None, availability=None):
        row, protected = await self._vault.get(uow=uow, scope=scope, request_id=request_id)
        if protected is None or str(protected["request"].get("ops_run_id")) != str(run.ops_run_id):
            raise state_conflict("计划采集审批来源无法复核")
        request = protected["request"]
        artifact = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=PREFIX + "request:" + str(request_id))
        state, decision = protected["state"], protected["decision"]
        if (artifact is None or artifact.schema_version != REQUEST_SCHEMA or artifact.trust_level != "SYSTEM"
            or canonical_hash(artifact.payload_json) != artifact.content_hash or artifact.payload_json != request
            or artifact.security_level != request["security_level"]):
            state = "UNAVAILABLE"
        if decision is not None:
            recorded = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=PREFIX + "decision:" + str(request_id))
            if (recorded is None or recorded.schema_version != DECISION_SCHEMA or recorded.trust_level != "SYSTEM"
                or canonical_hash(recorded.payload_json) != recorded.content_hash or recorded.payload_json != decision
                or recorded.security_level != request["security_level"]):
                state = "UNAVAILABLE"
        if state in {"PENDING", "APPROVED"}:
            if now >= datetime.fromisoformat(request["expires_at"]):
                state = "EXPIRED"
            elif availability != "AVAILABLE" or current != request["frozen"]:
                state = "UNAVAILABLE"
            else:
                try:
                    receipt, _ = await self._material(uow, scope, run, UUID(request["material_id"]), now, current)
                    if (receipt.content_hash != request["material_receipt_hash"]
                        or receipt.security_level != request["security_level"]):
                        state = "UNAVAILABLE"
                except AIOpsApplicationError:
                    state = "UNAVAILABLE"
        return row, protected, self._view(request, state, decision)

    async def request(self, *, scope, run_id, command, agent_ids=()):
        now = self._clock()
        async with self._uow_factory() as uow:
            run = await self._run(uow, scope, run_id, owner=True, agent_ids=agent_ids)
            policy, frozen, available = await self._policy(uow, run)
            if available != "AVAILABLE":
                raise state_conflict("尚未开启计划采集专用策略或审批开关；普通只读权限不授权计划采集")
            if command.statement_timeout_seconds > policy.max_statement_timeout_seconds:
                raise state_conflict("请求时限超过专用采集策略")
            receipt, material = await self._material(uow, scope, run, command.material_id, now, frozen)
            existing = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id,
                artifact_key=PREFIX + "request:" + str(command.request_id))
            if existing is not None:
                raise state_conflict("计划采集请求标识已使用，请刷新审批状态")
            artifacts = await uow.runs.list_artifacts(ops_run_id=run.ops_run_id)
            requests = [item for item in artifacts if item.schema_version == REQUEST_SCHEMA]
            if len(requests) >= 16:
                raise state_conflict("每轮最多16份计划采集请求")
            for item in requests:
                _, _, previous = await self._load(uow=uow, scope=scope, run=run,
                    request_id=self._request_id(item), now=now, current=frozen, availability=available)
                if previous.material_id == command.material_id and previous.state in {"PENDING", "APPROVED"}:
                    raise state_conflict("该材料已有有效采集请求，请先撤回旧请求")
            payload = {"schema_version": REQUEST_SCHEMA, "request_id": str(command.request_id),
                "ops_run_id": str(run.ops_run_id), "material_id": str(command.material_id),
                "requested_by": str(scope.actor_id), "source": material.source.model_dump(mode="json"),
                "material_receipt_hash": receipt.content_hash, "mode": command.mode,
                "statement_timeout_seconds": command.statement_timeout_seconds, "security_level": receipt.security_level,
                "oracle_plan_table_write": material.source.database_type == "ORACLE", "frozen": frozen,
                "created_at": now.isoformat(), "expires_at": min(material.expires_at, now + timedelta(seconds=300)).isoformat()}
            await self._vault.create(uow=uow, scope=scope, request=payload)
            await self._add(uow, run, PREFIX + "request:" + str(command.request_id), REQUEST_SCHEMA, payload,
                security_level=receipt.security_level)
            await uow.runs.append_event(ops_run_id=run.ops_run_id, event_type="sql.plan_capture.requested",
                event_key=PREFIX + str(command.request_id) + ":requested", visibility="USER",
                payload_json={"request_id": str(command.request_id), "execution_authorized": False,
                    "public_summary": "计划采集请求已冻结，等待独立风险审批；未执行 SQL"})
            await uow.commit()
            return self._view(payload, "PENDING")

    async def list(self, *, scope, run_id, review=False, agent_ids=()):
        now = self._clock()
        async with self._uow_factory() as uow:
            run = await self._run(uow, scope, run_id, owner=not review, agent_ids=agent_ids)
            current, availability = None, "ACCESS_CHANGED"
            try:
                _, current, availability = await self._policy(uow, run)
            except AIOpsApplicationError:
                pass
            artifacts = await uow.runs.list_artifacts(ops_run_id=run.ops_run_id)
            requests = [item for item in artifacts if item.schema_version == REQUEST_SCHEMA]
            if len(requests) > 16:
                raise state_conflict("计划采集请求数量超过冻结上限")
            views = []
            for item in requests:
                _, _, view = await self._load(uow=uow, scope=scope, run=run,
                    request_id=self._request_id(item), now=now, current=current, availability=availability)
                views.append(view)
            await uow.commit()
            return SqlPlanCapturePage(items=tuple(views), can_request=not review and availability == "AVAILABLE",
                can_review=review, availability=availability,
                max_statement_timeout_seconds=current["capture_policy"]["max_statement_timeout_seconds"] if current else None)

    async def decide(self, *, scope, run_id, request_id, command, agent_ids=()):
        now = self._clock()
        async with self._uow_factory() as uow:
            run = await self._run(uow, scope, run_id, owner=False, agent_ids=agent_ids)
            _, current, availability = await self._policy(uow, run)
            row, protected, view = await self._load(uow=uow, scope=scope, run=run,
                request_id=request_id, now=now, current=current, availability=availability)
            if view.request_sha256 != command.request_sha256:
                raise state_conflict("审批所见请求与冻结内容不一致，请刷新后重新审阅")
            if view.state != "PENDING":
                raise state_conflict("该计划采集请求已处理、过期或失效，不能重复审批")
            if command.decision == "APPROVE":
                if view.require_distinct_approver and str(scope.actor_id) == view.requested_by:
                    raise state_conflict("当前策略要求由另一位具备审批权限的操作人审阅")
                if (not command.acknowledge_estimate_only
                    or command.acknowledge_oracle_plan_table_write != view.oracle_plan_table_write):
                    raise state_conflict("请明确确认估算计划风险及对应引擎的计划表写入边界")
            decision = {"schema_version": DECISION_SCHEMA, "request_id": str(request_id),
                "request_sha256": view.request_sha256, "decision": command.decision,
                "decided_by": str(scope.actor_id), "decided_at": now.isoformat(),
                "acknowledge_estimate_only": command.acknowledge_estimate_only,
                "acknowledge_oracle_plan_table_write": command.acknowledge_oracle_plan_table_write}
            protected = {**protected, "state": "APPROVED" if command.decision == "APPROVE" else "REJECTED", "decision": decision}
            self._vault.update(row=row, scope=scope, payload=protected)
            await self._add(uow, run, PREFIX + "decision:" + str(request_id), DECISION_SCHEMA, decision,
                security_level=protected["request"]["security_level"])
            await uow.runs.append_event(ops_run_id=run.ops_run_id, event_type="sql.plan_capture.decided",
                event_key=PREFIX + str(request_id) + ":decided", visibility="USER",
                payload_json={"request_id": str(request_id), "state": protected["state"], "execution_authorized": False,
                    "public_summary": "计划采集风险审批已记录；尚未调度采集，不授权业务 SQL"})
            await uow.commit()
            return self._view(protected["request"], protected["state"], decision)

    async def revoke(self, *, scope, run_id, request_id, agent_ids=()):
        async with self._uow_factory() as uow:
            run = await self._run(uow, scope, run_id, owner=True, agent_ids=agent_ids)
            row, protected = await self._vault.get(uow=uow, scope=scope, request_id=request_id)
            if protected is None or str(protected["request"].get("ops_run_id")) != str(run.ops_run_id):
                raise resource_not_found("计划采集审批")
            if protected["state"] != "REVOKED":
                protected = {**protected, "state": "REVOKED"}
                self._vault.update(row=row, scope=scope, payload=protected)
                await uow.runs.append_event(ops_run_id=run.ops_run_id, event_type="sql.plan_capture.revoked",
                    event_key=PREFIX + str(request_id) + ":revoked", visibility="USER",
                    payload_json={"request_id": str(request_id), "execution_authorized": False,
                        "public_summary": "计划采集请求已撤回；原参数材料可另外撤回清除"})
            await uow.commit()
            return self._view(protected["request"], "REVOKED", protected["decision"])
