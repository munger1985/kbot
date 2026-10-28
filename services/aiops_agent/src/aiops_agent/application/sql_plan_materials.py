"""代表参数安全提交与撤回；保护材料与执行审批严格分离。"""

from datetime import UTC, datetime, timedelta
from dataclasses import asdict
import json
from uuid import UUID

from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found, state_conflict, validation_failed
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.domain.evidence.sql_plan_material import parse_sql_plan_material, SqlPlanMaterialInvalid
from aiops_agent.domain.evidence.sql_plan_capture import inspect_plan_capture, source_sql_text
from aiops_agent.domain.sql_subjects import normalize_sql_subject, SQL_STATISTICS_TOOLS
from aiops_agent.entities import OpsArtifactEntity
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_material import (
    SqlPlanCapturePreflight, SqlPlanMaterialPage, SqlPlanMaterialSource, SqlPlanMaterialSubmit, SqlPlanMaterialView,
)
from platform_core.identity import uuid7


MATERIAL_SCHEMA = "SQL_PLAN_MATERIAL_RECEIPT.v2"
MATERIAL_PREFIX = "sql-plan-material:"
_REQUIRED = {
    "POSTGRESQL": ("query_id", "database_oid", "user_oid"),
    "MYSQL": ("digest", "schema_name"),
    "ORACLE": ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value"),
}
_KIND = {"POSTGRESQL": "POSTGRESQL_QUERY_ID", "MYSQL": "MYSQL_DIGEST", "ORACLE": "ORACLE_SQL_ID"}


def _source_rows(artifact, run, database_type):
    """只接受同 Run/Target 的原始受控语句统计行，不使用模型副本或人工声明。"""
    if (str(artifact.ops_run_id) != str(run.ops_run_id)
        or artifact.schema_version != "DBA_TOOL_RESULT.v1" or artifact.trust_level != "SOURCE_VERIFIED"
        or canonical_hash(artifact.payload_json) != artifact.content_hash):
        return ()
    try:
        result = DbaToolResult.model_validate(artifact.payload_json)
    except (ValueError, TypeError):
        return ()
    if (result.status not in {"SUCCEEDED", "PARTIAL"}
        or len({item.step_id for item in result.tool_outcomes}) != len(result.tool_outcomes)):
        return ()
    sources = []
    for outcome in result.tool_outcomes:
        observation = outcome.observation
        if (outcome.status != "SUCCEEDED" or observation is None
            or outcome.tool_id != observation.tool_id or outcome.tool_version != observation.tool_version
            or observation.tool_id not in SQL_STATISTICS_TOOLS[database_type]
            or observation.db_type != database_type or str(observation.target_id) != str(run.target_id)
            or observation.truncated or observation.row_count != len(observation.rows)
            or not 1 <= len(outcome.step_id) <= 128):
            continue
        names = [column.name.lower() for column in observation.columns]
        if len(names) != len(set(names)):
            continue
        for index, row in enumerate(observation.rows[:256]):
            if len(row) != len(names):
                continue
            fields = dict(zip(names, row, strict=True))
            if not all(key in fields for key in _REQUIRED[database_type]):
                continue
            subject = normalize_sql_subject({"subject_type": _KIND[database_type],
                **{key: fields[key] for key in (*_REQUIRED[database_type], "top_level") if key in fields}}, database_type)
            if subject is None:
                continue
            sources.append(SqlPlanMaterialSource(
                source_artifact_id=artifact.artifact_id, step_id=outcome.step_id, row_index=index,
                database_type=database_type,
                subject={key: entry if isinstance(entry, bool) or entry is None else str(entry)
                    for key, entry in subject.items()},
            ))
    return tuple(sources)


class SqlPlanMaterialService:
    """公开接口只返回不透明引用和安全摘要；不提供参数读取或普通凭据引用。"""

    def __init__(self, *, uow_factory, vault, clock=None):
        self._uow_factory = uow_factory
        self._vault = vault
        self._clock = clock or (lambda: datetime.now(UTC))

    async def _owner(self, uow, scope, run_id, *, lock=False, agent_ids=()):
        run = await uow.runs.get_run_scoped(ops_run_id=run_id, domain_id=scope.domain_id, lock=lock)
        if (run is None or not self._vault.owns_run(scope=scope, run=run)
            or agent_ids and str(run.agent_id) not in {str(item) for item in agent_ids}):
            raise resource_not_found("SQL 计划材料")
        return run

    async def _access(self, uow, run):
        target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=run.domain_id)
        if (run.cancel_requested_at is not None or run.status in {"CANCELLED", "CANCELLING", "FAILED", "EXPIRED"}
            or target is None or target.status != "ENABLED" or not target.readonly_connection_enabled
            or target.diagnostic_credential_id is None):
            raise state_conflict("目标或原诊断已不可用于提交计划材料")
        frozen_database = dict(dict(dict(run.plan_snapshot_json or {}).get("investigation_execution") or {}).get("database") or {})
        if (frozen_database.get("automatic_access_enabled") is not True
            or str(frozen_database.get("target_row_version")) != str(target.row_version)
            or frozen_database.get("db_type") != target.db_type
            or frozen_database.get("connection_profile") != dict(target.endpoint_json or {})
            or str(frozen_database.get("diagnostic_credential_id")) != str(target.diagnostic_credential_id)):
            raise state_conflict("原诊断的目标版本、连接或凭据引用已变化，请重新取证后提交材料")
        binding = await uow.agents.get_active(domain_id=run.domain_id, agent_id=run.agent_id, target_id=run.target_id)
        if binding is None or str(binding.binding_id) != str(run.agent_version_id):
            raise state_conflict("Agent 与目标的当前绑定已变化")
        frozen_binding = dict(dict(run.plan_snapshot_json or {}).get("binding") or {})
        if str(binding.policy_id or "") != str(frozen_binding.get("policy_id") or ""):
            raise state_conflict("原诊断的执行策略已变化")
        policy_hash = None
        if binding.policy_id is not None:
            policy = await uow.policies.get_scoped(policy_id=binding.policy_id, domain_id=run.domain_id)
            if (policy is None or policy.status != "ACTIVE"
                or not dict(policy.rules_json or {}).get("readonly_database_enabled", False)):
                raise state_conflict("当前策略不允许数据库只读取证")
            policy_hash = policy.policy_hash
        return target, {
            "target_state_hash": canonical_hash({
                "target_id": str(target.target_id), "row_version": int(target.row_version),
                "database_type": target.db_type, "endpoint": target.endpoint_json,
                "diagnostic_credential_id": str(target.diagnostic_credential_id),
            }),
            "binding_state_hash": canonical_hash({
                "binding_id": str(binding.binding_id), "policy_id": str(binding.policy_id or ""),
                "policy_hash": policy_hash,
            }),
        }

    async def sources(self, *, scope, run_id, agent_ids=()):
        async with self._uow_factory() as uow:
            run = await self._owner(uow, scope, run_id, agent_ids=agent_ids)
            target, _ = await self._access(uow, run)
            artifacts = await uow.runs.list_artifacts(ops_run_id=run.ops_run_id)
            sources = []
            for artifact in reversed(artifacts):
                sources.extend(_source_rows(artifact, run, target.db_type))
                if len(sources) >= 64:
                    break
            return tuple(sources[:64])

    async def submit(self, *, scope, run_id, command: SqlPlanMaterialSubmit, agent_ids=()):
        try:
            protected = parse_sql_plan_material(command.protected_document.get_secret_value())
        except SqlPlanMaterialInvalid:
            raise validation_failed("计划参数材料格式无效；数值绑定请使用精确字符串，不通过聊天或普通附件提交原值") from None
        now = self._clock()
        async with self._uow_factory() as uow:
            run = await self._owner(uow, scope, run_id, lock=True, agent_ids=agent_ids)
            target, access = await self._access(uow, run)
            key = MATERIAL_PREFIX + str(command.submission_id)
            if await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=key) is not None:
                raise state_conflict("材料提交标识已使用，请刷新查看，不重复保存")
            artifacts = await uow.runs.list_artifacts(ops_run_id=run.ops_run_id)
            receipts = [item for item in artifacts if item.schema_version == MATERIAL_SCHEMA]
            if len(receipts) >= 16:
                raise state_conflict("每轮最多保存16份参数材料；请在新的诊断轮次补充")
            source_artifact = await uow.runs.get_artifact(artifact_id=command.source_artifact_id)
            candidates = _source_rows(source_artifact, run, target.db_type) if source_artifact else ()
            source = next((item for item in candidates
                if item.step_id == command.step_id and item.row_index == command.row_index), None)
            if source is None:
                raise state_conflict("所选 SQL 原始证据已不可用或统计身份不完整，请重新取证")
            native = next(item for item in source_artifact.payload_json["tool_outcomes"] if item["step_id"] == source.step_id)["observation"]
            fields = dict(zip((column["name"].lower() for column in native["columns"]), native["rows"][source.row_index], strict=True))
            inspection = inspect_plan_capture(
                database_type=target.db_type, sql=source_sql_text(fields, target.db_type),
                parameter_count=len(protected["parameters"]))
            preflight = SqlPlanCapturePreflight.model_validate(asdict(inspection))
            active = 0
            for receipt in receipts:
                view = await self._view(uow=uow, scope=scope, run=run, artifact=receipt, now=now, access=access)
                active += view.state == "STORED"
            if active >= 4:
                raise state_conflict("每轮最多同时保留4份有效材料；请先撤回旧材料")
            payload = {
                "schema_version": MATERIAL_SCHEMA, "material_id": str(command.submission_id),
                "ops_run_id": str(run.ops_run_id), "actor_id": str(scope.actor_id),
                "source": source.model_dump(mode="json"), "source_hash": source_artifact.content_hash,
                "parameter_count": len(protected["parameters"]), "created_at": now.isoformat(),
                "capture_preflight": preflight.model_dump(mode="json"),
                "expires_at": (now + timedelta(seconds=command.ttl_seconds)).isoformat(), **access,
            }
            await self._vault.seal(uow=uow, scope=scope, material_id=command.submission_id,
                payload={"binding": payload, "document": protected})
            artifact = await uow.runs.add_artifact(OpsArtifactEntity(
                artifact_id=uuid7(), ops_run_id=run.ops_run_id, ops_task_id=None, artifact_key=key,
                artifact_type="EVIDENCE", schema_version=MATERIAL_SCHEMA, payload_json=payload,
                content_hash=canonical_hash(payload), byte_size=len(json.dumps(payload, ensure_ascii=False).encode()),
                provenance_json={"producer": "aiops.sql-plan-material", "contains_parameter_values": False},
                trust_level="USER_PROVIDED", security_level=source_artifact.security_level,
            ))
            await uow.runs.append_event(ops_run_id=run.ops_run_id, event_type="sql.plan_material.stored",
                event_key=key + ":stored", visibility="USER",
                payload_json={"material_id": str(command.submission_id), "parameter_count": payload["parameter_count"],
                    "expires_at": payload["expires_at"], "execution_authorized": False,
                    "public_summary": "代表参数已保护保存；未执行计划采集，参数及解析环境尚未经核验"})
            await uow.commit()
            return self._public(run, payload, "STORED")

    def _public(self, run, payload, state):
        return SqlPlanMaterialView(material_id=payload["material_id"], ops_run_id=run.ops_run_id,
            source=payload["source"], parameter_count=payload["parameter_count"],
            capture_preflight=payload["capture_preflight"] if state == "STORED" else None,
            created_at=payload["created_at"], expires_at=payload["expires_at"], state=state)

    async def _view(self, *, uow, scope, run, artifact, now, access):
        payload = artifact.payload_json
        if (artifact.artifact_key != MATERIAL_PREFIX + payload.get("material_id", "")
            or canonical_hash(payload) != artifact.content_hash
            or str(payload.get("ops_run_id")) != str(run.ops_run_id)
            or str(payload.get("actor_id")) != str(scope.actor_id)):
            raise state_conflict("计划材料回执无法复核")
        # 先核对独立密文中的绑定，不能靠可修改的回执为自身证明来源或期限。
        state = await self._vault.state(uow=uow, scope=scope, material_id=UUID(payload["material_id"]), binding=payload, now=now)
        if state == "STORED":
            source = SqlPlanMaterialSource.model_validate(payload["source"])
            original = await uow.runs.get_artifact(artifact_id=source.source_artifact_id)
            if (access is None or any(payload.get(key) != value for key, value in access.items())
                or original is None or original.content_hash != payload["source_hash"]
                or source not in _source_rows(original, run, source.database_type)):
                state = "UNAVAILABLE"
        return self._public(run, payload, state)

    async def list(self, *, scope, run_id, agent_ids=()):
        now = self._clock()
        async with self._uow_factory() as uow:
            run = await self._owner(uow, scope, run_id, lock=True, agent_ids=agent_ids)
            try:
                _, access = await self._access(uow, run)
            except AIOpsApplicationError:
                access = None
            artifacts = await uow.runs.list_artifacts(ops_run_id=run.ops_run_id)
            receipts = [item for item in artifacts if item.schema_version == MATERIAL_SCHEMA]
            if len(receipts) > 16:
                raise state_conflict("计划材料数量超过冻结上限")
            items = tuple([await self._view(uow=uow, scope=scope, run=run, artifact=item, now=now, access=access) for item in receipts])
            await uow.commit()
            return SqlPlanMaterialPage(items=items)

    async def revoke(self, *, scope, run_id, material_id, agent_ids=()):
        now = self._clock()
        async with self._uow_factory() as uow:
            # 撤回不依赖当前取证权限，Target 下线、来源失效后所有者仍可清除在线密文。
            run = await self._owner(uow, scope, run_id, lock=True, agent_ids=agent_ids)
            artifact = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=MATERIAL_PREFIX + str(material_id))
            if artifact is None or artifact.schema_version != MATERIAL_SCHEMA:
                raise resource_not_found("SQL 计划材料")
            await self._vault.erase(uow=uow, scope=scope, material_id=material_id, binding=artifact.payload_json)
            await uow.runs.append_event(ops_run_id=run.ops_run_id, event_type="sql.plan_material.revoked",
                event_key=MATERIAL_PREFIX + str(material_id) + ":revoked", visibility="USER",
                payload_json={"material_id": str(material_id), "execution_authorized": False,
                    "public_summary": "参数材料已撤回，在线密文已清除；平台备份仍按既有留存策略处理"})
            await uow.commit()
            return self._public(run, artifact.payload_json, "REVOKED")
