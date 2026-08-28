"""Turn 的 Intent、能力快照和 Skill Plan 应用服务。"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict
from aiops_agent.application.conversation_inputs import (
    ConversationUploadSource,
    ConversationInputResolver,
    ResolvedConversationUpload,
)
from aiops_agent.entities import (
    OpsArtifactEntity,
    OpsInvestigationRevisionEntity,
    OpsPlaybookInvocationEntity,
    OpsTaskEntity,
    OpsToolInvocationEntity,
    OpsTurnEventEntity,
    OpsTurnEvidenceEntity,
)
from aiops_agent.investigation import InvestigationReasoner
from aiops_agent.investigation.reasoner import (
    InvestigationPlanValidationError,
)
from aiops_agent.diagnostics import (
    DynamicQueryPolicySnapshot,
    DynamicQueryRejected,
    OracleDynamicQueryPolicy,
)
from aiops_agent.monitoring import (
    LogQueryPolicy,
    LogQueryPolicySnapshot,
    MonitoringQueryRejected,
    PromQueryPolicy,
    PromQueryPolicySnapshot,
)
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_LOG_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
)
from aiops_agent.skills import (
    DbaSkillRegistry,
    SkillExecutionSnapshotBuilder,
    SkillPlanCompiler,
    build_capability_snapshot,
    canonical_hash,
)
from platform_core.contracts.aiops.skills import (
    DbaCapabilitySnapshot,
    DbaSkillPlan,
)
from platform_core.identity import uuid7


@dataclass(frozen=True, slots=True)
class TurnPlanningContext:
    domain_id: int
    turn_id: UUID
    conversation_id: UUID
    ops_run_id: UUID
    agent_id: UUID
    target_id: UUID
    source_ids: tuple[UUID, ...]
    actor_id: str
    question: str
    content: tuple[dict, ...]
    image_capabilities: dict
    recent_context: tuple[str, ...]
    trace_id: str
    deadline: datetime | None
    capabilities: DbaCapabilitySnapshot
    database_execution: dict
    change_context: dict
    source_run_evidence: dict | None = None
    raw_uploads: tuple[ConversationUploadSource, ...] = ()
    resolved_uploads: tuple[ResolvedConversationUpload, ...] = ()
    input_artifact_id: UUID | None = None
    upload_artifact_ids: tuple[tuple[int, UUID, UUID | None], ...] = ()


class _PlanningAlreadyApplied(Exception):
    def __init__(self, result: dict) -> None:
        super().__init__("Turn 计划已经持久化")
        self.result = result


class TurnPlanningStageError(RuntimeError):
    """保留规划内部故障的安全阶段信息，不携带用户输入或凭据。"""

    code = "AIOPS_INVESTIGATION_PLAN_INTERNAL_ERROR"

    _STAGE_BY_FUNCTION = {
        "_prepare": "PREPARE_CONTEXT",
        "_persist_raw_input": "PERSIST_RAW_INPUT",
        "_persist_input_extractions": "PERSIST_INPUT_EXTRACTIONS",
        "plan": "MODEL_PLANNING",
        "_prepare_dynamic_queries": "VALIDATE_DYNAMIC_SQL",
        "_prepare_source_queries": "VALIDATE_MONITORING_QUERY",
        "_build_playbook_plan": "BUILD_PLAYBOOK_PLAN",
        "_prepare_monitoring": "PREPARE_MONITORING",
        "compile": "COMPILE_TASK_PLAN",
        "build": "BUILD_EXECUTION_SNAPSHOT",
        "_persist": "PERSIST_INVESTIGATION_PLAN",
    }

    def __init__(self, cause: Exception) -> None:
        frames = traceback.extract_tb(cause.__traceback__)
        frame = frames[-1] if frames else None
        stage = "PROCESS_INVESTIGATION_PLAN"
        for candidate in reversed(frames):
            mapped = self._STAGE_BY_FUNCTION.get(candidate.name)
            if mapped is not None:
                stage = mapped
                break
        location = (
            f"{frame.name}:{frame.lineno}" if frame is not None else "unknown"
        )
        detail = self._safe_detail(cause)
        super().__init__(
            f"stage={stage}; cause={type(cause).__name__}; "
            f"detail={detail}; location={location}"
        )
        self.stage = stage
        self.cause_type = type(cause).__name__
        self.safe_detail = detail
        self.location = location

    @staticmethod
    def _safe_detail(cause: Exception) -> str:
        if not isinstance(cause, KeyError) or not cause.args:
            return "not-recorded"
        key = str(cause.args[0])
        if re.fullmatch(r"[A-Za-z0-9_.:@/-]{1,120}", key):
            return f"missing-key:{key}"
        return "missing-key:<non-identifier>"


class TurnPlanningService:
    """在外部模型调用两侧使用短事务冻结并持久化计划。"""

    def __init__(
        self,
        *,
        uow_factory,
        investigation_reasoner: InvestigationReasoner,
        playbook_registry: DbaSkillRegistry,
        skill_compiler: SkillPlanCompiler,
        execution_snapshot_builder: SkillExecutionSnapshotBuilder,
        agent_catalog,
        monitoring_snapshot_builder=None,
        conversation_input_resolver: ConversationInputResolver | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._investigation_reasoner = investigation_reasoner
        self._playbook_registry = playbook_registry
        self._skill_compiler = skill_compiler
        self._execution_snapshot_builder = execution_snapshot_builder
        self._agent_catalog = agent_catalog
        self._monitoring_snapshot_builder = monitoring_snapshot_builder
        self._conversation_input_resolver = conversation_input_resolver

    async def execute(self, payload: dict) -> dict:
        """执行首轮规划，并把未预期异常转换为可审计的阶段错误。"""
        try:
            return await self._execute_once(payload)
        except (InvestigationPlanValidationError, TurnPlanningStageError):
            raise
        except Exception as exc:
            raise TurnPlanningStageError(exc) from exc

    async def _execute_once(self, payload: dict) -> dict:
        try:
            context = await self._prepare(payload)
        except _PlanningAlreadyApplied as applied:
            return applied.result
        if self._conversation_input_resolver is not None:
            context = replace(
                context,
                raw_uploads=self._conversation_input_resolver.describe_sources(
                    domain_id=context.domain_id,
                    actor_id=context.actor_id,
                    content=context.content,
                ),
            )
        context = await self._persist_raw_input(context)
        if self._conversation_input_resolver is not None and any(
            item.get("upload_id") for item in context.content
        ):
            content, uploads = await self._conversation_input_resolver.resolve(
                domain_id=context.domain_id,
                actor_id=context.actor_id,
                content=context.content,
                image_capabilities=context.image_capabilities,
            )
            context = replace(
                context, content=content, resolved_uploads=uploads
            )
            context = await self._persist_input_extractions(context)
        model_snapshot = await self._agent_catalog.resolve_diagnosis_model(
            agent_id=context.agent_id,
            domain_id=context.domain_id,
            trace_id=context.trace_id,
        )
        planned = await self._investigation_reasoner.plan(
            content=context.content,
            conversation_context=context.recent_context,
            source_run_evidence=context.source_run_evidence,
            available_tools=self._available_tools(context.capabilities),
            available_playbooks=self._available_playbooks(context.capabilities),
            model_snapshot=model_snapshot,
            deadline=context.deadline,
            idempotency_key=f"turn:{context.turn_id}:investigation:1",
        )
        investigation, dynamic_queries = self._prepare_dynamic_queries(
            planned.output
        )
        investigation, source_queries = self._prepare_source_queries(
            investigation
        )
        playbook_plan = self._build_playbook_plan(
            investigation=investigation,
            capabilities=context.capabilities,
        )
        monitoring_requested = any(
            action.tool_id.startswith("monitor.")
            or action.tool_id.startswith("prometheus.")
            or action.tool_id.startswith("loki.")
            for action in investigation.plan.actions
        )
        monitoring_execution = (
            await self._prepare_monitoring(context)
            if monitoring_requested
            else {}
        )
        if monitoring_requested:
            monitoring_execution = {
                **monitoring_execution,
                **source_queries,
            }
        monitoring_binding_ids = (
            tuple(
                item["binding_id"]
                for item in monitoring_execution.get("bindings", ())
                if CAPABILITY_METRIC_QUERY_RANGE
                in item.get("effective_capabilities", ())
            )
            if monitoring_requested
            else ()
        )
        log_binding_ids = (
            tuple(monitoring_execution.get("log_binding_ids", ()))
            if monitoring_requested
            else ()
        )
        compiled = self._skill_compiler.compile(
            playbook_plan,
            monitoring_binding_ids=monitoring_binding_ids,
            log_binding_ids=log_binding_ids,
            user_evidence_artifact_keys=(
                "turn-user-input:1",
                "turn-input-analysis:1",
                *(
                    ("turn-source-run-evidence:1",)
                    if context.source_run_evidence is not None
                    else ()
                ),
            ),
            include_change=bool(investigation.task_frame.requires_change),
            investigation_actions=investigation.plan.actions,
        )
        execution_snapshot = self._execution_snapshot_builder.build(
            plan=playbook_plan,
            compiled=compiled,
            capabilities=context.capabilities,
            database_execution=context.database_execution,
            dynamic_queries=dynamic_queries,
            direct_actions=tuple(
                action
                for action in investigation.plan.actions
                if action.tool_id
                not in {
                    "monitor.query_range",
                    "loki.query_range",
                    "db.oracle.readonly_query",
                }
            ),
        )
        return await self._persist(
            context=context,
            investigation=investigation,
            planning_receipt=planned.receipt,
            playbook_plan=playbook_plan,
            compiled=compiled,
            execution_snapshot=execution_snapshot,
            model_snapshot=model_snapshot,
            monitoring_requested=monitoring_requested,
            monitoring_execution=monitoring_execution,
        )

    def _available_tools(
        self, capabilities: DbaCapabilitySnapshot
    ) -> tuple[dict, ...]:
        """向模型暴露当前数据库类型可用的原子只读工具，不暴露 SQL 模板。"""
        tools = {
            (item["tool_id"], item["version"]): item
            for item in self._execution_snapshot_builder.discover_tools(
                capabilities
            )
        }
        if CAPABILITY_METRIC_QUERY_RANGE in capabilities.available_source_capabilities:
            tools[("monitor.query_range", "1.0.0")] = {
                "tool_id": "monitor.query_range",
                "version": "1.0.0",
                "tool_class": "PROMETHEUS",
                "description": (
                    "执行受控 PromQL 时间序列查询；每个向量选择器必须用"
                    " instance=\"${external_target}\" 或"
                    " target_key=\"${host_target}\" 精确绑定当前 Target"
                ),
                "input": {
                    "query": "带 Target 占位符的 PromQL",
                    "window_seconds": "60 到 3600 秒",
                },
            }
        if CAPABILITY_LOG_QUERY in capabilities.available_source_capabilities:
            tools[("loki.query_range", "1.0.0")] = {
                "tool_id": "loki.query_range",
                "version": "1.0.0",
                "tool_class": "LOKI",
                "description": (
                    "执行受控 LogQL；必须以 ${binding_selector} 开始，"
                    "仅允许 |= 或 != 字面量行过滤"
                ),
                "input": {
                    "query": "${binding_selector} 加字面量过滤",
                    "window_seconds": "60 到 3600 秒",
                },
            }
        if (
            str(capabilities.database_type) == "ORACLE"
            and "DB_READONLY" in capabilities.target_capabilities
        ):
            tools[("db.oracle.readonly_query", "1.0.0")] = {
                "tool_id": "db.oracle.readonly_query",
                "version": "1.0.0",
                "tool_class": "ORACLE_SQL_DYNAMIC",
                "description": (
                    "在只读事务中执行一条受 AST 策略约束的 Oracle 诊断 SELECT；"
                    "仅在固定目录工具不能回答问题时使用，必须显式投影并使用 bind 参数"
                ),
                "input": {
                    "sql": "Oracle SELECT，必须为每个计算表达式提供别名",
                    "parameters": "与 SQL bind 名称完全一致的标量对象",
                },
            }
        return tuple(tools[key] for key in sorted(tools))

    @staticmethod
    def _prepare_dynamic_queries(investigation):
        """规划端先验证并规范化动态 SQL，再冻结供 Executor 重放。"""
        snapshot = DynamicQueryPolicySnapshot()
        policy = OracleDynamicQueryPolicy(snapshot)
        actions = []
        frozen = []
        for action in investigation.plan.actions:
            if action.tool_id != "db.oracle.readonly_query":
                actions.append(action)
                continue
            payload = dict(action.input)
            if set(payload) != {"sql", "parameters"}:
                raise InvestigationPlanValidationError(
                    "动态查询输入必须且只能包含 sql 与 parameters"
                )
            sql = payload.get("sql")
            parameters = payload.get("parameters")
            if not isinstance(sql, str) or not isinstance(parameters, dict):
                raise InvestigationPlanValidationError(
                    "动态查询 sql 或 parameters 类型无效"
                )
            try:
                validated = policy.validate(sql, parameters)
            except DynamicQueryRejected as exc:
                raise InvestigationPlanValidationError(
                    f"动态查询未通过策略：{exc.code}"
                ) from exc
            actions.append(
                action.model_copy(
                    update={
                        "input": {
                            "sql": validated.normalized_sql,
                            "parameters": dict(validated.parameters),
                        }
                    }
                )
            )
            frozen.append(
                {
                    "action_id": action.action_id,
                    "question": action.question,
                    "measurement_semantics": action.measurement_semantics,
                    "policy_snapshot": snapshot.model_dump(mode="json"),
                    "validated_query": validated.model_dump(mode="json"),
                    "limits": {
                        "statement_timeout_seconds": 20,
                        "max_result_rows": validated.max_rows,
                        "max_result_bytes": 1048576,
                        "max_columns": 64,
                        "max_cell_chars": 32768,
                    },
                }
            )
        updated_plan = investigation.plan.model_copy(
            update={"actions": tuple(actions)}
        )
        return (
            investigation.model_copy(update={"plan": updated_plan}),
            tuple(frozen),
        )

    @staticmethod
    def _prepare_source_queries(investigation):
        """校验并冻结模型提出的临时 PromQL 与 LogQL。"""
        prom_policy = PromQueryPolicy(PromQueryPolicySnapshot())
        log_policy = LogQueryPolicy(LogQueryPolicySnapshot())
        actions = []
        prom_queries = []
        log_queries = []
        for action in investigation.plan.actions:
            if action.tool_id not in {
                "monitor.query_range",
                "loki.query_range",
            }:
                actions.append(action)
                continue
            payload = dict(action.input)
            if (
                set(payload) - {"query", "window_seconds"}
                or "query" not in payload
            ):
                raise InvestigationPlanValidationError(
                    "监控查询输入只能包含 query 与 window_seconds"
                )
            query = payload.get("query")
            window = payload.get("window_seconds")
            if not isinstance(query, str) or (
                "window_seconds" in payload
                and not isinstance(window, int)
            ):
                raise InvestigationPlanValidationError(
                    "监控查询 query 或 window_seconds 类型无效"
                )
            try:
                if action.tool_id == "monitor.query_range":
                    validated = prom_policy.validate(
                        query, window_seconds=window
                    )
                    target = prom_queries
                else:
                    validated = log_policy.validate(
                        query, window_seconds=window
                    )
                    target = log_queries
            except MonitoringQueryRejected as exc:
                raise InvestigationPlanValidationError(
                    f"监控查询未通过策略：{exc.code}"
                ) from exc
            normalized_input = {
                "query": validated.normalized_query,
                "window_seconds": validated.window_seconds,
            }
            actions.append(
                action.model_copy(update={"input": normalized_input})
            )
            target.append(
                {
                    "action_id": action.action_id,
                    "question": action.question,
                    "measurement_semantics": action.measurement_semantics,
                    "validated_query": validated.model_dump(mode="json"),
                }
            )
        updated_plan = investigation.plan.model_copy(
            update={"actions": tuple(actions)}
        )
        if len(prom_queries) > 4 or len(log_queries) > 4:
            raise InvestigationPlanValidationError(
                "单轮临时 PromQL 或 LogQL 查询不能超过 4 条"
            )
        return (
            investigation.model_copy(update={"plan": updated_plan}),
            {
                "ad_hoc_prometheus_queries": prom_queries,
                "ad_hoc_log_queries": log_queries,
            },
        )

    def _available_playbooks(
        self, capabilities: DbaCapabilitySnapshot
    ) -> tuple[dict, ...]:
        """Playbook 只提供调查经验，不决定 Agent 是否能够回答。"""
        return tuple(
            {
                "playbook_id": manifest.skill_id,
                "version": manifest.version,
                "tools": [step.tool_id for step in manifest.tool_dag],
                "subjects": list(manifest.subjects),
            }
            for manifest in self._playbook_registry.manifests()
            if self._manifest_applicable(manifest, capabilities)
        )

    @staticmethod
    def _manifest_applicable(manifest, capabilities: DbaCapabilitySnapshot) -> bool:
        """只按确定性能力与版本边界筛选Playbook，不使用意图作为准入条件。"""
        if capabilities.database_type not in manifest.database_types:
            return False
        if not set(manifest.required_target_capabilities) <= set(
            capabilities.target_capabilities
        ):
            return False
        if not set(manifest.required_source_capabilities) <= set(
            capabilities.available_source_capabilities
        ):
            return False
        if not set(manifest.required_entitlements) <= set(
            capabilities.entitlements
        ):
            return False
        configured_privileges = set(capabilities.privileges)
        if configured_privileges and not set(manifest.required_privileges) <= (
            configured_privileges
        ):
            return False
        configured_version = capabilities.database_version
        if configured_version is None:
            return (
                manifest.version_range.minimum is None
                and manifest.version_range.maximum is None
            )
        version_match = re.search(r"\d+", configured_version)
        if version_match is None:
            return False
        major = int(version_match.group(0))
        minimum = manifest.version_range.minimum
        maximum = manifest.version_range.maximum
        return (
            (minimum is None or major >= int(minimum))
            and (maximum is None or major <= int(maximum))
        )

    def _build_playbook_plan(
        self, *, investigation, capabilities: DbaCapabilitySnapshot
    ) -> DbaSkillPlan:
        """保存Playbook目录快照；原子Tool执行不再要求隶属Playbook。"""
        del investigation, capabilities
        return DbaSkillPlan(
            catalog_hash=self._playbook_registry.catalog_hash,
            items=(),
        )

    async def execute_replan(self, payload: dict) -> dict:
        """根据上一轮Evidence Assessment生成并持久化下一轮调查DAG。"""
        revision_no = int(payload["revision_no"])
        if revision_no != 2:
            raise state_conflict("当前调查预算最多允许两轮")
        context = await self._prepare(payload, revision_no=revision_no)
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=context.domain_id,
                turn_id=context.turn_id,
            )
            assessment_artifact = await uow.runs.get_artifact(
                artifact_id=UUID(str(payload["assessment_artifact_id"]))
            )
            plan_artifact = (
                await uow.runs.get_artifact(
                    artifact_id=turn.current_plan_artifact_id
                )
                if turn is not None
                and turn.current_plan_artifact_id is not None
                else None
            )
            task_frame_artifact = (
                await uow.runs.get_artifact(
                    artifact_id=turn.task_frame_artifact_id
                )
                if turn is not None
                and turn.task_frame_artifact_id is not None
                else None
            )
            prior_artifacts = await uow.runs.list_artifacts(
                ops_run_id=context.ops_run_id
            )
        if (
            assessment_artifact is None
            or assessment_artifact.schema_version != "DBA_SUFFICIENCY.v1"
            or plan_artifact is None
            or task_frame_artifact is None
        ):
            raise state_conflict("重规划缺少上一轮评估或调查计划")
        model_snapshot = await self._agent_catalog.resolve_diagnosis_model(
            agent_id=context.agent_id,
            domain_id=context.domain_id,
            trace_id=context.trace_id,
        )
        prior_plan = dict(plan_artifact.payload_json or {})
        planned = await self._investigation_reasoner.replan(
            content=context.content,
            conversation_context=context.recent_context,
            source_run_evidence=context.source_run_evidence,
            task_frame=dict(task_frame_artifact.payload_json or {}),
            prior_plan=prior_plan,
            assessment=dict(assessment_artifact.payload_json or {}),
            available_tools=self._available_tools(context.capabilities),
            available_playbooks=self._available_playbooks(
                context.capabilities
            ),
            model_snapshot=model_snapshot,
            deadline=context.deadline,
            idempotency_key=(
                f"turn:{context.turn_id}:investigation:{revision_no}"
            ),
            revision_no=revision_no,
        )
        investigation, dynamic_queries = self._prepare_dynamic_queries(
            planned.output
        )
        investigation, source_queries = self._prepare_source_queries(
            investigation
        )
        playbook_plan = self._build_playbook_plan(
            investigation=investigation,
            capabilities=context.capabilities,
        )
        monitoring_requested = any(
            action.tool_id in {"monitor.query_range", "loki.query_range"}
            for action in investigation.plan.actions
        )
        monitoring_execution = (
            await self._prepare_monitoring(context)
            if monitoring_requested
            else {}
        )
        if monitoring_requested:
            monitoring_execution = {
                **monitoring_execution,
                **source_queries,
            }
        monitoring_binding_ids = tuple(
            item["binding_id"]
            for item in monitoring_execution.get("bindings", ())
            if CAPABILITY_METRIC_QUERY_RANGE
            in item.get("effective_capabilities", ())
        )
        log_binding_ids = tuple(
            monitoring_execution.get("log_binding_ids", ())
        )
        evidence_keys = tuple(
            item.artifact_key
            for item in prior_artifacts
            if item.schema_version
            in {
                "USER_PROVIDED_INPUT.v1",
                "SOURCE_RUN_EVIDENCE.v1",
                "DBA_SKILL_RESULT.v1",
                "OBSERVATION_SET.v1",
                "LOG_EVIDENCE_SET.v1",
            }
        )
        compiled = self._skill_compiler.compile(
            playbook_plan,
            monitoring_binding_ids=monitoring_binding_ids,
            log_binding_ids=log_binding_ids,
            user_evidence_artifact_keys=evidence_keys,
            revision_no=revision_no,
            include_answer=False,
            investigation_actions=investigation.plan.actions,
        )
        execution_snapshot = self._execution_snapshot_builder.build(
            plan=playbook_plan,
            compiled=compiled,
            capabilities=context.capabilities,
            database_execution=context.database_execution,
            dynamic_queries=dynamic_queries,
            direct_actions=tuple(
                action
                for action in investigation.plan.actions
                if action.tool_id
                not in {
                    "monitor.query_range",
                    "loki.query_range",
                    "db.oracle.readonly_query",
                }
            ),
        )
        return await self._persist_replan(
            context=context,
            revision_no=revision_no,
            investigation=investigation,
            planning_receipt=planned.receipt,
            playbook_plan=playbook_plan,
            compiled=compiled,
            execution_snapshot=execution_snapshot,
            model_snapshot=model_snapshot,
            monitoring_execution=monitoring_execution,
        )

    async def _persist_replan(
        self,
        *,
        context: TurnPlanningContext,
        revision_no: int,
        investigation,
        planning_receipt,
        playbook_plan,
        compiled,
        execution_snapshot: dict,
        model_snapshot: dict,
        monitoring_execution: dict,
    ) -> dict:
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=context.domain_id,
                turn_id=context.turn_id,
                lock=True,
            )
            run = await uow.runs.get_run(
                ops_run_id=context.ops_run_id,
                lock=True,
            )
            if turn is None or run is None:
                raise resource_not_found("Turn Primary Run")
            if int(turn.current_plan_revision or 1) >= revision_no:
                return {
                    "turn_id": str(turn.turn_id),
                    "ops_run_id": str(run.ops_run_id),
                    "status": turn.status,
                }
            if turn.status != "REPLANNING" or run.status != "RUNNING":
                raise state_conflict("Turn重规划提交时状态已变化")
            task_frame_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key=f"turn-task-frame:{revision_no}",
                artifact_type="DBA_TASK_FRAME",
                schema_version=investigation.task_frame.schema_version,
                payload=investigation.task_frame.model_dump(mode="json"),
                producer="aiops.task-reframer",
            )
            plan_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key=f"turn-investigation-plan:{revision_no}",
                artifact_type="DBA_INVESTIGATION_PLAN",
                schema_version=investigation.plan.schema_version,
                payload=investigation.plan.model_dump(mode="json"),
                producer="aiops.investigation-replanner",
            )
            playbook_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key=f"turn-playbook-plan:{revision_no}",
                artifact_type="DBA_PLAYBOOK_PLAN",
                schema_version=playbook_plan.schema_version,
                payload=playbook_plan.model_dump(mode="json"),
                producer="aiops.playbook-selector",
            )
            for artifact in (
                task_frame_artifact,
                plan_artifact,
                playbook_artifact,
            ):
                await uow.runs.add_artifact(artifact)
            revision = OpsInvestigationRevisionEntity(
                revision_id=uuid7(),
                turn_id=turn.turn_id,
                revision_no=revision_no,
                revision_type="EVIDENCE_REPLAN",
                trigger_reason="上一轮评估仍有可由系统自动补齐的关键证据缺口",
                task_frame_artifact_id=task_frame_artifact.artifact_id,
                plan_artifact_id=plan_artifact.artifact_id,
                created_by="aiops.investigation-replanner",
            )
            await uow.turns.add_investigation_revision(revision)
            task_ids = {
                spec.task_key: uuid7() for spec in compiled.tasks
            }
            task_specs = {spec.task_key: spec for spec in compiled.tasks}
            tasks = [
                OpsTaskEntity(
                    ops_task_id=task_ids[spec.task_key],
                    ops_run_id=run.ops_run_id,
                    parent_task_id=(
                        task_ids[spec.depends_on[0]]
                        if spec.depends_on
                        else None
                    ),
                    task_key=spec.task_key,
                    task_type=spec.task_type,
                    handler_id=spec.handler_id,
                    handler_version=spec.handler_version,
                    input_schema_version=spec.input_schema_version,
                    output_schema_version=spec.output_schema_version,
                    depends_on_json=list(spec.depends_on),
                    input_artifacts_json=list(spec.input_artifact_keys),
                    status="READY" if not spec.depends_on else "PENDING",
                    priority=spec.priority,
                    max_attempts=spec.max_attempts,
                    timeout_seconds=spec.timeout_seconds,
                )
                for spec in compiled.tasks
            ]
            await uow.runs.add_tasks(tasks)
            invocation_by_action = {}
            for item, task_key in zip(
                playbook_plan.items,
                compiled.invocation_task_keys,
                strict=True,
            ):
                invocation_id = uuid7()
                await uow.turns.add_playbook_invocation(
                    OpsPlaybookInvocationEntity(
                        playbook_invocation_id=invocation_id,
                        turn_id=turn.turn_id,
                        revision_id=revision.revision_id,
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task_ids[task_key],
                        ordinal=item.ordinal,
                        playbook_id=item.skill_id,
                        playbook_version=item.skill_version,
                        manifest_hash=item.manifest_hash,
                        status="PLANNED",
                        input_schema_version=task_specs[
                            task_key
                        ].input_schema_version,
                        input_json=dict(item.input),
                    )
                )
                if item.action_id is not None:
                    invocation_by_action[item.action_id] = (
                        task_ids[task_key],
                        invocation_id,
                    )
            monitoring_task_id = (
                task_ids[compiled.monitoring_task_keys[0]]
                if compiled.monitoring_task_keys
                else None
            )
            log_task_id = (
                task_ids[compiled.log_task_keys[0]]
                if compiled.log_task_keys
                else None
            )
            dynamic_task_by_action = {
                action.action_id: task_ids[task_key]
                for action, task_key in zip(
                    (
                        item
                        for item in investigation.plan.actions
                        if item.tool_id == "db.oracle.readonly_query"
                    ),
                    compiled.dynamic_task_keys,
                    strict=True,
                )
            }
            diagnostic_task_by_action = {
                action.action_id: task_ids[task_key]
                for action, task_key in zip(
                    (
                        item
                        for item in investigation.plan.actions
                        if item.tool_id
                        not in {
                            "monitor.query_range",
                            "loki.query_range",
                            "db.oracle.readonly_query",
                        }
                    ),
                    compiled.diagnostic_task_keys,
                    strict=True,
                )
            }
            for ordinal, action in enumerate(
                investigation.plan.actions,
                start=1,
            ):
                task_id, invocation_id = invocation_by_action.get(
                    action.action_id,
                    (None, None),
                )
                if action.tool_id == "monitor.query_range":
                    task_id = monitoring_task_id
                elif action.tool_id == "loki.query_range":
                    task_id = log_task_id
                elif action.tool_id == "db.oracle.readonly_query":
                    task_id = dynamic_task_by_action[action.action_id]
                else:
                    task_id = diagnostic_task_by_action[action.action_id]
                await uow.turns.add_tool_invocation(
                    OpsToolInvocationEntity(
                        tool_invocation_id=uuid7(),
                        turn_id=turn.turn_id,
                        revision_id=revision.revision_id,
                        playbook_invocation_id=invocation_id,
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task_id,
                        ordinal=ordinal,
                        action_id=action.action_id,
                        tool_id=action.tool_id,
                        tool_version="1.0.0",
                        tool_class=(
                            "PROMETHEUS"
                            if action.tool_id == "monitor.query_range"
                            else "LOKI"
                            if action.tool_id == "loki.query_range"
                            else "ORACLE_SQL_DYNAMIC"
                            if action.tool_id == "db.oracle.readonly_query"
                            else "ORACLE_SQL"
                        ),
                        status="PLANNED",
                        input_json=dict(action.input),
                        policy_hash=canonical_hash(
                            {
                                "capabilities": context.capabilities.model_dump(
                                    mode="json"
                                ),
                                "readonly": True,
                                "tool_id": action.tool_id,
                            }
                        ),
                    )
                )
            existing_tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id,
                lock=True,
            )
            answer = next(
                (
                    item
                    for item in existing_tasks
                    if item.task_key == "answer:compose"
                ),
                None,
            )
            if answer is None or answer.status != "PENDING":
                raise state_conflict("重规划缺少待执行回答Task")
            action_plan = next(
                (
                    item
                    for item in existing_tasks
                    if item.task_key == "change:action-plan"
                ),
                None,
            )
            if action_plan is not None:
                if action_plan.status != "PENDING":
                    raise state_conflict("重规划缺少待执行动作计划Task")
                action_plan.depends_on_json = [compiled.assessment_task_key]
                action_plan.input_artifacts_json = [
                    compiled.assessment_task_key
                ]
                answer.input_artifacts_json = [
                    compiled.assessment_task_key,
                    "change:proposal",
                ]
            else:
                answer.depends_on_json = [compiled.assessment_task_key]
                answer.input_artifacts_json = [
                    compiled.assessment_task_key
                ]
            old_execution = dict(
                dict(run.plan_snapshot_json or {}).get(
                    "skill_execution", {}
                )
            )
            execution_snapshot["invocations"] = {
                **dict(old_execution.get("invocations", {})),
                **dict(execution_snapshot.get("invocations", {})),
            }
            execution_snapshot["direct_invocations"] = {
                **dict(old_execution.get("direct_invocations", {})),
                **dict(execution_snapshot.get("direct_invocations", {})),
            }
            execution_snapshot["dynamic_invocations"] = {
                **dict(old_execution.get("dynamic_invocations", {})),
                **dict(execution_snapshot.get("dynamic_invocations", {})),
            }
            run.plan_snapshot_json = {
                **dict(run.plan_snapshot_json or {}),
                "skill_execution": execution_snapshot,
                "monitoring": monitoring_execution,
                "answer_context": {
                    "question": context.question,
                    "input_envelope": investigation.input_envelope.model_dump(
                        mode="json"
                    ),
                    "task_frame": investigation.task_frame.model_dump(
                        mode="json"
                    ),
                    "investigation_plan": investigation.plan.model_dump(
                        mode="json"
                    ),
                    "model": dict(model_snapshot),
                },
                "investigation_model_receipt": (
                    planning_receipt.model_dump(mode="json")
                ),
            }
            turn.task_frame_artifact_id = task_frame_artifact.artifact_id
            turn.current_plan_artifact_id = plan_artifact.artifact_id
            turn.current_plan_revision = revision_no
            turn.investigation_round = revision_no
            turn.tool_call_count = int(turn.tool_call_count or 0) + len(
                investigation.plan.actions
            )
            turn.status = "COLLECTING"
            await self._append_event(
                uow,
                turn,
                event_type="investigation.replanned",
                payload={
                    "revision_no": revision_no,
                    "action_count": len(investigation.plan.actions),
                    "public_summary": "已根据证据缺口调整调查计划，正在补充取证",
                },
            )
            await self._append_event(
                uow,
                turn,
                event_type="turn.status",
                payload={
                    "status": "COLLECTING",
                    "public_summary": "正在执行第二轮补充调查",
                },
            )
            await uow.commit()
            return {
                "turn_id": str(turn.turn_id),
                "ops_run_id": str(run.ops_run_id),
                "status": turn.status,
            }

    async def fail_terminal(
        self,
        payload: dict,
        *,
        error_code: str,
        error_message: str,
    ) -> dict:
        """把不可重试的规划错误收敛为用户可见终态。"""
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
                lock=True,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.status in {
                "COMPLETED",
                "PARTIAL",
                "FAILED",
                "CANCELLED",
            }:
                return {"turn_id": str(turn.turn_id), "status": turn.status}
            link = await uow.turns.get_run_link(
                turn_id=turn_id,
                purpose="PRIMARY",
            )
            run = (
                await uow.runs.get_run(
                    ops_run_id=link.ops_run_id,
                    lock=True,
                )
                if link is not None
                else None
            )
            now = datetime.now(UTC)
            public_summary = self._terminal_failure_summary(error_code)
            turn.status = "FAILED"
            turn.error_domain = "PLANNING"
            turn.error_code = error_code
            turn.error_message = public_summary
            turn.completed_at = now
            if run is not None and run.status not in {
                "COMPLETED",
                "PARTIAL",
                "FAILED",
                "CANCELLED",
                "EXPIRED",
            }:
                run.status = "FAILED"
                run.error_code = error_code
                run.error_message = error_message[:2000]
                run.completed_at = now
            await self._append_event(
                uow,
                turn,
                event_type="turn.status",
                payload={
                    "status": "FAILED",
                    "error_domain": "PLANNING",
                    "error_code": error_code,
                    "public_summary": public_summary,
                },
            )
            await uow.commit()
            return {"turn_id": str(turn.turn_id), "status": turn.status}

    async def fall_back_from_replan(
        self,
        payload: dict,
        *,
        error_code: str,
    ) -> dict:
        """重规划不可用时使用首轮真实证据回答，避免Turn永久卡住。"""
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        run_id = UUID(str(payload["ops_run_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
                lock=True,
            )
            run = await uow.runs.get_run(
                ops_run_id=run_id,
                lock=True,
            )
            if turn is None or run is None:
                raise resource_not_found("Turn Primary Run")
            if turn.status != "REPLANNING":
                return {"turn_id": str(turn.turn_id), "status": turn.status}
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id,
                lock=True,
            )
            assessment = next(
                (
                    item
                    for item in tasks
                    if item.output_artifact_id
                    == UUID(str(payload["assessment_artifact_id"]))
                ),
                None,
            )
            answer = next(
                (item for item in tasks if item.task_key == "answer:compose"),
                None,
            )
            if assessment is None or answer is None:
                raise state_conflict("重规划回退缺少评估或回答Task")
            action_plan = next(
                (
                    item
                    for item in tasks
                    if item.task_key == "change:action-plan"
                ),
                None,
            )
            ready_task = action_plan or answer
            if action_plan is not None:
                action_plan.depends_on_json = [assessment.task_key]
                action_plan.input_artifacts_json = [assessment.task_key]
                answer.input_artifacts_json = [
                    assessment.task_key,
                    "change:proposal",
                ]
            else:
                answer.depends_on_json = [assessment.task_key]
                answer.input_artifacts_json = [assessment.task_key]
            ready_task.status = "READY"
            ready_task.available_at = await uow.runs.database_now()
            turn.status = "ANSWERING"
            await self._append_event(
                uow,
                turn,
                event_type="investigation.replanned",
                payload={
                    "revision_no": int(payload["revision_no"]),
                    "action_count": 0,
                    "error_code": error_code,
                    "public_summary": "没有形成安全且有进展的补充计划，将依据现有证据回答",
                },
            )
            await self._append_event(
                uow,
                turn,
                event_type="turn.status",
                payload={
                    "status": "ANSWERING",
                    "public_summary": "正在依据已取得的证据形成回答",
                },
            )
            await uow.commit()
            return {"turn_id": str(turn.turn_id), "status": turn.status}

    @staticmethod
    def _terminal_failure_summary(error_code: str) -> str:
        if error_code == "AIOPS_INVESTIGATION_PLAN_INTERNAL_ERROR":
            return (
                "调查计划处理发生内部错误，本轮没有执行诊断工具。"
                "请重试；若仍然失败，请将错误编号提供给管理员。"
            )
        if error_code == "AIOPS_SKILL_UNAVAILABLE":
            return (
                "当前 Agent 暂时没有可用于本轮问题的诊断工具。"
                "请补充问题范围或检查 Agent 的 Target 与监控源配置。"
            )
        return (
            "本轮输入未能形成通过安全校验的调查计划。"
            "系统没有执行越界工具，请重试或补充问题范围。"
        )

    async def _prepare(
        self,
        payload: dict,
        *,
        revision_no: int = 1,
    ) -> TurnPlanningContext:
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if revision_no == 1 and turn.current_plan_artifact_id is not None:
                link = await uow.turns.get_run_link(
                    turn_id=turn_id,
                    purpose="PRIMARY",
                )
                raise _PlanningAlreadyApplied(
                    {
                        "turn_id": str(turn.turn_id),
                        "ops_run_id": (
                            str(link.ops_run_id) if link is not None else None
                        ),
                        "status": turn.status,
                    }
                )
            allowed_statuses = (
                {"UNDERSTANDING", "PLANNING"}
                if revision_no == 1
                else {"REPLANNING"}
            )
            if turn.status not in allowed_statuses:
                raise state_conflict(
                    f"Turn当前状态不能生成第{revision_no}版计划：{turn.status}"
                )
            link = await uow.turns.get_run_link(
                turn_id=turn_id,
                purpose="PRIMARY",
            )
            run = (
                await uow.runs.get_run(ops_run_id=link.ops_run_id)
                if link is not None
                else None
            )
            if run is None or run.status != "RUNNING":
                raise state_conflict("Turn Primary Run 不可规划")
            messages = await uow.turns.list_messages(turn_id=turn_id)
            user_message = next(
                (
                    row
                    for row in messages
                    if row.message_type == "USER_MESSAGE"
                ),
                None,
            )
            if user_message is None:
                raise state_conflict("Turn 缺少唯一用户问题")
            recent = await uow.turns.list_recent_conversation_messages(
                conversation_id=turn.conversation_id,
                before_sequence_no=int(user_message.sequence_no),
            )
            agent_version = await uow.agents.version(
                agent_id=run.agent_id,
                agent_version_id=run.agent_version_id,
            )
            target = await uow.targets.get_scoped(
                target_id=run.target_id,
                domain_id=domain_id,
            )
            if agent_version is None or target is None:
                raise resource_not_found("Turn 规划配置")
            policy = await uow.policies.get_scoped(
                policy_id=agent_version.policy_id,
                domain_id=domain_id,
            )
            policy_rules = dict(policy.rules_json or {}) if policy else {}
            readonly_allowed = bool(
                policy_rules.get("readonly_database_enabled", False)
            )
            target_enabled = str(target.status) == "ENABLED"
            target_reachable = str(target.connectivity_status) in {
                "CONNECTED",
                "DEGRADED",
            }
            access_gaps = []
            readonly_configured = bool(
                getattr(target, "readonly_connection_enabled", False)
            )
            if not readonly_configured:
                access_gaps.append(
                    {
                        "code": "DB_DIRECT_NOT_CONFIGURED",
                        "detail": "该逻辑 Target 采用仅监控模式，未配置数据库直连",
                        "retryable": False,
                    }
                )
            if not readonly_allowed:
                access_gaps.append(
                    {
                        "code": "DIAGNOSTIC_POLICY_DENIED",
                        "detail": "当前 Agent 策略禁止数据库直连诊断",
                        "retryable": False,
                    }
                )
            if not target_enabled:
                access_gaps.append(
                    {
                        "code": "TARGET_INACTIVE",
                        "detail": "Target 当前未启用，禁止数据库直连诊断",
                        "retryable": False,
                    }
                )
            if readonly_configured and not target_reachable:
                access_gaps.append(
                    {
                        "code": "TARGET_CONNECTIVITY_UNAVAILABLE",
                        "detail": "Target 当前不可连接，数据库直连取证可能失败",
                        "retryable": True,
                    }
                )
            if readonly_configured and target.diagnostic_credential_id is None:
                access_gaps.append(
                    {
                        "code": "DIAGNOSTIC_SECRET_MISSING",
                        "detail": "Target 未配置只读诊断凭据",
                        "retryable": False,
                    }
                )
            if readonly_configured and not target.endpoint_json:
                access_gaps.append(
                    {
                        "code": "TARGET_ENDPOINT_MISSING",
                        "detail": "Target 未配置数据库地址",
                        "retryable": False,
                    }
                )
            source_ids = await uow.agents.version_source_ids(
                agent_version_id=agent_version.agent_version_id
            )
            sources = []
            for source_id in source_ids:
                source = await uow.diagnostic_sources.get_scoped(
                    diagnostic_source_id=source_id,
                    domain_id=domain_id,
                )
                if source is not None:
                    sources.append(source)
            source_run_evidence = None
            source_run_id = dict(run.plan_snapshot_json or {}).get(
                "source_run_id"
            )
            if source_run_id:
                source_run = await uow.runs.get_run(
                    ops_run_id=UUID(str(source_run_id))
                )
                source_artifact = (
                    await uow.runs.get_artifact(
                        artifact_id=source_run.final_artifact_id
                    )
                    if source_run is not None
                    and source_run.final_artifact_id is not None
                    and int(source_run.domain_id) == domain_id
                    else None
                )
                if source_artifact is not None:
                    source_run_evidence = {
                        "source_run_id": str(source_run.ops_run_id),
                        "source_artifact_id": str(source_artifact.artifact_id),
                        "source_schema_version": source_artifact.schema_version,
                        "source_trust_level": source_artifact.trust_level,
                        "captured_at": source_artifact.created_at.isoformat(),
                        "payload": source_artifact.payload_json,
                    }
            return TurnPlanningContext(
                domain_id=domain_id,
                turn_id=turn_id,
                conversation_id=turn.conversation_id,
                ops_run_id=run.ops_run_id,
                agent_id=run.agent_id,
                target_id=target.target_id,
                source_ids=tuple(source_ids),
                actor_id=str(
                    getattr(turn, "created_by", None)
                    or getattr(user_message, "created_by", None)
                    or ""
                ),
                question=str(user_message.payload_json["text"]),
                content=tuple(user_message.payload_json["content"]),
                image_capabilities=dict(
                    getattr(agent_version, "image_capabilities_json", None)
                    or {}
                ),
                recent_context=tuple(
                    str(row.payload_json.get("text", ""))
                    for row in recent
                    if row.payload_json.get("text")
                ),
                trace_id=run.trace_id,
                deadline=run.deadline_at,
                capabilities=build_capability_snapshot(
                    agent_id=run.agent_id,
                    agent_version=agent_version,
                    target=target,
                    sources=sources,
                ),
                database_execution={
                    "domain_id": int(target.domain_id),
                    "target_row_version": int(target.row_version),
                    "db_type": str(target.db_type),
                    "configured_version": target.version_code,
                    "connection_profile": dict(target.endpoint_json or {}),
                    "diagnostic_credential_id": str(
                        target.diagnostic_credential_id
                    ),
                    "automatic_access_enabled": (
                        readonly_allowed
                        and readonly_configured
                        and target_enabled
                        and target.diagnostic_credential_id is not None
                        and bool(target.endpoint_json)
                    ),
                    "initial_gaps": access_gaps,
                },
                change_context={
                    "target": {
                        "target_id": str(target.target_id),
                        "row_version": int(target.row_version),
                        "db_type": str(target.db_type),
                        "version_code": target.version_code,
                        "environment": str(
                            getattr(target, "environment", "PROD")
                        ),
                        "status": str(
                            getattr(target, "status", "DISABLED")
                        ),
                        "connectivity_status": str(
                            target.connectivity_status
                        ),
                        "execution_secret_configured": bool(
                            getattr(target, "controlled_change_enabled", False)
                            and
                            getattr(
                                target, "execution_credential_id", None
                            )
                        ),
                        "security_level": int(
                            getattr(target, "security_level", 1)
                        ),
                        "capabilities": dict(
                            getattr(target, "capabilities_json", None) or {}
                        ),
                    },
                    "policy": {
                        "policy_id": (
                            str(getattr(policy, "policy_id"))
                            if policy is not None
                            and getattr(policy, "policy_id", None) is not None
                            else None
                        ),
                        "policy_hash": (
                            getattr(policy, "policy_hash", None)
                            if policy is not None
                            else None
                        ),
                        "rules": policy_rules,
                    },
                },
                source_run_evidence=source_run_evidence,
            )

    async def _prepare_monitoring(
        self,
        context: TurnPlanningContext,
    ) -> dict:
        if self._monitoring_snapshot_builder is None:
            return {}
        async with self._uow_factory() as uow:
            target = await uow.targets.get_scoped(
                target_id=context.target_id,
                domain_id=context.domain_id,
            )
            if target is None:
                raise resource_not_found("Turn Target")
            snapshot = await self._monitoring_snapshot_builder.build(
                uow=uow,
                domain_id=context.domain_id,
                target=target,
                now=await uow.runs.database_now(),
                allowed_source_ids=context.source_ids,
            )
            if not any(
                CAPABILITY_METRIC_QUERY_RANGE
                in item.get("effective_capabilities", ())
                for item in snapshot.get("bindings", ())
            ):
                snapshot["initial_gaps"].append(
                    {
                        "binding_id": "monitoring",
                        "source_id": "",
                        "code": "METRIC_SOURCE_UNAVAILABLE",
                        "detail": (
                            "当前 Agent 与 Target 没有可用的时序指标监控绑定"
                        ),
                    }
                )
            return snapshot

    async def _persist_raw_input(
        self, context: TurnPlanningContext
    ) -> TurnPlanningContext:
        """在任何模型调用前保存原始输入和上传文件定位。"""
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=context.domain_id,
                turn_id=context.turn_id,
                lock=True,
            )
            run = await uow.runs.get_run(
                ops_run_id=context.ops_run_id,
                lock=True,
            )
            if turn is None or run is None:
                raise resource_not_found("Turn Primary Run")
            input_artifact = await uow.runs.get_artifact_by_key(
                ops_run_id=run.ops_run_id,
                artifact_key="turn-user-input:1",
            )
            created = input_artifact is None
            if input_artifact is None:
                input_artifact = self._artifact(
                    ops_run_id=run.ops_run_id,
                    artifact_key="turn-user-input:1",
                    artifact_type="USER_PROVIDED_INPUT",
                    schema_version="USER_PROVIDED_INPUT.v1",
                    payload={
                        "text": context.question,
                        "content": list(context.content),
                        "received_at": datetime.now(UTC).isoformat(),
                    },
                    producer="aiops.turn-intake",
                    trust_level="USER_PROVIDED",
                )
                await uow.runs.add_artifact(input_artifact)

            upload_ids = []
            input_rows = {
                int(item.item_no): item
                for item in await uow.turns.list_input_items(
                    turn_id=turn.turn_id
                )
            }
            for upload in context.raw_uploads:
                artifact_key = f"turn-upload-source:{upload.item_no}"
                source_artifact = await uow.runs.get_artifact_by_key(
                    ops_run_id=run.ops_run_id,
                    artifact_key=artifact_key,
                )
                if source_artifact is None:
                    source_artifact = self._uri_artifact(
                        ops_run_id=run.ops_run_id,
                        artifact_key=artifact_key,
                        artifact_type="USER_UPLOAD_SOURCE",
                        schema_version="USER_UPLOAD_SOURCE.v1",
                        payload_uri=upload.payload_uri,
                        content_hash=upload.content_hash,
                        byte_size=upload.byte_size,
                        provenance={
                            "producer": "aiops.conversation-upload",
                            "upload_id": upload.upload_id,
                            "file_name": upload.file_name,
                            "media_type": upload.media_type,
                        },
                        trust_level="USER_PROVIDED",
                    )
                    await uow.runs.add_artifact(source_artifact)
                row = input_rows.get(upload.item_no)
                if row is not None:
                    row.source_artifact_id = source_artifact.artifact_id
                upload_ids.append(
                    (upload.item_no, source_artifact.artifact_id, None)
                )
            if created:
                await self._append_event(
                    uow,
                    turn,
                    event_type="input.analysis.started",
                    payload={
                        "content_count": len(context.content),
                        "upload_count": len(context.raw_uploads),
                        "public_summary": "输入材料已安全保存，正在识别内容",
                    },
                )
            await uow.commit()
            return replace(
                context,
                input_artifact_id=input_artifact.artifact_id,
                upload_artifact_ids=tuple(upload_ids),
            )

    async def _persist_input_extractions(
        self, context: TurnPlanningContext
    ) -> TurnPlanningContext:
        """在调查规划前保存文件解析结果，原始Artifact保持不变。"""
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=context.domain_id,
                turn_id=context.turn_id,
                lock=True,
            )
            run = await uow.runs.get_run(
                ops_run_id=context.ops_run_id,
                lock=True,
            )
            if turn is None or run is None:
                raise resource_not_found("Turn Primary Run")
            source_ids = {
                item_no: source_id
                for item_no, source_id, _ in context.upload_artifact_ids
            }
            input_rows = {
                int(item.item_no): item
                for item in await uow.turns.list_input_items(
                    turn_id=turn.turn_id
                )
            }
            persisted = []
            for upload in context.resolved_uploads:
                artifact_key = f"turn-upload-extract:{upload.item_no}"
                extracted = await uow.runs.get_artifact_by_key(
                    ops_run_id=run.ops_run_id,
                    artifact_key=artifact_key,
                )
                if extracted is None:
                    extracted = self._artifact(
                        ops_run_id=run.ops_run_id,
                        artifact_key=artifact_key,
                        artifact_type="USER_UPLOAD_EXTRACT",
                        schema_version="USER_UPLOAD_EXTRACT.v1",
                        payload={
                            "item_no": upload.item_no,
                            "file_name": upload.file_name,
                            "media_type": upload.media_type,
                            "text": upload.extracted_text,
                            "extraction_mode": upload.extraction_mode,
                            "model_id": (
                                str(upload.model_id)
                                if upload.model_id
                                else None
                            ),
                            "model_revision": upload.model_revision,
                            "extraction_error": upload.extraction_error,
                        },
                        producer="aiops.input-extraction",
                        trust_level="USER_PROVIDED",
                    )
                    await uow.runs.add_artifact(extracted)
                row = input_rows.get(upload.item_no)
                if row is not None:
                    row.extracted_artifact_id = extracted.artifact_id
                existing = await uow.turns.get_evidence_by_artifact(
                    turn_id=turn.turn_id,
                    artifact_id=extracted.artifact_id,
                )
                if existing is None:
                    await uow.turns.add_evidence(
                        OpsTurnEvidenceEntity(
                            turn_evidence_id=uuid7(),
                            turn_id=turn.turn_id,
                            artifact_id=extracted.artifact_id,
                            source_kind="USER",
                            evidence_kind=(
                                "SCREENSHOT"
                                if upload.media_type.startswith("image/")
                                else "USER_FILE"
                            ),
                            confidence=(
                                0.8
                                if upload.extraction_mode in {"OCR", "VLM"}
                                else 1
                            ),
                            extraction_artifact_id=extracted.artifact_id,
                            evidence_role="USER_PROVIDED",
                            measurement_semantics="NOT_APPLICABLE",
                            freshness_status="UNKNOWN",
                            usage_reason=(
                                f"用户上传的 {upload.file_name} 已作为本轮诊断材料"
                            ),
                            linked_by="aiops.input-extraction",
                        )
                    )
                persisted.append(
                    (
                        upload.item_no,
                        source_ids[upload.item_no],
                        extracted.artifact_id,
                    )
                )
            await uow.commit()
            return replace(context, upload_artifact_ids=tuple(persisted))

    async def _persist(
        self,
        *,
        context: TurnPlanningContext,
        investigation,
        planning_receipt,
        playbook_plan,
        compiled,
        execution_snapshot: dict,
        model_snapshot: dict,
        monitoring_requested: bool,
        monitoring_execution: dict,
    ) -> dict:
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=context.domain_id,
                turn_id=context.turn_id,
                lock=True,
            )
            run = await uow.runs.get_run(
                ops_run_id=context.ops_run_id,
                lock=True,
            )
            if turn is None or run is None:
                raise resource_not_found("Turn Primary Run")
            if turn.current_plan_artifact_id is not None:
                return {
                    "turn_id": str(turn.turn_id),
                    "ops_run_id": str(run.ops_run_id),
                    "status": turn.status,
                }
            if turn.status not in {"UNDERSTANDING", "PLANNING"} or run.status != "RUNNING":
                raise state_conflict("Turn 计划提交时状态已变化")

            contains_user_evidence = any(
                item.contains_user_evidence
                for item in investigation.input_envelope.materials
            )
            input_artifact = await uow.runs.get_artifact(
                artifact_id=context.input_artifact_id
            )
            if input_artifact is None:
                raise state_conflict("Turn 原始输入Artifact尚未持久化")
            input_analysis_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key="turn-input-analysis:1",
                artifact_type="TURN_INPUT_ENVELOPE",
                schema_version=investigation.input_envelope.schema_version,
                payload=investigation.input_envelope.model_dump(mode="json"),
                producer="aiops.input-understanding",
            )
            task_frame_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key="turn-task-frame:1",
                artifact_type="DBA_TASK_FRAME",
                schema_version=investigation.task_frame.schema_version,
                payload=investigation.task_frame.model_dump(mode="json"),
                producer="aiops.task-framer",
            )
            plan_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key="turn-investigation-plan:1",
                artifact_type="DBA_INVESTIGATION_PLAN",
                schema_version=investigation.plan.schema_version,
                payload=investigation.plan.model_dump(mode="json"),
                producer="aiops.investigation-planner",
            )
            playbook_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key="turn-playbook-plan:1",
                artifact_type="DBA_PLAYBOOK_PLAN",
                schema_version=playbook_plan.schema_version,
                payload=playbook_plan.model_dump(mode="json"),
                producer="aiops.playbook-selector",
            )
            source_run_artifact = (
                self._artifact(
                    ops_run_id=run.ops_run_id,
                    artifact_key="turn-source-run-evidence:1",
                    artifact_type="SOURCE_RUN_EVIDENCE",
                    schema_version="SOURCE_RUN_EVIDENCE.v1",
                    payload=context.source_run_evidence,
                    producer="aiops.source-run-linker",
                    trust_level=str(
                        context.source_run_evidence.get(
                            "source_trust_level", "MODEL_INFERENCE"
                        )
                    ),
                )
                if context.source_run_evidence is not None
                else None
            )
            for artifact in (
                input_analysis_artifact,
                task_frame_artifact,
                plan_artifact,
                playbook_artifact,
                source_run_artifact,
            ):
                if artifact is not None:
                    await uow.runs.add_artifact(artifact)

            revision = OpsInvestigationRevisionEntity(
                revision_id=uuid7(),
                turn_id=turn.turn_id,
                revision_no=1,
                revision_type="INITIAL",
                trigger_reason="完成用户输入理解并形成首轮调查计划",
                task_frame_artifact_id=task_frame_artifact.artifact_id,
                plan_artifact_id=plan_artifact.artifact_id,
                created_by="aiops.investigation-planner",
            )
            await uow.turns.add_investigation_revision(revision)
            input_rows = await uow.turns.list_input_items(
                turn_id=turn.turn_id
            )
            materials_by_no = {
                item.item_no: item
                for item in investigation.input_envelope.materials
            }
            for input_row in input_rows:
                material = materials_by_no.get(int(input_row.item_no))
                if material is None:
                    continue
                input_row.detected_kind = str(material.material_kind)
                input_row.detection_confidence = material.confidence
            if contains_user_evidence:
                existing = await uow.turns.get_evidence_by_artifact(
                    turn_id=turn.turn_id,
                    artifact_id=input_artifact.artifact_id,
                )
                if existing is None:
                    await uow.turns.add_evidence(OpsTurnEvidenceEntity(
                        turn_evidence_id=uuid7(),
                        turn_id=turn.turn_id,
                        artifact_id=input_artifact.artifact_id,
                        source_kind="USER",
                        evidence_kind="USER_PROVIDED",
                        confidence=1,
                        evidence_role="USER_PROVIDED",
                        measurement_semantics="NOT_APPLICABLE",
                        freshness_status="UNKNOWN",
                        usage_reason="用户在本轮对话中直接提供的日志、查询或命令结果",
                        linked_by="aiops.input-understanding",
                    ))
            if source_run_artifact is not None:
                await uow.turns.add_evidence(
                    OpsTurnEvidenceEntity(
                        turn_evidence_id=uuid7(),
                        turn_id=turn.turn_id,
                        artifact_id=source_run_artifact.artifact_id,
                        source_kind="RUN",
                        evidence_kind="INHERITED_DIAGNOSIS",
                        confidence=0.9,
                        evidence_role="CONTEXT",
                        measurement_semantics="NOT_APPLICABLE",
                        freshness_status="UNKNOWN",
                        usage_reason="从告警或巡检来源Run继承的已持久化诊断结果",
                        linked_by="aiops.source-run-linker",
                    )
                )

            task_ids = {
                task.task_key: uuid7() for task in compiled.tasks
            }
            task_specs = {
                task.task_key: task for task in compiled.tasks
            }
            tasks = [
                OpsTaskEntity(
                    ops_task_id=task_ids[spec.task_key],
                    ops_run_id=run.ops_run_id,
                    parent_task_id=(
                        task_ids[spec.depends_on[0]]
                        if spec.depends_on
                        else None
                    ),
                    task_key=spec.task_key,
                    task_type=spec.task_type,
                    handler_id=spec.handler_id,
                    handler_version=spec.handler_version,
                    input_schema_version=spec.input_schema_version,
                    output_schema_version=spec.output_schema_version,
                    depends_on_json=list(spec.depends_on),
                    input_artifacts_json=list(spec.input_artifact_keys),
                    status="READY" if not spec.depends_on else "PENDING",
                    priority=spec.priority,
                    max_attempts=spec.max_attempts,
                    timeout_seconds=spec.timeout_seconds,
                )
                for spec in compiled.tasks
            ]
            await uow.runs.add_tasks(tasks)
            playbook_invocation_ids: dict[int, UUID] = {}
            for item, task_key in zip(
                playbook_plan.items,
                compiled.invocation_task_keys,
                strict=True,
            ):
                playbook_invocation_id = uuid7()
                playbook_invocation_ids[item.ordinal] = playbook_invocation_id
                await uow.turns.add_playbook_invocation(
                    OpsPlaybookInvocationEntity(
                        playbook_invocation_id=playbook_invocation_id,
                        turn_id=turn.turn_id,
                        revision_id=revision.revision_id,
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task_ids[task_key],
                        ordinal=item.ordinal,
                        playbook_id=item.skill_id,
                        playbook_version=item.skill_version,
                        manifest_hash=item.manifest_hash,
                        status="PLANNED",
                        input_schema_version=task_specs[
                            task_key
                        ].input_schema_version,
                        input_json=dict(item.input),
                    )
                )

            playbook_context_by_action = {
                item.action_id: (
                    task_ids[task_key],
                    playbook_invocation_ids[item.ordinal],
                )
                for item, task_key in zip(
                    playbook_plan.items,
                    compiled.invocation_task_keys,
                    strict=True,
                )
                if item.action_id is not None
            }
            monitoring_task_id = (
                task_ids[compiled.monitoring_task_keys[0]]
                if compiled.monitoring_task_keys
                else None
            )
            log_task_id = (
                task_ids[compiled.log_task_keys[0]]
                if compiled.log_task_keys
                else None
            )
            dynamic_task_by_action = {
                action.action_id: task_ids[task_key]
                for action, task_key in zip(
                    (
                        item
                        for item in investigation.plan.actions
                        if item.tool_id == "db.oracle.readonly_query"
                    ),
                    compiled.dynamic_task_keys,
                    strict=True,
                )
            }
            for ordinal, action in enumerate(
                investigation.plan.actions, start=1
            ):
                playbook_context = playbook_context_by_action.get(
                    action.action_id
                )
                task_id = playbook_context[0] if playbook_context else None
                playbook_invocation_id = (
                    playbook_context[1] if playbook_context else None
                )
                if action.tool_id == "monitor.query_range":
                    task_id = monitoring_task_id
                elif action.tool_id == "loki.query_range":
                    task_id = log_task_id
                elif action.tool_id == "db.oracle.readonly_query":
                    task_id = dynamic_task_by_action[action.action_id]
                else:
                    task_id = next(
                        task_ids[task_key]
                        for candidate, task_key in zip(
                            (
                                item
                                for item in investigation.plan.actions
                                if item.tool_id
                                not in {
                                    "monitor.query_range",
                                    "loki.query_range",
                                    "db.oracle.readonly_query",
                                }
                            ),
                            compiled.diagnostic_task_keys,
                            strict=True,
                        )
                        if candidate.action_id == action.action_id
                    )
                await uow.turns.add_tool_invocation(
                    OpsToolInvocationEntity(
                        tool_invocation_id=uuid7(),
                        turn_id=turn.turn_id,
                        revision_id=revision.revision_id,
                        playbook_invocation_id=playbook_invocation_id,
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task_id,
                        ordinal=ordinal,
                        action_id=action.action_id,
                        tool_id=action.tool_id,
                        tool_version="1.0.0",
                        tool_class=(
                            "PROMETHEUS" if action.tool_id == "monitor.query_range"
                            else "LOKI" if action.tool_id == "loki.query_range"
                            else "ORACLE_SQL_DYNAMIC"
                            if action.tool_id == "db.oracle.readonly_query"
                            else "ORACLE_SQL"
                        ),
                        status="PLANNED",
                        input_json=dict(action.input),
                        policy_hash=canonical_hash(
                            {
                                "capabilities": context.capabilities.model_dump(
                                    mode="json"
                                ),
                                "readonly": True,
                                "tool_id": action.tool_id,
                            }
                        ),
                    )
                )

            turn.input_analysis_artifact_id = input_analysis_artifact.artifact_id
            turn.task_frame_artifact_id = task_frame_artifact.artifact_id
            turn.current_plan_artifact_id = plan_artifact.artifact_id
            turn.current_plan_revision = 1
            turn.investigation_round = 1
            turn.tool_call_count = len(investigation.plan.actions)
            turn.status = "COLLECTING"
            run.plan_snapshot_json = {
                **dict(run.plan_snapshot_json or {}),
                "capability_snapshot": context.capabilities.model_dump(
                    mode="json"
                ),
                "input_analysis_artifact_id": str(input_analysis_artifact.artifact_id),
                "task_frame_artifact_id": str(task_frame_artifact.artifact_id),
                "investigation_plan_artifact_id": str(plan_artifact.artifact_id),
                "playbook_plan_artifact_id": str(playbook_artifact.artifact_id),
                "playbook_catalog_hash": playbook_plan.catalog_hash,
                "investigation_model_receipt": planning_receipt.model_dump(mode="json"),
                "skill_execution": execution_snapshot,
                **(
                    {"monitoring": monitoring_execution}
                    if monitoring_requested
                    else {}
                ),
                "answer_context": {
                    "question": context.question,
                    "input_envelope": investigation.input_envelope.model_dump(mode="json"),
                    "task_frame": investigation.task_frame.model_dump(mode="json"),
                    "investigation_plan": investigation.plan.model_dump(
                        mode="json"
                    ),
                    "model": dict(model_snapshot),
                },
                "change_context": dict(context.change_context),
            }
            run.policy_snapshot_json = dict(
                context.change_context.get("policy", {})
            )
            await self._append_event(
                uow,
                turn,
                event_type="input.analysis.completed",
                payload={
                    "material_count": len(investigation.input_envelope.materials),
                    "contains_user_evidence": contains_user_evidence,
                    "public_summary": "已识别输入材料，正在形成调查任务",
                },
            )
            await self._append_event(
                uow,
                turn,
                event_type="task.frame.completed",
                payload={
                    "objectives": [
                        str(item)
                        for item in investigation.task_frame.objectives
                    ],
                    "public_summary": "已明确本轮问题、已知事实和待验证项",
                },
            )
            await self._append_event(
                uow,
                turn,
                event_type="investigation.planned",
                payload={
                    "revision_no": 1,
                    "action_count": len(investigation.plan.actions),
                    "playbook_count": len(playbook_plan.items),
                    "public_summary": "调查计划已建立，正在调用只读工具取证",
                },
            )
            await self._append_event(
                uow,
                turn,
                event_type="turn.status",
                payload={
                    "status": "COLLECTING",
                    "public_summary": "已理解输入材料，正在执行首轮调查",
                },
            )
            await uow.commit()
            return {
                "turn_id": str(turn.turn_id),
                "ops_run_id": str(run.ops_run_id),
                "status": turn.status,
            }

    @staticmethod
    def _artifact(
        *,
        ops_run_id: UUID,
        artifact_key: str,
        artifact_type: str,
        schema_version: str,
        payload: dict,
        producer: str,
        trust_level: str = "MODEL_INFERENCE",
    ) -> OpsArtifactEntity:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return OpsArtifactEntity(
            artifact_id=uuid7(),
            ops_run_id=ops_run_id,
            artifact_key=artifact_key,
            artifact_type=artifact_type,
            schema_version=schema_version,
            payload_json=payload,
            content_hash=canonical_hash(payload),
            byte_size=len(encoded),
            provenance_json={"producer": producer},
            trust_level=trust_level,
            security_level=1,
        )

    @staticmethod
    def _uri_artifact(
        *,
        ops_run_id: UUID,
        artifact_key: str,
        artifact_type: str,
        schema_version: str,
        payload_uri: str,
        content_hash: str,
        byte_size: int,
        provenance: dict,
        trust_level: str,
    ) -> OpsArtifactEntity:
        return OpsArtifactEntity(
            artifact_id=uuid7(),
            ops_run_id=ops_run_id,
            artifact_key=artifact_key,
            artifact_type=artifact_type,
            schema_version=schema_version,
            payload_uri=payload_uri,
            content_hash=content_hash,
            byte_size=byte_size,
            provenance_json=provenance,
            trust_level=trust_level,
            security_level=1,
        )

    @staticmethod
    async def _append_event(uow, turn, *, event_type: str, payload: dict) -> None:
        turn.event_cursor = int(turn.event_cursor) + 1
        await uow.turns.add_event(
            OpsTurnEventEntity(
                turn_id=turn.turn_id,
                sequence_no=turn.event_cursor,
                event_type=event_type,
                event_key=f"{event_type}:{turn.turn_id}:{turn.event_cursor}",
                visibility="USER",
                payload_json=payload,
            )
        )
