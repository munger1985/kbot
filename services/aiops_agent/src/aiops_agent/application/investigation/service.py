"""Turn 调查上下文冻结、计划编译和持久化应用服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import UUID

from loguru import logger

from aiops_agent.application.conversation_inputs import ConversationInputResolver
from aiops_agent.application.errors import (
    AIOpsSchemaNotReadyError,
    resource_not_found,
    state_conflict,
)
from aiops_agent.application.investigation.context import (
    PlanningAlreadyApplied,
    TurnPlanningContext,
)
from aiops_agent.application.investigation.discovery import (
    available_playbooks,
    available_tools,
    build_playbook_plan,
    compact_tool_cards,
    select_planning_candidates,
)
from aiops_agent.application.investigation.errors import TurnPlanningStageError
from aiops_agent.application.investigation.query_freezing import (
    prepare_dynamic_queries,
    prepare_source_queries,
)
from aiops_agent.application.investigation.projection import (
    safe_plan_projection,
    tool_class_for,
)
from aiops_agent.application.investigation.reasoner import (
    InvestigationPlanValidationError,
    InvestigationReasoner,
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
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.ports.diagnostic_source import CAPABILITY_METRIC_QUERY_RANGE
from aiops_agent.playbooks import PlaybookRegistry, canonical_hash
from aiops_agent.tools import (
    InvestigationTaskCompiler,
    ToolExecutionSnapshotBuilder,
    build_capability_snapshot,
)
from platform_core.contracts.aiops import (
    ActionIntent,
    CompactPlanningMode,
    InputMaterial,
    InvestigationAction,
    InvestigationPlan,
    InvestigationPlanningOutput,
    MaterialKind,
    MeasurementSemantics,
    TaskFrame,
    TaskObjective,
    TurnInputEnvelope,
)
from platform_core.identity import uuid7


@dataclass(frozen=True)
class _ReplanInputSnapshot:
    """冻结重规划所需的持久化数据，禁止把 ORM 实体带出 UoW。"""

    prior_plan: dict
    task_frame: dict
    assessment: dict
    prior_artifacts: tuple[tuple[str, str], ...]


class TurnPlanningService:
    """在外部模型调用两侧使用短事务冻结并持久化计划。"""

    def __init__(
        self,
        *,
        uow_factory,
        investigation_reasoner: InvestigationReasoner,
        playbook_registry: PlaybookRegistry,
        task_compiler: InvestigationTaskCompiler,
        tool_snapshot_builder: ToolExecutionSnapshotBuilder,
        agent_catalog,
        monitoring_snapshot_builder=None,
        conversation_input_resolver: ConversationInputResolver | None = None,
        schema_ready_check=None,
    ) -> None:
        self._uow_factory = uow_factory
        self._investigation_reasoner = investigation_reasoner
        self._playbook_registry = playbook_registry
        self._task_compiler = task_compiler
        self._tool_snapshot_builder = tool_snapshot_builder
        self._agent_catalog = agent_catalog
        self._monitoring_snapshot_builder = monitoring_snapshot_builder
        self._conversation_input_resolver = conversation_input_resolver
        self._schema_ready_check = schema_ready_check

    async def execute(self, payload: dict) -> dict:
        """执行首轮规划，并把未预期异常转换为可审计的阶段错误。"""
        try:
            await self._require_schema_ready()
            return await self._execute_once(payload)
        except (
            AIOpsSchemaNotReadyError,
            InvestigationPlanValidationError,
            TurnPlanningStageError,
        ):
            raise
        except Exception as exc:
            raise TurnPlanningStageError(exc) from exc

    async def _require_schema_ready(self) -> None:
        schema_ready_check = getattr(self, "_schema_ready_check", None)
        if schema_ready_check is None:
            return
        checks = await schema_ready_check()
        if not checks or any(status != "ok" for status in checks.values()):
            raise AIOpsSchemaNotReadyError(checks)

    async def _execute_once(self, payload: dict) -> dict:
        try:
            context = await self._prepare(payload)
        except PlanningAlreadyApplied as applied:
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
        planner_model_snapshot = await self._agent_catalog.resolve_planner_model(
            agent_id=context.agent_id,
            domain_id=context.domain_id,
            trace_id=context.trace_id,
        )
        diagnosis_model_snapshot = (
            await self._agent_catalog.resolve_diagnosis_model(
                agent_id=context.agent_id,
                domain_id=context.domain_id,
                trace_id=context.trace_id,
            )
        )
        discovered_tools = available_tools(
            self._tool_snapshot_builder, context.capabilities
        )
        discovered_playbooks = available_playbooks(
            self._playbook_registry, context.capabilities
        )
        planned, planning_tools, planning_playbooks, planning_route = (
            await self._plan_initial(
                context=context,
                available_tools=discovered_tools,
                available_playbooks=discovered_playbooks,
                planner_model_snapshot=planner_model_snapshot,
            )
        )
        planned, investigation, dynamic_queries, source_queries = (
            await self._prepare_queries_with_repair(
                context=context,
                planned=planned,
                available_tools=planning_tools,
                available_playbooks=planning_playbooks,
                model_snapshot=planner_model_snapshot,
                revision_no=1,
            )
        )
        playbook_plan = build_playbook_plan(self._playbook_registry)
        alert_diagnosis = bool(
            context.source_run_evidence
            and context.source_run_evidence.get("source_kind") == "SITUATION"
        )
        monitoring_requested = alert_diagnosis or any(
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
        compiled = self._task_compiler.compile(
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
            include_change=(
                investigation.task_frame.action_intent != ActionIntent.NONE
            ),
            investigation_actions=investigation.plan.actions,
        )
        execution_snapshot = self._tool_snapshot_builder.build(
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
            diagnosis_model_snapshot=diagnosis_model_snapshot,
            planner_model_snapshot=planner_model_snapshot,
            planning_route=planning_route,
            monitoring_requested=monitoring_requested,
            monitoring_execution=monitoring_execution,
        )

    async def _plan_initial(
        self,
        *,
        context: TurnPlanningContext,
        available_tools: tuple[dict, ...],
        available_playbooks: tuple[dict, ...],
        planner_model_snapshot: dict,
    ):
        """对纯文本问题先做精简语义规划，复杂材料直接进入完整规划。"""
        compact_question = self._compact_question(context)
        if compact_question is None:
            await self._record_planning_route(
                context=context,
                mode="FULL_INVESTIGATION",
                public_summary="输入包含诊断材料，将进行完整调查规划",
                public_sections=[
                    {
                        "title": "为什么进入完整调查",
                        "items": [
                            "需要同时理解用户材料、问题上下文和可用证据，不能直接套用单一查询"
                        ],
                    },
                    {
                        "title": "接下来会做什么",
                        "items": [
                            "识别已知事实与证据缺口",
                            "形成候选假设并选择能够区分假设的只读工具",
                        ],
                    },
                ],
            )
            planned = await self._investigation_reasoner.plan(
                content=context.content,
                conversation_context=context.recent_context,
                target_context=context.target_context,
                prompt_snapshot=context.prompt_snapshot,
                source_run_evidence=context.source_run_evidence,
                available_tools=available_tools,
                available_playbooks=available_playbooks,
                model_snapshot=planner_model_snapshot,
                deadline=context.deadline,
                idempotency_key=f"turn:{context.turn_id}:investigation:1",
            )
            return planned, available_tools, available_playbooks, {
                "mode": "FULL_INVESTIGATION",
                "public_summary": "输入包含诊断材料，正在进行完整调查规划",
            }

        routed = await self._investigation_reasoner.plan_compact(
            question=compact_question,
            conversation_context=context.recent_context,
            target_context=context.target_context,
            prompt_snapshot=context.prompt_snapshot,
            tool_cards=compact_tool_cards(available_tools),
            available_playbooks=available_playbooks,
            model_snapshot=planner_model_snapshot,
            deadline=context.deadline,
            idempotency_key=f"turn:{context.turn_id}:compact-planning:1",
        )
        compact = routed.output
        candidate_tool_ids = tuple(
            dict.fromkeys(
                (
                    *compact.selected_tool_ids,
                    *(action.tool_id for action in compact.actions),
                )
            )
        )
        selected_tools, selected_playbooks = select_planning_candidates(
            tools=available_tools,
            playbooks=available_playbooks,
            tool_ids=candidate_tool_ids,
            playbook_ids=compact.selected_playbook_ids,
        )
        compact_actions_missing = (
            compact.planning_mode
            in {
                CompactPlanningMode.READ_ONLY_LOOKUP,
                CompactPlanningMode.CONTROLLED_ACTION,
            }
            and not compact.actions
        )
        compact_route_mismatch = (
            (
                compact.planning_mode
                == CompactPlanningMode.CONTROLLED_ACTION
            )
            != (compact.action_intent != ActionIntent.NONE)
        )
        compact_route_incomplete = (
            compact_actions_missing or compact_route_mismatch
        )
        if compact_route_incomplete:
            # 精简模型已经完成语义选路，但可能因“该表”“按前述方案”等
            # 对话指代而无法安全生成参数。统一升级到携带完整上下文的 Planner；
            # 若精简模型连候选能力也未选出，则恢复完整能力集，避免二次空计划。
            if not selected_tools:
                selected_tools = available_tools
            if not selected_playbooks:
                selected_playbooks = available_playbooks
        selected_tools = self._include_identity_tool(
            selected_tools, available_tools
        )
        effective_mode = (
            CompactPlanningMode.FULL_INVESTIGATION
            if compact_route_incomplete
            else compact.planning_mode
        )
        public_summary = compact.public_reasoning_summary
        if compact_route_incomplete:
            public_summary = (
                "精简路由已识别任务方向，但尚未形成可执行的前置核验，"
                "正在结合对话上下文生成完整调查计划"
                if compact_actions_missing
                else "精简路由与动作意图不一致，正在由完整 Planner 重新确认用户诉求"
            )
        route_snapshot = {
            "mode": str(effective_mode),
            "public_summary": public_summary,
            "selected_tool_ids": [
                str(item["tool_id"]) for item in selected_tools
            ],
            "selected_playbook_ids": [
                str(item["playbook_id"]) for item in selected_playbooks
            ],
            "model_receipt": routed.receipt.model_dump(mode="json"),
        }
        if compact_route_incomplete:
            route_snapshot.update(
                {
                    "compact_mode": str(compact.planning_mode),
                    "fallback_reason": (
                        "COMPACT_ACTIONS_MISSING"
                        if compact_actions_missing
                        else "COMPACT_ACTION_INTENT_MISMATCH"
                    ),
                }
            )
        await self._record_planning_route(
            context=context,
            mode=str(effective_mode),
            public_summary=public_summary,
            public_sections=[
                {
                    "title": "问题判断",
                    "items": [compact.problem_statement],
                },
                {
                    "title": "规划选择",
                    "items": [public_summary],
                },
                {
                    "title": "完成标准",
                    "items": list(compact.success_criteria),
                },
            ],
        )
        if not compact_route_incomplete and compact.planning_mode in {
            CompactPlanningMode.READ_ONLY_LOOKUP,
            CompactPlanningMode.CONTROLLED_ACTION,
        }:
            output = self._compact_investigation_output(
                question=compact_question,
                compact=compact,
                target_context=context.target_context,
            )
            return (
                StructuredModelResult(output=output, receipt=routed.receipt),
                selected_tools,
                selected_playbooks,
                route_snapshot,
            )
        planned = await self._investigation_reasoner.plan(
            content=context.content,
            conversation_context=context.recent_context,
            target_context=context.target_context,
            prompt_snapshot=context.prompt_snapshot,
            source_run_evidence=context.source_run_evidence,
            available_tools=selected_tools,
            available_playbooks=selected_playbooks,
            model_snapshot=planner_model_snapshot,
            deadline=context.deadline,
            idempotency_key=f"turn:{context.turn_id}:investigation:1",
        )
        if (
            compact.action_intent != ActionIntent.NONE
            and planned.output.task_frame.action_intent
            == ActionIntent.NONE
        ):
            planned = StructuredModelResult(
                output=planned.output.model_copy(
                    update={
                        "task_frame": planned.output.task_frame.model_copy(
                            update={
                                "action_intent": compact.action_intent,
                                "requires_change": (
                                    compact.action_intent
                                    == ActionIntent.EXECUTE
                                ),
                            }
                        )
                    }
                ),
                receipt=planned.receipt,
            )
        return planned, selected_tools, selected_playbooks, route_snapshot

    async def _record_planning_route(
        self,
        *,
        context: TurnPlanningContext,
        mode: str,
        public_summary: str,
        public_sections: list[dict[str, object]],
    ) -> None:
        """提交模型可公开的规划决策摘要，不记录隐藏推理。"""
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=context.domain_id,
                turn_id=context.turn_id,
                lock=True,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            await self._append_event(
                uow,
                turn,
                event_type="planning.route.selected",
                payload={
                    "planning_mode": mode,
                    "public_summary": public_summary,
                    "public_sections": public_sections,
                },
            )
            await uow.commit()

    @staticmethod
    def _compact_question(context: TurnPlanningContext) -> str | None:
        """只按输入结构决定是否启用精简语义路由，不匹配领域关键词。"""
        if context.source_run_evidence is not None or context.resolved_uploads:
            return None
        if len(context.content) != 1:
            return None
        item = context.content[0]
        if str(item.get("content_type")) != "TEXT":
            return None
        question = str(item.get("text") or "").strip()
        if question != str(context.question or "").strip():
            return None
        return question if 0 < len(question) <= 4000 else None

    @staticmethod
    def _include_identity_tool(
        selected_tools: tuple[dict, ...],
        all_tools: tuple[dict, ...],
    ) -> tuple[dict, ...]:
        """数据库动作的实例身份前置工具由服务端补入，不占模型筛选名额。"""
        if not any(
            str(item.get("tool_id", "")).startswith("db.")
            for item in selected_tools
        ):
            return selected_tools
        identity = next(
            (
                item
                for item in all_tools
                if str(item.get("tool_id")) == "db.instance.identity"
            ),
            None,
        )
        if identity is None or any(
            str(item.get("tool_id")) == "db.instance.identity"
            for item in selected_tools
        ):
            return selected_tools
        return (identity, *selected_tools)

    @staticmethod
    def _compact_investigation_output(
        *,
        question: str,
        compact,
        target_context: dict,
    ) -> InvestigationPlanningOutput:
        """把精简查询或受控动作提升为统一调查契约。"""
        display_name = str(
            target_context.get("display_name")
            or target_context.get("target_id")
            or "当前 Target"
        )
        action_intent = compact.action_intent
        controlled_action = action_intent != ActionIntent.NONE
        return InvestigationPlanningOutput(
            input_envelope=TurnInputEnvelope(
                materials=(
                    InputMaterial(
                        item_no=1,
                        material_kind=MaterialKind.QUESTION,
                        summary=question[:2000],
                        key_facts=(f"用户已选择逻辑 Target：{display_name}",),
                        confidence=1,
                        contains_user_evidence=False,
                    ),
                ),
                explicit_question=question,
            ),
            task_frame=TaskFrame(
                objectives=(
                    (TaskObjective.CHANGE,)
                    if action_intent == ActionIntent.EXECUTE
                    else (TaskObjective.PLAN,)
                    if action_intent == ActionIntent.ADVISORY
                    else (TaskObjective.UNDERSTAND,)
                ),
                problem_statement=compact.problem_statement,
                database_context=dict(target_context),
                known_facts=(f"当前逻辑 Target 为 {display_name}",),
                unknowns=(),
                constraints=(
                    (
                        "仅自动执行当前 Target 的只读前置核验；"
                        + (
                            "只生成登记模板语句，不申请执行"
                            if action_intent == ActionIntent.ADVISORY
                            else "已登记动作必须等待人工审批"
                        )
                        if controlled_action
                        else "仅执行当前 Target 的只读诊断查询"
                    ),
                ),
                success_criteria=compact.success_criteria,
                action_intent=action_intent,
                requires_change=(action_intent == ActionIntent.EXECUTE),
            ),
            plan=InvestigationPlan(
                revision_no=1,
                actions=compact.actions,
            ),
            suggested_playbook_ids=compact.selected_playbook_ids,
        )

    async def execute_replan(self, payload: dict) -> dict:
        """根据上一轮Evidence Assessment生成并持久化下一轮调查DAG。"""
        revision_no = int(payload["revision_no"])
        if revision_no != 2:
            raise state_conflict("当前调查预算最多允许两轮")
        context = await self._prepare(payload, revision_no=revision_no)
        inputs = await self._load_replan_inputs(
            context=context,
            assessment_artifact_id=UUID(
                str(payload["assessment_artifact_id"])
            ),
        )
        planner_model_snapshot = await self._agent_catalog.resolve_planner_model(
            agent_id=context.agent_id,
            domain_id=context.domain_id,
            trace_id=context.trace_id,
        )
        diagnosis_model_snapshot = (
            await self._agent_catalog.resolve_diagnosis_model(
                agent_id=context.agent_id,
                domain_id=context.domain_id,
                trace_id=context.trace_id,
            )
        )
        discovered_tools = available_tools(
            self._tool_snapshot_builder, context.capabilities
        )
        discovered_playbooks = available_playbooks(
            self._playbook_registry, context.capabilities
        )
        planned = await self._investigation_reasoner.replan(
            content=context.content,
            conversation_context=context.recent_context,
            target_context=context.target_context,
            prompt_snapshot=context.prompt_snapshot,
            source_run_evidence=context.source_run_evidence,
            task_frame=inputs.task_frame,
            prior_plan=inputs.prior_plan,
            assessment=inputs.assessment,
            available_tools=discovered_tools,
            available_playbooks=discovered_playbooks,
            model_snapshot=planner_model_snapshot,
            deadline=context.deadline,
            idempotency_key=(
                f"turn:{context.turn_id}:investigation:{revision_no}"
            ),
            revision_no=revision_no,
        )
        planned, investigation, dynamic_queries, source_queries = (
            await self._prepare_queries_with_repair(
                context=context,
                planned=planned,
                available_tools=discovered_tools,
                available_playbooks=discovered_playbooks,
                model_snapshot=planner_model_snapshot,
                revision_no=revision_no,
            )
        )
        playbook_plan = build_playbook_plan(self._playbook_registry)
        alert_diagnosis = bool(
            context.source_run_evidence
            and context.source_run_evidence.get("source_kind") == "SITUATION"
        )
        monitoring_requested = alert_diagnosis or any(
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
            artifact_key
            for artifact_key, schema_version in inputs.prior_artifacts
            if schema_version
            in {
                "USER_PROVIDED_INPUT.v1",
                "SOURCE_RUN_EVIDENCE.v1",
                "SITUATION_EVIDENCE.v1",
                "DBA_TOOL_RESULT.v1",
                "OBSERVATION_SET.v1",
                "LOG_EVIDENCE_SET.v1",
            }
        )
        compiled = self._task_compiler.compile(
            playbook_plan,
            monitoring_binding_ids=monitoring_binding_ids,
            log_binding_ids=log_binding_ids,
            user_evidence_artifact_keys=evidence_keys,
            revision_no=revision_no,
            include_answer=False,
            investigation_actions=investigation.plan.actions,
        )
        execution_snapshot = self._tool_snapshot_builder.build(
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
            diagnosis_model_snapshot=diagnosis_model_snapshot,
            planner_model_snapshot=planner_model_snapshot,
            monitoring_execution=monitoring_execution,
        )

    async def _load_replan_inputs(
        self,
        *,
        context: TurnPlanningContext,
        assessment_artifact_id: UUID,
    ) -> _ReplanInputSnapshot:
        """在同一只读 UoW 内验证并冻结重规划输入。"""
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=context.domain_id,
                turn_id=context.turn_id,
            )
            assessment_artifact = await uow.runs.get_artifact(
                artifact_id=assessment_artifact_id
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
                or assessment_artifact.schema_version
                != "DBA_SUFFICIENCY.v1"
                or plan_artifact is None
                or task_frame_artifact is None
            ):
                raise state_conflict("重规划缺少上一轮评估或调查计划")
            return _ReplanInputSnapshot(
                prior_plan=dict(plan_artifact.payload_json or {}),
                task_frame=dict(task_frame_artifact.payload_json or {}),
                assessment=dict(assessment_artifact.payload_json or {}),
                prior_artifacts=tuple(
                    (str(item.artifact_key), str(item.schema_version))
                    for item in prior_artifacts
                ),
            )

    async def _prepare_queries_with_repair(
        self,
        *,
        context: TurnPlanningContext,
        planned,
        available_tools: tuple[dict, ...],
        available_playbooks: tuple[dict, ...],
        model_snapshot: dict,
        revision_no: int,
    ):
        """Tool 输入首稿越界时，带策略反馈执行一次受控修正。"""
        planned = StructuredModelResult(
            output=self._bind_target_to_plan(
                investigation=planned.output,
                target_context=context.target_context,
                available_tools=available_tools,
            ),
            receipt=planned.receipt,
        )
        try:
            investigation, dynamic_queries, source_queries = (
                self._prepare_query_inputs(
                    investigation=planned.output,
                    context=context,
                )
            )
            return planned, investigation, dynamic_queries, source_queries
        except InvestigationPlanValidationError as exc:
            logger.warning(
                "AIOps Tool 输入未通过策略，正在请求模型修正："
                "turn_id={} revision_no={} reason={}",
                context.turn_id,
                revision_no,
                str(exc),
            )
            repaired = (
                await self._investigation_reasoner.repair_policy_invalid_plan(
                    content=context.content,
                    conversation_context=context.recent_context,
                    target_context=context.target_context,
                    prompt_snapshot=context.prompt_snapshot,
                    source_run_evidence=context.source_run_evidence,
                    invalid_output=planned.output,
                    validation_error=str(exc),
                    available_tools=available_tools,
                    available_playbooks=available_playbooks,
                    model_snapshot=model_snapshot,
                    deadline=context.deadline,
                    idempotency_key=(
                        f"turn:{context.turn_id}:investigation:"
                        f"{revision_no}:policy-repair"
                    ),
                )
            )
            repaired = StructuredModelResult(
                output=self._bind_target_to_plan(
                    investigation=repaired.output,
                    target_context=context.target_context,
                    available_tools=available_tools,
                ),
                receipt=repaired.receipt,
            )
            try:
                investigation, dynamic_queries, source_queries = (
                    self._prepare_query_inputs(
                        investigation=repaired.output,
                        context=context,
                    )
                )
            except InvestigationPlanValidationError:
                investigation, dynamic_queries, source_queries = (
                    self._prepare_valid_query_subset(
                        investigation=repaired.output,
                        context=context,
                        revision_no=revision_no,
                    )
                )
            return repaired, investigation, dynamic_queries, source_queries

    def _prepare_valid_query_subset(
        self,
        *,
        investigation: InvestigationPlanningOutput,
        context: TurnPlanningContext,
        revision_no: int,
    ):
        """修正计划仍越界时，保留独立且合规的只读调查动作。"""
        normalized_actions: dict[str, InvestigationAction] = {}
        rejected_reasons: dict[str, str] = {}
        source_actions = tuple(investigation.plan.actions)

        for action in source_actions:
            isolated_action = action.model_copy(update={"depends_on": ()})
            isolated_plan = investigation.plan.model_copy(
                update={"actions": (isolated_action,)}
            )
            isolated = investigation.model_copy(
                update={"plan": isolated_plan}
            )
            try:
                prepared, _, _ = self._prepare_query_inputs(
                    investigation=isolated,
                    context=context,
                )
            except InvestigationPlanValidationError as exc:
                rejected_reasons[action.action_id] = str(exc)
                continue
            normalized_actions[action.action_id] = (
                prepared.plan.actions[0].model_copy(
                    update={"depends_on": action.depends_on}
                )
            )

        for tool_id in ("monitor.query_range", "loki.query_range"):
            action_ids = [
                action.action_id
                for action in source_actions
                if action.tool_id == tool_id
                and action.action_id in normalized_actions
            ]
            for action_id in action_ids[4:]:
                normalized_actions.pop(action_id, None)
                rejected_reasons[action_id] = (
                    "同类临时监控查询超过单轮 4 条限制"
                )

        rejected_ids = set(rejected_reasons)
        changed = True
        while changed:
            changed = False
            for action in source_actions:
                if action.action_id in rejected_ids:
                    continue
                unavailable = sorted(set(action.depends_on) & rejected_ids)
                if not unavailable:
                    continue
                rejected_ids.add(action.action_id)
                normalized_actions.pop(action.action_id, None)
                rejected_reasons[action.action_id] = (
                    "依赖的调查动作不可执行：" + ", ".join(unavailable)
                )
                changed = True

        retained_actions = tuple(
            normalized_actions[action.action_id]
            for action in source_actions
            if action.action_id in normalized_actions
            and action.action_id not in rejected_ids
        )
        if not retained_actions:
            details = "；".join(
                f"{action_id}={reason}"
                for action_id, reason in rejected_reasons.items()
            )
            raise InvestigationPlanValidationError(
                "修正后的调查计划没有可执行动作"
                + (f"：{details}" if details else "")
            )

        logger.warning(
            "AIOps 修正计划仍有越界动作，已保留独立合规动作："
            "turn_id={} revision_no={} retained={} rejected={}",
            context.turn_id,
            revision_no,
            [action.action_id for action in retained_actions],
            rejected_reasons,
        )
        retained_plan = investigation.plan.model_copy(
            update={"actions": retained_actions}
        )
        retained = investigation.model_copy(update={"plan": retained_plan})
        return self._prepare_query_inputs(
            investigation=retained,
            context=context,
        )

    @staticmethod
    def _bind_target_to_plan(
        *,
        investigation: InvestigationPlanningOutput,
        target_context: dict[str, object],
        available_tools: tuple[dict, ...],
    ) -> InvestigationPlanningOutput:
        """冻结目标语义，并为数据库调查补充可审计的实例身份前置步骤。"""
        task_frame = investigation.task_frame.model_copy(
            update={"database_context": dict(target_context)}
        )
        if task_frame.action_intent == ActionIntent.ADVISORY:
            return investigation.model_copy(update={"task_frame": task_frame})
        actions = list(investigation.plan.actions)
        database_actions = [
            action for action in actions if action.tool_id.startswith("db.")
        ]
        identity_available = any(
            str(tool.get("tool_id")) == "db.instance.identity"
            for tool in available_tools
        )
        if not database_actions or not identity_available:
            return investigation.model_copy(update={"task_frame": task_frame})

        identity_actions = [
            action
            for action in actions
            if action.tool_id == "db.instance.identity"
        ]
        retained = [
            action
            for action in actions
            if action.tool_id != "db.instance.identity"
        ]
        display_name = str(
            target_context.get("display_name")
            or target_context.get("target_id")
            or "当前 Target"
        )
        database_type = str(target_context.get("db_type") or "数据库")
        identity_question = (
            f"核验已绑定 Target“{display_name}”当前实际连接的"
            f"{database_type}实例、数据库容器、版本及启动上下文。"
        )
        identity = (
            identity_actions[0].model_copy(
                update={
                    "question": identity_question,
                    "input": {},
                    "depends_on": (),
                    "optional": False,
                }
            )
            if identity_actions
            else InvestigationAction(
                action_id="a1",
                question=identity_question,
                tool_id="db.instance.identity",
                input={},
                expected_evidence_kind="DATABASE_IDENTITY",
                measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
                depends_on=(),
                optional=False,
            )
        )
        ordered = [identity, *retained]
        action_id_map = {
            action.action_id: f"a{index}"
            for index, action in enumerate(ordered, start=1)
        }
        action_id_map.update(
            {action.action_id: "a1" for action in identity_actions}
        )
        normalized = []
        for index, action in enumerate(ordered, start=1):
            dependencies = tuple(
                dict.fromkeys(
                    action_id_map[value]
                    for value in action.depends_on
                    if value in action_id_map
                )
            )
            if action.tool_id.startswith("db.") and index != 1:
                dependencies = tuple(dict.fromkeys(("a1", *dependencies)))
            normalized.append(
                action.model_copy(
                    update={
                        "action_id": f"a{index}",
                        "depends_on": dependencies,
                    }
                )
            )
        plan = investigation.plan.model_copy(
            update={"actions": tuple(normalized)}
        )
        return investigation.model_copy(
            update={"task_frame": task_frame, "plan": plan}
        )

    def _prepare_query_inputs(
        self,
        *,
        investigation,
        context: TurnPlanningContext,
    ):
        """在任务编译前冻结所有模型生成的 Tool 输入。"""
        investigation, dynamic_queries = prepare_dynamic_queries(
            investigation
        )
        investigation, source_queries = prepare_source_queries(investigation)
        direct_actions = tuple(
            action
            for action in investigation.plan.actions
            if action.tool_id
            not in {
                "monitor.query_range",
                "loki.query_range",
                "db.oracle.readonly_query",
            }
        )
        if not direct_actions:
            return investigation, dynamic_queries, source_queries
        try:
            normalized = self._tool_snapshot_builder.validate_direct_actions(
                actions=direct_actions,
                capabilities=context.capabilities,
            )
        except ValueError as exc:
            raise InvestigationPlanValidationError(
                f"固定诊断工具输入未通过目录约束：{exc}"
            ) from exc
        actions = tuple(
            action.model_copy(update={"input": normalized[action.action_id]})
            if action.action_id in normalized
            else action
            for action in investigation.plan.actions
        )
        plan = investigation.plan.model_copy(update={"actions": actions})
        return (
            investigation.model_copy(update={"plan": plan}),
            dynamic_queries,
            source_queries,
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
        diagnosis_model_snapshot: dict,
        planner_model_snapshot: dict,
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
                revision_type="EVIDENCE_DRIVEN",
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
                        playbook_id=item.playbook_id,
                        playbook_version=item.playbook_version,
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
                        tool_class=tool_class_for(action.tool_id),
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
                    "investigation_execution", {}
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
                "investigation_execution": execution_snapshot,
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
                    "model": dict(diagnosis_model_snapshot),
                    "planner_model": dict(planner_model_snapshot),
                    "prompts": dict(context.prompt_snapshot),
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
                    "plan": safe_plan_projection(
                        investigation.plan.model_dump(mode="json"),
                        execution_snapshot=execution_snapshot,
                    ),
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
        if error_code == "AIOPS_INVESTIGATION_PLAN_INVALID":
            return (
                "本轮输入未能形成通过安全校验的调查计划。"
                "系统没有执行越界工具，请重试或补充问题范围。"
            )
        if error_code == "AIOPS_SCHEMA_INTEGRITY_ERROR":
            return (
                "AIOps数据库结构与当前服务合同不一致，本轮没有执行诊断工具。"
                "请管理员检查Schema版本和完整性后重试。"
            )
        if error_code == "AIOPS_INVESTIGATION_CATALOG_CHANGED":
            return (
                "调查能力目录在规划期间发生变化，本轮没有执行诊断工具。"
                "请刷新后重试。"
            )
        if error_code == "AIOPS_INVESTIGATION_PLAN_INTERNAL_ERROR":
            return (
                "调查计划处理发生内部错误，本轮没有执行诊断工具。"
                "请重试；若仍然失败，请将错误编号提供给管理员。"
            )
        return (
            "调查计划处理发生内部错误，本轮没有执行诊断工具。"
            "请重试；若仍然失败，请将错误编号提供给管理员。"
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
                raise PlanningAlreadyApplied(
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
            target_action_policies = await uow.agents.version_target_policies(
                agent_version_id=agent_version.agent_version_id
            )
            controlled_action_execution = dict(
                target_action_policies.get(target.target_id, {})
            )
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
            source_situation_id = dict(run.plan_snapshot_json or {}).get(
                "source_situation_id"
            )
            if source_situation_id and source_run_evidence is None:
                situation = await uow.situations.get_situation_scoped(
                    situation_id=UUID(str(source_situation_id)),
                    domain_id=domain_id,
                )
                if situation is not None and situation.target_id == target.target_id:
                    signal_events = await uow.situations.list_events_for_situation(
                        situation_id=situation.situation_id,
                        limit=32,
                    )
                    source_run_evidence = {
                        "source_kind": "SITUATION",
                        "source_situation_id": str(situation.situation_id),
                        "source_schema_version": "SITUATION_EVIDENCE.v1",
                        "inheritance_schema_version": "SITUATION_EVIDENCE.v1",
                        "source_artifact_type": "SITUATION_EVIDENCE",
                        "source_trust_level": "SOURCE_VERIFIED",
                        "captured_at": situation.last_observed_at.isoformat(),
                        "payload": {
                            "title": situation.title,
                            "summary": situation.summary,
                            "status": situation.status,
                            "severity": situation.severity,
                            "first_observed_at": (
                                situation.first_observed_at.isoformat()
                            ),
                            "last_observed_at": (
                                situation.last_observed_at.isoformat()
                            ),
                            "signal_events": [
                                {
                                    "signal_event_id": str(item.signal_event_id),
                                    "diagnostic_source_id": str(
                                        item.diagnostic_source_id
                                    ),
                                    "source_event_key": item.source_event_key,
                                    "event_class": item.event_class,
                                    "severity": item.severity,
                                    "normalized_status": item.normalized_status,
                                    "summary": item.summary,
                                    "occurred_at": item.occurred_at.isoformat(),
                                    "evidence_locator": dict(
                                        item.evidence_locator_json or {}
                                    ),
                                    "provider_attributes": dict(
                                        dict(item.payload_json or {}).get(
                                            "provider_attributes"
                                        )
                                        or {}
                                    ),
                                }
                                for item in signal_events
                            ],
                        },
                    }
            existing_prompt_snapshot = dict(
                dict(
                    dict(run.plan_snapshot_json or {}).get(
                        "answer_context", {}
                    )
                ).get("prompts", {})
            )
            prompt_snapshot = (
                await self._investigation_reasoner.freeze_prompts(
                    existing_prompt_snapshot or None
                )
            )
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
                target_context={
                    "target_id": str(target.target_id),
                    "display_name": str(target.display_name),
                    "db_type": str(target.db_type),
                    "configured_version": target.version_code,
                    "environment": str(target.environment),
                    "db_role": str(target.db_role),
                    "status": str(target.status),
                    "connectivity_status": str(target.connectivity_status),
                    "selection_status": "BOUND",
                },
                prompt_snapshot=prompt_snapshot,
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
                    "controlled_action_execution": (
                        controlled_action_execution
                    ),
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
                        "public_sections": [
                            {
                                "title": "当前目标",
                                "items": [
                                    (
                                        f"{context.target_context['display_name']} · "
                                        f"{context.target_context['db_type']} · "
                                        f"{context.target_context['environment']}"
                                    )
                                ],
                            },
                            {
                                "title": "当前动作",
                                "items": [
                                    f"核对 {len(context.content)} 项输入内容"
                                    + (
                                        f"和 {len(context.raw_uploads)} 个上传文件"
                                        if context.raw_uploads
                                        else ""
                                    ),
                                    "冻结本轮 Prompt、Target 能力和只读执行边界",
                                    "下一步将判断直接查询还是进入完整调查",
                                ],
                            },
                        ],
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
        diagnosis_model_snapshot: dict,
        planner_model_snapshot: dict,
        planning_route: dict,
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
                    artifact_type=str(
                        context.source_run_evidence.get(
                            "source_artifact_type", "SOURCE_RUN_EVIDENCE"
                        )
                    ),
                    schema_version=str(
                        context.source_run_evidence.get(
                            "inheritance_schema_version",
                            "SOURCE_RUN_EVIDENCE.v1",
                        )
                    ),
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
                inherited_from_situation = (
                    context.source_run_evidence.get("source_kind")
                    == "SITUATION"
                )
                await uow.turns.add_evidence(
                    OpsTurnEvidenceEntity(
                        turn_evidence_id=uuid7(),
                        turn_id=turn.turn_id,
                        artifact_id=source_run_artifact.artifact_id,
                        source_kind=(
                            "SITUATION" if inherited_from_situation else "RUN"
                        ),
                        evidence_kind=(
                            "MONITORING_SIGNAL"
                            if inherited_from_situation
                            else "INHERITED_DIAGNOSIS"
                        ),
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
                        playbook_id=item.playbook_id,
                        playbook_version=item.playbook_version,
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
                        tool_class=tool_class_for(action.tool_id),
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
                "planning_route": dict(planning_route),
                "investigation_execution": execution_snapshot,
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
                    "model": dict(diagnosis_model_snapshot),
                    "planner_model": dict(planner_model_snapshot),
                    "prompts": dict(context.prompt_snapshot),
                },
                "change_context": dict(context.change_context),
            }
            run.policy_snapshot_json = dict(
                context.change_context.get("policy", {})
            )
            public_plan = safe_plan_projection(
                investigation.plan.model_dump(mode="json"),
                task_frame=investigation.task_frame.model_dump(mode="json"),
                execution_snapshot=execution_snapshot,
            )
            await self._append_event(
                uow,
                turn,
                event_type="input.analysis.completed",
                payload={
                    "material_count": len(investigation.input_envelope.materials),
                    "contains_user_evidence": contains_user_evidence,
                    "materials": [
                        {
                            "item_no": item.item_no,
                            "material_kind": str(item.material_kind),
                            "summary": item.summary,
                            "confidence": item.confidence,
                            "contains_user_evidence": (
                                item.contains_user_evidence
                            ),
                        }
                        for item in investigation.input_envelope.materials
                    ],
                    "explicit_question": (
                        investigation.input_envelope.explicit_question
                    ),
                    "inferred_question": (
                        investigation.input_envelope.inferred_question
                    ),
                    "ambiguities": list(
                        investigation.input_envelope.ambiguities
                    ),
                    "public_sections": [
                        {
                            "title": "识别到的材料",
                            "items": [
                                f"{item.material_kind}: {item.summary}"
                                for item in investigation.input_envelope.materials
                            ],
                        },
                        {
                            "title": "问题理解",
                            "items": [
                                str(
                                    investigation.input_envelope.explicit_question
                                    or investigation.input_envelope.inferred_question
                                    or investigation.task_frame.problem_statement
                                )
                            ],
                        },
                    ],
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
                    "task_frame": public_plan["task_frame"],
                    "public_sections": [
                        {
                            "title": "本轮要解决的问题",
                            "items": [
                                investigation.task_frame.problem_statement
                            ],
                        },
                        {
                            "title": "当前已知",
                            "items": list(
                                investigation.task_frame.known_facts
                            )
                            or ["尚无经过验证的事实，需先取证"],
                        },
                        {
                            "title": "待验证",
                            "items": list(investigation.task_frame.unknowns)
                            or ["没有额外待验证项"],
                        },
                        {
                            "title": "完成标准",
                            "items": list(
                                investigation.task_frame.success_criteria
                            ),
                        },
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
                    "planning_mode": planning_route.get("mode"),
                    "plan": public_plan,
                    "public_sections": [
                        {
                            "title": "规划方式",
                            "items": [
                                str(
                                    planning_route.get("public_summary")
                                    or "已生成最小充分调查计划"
                                )
                            ],
                        },
                        {
                            "title": "准备执行的步骤",
                            "items": [
                                str(item.get("question") or "执行诊断步骤")
                                for item in public_plan["actions"]
                            ]
                            or ["现有材料已经足够，不需要调用诊断工具"],
                        },
                    ],
                    "public_summary": str(
                        planning_route.get("public_summary")
                        or "调查计划已建立，正在调用只读工具取证"
                    ),
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
