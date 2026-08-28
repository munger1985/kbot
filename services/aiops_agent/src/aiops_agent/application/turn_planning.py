"""Turn 的 Intent、能力快照和 Skill Plan 应用服务。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict
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
    SkillPlanItem,
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
    question: str
    content: tuple[dict, ...]
    recent_context: tuple[str, ...]
    trace_id: str
    deadline: datetime | None
    capabilities: DbaCapabilitySnapshot
    database_execution: dict


class _PlanningAlreadyApplied(Exception):
    def __init__(self, result: dict) -> None:
        super().__init__("Turn 计划已经持久化")
        self.result = result


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
    ) -> None:
        self._uow_factory = uow_factory
        self._investigation_reasoner = investigation_reasoner
        self._playbook_registry = playbook_registry
        self._skill_compiler = skill_compiler
        self._execution_snapshot_builder = execution_snapshot_builder
        self._agent_catalog = agent_catalog
        self._monitoring_snapshot_builder = monitoring_snapshot_builder

    async def execute(self, payload: dict) -> dict:
        try:
            context = await self._prepare(payload)
        except _PlanningAlreadyApplied as applied:
            return applied.result
        model_snapshot = await self._agent_catalog.resolve_diagnosis_model(
            agent_id=context.agent_id,
            domain_id=context.domain_id,
            trace_id=context.trace_id,
        )
        planned = await self._investigation_reasoner.plan(
            content=context.content,
            conversation_context=context.recent_context,
            available_tools=self._available_tools(context.capabilities),
            available_playbooks=self._available_playbooks(context.capabilities),
            model_snapshot=model_snapshot,
            deadline=context.deadline,
            idempotency_key=f"turn:{context.turn_id}:investigation:1",
        )
        investigation = planned.output
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
            user_evidence_artifact_keys=("turn-user-input:1",),
        )
        execution_snapshot = self._execution_snapshot_builder.build(
            plan=playbook_plan,
            compiled=compiled,
            capabilities=context.capabilities,
            database_execution=context.database_execution,
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
        tools: dict[tuple[str, str], dict] = {}
        for manifest in self._playbook_registry.manifests():
            if not self._manifest_applicable(manifest, capabilities):
                continue
            for step in manifest.tool_dag:
                tools[(step.tool_id, step.tool_version)] = {
                    "tool_id": step.tool_id,
                    "version": step.tool_version,
                    "tool_class": "ORACLE_SQL",
                    "description": f"受控只读数据库观测：{step.tool_id}",
                    "input": dict(step.input),
                }
        if CAPABILITY_METRIC_QUERY_RANGE in capabilities.available_source_capabilities:
            tools[("monitor.query_range", "1.0.0")] = {
                "tool_id": "monitor.query_range",
                "version": "1.0.0",
                "tool_class": "PROMETHEUS",
                "description": "查询绑定 Target 的 Prometheus 时间序列",
                "input": {"window": "RECENT"},
            }
        if CAPABILITY_LOG_QUERY in capabilities.available_source_capabilities:
            tools[("loki.query_range", "1.0.0")] = {
                "tool_id": "loki.query_range",
                "version": "1.0.0",
                "tool_class": "LOKI",
                "description": "查询绑定 Target 的 Oracle Alert Log",
                "input": {"window": "RECENT"},
            }
        return tuple(tools[key] for key in sorted(tools))

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
        """把模型选择的原子工具映射到可复用 Playbook；空计划同样合法。"""
        requested_tools = {action.tool_id for action in investigation.plan.actions}
        suggested = set(investigation.suggested_playbook_ids)
        candidates = []
        for manifest in self._playbook_registry.manifests():
            if not self._manifest_applicable(manifest, capabilities):
                continue
            candidates.append(manifest)
        selected = [
            manifest for manifest in candidates
            if manifest.skill_id in suggested
            and requested_tools.intersection(
                step.tool_id for step in manifest.tool_dag
            )
        ]
        covered = {
            step.tool_id for manifest in selected for step in manifest.tool_dag
        }
        remaining = requested_tools - covered - {
            "monitor.query_range", "loki.query_range"
        }
        while remaining:
            ranked = sorted(
                (
                    manifest for manifest in candidates
                    if manifest not in selected
                    and remaining.intersection(
                        step.tool_id for step in manifest.tool_dag
                    )
                ),
                key=lambda manifest: (
                    -len(
                        remaining.intersection(
                            step.tool_id for step in manifest.tool_dag
                        )
                    ),
                    manifest.limits.cost_units,
                    manifest.skill_id,
                ),
            )
            if not ranked:
                break
            chosen = ranked[0]
            selected.append(chosen)
            covered.update(step.tool_id for step in chosen.tool_dag)
            remaining -= covered
        items = []
        for ordinal, manifest in enumerate(selected, start=1):
            action_inputs = {
                key: value
                for action in investigation.plan.actions
                if action.tool_id in {step.tool_id for step in manifest.tool_dag}
                for key, value in action.input.items()
            }
            items.append(
                SkillPlanItem(
                    ordinal=ordinal,
                    skill_id=manifest.skill_id,
                    skill_version=manifest.version,
                    manifest_hash=self._playbook_registry.manifest_hash(
                        manifest.skill_id, manifest.version
                    ),
                    reason="调查计划选择了该 Playbook 中的受控工具",
                    evidence_question="这些观测能否验证或排除当前调查假设？",
                    measurement_semantics=manifest.measurement_semantics,
                    input={**dict(manifest.defaults), **action_inputs},
                )
            )
        return DbaSkillPlan(
            catalog_hash=self._playbook_registry.catalog_hash,
            items=tuple(items),
        )

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

    @staticmethod
    def _terminal_failure_summary(error_code: str) -> str:
        return (
            "本轮输入未能形成通过安全校验的调查计划。"
            "系统没有执行越界工具，请重试或补充问题范围。"
        )

    async def _prepare(self, payload: dict) -> TurnPlanningContext:
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.current_plan_artifact_id is not None:
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
            if turn.status not in {"UNDERSTANDING", "PLANNING"}:
                raise state_conflict(
                    f"只有 UNDERSTANDING/PLANNING Turn 可以生成计划，当前状态为 {turn.status}"
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
            return TurnPlanningContext(
                domain_id=domain_id,
                turn_id=turn_id,
                conversation_id=turn.conversation_id,
                ops_run_id=run.ops_run_id,
                agent_id=run.agent_id,
                target_id=target.target_id,
                source_ids=tuple(source_ids),
                question=str(user_message.payload_json["text"]),
                content=tuple(user_message.payload_json["content"]),
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
                },
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
            input_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key="turn-user-input:1",
                artifact_type="USER_PROVIDED_INPUT",
                schema_version="USER_PROVIDED_INPUT.v1",
                payload={
                    "text": context.question,
                    "content": list(context.content),
                    "contains_evidence": contains_user_evidence,
                    "received_at": datetime.now(UTC).isoformat(),
                },
                producer="aiops.input-understanding",
                trust_level="USER_PROVIDED",
            )
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
            for artifact in (
                input_artifact,
                input_analysis_artifact,
                task_frame_artifact,
                plan_artifact,
                playbook_artifact,
            ):
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
                await uow.turns.add_evidence(
                    OpsTurnEvidenceEntity(
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
            playbook_invocation_ids: dict[tuple[str, str], UUID] = {}
            for item, task_key in zip(
                playbook_plan.items,
                compiled.invocation_task_keys,
                strict=True,
            ):
                playbook_invocation_id = uuid7()
                playbook_invocation_ids[(item.skill_id, item.skill_version)] = (
                    playbook_invocation_id
                )
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

            playbook_context_by_tool = {
                step.tool_id: (
                    task_ids[task_key],
                    playbook_invocation_ids[
                        (item.skill_id, item.skill_version)
                    ],
                )
                for item, task_key in zip(
                    playbook_plan.items,
                    compiled.invocation_task_keys,
                    strict=True,
                )
                for step in self._playbook_registry.resolve(
                    item.skill_id, item.skill_version
                ).tool_dag
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
            for ordinal, action in enumerate(
                investigation.plan.actions, start=1
            ):
                playbook_context = playbook_context_by_tool.get(action.tool_id)
                task_id = playbook_context[0] if playbook_context else None
                playbook_invocation_id = (
                    playbook_context[1] if playbook_context else None
                )
                if action.tool_id == "monitor.query_range":
                    task_id = monitoring_task_id
                elif action.tool_id == "loki.query_range":
                    task_id = log_task_id
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
                    "model": dict(model_snapshot),
                },
            }
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
                    "objective": str(investigation.task_frame.objective),
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
