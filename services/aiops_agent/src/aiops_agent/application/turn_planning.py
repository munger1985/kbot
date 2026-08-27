"""Turn 的 Intent、能力快照和 Skill Plan 应用服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict
from aiops_agent.entities import (
    OpsArtifactEntity,
    OpsSkillInvocationEntity,
    OpsTaskEntity,
    OpsTurnEventEntity,
)
from aiops_agent.skills import (
    DbaIntentRouter,
    DbaSkillPlanner,
    SkillExecutionSnapshotBuilder,
    SkillPlanCompiler,
    build_capability_snapshot,
    canonical_hash,
)
from platform_core.contracts.aiops.skills import DbaCapabilitySnapshot
from platform_core.identity import uuid7


@dataclass(frozen=True, slots=True)
class TurnPlanningContext:
    domain_id: int
    turn_id: UUID
    conversation_id: UUID
    ops_run_id: UUID
    agent_id: UUID
    question: str
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
        intent_router: DbaIntentRouter,
        skill_planner: DbaSkillPlanner,
        skill_compiler: SkillPlanCompiler,
        execution_snapshot_builder: SkillExecutionSnapshotBuilder,
        agent_catalog,
    ) -> None:
        self._uow_factory = uow_factory
        self._intent_router = intent_router
        self._skill_planner = skill_planner
        self._skill_compiler = skill_compiler
        self._execution_snapshot_builder = execution_snapshot_builder
        self._agent_catalog = agent_catalog

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
        routed = await self._intent_router.route(
            question=context.question,
            conversation_context=context.recent_context,
            model_snapshot=model_snapshot,
            deadline=context.deadline,
            idempotency_key=f"turn:{context.turn_id}:intent:1",
        )
        intent_plan = routed.output
        skill_plan = self._skill_planner.plan(
            intent=intent_plan,
            capabilities=context.capabilities,
        )
        compiled = self._skill_compiler.compile(skill_plan)
        execution_snapshot = self._execution_snapshot_builder.build(
            plan=skill_plan,
            compiled=compiled,
            capabilities=context.capabilities,
            database_execution=context.database_execution,
        )
        return await self._persist(
            context=context,
            intent_plan=intent_plan,
            intent_receipt=routed.receipt,
            skill_plan=skill_plan,
            compiled=compiled,
            execution_snapshot=execution_snapshot,
            model_snapshot=model_snapshot,
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
            if turn.intent_plan_artifact_id is not None:
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
            if turn.status != "PLANNING":
                raise state_conflict(
                    f"只有 PLANNING Turn 可以生成计划，当前状态为 {turn.status}"
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
                question=str(user_message.payload_json["text"]),
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

    async def _persist(
        self,
        *,
        context: TurnPlanningContext,
        intent_plan,
        intent_receipt,
        skill_plan,
        compiled,
        execution_snapshot: dict,
        model_snapshot: dict,
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
            if turn.intent_plan_artifact_id is not None:
                return {
                    "turn_id": str(turn.turn_id),
                    "ops_run_id": str(run.ops_run_id),
                    "status": turn.status,
                }
            if turn.status != "PLANNING" or run.status != "RUNNING":
                raise state_conflict("Turn 计划提交时状态已变化")

            intent_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key="turn-intent-plan:1",
                artifact_type="DBA_INTENT_PLAN",
                schema_version=intent_plan.schema_version,
                payload=intent_plan.model_dump(mode="json"),
                producer="aiops.intent-router",
            )
            skill_artifact = self._artifact(
                ops_run_id=run.ops_run_id,
                artifact_key="turn-skill-plan:1",
                artifact_type="DBA_SKILL_PLAN",
                schema_version=skill_plan.schema_version,
                payload=skill_plan.model_dump(mode="json"),
                producer="aiops.skill-planner",
            )
            await uow.runs.add_artifact(intent_artifact)
            await uow.runs.add_artifact(skill_artifact)

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
            for item, task_key in zip(
                skill_plan.items,
                compiled.invocation_task_keys,
                strict=True,
            ):
                await uow.turns.add_skill_invocation(
                    OpsSkillInvocationEntity(
                        skill_invocation_id=uuid7(),
                        turn_id=turn.turn_id,
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task_ids[task_key],
                        ordinal=item.ordinal,
                        skill_id=item.skill_id,
                        skill_version=item.skill_version,
                        manifest_hash=item.manifest_hash,
                        primary_intent=intent_plan.primary_intent,
                        primary_domain=intent_plan.primary_domain,
                        status="PLANNED",
                        input_schema_version=task_specs[
                            task_key
                        ].input_schema_version,
                        input_json=dict(item.input),
                    )
                )

            turn.primary_intent = intent_plan.primary_intent
            turn.primary_domain = intent_plan.primary_domain
            turn.subject = intent_plan.subject
            turn.intent_schema_version = intent_plan.schema_version
            turn.intent_plan_json = intent_plan.model_dump(mode="json")
            turn.intent_plan_artifact_id = intent_artifact.artifact_id
            turn.skill_plan_schema_version = skill_plan.schema_version
            turn.skill_plan_json = skill_plan.model_dump(mode="json")
            turn.skill_plan_artifact_id = skill_artifact.artifact_id
            turn.status = "COLLECTING"
            run.plan_snapshot_json = {
                **dict(run.plan_snapshot_json or {}),
                "capability_snapshot": context.capabilities.model_dump(
                    mode="json"
                ),
                "intent_plan_artifact_id": str(intent_artifact.artifact_id),
                "skill_plan_artifact_id": str(skill_artifact.artifact_id),
                "skill_catalog_hash": skill_plan.catalog_hash,
                "intent_model_receipt": intent_receipt.model_dump(mode="json"),
                "skill_execution": execution_snapshot,
                "answer_context": {
                    "question": context.question,
                    "intent": intent_plan.model_dump(mode="json"),
                    "model": dict(model_snapshot),
                },
            }
            await self._append_event(
                uow,
                turn,
                event_type="intent.updated",
                payload={
                    "primary_intent": intent_plan.primary_intent,
                    "primary_domain": intent_plan.primary_domain,
                    "subject": intent_plan.subject,
                },
            )
            await self._append_event(
                uow,
                turn,
                event_type="skill.plan.created",
                payload={
                    "catalog_hash": skill_plan.catalog_hash,
                    "skill_count": len(skill_plan.items),
                },
            )
            await self._append_event(
                uow,
                turn,
                event_type="turn.status",
                payload={
                    "status": "COLLECTING",
                    "public_summary": "已理解问题，正在采集最小充分证据",
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
            trust_level="MODEL_INFERENCE",
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
