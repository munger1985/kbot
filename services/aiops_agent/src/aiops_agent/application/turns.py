"""专业 DBA Conversation 与 Turn 的命令和查询用例。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from aiops_agent.application.inspections.check_catalog import (
    compile_selected_check_steps,
    inspection_template_steps,
)
from aiops_agent.application.conversation_starters import (
    ConversationStarterCatalog,
)
from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.configuration.projections import _target_fact_view
from aiops_agent.application.errors import (
    AIOpsApplicationError,
    resource_not_found,
    state_conflict,
    validation_failed,
)
from aiops_agent.application.display_dictionary import dictionary_for_payload
from aiops_agent.application.sql_selection import resolve_sql_selection
from aiops_agent.application.targets.facts import create_confirmed_fact
from aiops_agent.application.investigation.projection import (
    safe_plan_projection,
)
from aiops_agent.application.implementation.pdf import (
    render_implementation_runbook_pdf,
)
from aiops_agent.application.implementation.markdown import (
    render_implementation_runbook_markdown,
)
from aiops_agent.application.implementation.artifacts import (
    render_runbook_artifact_zip,
)
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.entities import (
    OpsConversationEntity,
    OpsConversationMessageEntity,
    OpsConversationTurnEntity,
    OpsTurnInputItemEntity,
    OpsTurnEventEntity,
    OutboxEntity,
)
from platform_core.contracts.aiops import (
    ClarificationResponseV1,
    ConversationCreate,
    ConversationSourceContext,
    ConversationSourceType,
    TrustedContextHint,
    TargetFactConfirmCommand,
    TargetFactView,
    InputContent,
    TurnCreate,
)
from platform_core.contracts.aiops.types import WorkflowKind
from platform_core.identity import uuid7


class ConversationTurnService:
    """在单一 UoW 内接收一轮用户问题并可靠投递规划任务。"""

    def __init__(
        self,
        *,
        uow_factory,
        upload_store=None,
        conversation_starter_catalog=None,
        diagnosis_config=None,
    ):
        self._uow_factory = uow_factory
        self._upload_store = upload_store
        self._turn_budget_seconds = int(getattr(diagnosis_config, "interactive_turn_budget_seconds", 90))
        self._sql_selection_ttl = int(getattr(diagnosis_config, "sql_selection_ttl_seconds", 86400))
        self._conversation_starter_catalog = (
            conversation_starter_catalog or ConversationStarterCatalog()
        )

    async def start(
        self,
        *,
        domain_id: int,
        actor_id: str,
        trace_id: str,
        conversation_create: ConversationCreate,
        first_turn: TurnCreate,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            receipt = await self._start_in_uow(
                uow=uow,
                domain_id=domain_id,
                actor_id=actor_id,
                trace_id=trace_id,
                conversation_create=conversation_create,
                first_turn=first_turn,
            )
            await uow.commit()
            return receipt

    async def start_alert_diagnosis(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """把已选中 Agent 的告警转换为标准智能运维 Turn。"""
        domain_id = int(payload["domain_id"])
        agent_id = UUID(str(payload["agent_id"]))
        target_id = UUID(str(payload["target_id"]))
        situation_id = UUID(str(payload["situation_id"]))
        signal_event_id = UUID(str(payload["signal_event_id"]))
        actor_id = "system:signal-intake"
        trace_id = str(payload["trace_id"])
        turn_idempotency_key = f"alert-signal:{signal_event_id}"
        async with self._uow_factory() as uow:
            situation = await uow.situations.get_situation_scoped(
                situation_id=situation_id,
                domain_id=domain_id,
            )
            if situation is None:
                raise self._error(
                    "AIOPS_SOURCE_SITUATION_INVALID",
                    "来源 Situation 不存在",
                )
            task_text = (
                f"请基于告警情境“{situation.title}”及其关联监控信号"
                "自动开始诊断。请核验告警、规划并执行必要的只读取证，"
                "分析可能原因、影响和处置建议，并给出有证据支持的结论。"
            )
            command = TurnCreate(
                content=(
                    {
                        "content_type": "TEXT",
                        "text": task_text,
                    },
                ),
                idempotency_key=turn_idempotency_key,
            )
            execution_context = {
                "trigger_type": "ALERT",
                "interaction_mode": "AUTONOMOUS",
                "workflow_kind": WorkflowKind.ALERT_DIAGNOSIS.value,
                "trigger_signal_event_id": str(signal_event_id),
            }
            existing_conversation = (
                await uow.conversations.get_auto_situation_conversation(
                    domain_id=domain_id,
                    situation_id=situation_id,
                    agent_id=agent_id,
                    actor_id=actor_id,
                    lock=True,
                )
            )
            if existing_conversation is not None:
                existing_turn = await uow.turns.get_by_idempotency(
                    conversation_id=existing_conversation.conversation_id,
                    idempotency_key=turn_idempotency_key,
                )
                if existing_turn is not None:
                    return self._receipt(existing_turn)
                version = await uow.agents.version(
                    agent_id=existing_conversation.agent_id,
                    agent_version_id=existing_conversation.agent_version_id,
                )
                if version is None:
                    raise resource_not_found("Agent Version")
                receipt = await self._create_turn(
                    uow=uow,
                    conversation=existing_conversation,
                    version=version,
                    command=command,
                    actor_id=actor_id,
                    trace_id=trace_id,
                    source_run=None,
                    execution_context=execution_context,
                )
                await uow.commit()
                return receipt

            receipt = await self._start_in_uow(
                uow=uow,
                domain_id=domain_id,
                actor_id=actor_id,
                trace_id=trace_id,
                conversation_create=ConversationCreate(
                    agent_id=agent_id,
                    title=f"{situation.title} · 自动诊断"[:256],
                    context_hint=await self._context_hint_for_target(
                        uow=uow,
                        domain_id=domain_id,
                        target_id=target_id,
                        source="SITUATION",
                        source_resource_id=situation_id,
                    ),
                    source=ConversationSourceContext(
                        source_type=ConversationSourceType.SITUATION,
                        situation_id=situation_id,
                    ),
                ),
                first_turn=command,
                execution_context=execution_context,
            )
            await uow.commit()
            return receipt

    async def _start_in_uow(
        self,
        *,
        uow,
        domain_id: int,
        actor_id: str,
        trace_id: str,
        conversation_create: ConversationCreate,
        first_turn: TurnCreate,
        execution_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """复用人工与自动入口的 Conversation/Turn 创建合同。"""
        agent = await uow.agents.get(
            domain_id=domain_id,
            agent_id=conversation_create.agent_id,
        )
        if (
            agent is None
            or agent.status != "ACTIVE"
            or agent.current_version_id is None
        ):
            raise resource_not_found("Active AIOps Agent")
        version = await uow.agents.version(
            agent_id=agent.agent_id,
            agent_version_id=agent.current_version_id,
        )
        if version is None:
            raise resource_not_found("Agent Version")
        await self._validate_context_hint(
            uow=uow,
            domain_id=domain_id,
            agent_version_id=version.agent_version_id,
            context_hint=conversation_create.context_hint,
        )
        source = conversation_create.source
        source_run = None
        focused_target_id = await self._focused_target_from_context_hint(
            uow=uow,
            context_hint=conversation_create.context_hint,
        )
        if source.source_type == ConversationSourceType.RUN:
            source_run = await uow.runs.get_run_scoped(
                ops_run_id=source.run_id,
                domain_id=domain_id,
            )
            if source_run is None or source_run.final_artifact_id is None:
                raise self._error(
                    "AIOPS_SOURCE_RUN_INVALID",
                    "来源 Run 必须存在且已经产生可用结果",
                )
            if not await uow.agents.version_has_target(
                agent_version_id=version.agent_version_id,
                target_id=source_run.target_id,
            ):
                raise self._error(
                    "AIOPS_SOURCE_SCOPE_DENIED",
                    "来源 Run 不属于当前 Agent 获授权的业务系统",
                )
        elif source.source_type == ConversationSourceType.SITUATION:
            source_situation = await uow.situations.get_situation_scoped(
                situation_id=source.situation_id,
                domain_id=domain_id,
            )
            if source_situation is None:
                raise self._error(
                    "AIOPS_SOURCE_SITUATION_INVALID",
                    "来源 Situation 不存在",
                )
            if not await uow.agents.version_has_target(
                agent_version_id=version.agent_version_id,
                target_id=source_situation.target_id,
            ):
                raise self._error(
                    "AIOPS_SOURCE_SCOPE_DENIED",
                    "来源 Situation 不属于当前 Agent 获授权的业务系统",
                )
        conversation = OpsConversationEntity(
            domain_id=domain_id,
            agent_id=agent.agent_id,
            agent_version_id=version.agent_version_id,
            context_hint_json=(
                conversation_create.context_hint.model_dump(mode="json")
                if conversation_create.context_hint is not None
                else None
            ),
            title=conversation_create.title,
            status="ACTIVE",
            source_type=str(source.source_type),
            source_situation_id=source.situation_id,
            source_run_id=source.run_id,
            source_report_id=source.report_id,
            source_inspection_fire_id=source.inspection_fire_id,
            last_turn_no=0,
            last_message_no=0,
            created_by=actor_id,
            updated_by=actor_id,
        )
        await uow.conversations.add_conversation(conversation)
        return await self._create_turn(
            uow=uow,
            conversation=conversation,
            version=version,
            command=first_turn,
            actor_id=actor_id,
            trace_id=trace_id,
            source_run=source_run,
            focused_target_id=focused_target_id,
            execution_context=execution_context,
        )

    async def start_scheduled_inspection(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """把一次定时触发转换为标准 Agent Conversation Turn。"""
        domain_id = int(payload["domain_id"])
        agent_id = UUID(str(payload["agent_id"]))
        fire_id = UUID(str(payload["inspection_fire_id"]))
        actor_id = str(payload["actor_id"])
        trace_id = str(payload["trace_id"])
        try:
            template_definition = dict(
                payload.get("inspection_template_definition") or {}
            )
            selected_check_ids = tuple(
                template_definition.get("selected_check_ids") or ()
            )
        except ValueError as exc:
            raise self._error(
                "AIOPS_INSPECTION_CHECK_SELECTION_INVALID",
                str(exc),
            ) from exc
        async with self._uow_factory() as uow:
            fire = await uow.inspections.get_fire(
                inspection_fire_id=fire_id,
                lock=True,
            )
            if fire is None:
                raise resource_not_found("Inspection Fire")
            existing = await uow.conversations.list_for_inspection_fire(
                inspection_fire_id=fire_id
            )
            if existing:
                return {
                    "inspection_fire_id": str(fire_id),
                    "conversation_count": len(existing),
                    "agent_version_id": dict(fire.plan_snapshot_json).get(
                        "agent_version_id"
                    ),
                }
            agent = await uow.agents.get(
                domain_id=domain_id,
                agent_id=agent_id,
            )
            if (
                agent is None
                or agent.status != "ACTIVE"
                or agent.current_version_id is None
            ):
                raise resource_not_found("Active AIOps Agent")
            version = await uow.agents.version(
                agent_id=agent.agent_id,
                agent_version_id=agent.current_version_id,
            )
            if version is None:
                raise resource_not_found("Agent Version")
            target_ids = await uow.agents.active_version_target_ids(
                domain_id=domain_id,
                agent_version_id=version.agent_version_id,
            )
            if not target_ids:
                raise self._error(
                    "AIOPS_INSPECTION_AGENT_TARGET_REQUIRED",
                    "巡检触发时 Agent 当前版本没有已启用的 Target",
                )
            period_start = datetime.fromisoformat(str(payload["period_start"]))
            period_end = datetime.fromisoformat(str(payload["period_end"]))
            schedule_type = str(payload["schedule_type"])
            target_work: list[tuple[Any, tuple[dict[str, Any], ...]]] = []
            uncovered: list[str] = []
            for target_id in target_ids:
                target = await self._require_existing_target(
                    uow=uow,
                    domain_id=domain_id,
                    target_id=target_id,
                )
                evidence_steps = inspection_template_steps(
                    template_definition,
                    schedule_type=schedule_type,
                    database_type=str(target.db_type),
                )
                if not evidence_steps:
                    uncovered.append(
                        f"{target.display_name}（{target.db_type}）"
                    )
                    continue
                target_work.append((target, evidence_steps))
            if uncovered:
                raise self._error(
                    "AIOPS_INSPECTION_TARGET_COVERAGE_REQUIRED",
                    "巡检模板没有覆盖 Agent 当前访问范围内的 Target："
                    + "、".join(uncovered),
                )
            task_text = (
                "请对你负责的数据库执行本期健康巡检并生成巡检报告。"
                f"观测窗口为 [{period_start.isoformat()}, {period_end.isoformat()})，"
                f"巡检周期类型为 {schedule_type}。"
                "本次巡检将执行已勾选检查项对应的固定只读取证步骤，"
                "并基于实际证据评估健康风险、结论和建议。"
            )
            deadline_at = datetime.now(UTC) + timedelta(
                seconds=int(payload["timeout_seconds"])
            )
            receipts: list[dict[str, Any]] = []
            for target, evidence_steps in target_work:
                target_id = target.target_id
                recovery_repository = getattr(uow, "recovery", None)
                business_system_repository = getattr(uow, "business_systems", None)
                resource = (
                    await business_system_repository.get_resource_for_target(
                        target_id=target_id,
                        domain_id=domain_id,
                    )
                    if business_system_repository is not None
                    else None
                )
                recovery_profile = (
                    await recovery_repository.get_active_profile(
                        resource_id=resource.resource_id,
                        domain_id=domain_id,
                    )
                    if recovery_repository is not None and resource is not None
                    else None
                )
                backup_policy = None
                if recovery_profile is not None:
                    backup_policy = {
                        "enabled": bool(recovery_profile.backup_policy_enabled),
                        "frequency_hours": int(recovery_profile.backup_frequency_hours),
                        "success_max_age_hours": int(recovery_profile.backup_success_max_age_hours),
                        "retention_days": int(recovery_profile.backup_retention_days),
                        "source_types": list(recovery_profile.required_backup_source_types_json),
                        "profile_id": str(recovery_profile.recovery_profile_id),
                        "profile_version": int(recovery_profile.version_no),
                    }
                if schedule_type == "DAILY" and backup_policy and backup_policy["enabled"]:
                    backup_check_id = {
                        "ORACLE": "oracle.backup.recent_jobs",
                        "POSTGRESQL": "postgresql.backup.archive_chain",
                        "MYSQL": "mysql.backup.binlog_chain",
                    }.get(str(target.db_type))
                    existing_tool_ids = {str(item.get("tool_id") or "") for item in evidence_steps}
                    if backup_check_id:
                        backup_steps = compile_selected_check_steps(
                            (backup_check_id,),
                            schedule_type=schedule_type,
                            database_type=str(target.db_type),
                        )
                        evidence_steps = tuple(evidence_steps) + tuple(
                            item for item in backup_steps
                            if str(item.get("tool_id") or "") not in existing_tool_ids
                        )
                    if str(target.db_type) == "ORACLE":
                        query_days = min(
                            90,
                            max(
                                7,
                                (int(backup_policy["success_max_age_hours"]) + 23)
                                // 24
                                + 1,
                            ),
                        )
                        evidence_steps = tuple(
                            {
                                **item,
                                "input": {
                                    **dict(item.get("input") or {}),
                                    "days": query_days,
                                },
                            }
                            if str(item.get("tool_id") or "") == "db.backup.recent_jobs"
                            else item
                            for item in evidence_steps
                        )
                conversation = OpsConversationEntity(
                    domain_id=domain_id,
                    agent_id=agent.agent_id,
                    agent_version_id=version.agent_version_id,
                    context_hint_json=(
                        TrustedContextHint(
                            source_kind="INSPECTION",
                            resource_ids=(resource.resource_id,),
                            source_resource_id=fire_id,
                        ).model_dump(mode="json")
                        if resource is not None
                        else None
                    ),
                    title=f"{payload['plan_display_name']} · 定期巡检",
                    status="ACTIVE",
                    source_type="INSPECTION",
                    source_inspection_fire_id=fire_id,
                    last_turn_no=0,
                    last_message_no=0,
                    created_by=actor_id,
                    updated_by=actor_id,
                )
                await uow.conversations.add_conversation(conversation)
                receipts.append(
                    await self._create_turn(
                        uow=uow,
                        conversation=conversation,
                        version=version,
                        command=TurnCreate(
                            content=(
                                {
                                    "content_type": "TEXT",
                                    "text": task_text,
                                },
                            ),
                            idempotency_key=(
                                f"inspection:{fire_id}:target:{target_id}"
                            ),
                        ),
                        actor_id=actor_id,
                        trace_id=trace_id,
                        source_run=None,
                        focused_target_id=target_id,
                        execution_context={
                            "trigger_type": "SCHEDULE",
                            "interaction_mode": "AUTONOMOUS",
                            "workflow_kind": WorkflowKind.INSPECTION.value,
                            "inspection_fire_id": str(fire_id),
                            "deadline_at": deadline_at.isoformat(),
                            "observation_start": period_start.isoformat(),
                            "observation_end": period_end.isoformat(),
                            "inspection": {
                                "template_id": payload[
                                    "inspection_template_id"
                                ],
                                "template_version": str(payload[
                                    "inspection_template_version"
                                ]),
                                "template_version_id": payload[
                                    "inspection_template_version_id"
                                ],
                                "template_name": payload[
                                    "inspection_template_name"
                                ],
                                "template_hash": payload[
                                    "inspection_template_hash"
                                ],
                                "schedule_resolver_version": payload[
                                    "schedule_resolver_version"
                                ],
                                "schedule_type": schedule_type,
                                "database_type": str(target.db_type),
                                "timezone": payload["timezone"],
                                "period_start": period_start.isoformat(),
                                "period_end": period_end.isoformat(),
                                "selected_check_ids": list(
                                    selected_check_ids
                                ),
                                "evidence_steps": evidence_steps,
                                "backup_policy": backup_policy,
                            },
                        },
                    )
                )
            snapshot = dict(fire.plan_snapshot_json)
            snapshot["agent_version_id"] = str(version.agent_version_id)
            fire.plan_snapshot_json = snapshot
            fire.target_count = len(receipts)
            fire.updated_at = datetime.now(UTC)
            await uow.commit()
            return {
                "inspection_fire_id": str(fire_id),
                "conversation_count": len(receipts),
                "agent_version_id": str(version.agent_version_id),
            }

    async def create_turn(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        actor_id: str,
        trace_id: str,
        command: TurnCreate,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
                lock=True,
            )
            if conversation is None or conversation.created_by != actor_id:
                raise resource_not_found("Conversation")
            if conversation.status == "ARCHIVED":
                raise state_conflict("归档后的 Conversation 不能继续提问")
            await self._require_conversation_runnable(
                uow=uow,
                domain_id=domain_id,
                conversation=conversation,
            )
            existing = await uow.turns.get_by_idempotency(
                conversation_id=conversation_id,
                idempotency_key=command.idempotency_key,
            )
            if existing is not None:
                messages = await uow.turns.list_messages(turn_id=existing.turn_id)
                original = next((item for item in messages if item.message_type == "USER_MESSAGE"), None)
                if original is None or original.payload_json.get("request_sha256") != self._request_sha256(command):
                    raise self._error("AIOPS_IDEMPOTENCY_CONFLICT", "同一幂等键不能提交不同内容或 SQL 选择")
                return self._receipt(existing)
            version = await uow.agents.version(
                agent_id=conversation.agent_id,
                agent_version_id=conversation.agent_version_id,
            )
            if version is None:
                raise resource_not_found("Agent Version")
            source_run = None
            if command.source_run_id is not None:
                source_run = await uow.runs.get_run_scoped(
                    ops_run_id=command.source_run_id,
                    domain_id=domain_id,
                )
                if source_run is None or source_run.final_artifact_id is None:
                    raise self._error(
                        "AIOPS_SOURCE_RUN_INVALID",
                        "来源 Run 必须存在且已经产生可用结果",
                    )
            context_hint = (
                TrustedContextHint.model_validate(conversation.context_hint_json)
                if conversation.context_hint_json
                else None
            )
            receipt = await self._create_turn(
                uow=uow,
                conversation=conversation,
                version=version,
                command=command,
                actor_id=actor_id,
                trace_id=trace_id,
                source_run=source_run,
                focused_target_id=await self._focused_target_from_context_hint(
                    uow=uow,
                    context_hint=context_hint,
                ),
            )
            await uow.commit()
            return receipt

    async def list_conversations(
        self,
        *,
        domain_id: int,
        actor_id: str,
        agent_id: UUID | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await uow.conversations.list_conversations(
                domain_id=domain_id,
                created_by=actor_id,
                agent_id=agent_id,
                limit=limit,
            )
            return [self._conversation_summary(row) for row in rows]

    async def get_conversation(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        actor_id: str,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            row = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
            )
            if (
                row is None
                or row.created_by != actor_id
                or row.status == "ARCHIVED"
            ):
                raise resource_not_found("Conversation")
            return self._conversation_summary(row)

    async def archive_conversation(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        actor_id: str,
    ) -> dict[str, Any]:
        """从用户历史中移除会话，同时保留诊断和执行审计事实。"""
        async with self._uow_factory() as uow:
            row = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
                lock=True,
            )
            if row is None or row.created_by != actor_id:
                raise resource_not_found("Conversation")
            if row.status == "ARCHIVED":
                return self._conversation_summary(row)
            row.status = "ARCHIVED"
            row.updated_by = actor_id
            row.updated_at = datetime.now(UTC)
            await uow.commit()
            return self._conversation_summary(row)

    async def list_turns(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        actor_id: str,
        after_turn_no: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
            )
            if conversation is None or conversation.created_by != actor_id:
                raise resource_not_found("Conversation")
            turns = await uow.turns.list_turns(
                conversation_id=conversation_id,
                after_turn_no=after_turn_no,
                limit=limit,
            )
            return [self._turn_summary(row) for row in turns]

    async def get_turn(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
            )
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
            )
            if (
                conversation is None
                or conversation.created_by != actor_id
                or turn is None
                or turn.conversation_id != conversation_id
            ):
                raise resource_not_found("Conversation Turn")
            messages = await uow.turns.list_messages(turn_id=turn_id)
            blocks = await uow.turns.list_answer_blocks(turn_id=turn_id)
            citations = await uow.turns.list_answer_citations(
                answer_block_ids=tuple(
                    row.answer_block_id for row in blocks
                )
            )
            citations_by_block: dict[UUID, list] = {}
            for citation in citations:
                citations_by_block.setdefault(
                    citation.answer_block_id, []
                ).append(citation)
            link = await uow.turns.get_run_link(
                turn_id=turn.turn_id,
                purpose="PRIMARY",
            )
            run = (
                await uow.runs.get_run(ops_run_id=link.ops_run_id)
                if link is not None
                else None
            )
            plan_artifact = (
                await uow.runs.get_artifact(
                    artifact_id=turn.current_plan_artifact_id
                )
                if turn.current_plan_artifact_id is not None
                else None
            )
            task_frame_artifact = (
                await uow.runs.get_artifact(
                    artifact_id=turn.task_frame_artifact_id
                )
                if getattr(turn, "task_frame_artifact_id", None) is not None
                else None
            )
            plan_view = None
            if plan_artifact is not None:
                execution_snapshot = dict(
                    dict(getattr(run, "plan_snapshot_json", None) or {}).get(
                        "investigation_execution", {}
                    )
                )
                tool_invocations = await uow.turns.list_tool_invocations(
                    turn_id=turn.turn_id
                )
                plan_view = safe_plan_projection(
                    dict(plan_artifact.payload_json or {}),
                    task_frame=(
                        dict(task_frame_artifact.payload_json or {})
                        if task_frame_artifact is not None
                        else None
                    ),
                    execution_snapshot=execution_snapshot,
                    tool_invocations=tool_invocations,
                )
            block_views = [
                self._block_view(
                    row,
                    citations_by_block.get(row.answer_block_id, []),
                )
                for row in blocks
            ]
            turn_summary = self._turn_summary(turn)
            return {
                **turn_summary,
                "messages": [self._message_view(row) for row in messages],
                "answer_blocks": block_views,
                "display_dictionary": dictionary_for_payload(
                    {
                        "status": turn_summary["status"],
                        "sufficiency_status": turn_summary[
                            "sufficiency_status"
                        ],
                        "evidence_gaps": turn_summary["evidence_gaps"],
                        "answer_blocks": block_views,
                        "investigation_plan": plan_view,
                    }
                ),
                "investigation_plan": plan_view,
                "ops_run_id": (
                    str(link.ops_run_id) if link is not None else None
                ),
            }

    async def get_uploaded_input_content(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        item_no: int,
        actor_id: str,
    ) -> tuple[bytes, str]:
        """读取当前用户会话中已固化的图片证据，绝不暴露本地存储地址。"""
        if self._upload_store is None:
            raise self._error(
                "AIOPS_UPLOAD_STORE_UNAVAILABLE", "对话上传存储当前不可用"
            )
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
            )
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
            )
            if (
                conversation is None
                or conversation.created_by != actor_id
                or turn is None
                or turn.conversation_id != conversation_id
            ):
                raise resource_not_found("Conversation Turn")
            input_item = next(
                (
                    row
                    for row in await uow.turns.list_input_items(turn_id=turn_id)
                    if int(row.item_no) == item_no
                ),
                None,
            )
            if input_item is None or not str(
                input_item.media_type or ""
            ).startswith("image/"):
                raise resource_not_found("Conversation Image Input")
            link = await uow.turns.get_run_link(
                turn_id=turn_id,
                purpose="PRIMARY",
            )
            source = (
                await uow.runs.get_artifact(
                    artifact_id=input_item.source_artifact_id
                )
                if link is not None
                else None
            )
            media_type = str(input_item.media_type)
            if (
                source is not None
                and link is not None
                and source.ops_run_id == link.ops_run_id
                and source.artifact_type == "USER_UPLOAD_SOURCE"
                and source.payload_uri
            ):
                payload_uri = source.payload_uri
                content_hash = source.content_hash
                byte_size = int(source.byte_size)
                upload_id = None
            else:
                message = next(
                    (
                        row
                        for row in await uow.turns.list_messages(turn_id=turn_id)
                        if row.message_id == input_item.message_id
                    ),
                    None,
                )
                content = list(
                    dict(getattr(message, "payload_json", {}) or {}).get(
                        "content", []
                    )
                )
                item = (
                    content[item_no - 1]
                    if len(content) >= item_no and isinstance(
                        content[item_no - 1], dict
                    )
                    else {}
                )
                upload_id = str(item.get("upload_id") or "")
                payload_uri = None
                content_hash = None
                byte_size = None
                if not upload_id or str(item.get("media_type") or "") != media_type:
                    raise resource_not_found("Conversation Image Input")
        try:
            if upload_id is not None:
                stored = self._upload_store.get(
                    upload_id=upload_id,
                    domain_id=domain_id,
                    actor_id=actor_id,
                )
                return self._upload_store.read(stored), media_type
            return (
                self._upload_store.read_artifact(
                    payload_uri=payload_uri,
                    content_hash=content_hash,
                    byte_size=byte_size,
                ),
                media_type,
            )
        except (OSError, ValueError) as exc:
            raise resource_not_found("Conversation Image Input") from exc

    async def get_artifact_content(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
        artifact_id: UUID,
    ) -> tuple[bytes, str, str]:
        """按会话、Turn和Run关系读取不可变Artifact内容。"""
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id, conversation_id=conversation_id,
            )
            turn = await uow.turns.get_turn(
                domain_id=domain_id, turn_id=turn_id,
            )
            if (
                conversation is None
                or conversation.created_by != actor_id
                or turn is None
                or turn.conversation_id != conversation_id
            ):
                raise resource_not_found("Conversation Turn")
            link = await uow.turns.get_run_link(
                turn_id=turn_id, purpose="PRIMARY",
            )
            if link is None:
                raise resource_not_found("Artifact")
            artifact = await uow.runs.get_artifact(artifact_id=artifact_id)
            if artifact is None or artifact.ops_run_id != link.ops_run_id:
                raise resource_not_found("Artifact")
            provenance = dict(artifact.provenance_json or {})
            file_name = self._safe_artifact_file_name(
                str(
                    provenance.get("file_name")
                    or f"{artifact.artifact_type.lower()}-{artifact.artifact_id}.json"
                )
            )
            if artifact.payload_uri is not None:
                try:
                    content = self._upload_store.read_artifact(
                        payload_uri=artifact.payload_uri,
                        content_hash=artifact.content_hash,
                        byte_size=artifact.byte_size,
                    )
                except (OSError, ValueError) as exc:
                    raise resource_not_found("Artifact") from exc
                return (
                    content,
                    str(provenance.get("content_type") or "application/octet-stream"),
                    file_name,
                )
            if artifact.schema_version == "DBA_TOOL_RESULT.v1":
                result = DbaToolResult.model_validate(artifact.payload_json)
                reports: list[bytes] = []
                report_labels: list[str] = []
                for outcome in result.tool_outcomes:
                    observation = outcome.observation
                    if (
                        observation is None
                        or observation.truncated
                        or len(observation.columns) != 1
                        or observation.columns[0].name != "output"
                    ):
                        continue
                    content = "".join(
                        str(row[0]) for row in observation.rows
                        if row and row[0] is not None
                    ).encode("utf-8")
                    if content:
                        reports.append(content)
                        report_labels.append(outcome.tool_id)
                if len(reports) != 1:
                    raise state_conflict("报告Artifact必须且只能包含一份完整正文")
                if not reports[0].lstrip().lower().startswith(
                    (b"<!doctype html", b"<html")
                ):
                    raise state_conflict("报告Artifact正文不是受支持的HTML")
                return (
                    reports[0],
                    "text/html",
                    self._safe_artifact_file_name(
                        f"{report_labels[0].replace('.', '-')}-{artifact.artifact_id}.html"
                    ),
                )
            content = json.dumps(
                artifact.payload_json,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            return content, "application/json", file_name

    @staticmethod
    def _safe_artifact_file_name(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
        return (safe or "aiops-artifact.bin")[:256]

    async def _get_implementation_runbook_payload(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
    ) -> dict[str, Any]:
        """读取并鉴权会话中已固化的数据库实施 Runbook。"""
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
            )
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
            )
            if (
                conversation is None
                or conversation.created_by != actor_id
                or turn is None
                or turn.conversation_id != conversation_id
            ):
                raise resource_not_found("Conversation Turn")
            for block in await uow.turns.list_answer_blocks(turn_id=turn_id):
                if block.block_type != "IMPLEMENTATION_RUNBOOK":
                    continue
                payload = block.payload_json
                if isinstance(payload, dict):
                    return dict(payload)
                raise state_conflict("数据库实施文档内容无效")
        raise resource_not_found("Implementation Runbook")

    async def get_implementation_runbook_pdf(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
    ) -> bytes:
        """导出会话中已固化的数据库实施 Runbook，兼容历史 Block。"""
        payload = await self._get_implementation_runbook_payload(
            domain_id=domain_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            actor_id=actor_id,
        )
        return render_implementation_runbook_pdf(payload)

    async def get_implementation_runbook_markdown(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
    ) -> bytes:
        """导出会话中已固化 Runbook 的确定性 Markdown 投影。"""
        payload = await self._get_implementation_runbook_payload(
            domain_id=domain_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            actor_id=actor_id,
        )
        return render_implementation_runbook_markdown(payload).encode("utf-8")

    async def get_implementation_runbook_zip(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
    ) -> bytes:
        """导出当前 Turn 已固化脚本，不生成空 ZIP。"""
        payload = await self._get_implementation_runbook_payload(
            domain_id=domain_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            actor_id=actor_id,
        )
        descriptors = tuple(payload.get("artifacts") or ())
        if not descriptors:
            raise resource_not_found("Implementation Runbook Artifacts")
        async with self._uow_factory() as uow:
            link = await uow.turns.get_run_link(
                turn_id=turn_id,
                purpose="PRIMARY",
            )
            if link is None:
                raise resource_not_found("Implementation Runbook Artifacts")
            contents: dict[str, str] = {}
            for artifact in await uow.runs.list_artifacts(
                ops_run_id=link.ops_run_id
            ):
                if artifact.schema_version != "AIOPS_RUNBOOK_ARTIFACT.v1":
                    continue
                artifact_payload = dict(artifact.payload_json or {})
                artifact_id = str(artifact_payload.get("artifact_id") or "")
                content = artifact_payload.get("content")
                if artifact_id and isinstance(content, str):
                    contents[artifact_id] = content
        try:
            return render_runbook_artifact_zip(payload, contents)
        except ValueError as exc:
            raise state_conflict(str(exc)) from exc

    async def get_implementation_runbook_file_stem(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
    ) -> str:
        """返回与实施档案一致的安全下载文件名前缀。"""
        payload = await self._get_implementation_runbook_payload(
            domain_id=domain_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            actor_id=actor_id,
        )
        profile = str(payload.get("profile") or "database-implementation")
        return profile.lower().replace("_", "-") + f"-{turn_id}"

    async def list_events(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
        after_sequence: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
            )
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
            )
            if (
                conversation is None
                or conversation.created_by != actor_id
                or turn is None
                or turn.conversation_id != conversation_id
            ):
                raise resource_not_found("Conversation Turn")
            events = await uow.turns.list_events(
                turn_id=turn_id,
                after_sequence=after_sequence,
                limit=limit,
            )
            return {
                "events": [
                    {
                        "turn_id": str(row.turn_id),
                        "sequence_no": int(row.sequence_no),
                        "event_type": row.event_type,
                        "payload": dict(row.payload_json),
                        "occurred_at": (
                            row.created_at.isoformat() if row.created_at else None
                        ),
                    }
                    for row in events
                ],
                "next_sequence": int(turn.event_cursor),
                "terminal": turn.status
                in {
                    "WAITING_USER",
                    "COMPLETED",
                    "PARTIAL",
                    "FAILED",
                    "CANCELLED",
                },
            }

    async def respond_to_clarification(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
        trace_id: str,
        command: ClarificationResponseV1,
    ) -> dict[str, Any]:
        """幂等记录澄清回答，并让同一 Turn 从冻结候选继续。"""
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
                lock=True,
            )
            turn = await uow.turns.get_turn(
                domain_id=domain_id, turn_id=turn_id, lock=True
            )
            if (
                conversation is None
                or conversation.created_by != actor_id
                or turn is None
                or turn.conversation_id != conversation_id
            ):
                raise resource_not_found("Conversation Turn")
            messages = await uow.turns.list_messages(turn_id=turn_id)
            duplicate = next(
                (
                    row
                    for row in messages
                    if row.message_type == "CLARIFICATION_RESPONSE"
                    and row.payload_json.get("idempotency_key")
                    == command.idempotency_key
                ),
                None,
            )
            if duplicate is not None:
                return self._receipt(turn)
            if turn.status != "WAITING_USER" or not turn.clarification_json:
                raise state_conflict("当前 Turn 没有等待回答的范围澄清")
            clarification = dict(turn.clarification_json)
            if str(command.clarification_id) != str(
                clarification.get("clarification_id")
            ):
                raise validation_failed("澄清请求已经变化，请刷新后重试")
            if command.candidate_id is not None:
                candidate_ids = {
                    str(item.get("candidate_id"))
                    for item in clarification.get("candidates", [])
                    if isinstance(item, dict)
                }
                if command.candidate_id not in candidate_ids:
                    raise validation_failed("澄清候选不存在或已经失效")
            elif not clarification.get("allow_free_text", False):
                raise validation_failed("当前澄清不接受自由文字")
            conversation.last_message_no = int(conversation.last_message_no) + 1
            conversation.updated_by = actor_id
            conversation.updated_at = datetime.now(UTC)
            response_payload = command.model_dump(mode="json")
            await uow.turns.add_message(
                OpsConversationMessageEntity(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    sequence_no=conversation.last_message_no,
                    role="USER",
                    message_type="CLARIFICATION_RESPONSE",
                    payload_schema="AIOPS_CLARIFICATION_RESPONSE.v1",
                    payload_json=response_payload,
                    created_by=actor_id,
                )
            )
            turn.status = "ACCEPTED"
            turn.completed_at = None
            turn.event_cursor = int(turn.event_cursor) + 1
            await uow.turns.add_event(
                OpsTurnEventEntity(
                    turn_id=turn_id,
                    sequence_no=turn.event_cursor,
                    event_type="clarification.responded",
                    event_key=(
                        f"clarification.responded:{command.clarification_id}:"
                        f"{command.idempotency_key}"
                    )[:128],
                    visibility="USER",
                    payload_json={
                        "clarification_id": str(command.clarification_id),
                        "status": "ACCEPTED",
                    },
                )
            )
            created = await uow.outbox.get_by_idempotency(
                idempotency_key=f"turn-created:{turn_id}"
            )
            if created is None:
                raise state_conflict("Turn 缺少可重放的创建命令")
            outbox_payload = {
                **dict(created.payload_json or {}),
                "response": response_payload,
                "trace_id": trace_id,
            }
            encoded = json.dumps(
                outbox_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            await uow.outbox.add(
                OutboxEntity(
                    aggregate_type="CONVERSATION_TURN",
                    aggregate_id=turn_id,
                    event_type="aiops.turn.clarification_resolved",
                    idempotency_key=(
                        f"turn-clarification:{command.clarification_id}:"
                        f"{command.idempotency_key}"
                    )[:128],
                    payload_json=outbox_payload,
                    payload_hash=hashlib.sha256(encoded).hexdigest(),
                    trace_id=trace_id,
                )
            )
            await uow.commit()
            return self._receipt(turn)

    async def cancel_turn(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=conversation_id,
            )
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
                lock=True,
            )
            if (
                conversation is None
                or conversation.created_by != actor_id
                or turn is None
                or turn.conversation_id != conversation_id
            ):
                raise resource_not_found("Conversation Turn")
            if turn.status in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}:
                return self._turn_summary(turn)
            now = datetime.now(UTC)
            turn.cancel_requested_at = now
            turn.cancel_requested_by = actor_id
            turn.event_cursor = int(turn.event_cursor) + 1
            await uow.turns.add_event(
                OpsTurnEventEntity(
                    turn_id=turn.turn_id,
                    sequence_no=turn.event_cursor,
                    event_type="turn.cancel_requested",
                    event_key=f"turn.cancel_requested:{turn.turn_id}",
                    visibility="USER",
                    payload_json={"status": turn.status},
                )
            )
            payload = {
                "domain_id": domain_id,
                "turn_id": str(turn.turn_id),
                "requested_by": actor_id,
            }
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            await uow.outbox.add(
                OutboxEntity(
                    aggregate_type="CONVERSATION_TURN",
                    aggregate_id=turn.turn_id,
                    event_type="aiops.turn.cancel_requested",
                    idempotency_key=f"turn-cancel:{turn.turn_id}",
                    payload_json=payload,
                    payload_hash=hashlib.sha256(encoded).hexdigest(),
                    trace_id=trace_id,
                )
            )
            await uow.commit()
            return self._turn_summary(turn)

    async def confirm_target_fact(
        self,
        *,
        domain_id: int,
        conversation_id: UUID,
        turn_id: UUID,
        actor_id: str,
        trace_id: str,
        command: TargetFactConfirmCommand,
    ) -> TargetFactView:
        """把聊天确认卡中的运维事实写入 Target，不生成可执行 SQL。"""
        try:
            async with self._uow_factory() as uow:
                conversation = await uow.conversations.get_conversation(
                    domain_id=domain_id,
                    conversation_id=conversation_id,
                )
                turn = await uow.turns.get_turn(
                    domain_id=domain_id,
                    turn_id=turn_id,
                )
                if (
                    conversation is None
                    or conversation.created_by != actor_id
                    or turn is None
                    or turn.conversation_id != conversation_id
                ):
                    raise resource_not_found("Conversation Turn")
                allowed_targets: set[str] = set()
                if turn.focused_target_id is not None:
                    allowed_targets.add(str(turn.focused_target_id))
                for link in await uow.turns.list_run_links(turn_id=turn_id):
                    run = await uow.runs.get_run(ops_run_id=link.ops_run_id)
                    if run is not None and run.target_id is not None:
                        allowed_targets.add(str(run.target_id))
                if str(command.target_id) not in allowed_targets:
                    raise validation_failed("确认的 Target 必须属于当前 Turn 范围")
                scope = ConfigurationScope(
                    domain_id=domain_id,
                    principal_id=f"USER:{actor_id}",
                    actor_id=actor_id,
                    request_id=trace_id,
                    trace_id=trace_id,
                )
                details = {
                    "conversation_id": str(conversation_id),
                    "turn_id": str(turn_id),
                }
                if command.note:
                    details["note"] = command.note
                entity = await create_confirmed_fact(
                    uow=uow,
                    scope=scope,
                    target_id=command.target_id,
                    fact_type=command.fact_type,
                    fact_key=command.fact_key,
                    fact_value=command.fact_value,
                    now=datetime.now(UTC),
                    details=details,
                )
                await uow.commit()
                return _target_fact_view(entity)
        except IntegrityError as exc:
            raise state_conflict("配置自然键已存在或并发创建冲突") from exc

    @staticmethod
    def _request_sha256(command: TurnCreate) -> str:
        encoded = json.dumps(command.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    async def _create_turn(
        self,
        *,
        uow,
        conversation,
        version,
        command: TurnCreate,
        actor_id: str,
        trace_id: str,
        source_run,
        execution_context: dict[str, Any] | None = None,
        focused_target_id: UUID | None = None,
    ) -> dict[str, Any]:
        effective_context = dict(execution_context or {})
        if not effective_context.get("deadline_at") and str(effective_context.get("workflow_kind") or "CHAT_TURN") == "CHAT_TURN":
            effective_context["deadline_at"] = (datetime.now(UTC) + timedelta(seconds=self._turn_budget_seconds)).isoformat()
        if command.sql_diagnosis is not None:
            try:
                selected = await resolve_sql_selection(
                    uow=uow, reference=command.sql_diagnosis.selection_ref,
                    conversation=conversation, actor_id=actor_id, ttl_seconds=self._sql_selection_ttl,
                )
            except PermissionError as exc:
                raise self._error("AIOPS_SQL_SELECTION_FORBIDDEN", str(exc)) from exc
            except ValueError as exc:
                raise self._error("AIOPS_SQL_SELECTION_INVALID", str(exc)) from exc
            if focused_target_id is not None and str(focused_target_id) != selected["target_id"]:
                raise self._error("AIOPS_SQL_SELECTION_TARGET_MISMATCH", "所选 SQL 与当前诊断资源不一致")
            focused_target_id = UUID(selected["target_id"])
            effective_context["sql_selection"] = selected
        if source_run is not None:
            if focused_target_id is not None and focused_target_id != source_run.target_id:
                raise self._error("AIOPS_SQL_SELECTION_TARGET_MISMATCH", "来源 Run 与所选 SQL 资源不一致")
            focused_target_id = source_run.target_id
        if focused_target_id is not None:
            if not await uow.agents.version_has_target(
                agent_version_id=version.agent_version_id,
                target_id=focused_target_id,
            ):
                raise self._error(
                    "AIOPS_AGENT_SCOPE_DENIED",
                    "诊断焦点不属于当前 Agent 允许访问的业务系统",
                )
            await self._require_existing_target(
                uow=uow,
                domain_id=int(conversation.domain_id),
                target_id=focused_target_id,
            )
        content = tuple(command.content)
        if command.starter is not None:
            if focused_target_id is None:
                raise self._error(
                    "AIOPS_STARTER_SCOPE_REQUIRED",
                    "诊断功能入口必须由可信资源上下文确定数据库资源",
                )
            target = await uow.targets.get_scoped(
                target_id=focused_target_id,
                domain_id=int(conversation.domain_id),
            )
            if target is None:
                raise resource_not_found("Target")
            frozen_starter = self._conversation_starter_catalog.freeze(
                selection=command.starter,
                capabilities=set(version.capabilities_json or ()),
                target=target,
            )
            content = (
                InputContent(
                    content_type="TEXT",
                    text=str(frozen_starter["user_message"]),
                ),
                *tuple(
                    item for item in content
                    if str(item.content_type) != "TEXT"
                ),
            )
            effective_context["conversation_starter"] = frozen_starter
        upload_metadata = {}
        for item in content:
            if not item.upload_id:
                continue
            if self._upload_store is None:
                raise self._error(
                    "AIOPS_UPLOAD_STORE_UNAVAILABLE",
                    "对话上传存储当前不可用",
                )
            try:
                stored = self._upload_store.get(
                    upload_id=item.upload_id,
                    domain_id=int(conversation.domain_id),
                    actor_id=actor_id,
                )
            except PermissionError as exc:
                raise self._error(
                    "AIOPS_UPLOAD_FORBIDDEN", "不能引用其他用户的上传文件"
                ) from exc
            except ValueError as exc:
                raise self._error("AIOPS_UPLOAD_INVALID", str(exc)) from exc
            stored = self._upload_store.preserve(stored)
            if item.media_type and item.media_type != stored.media_type:
                raise self._error(
                    "AIOPS_UPLOAD_MEDIA_TYPE_MISMATCH",
                    "上传引用的媒体类型与登记信息不一致",
                )
            upload_metadata[item.upload_id] = stored
        text_parts = [
            str(item.text).strip()
            for item in content
            if item.text and item.text.strip()
        ]
        text_parts.extend(
            f"[用户上传文件：{stored.file_name}]"
            for stored in upload_metadata.values()
        )
        message = "\n\n".join(text_parts)
        if not message:
            raise self._error("AIOPS_TURN_CONTENT_REQUIRED", "诊断输入不能为空")
        conversation.last_turn_no = int(conversation.last_turn_no) + 1
        conversation.last_message_no = int(conversation.last_message_no) + 1
        conversation.updated_by = actor_id
        conversation.updated_at = datetime.now(UTC)
        if not conversation.title:
            conversation.title = self._title(message)
        turn = OpsConversationTurnEntity(
            turn_id=uuid7(),
            domain_id=conversation.domain_id,
            conversation_id=conversation.conversation_id,
            turn_no=conversation.last_turn_no,
            idempotency_key=command.idempotency_key,
            status="QUEUED",
            focused_target_id=focused_target_id,
            scope_resolution_json=None,
            event_cursor=1,
            created_by=actor_id,
        )
        user_message = OpsConversationMessageEntity(
            conversation_id=conversation.conversation_id,
            turn_id=turn.turn_id,
            sequence_no=conversation.last_message_no,
            role="USER",
            message_type="USER_MESSAGE",
            payload_schema="AIOPS_USER_MESSAGE.v2",
            payload_json={
                "text": message,
                "content": [item.model_dump(mode="json") for item in content],
                "request_sha256": self._request_sha256(command),
                "sql_selection": effective_context.get("sql_selection"),
            },
            created_by=actor_id,
        )
        event = OpsTurnEventEntity(
            turn_id=turn.turn_id,
            sequence_no=1,
            event_type="turn.created",
            event_key=f"turn.created:{turn.turn_id}",
            visibility="USER",
            payload_json={"status": "QUEUED", "turn_no": turn.turn_no},
        )
        outbox_payload = {
            "schema_version": "aiops.turn-command.v2",
            "domain_id": int(conversation.domain_id),
            "conversation_id": str(conversation.conversation_id),
            "turn_id": str(turn.turn_id),
            "focused_target_id": (
                str(focused_target_id) if focused_target_id else None
            ),
            "agent_id": str(conversation.agent_id),
            "agent_version_id": str(conversation.agent_version_id),
            "source_situation_id": (
                str(conversation.source_situation_id)
                if conversation.source_situation_id
                else None
            ),
            "source_run_id": str(source_run.ops_run_id) if source_run else None,
            "trace_id": trace_id,
        }
        if effective_context:
            outbox_payload["execution_context"] = effective_context
        encoded = json.dumps(
            outbox_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        outbox = OutboxEntity(
            aggregate_type="CONVERSATION_TURN",
            aggregate_id=turn.turn_id,
            event_type="aiops.turn.created",
            idempotency_key=f"turn-created:{turn.turn_id}",
            payload_json=outbox_payload,
            payload_hash=hashlib.sha256(encoded).hexdigest(),
            trace_id=trace_id,
        )
        await uow.turns.add_turn(turn)
        await uow.turns.add_message(user_message)
        for item_no, item in enumerate(content, start=1):
            await uow.turns.add_input_item(
                OpsTurnInputItemEntity(
                    input_item_id=uuid7(),
                    turn_id=turn.turn_id,
                    message_id=user_message.message_id,
                    item_no=item_no,
                    content_type=str(item.content_type),
                    media_type=(
                        upload_metadata[item.upload_id].media_type
                        if item.upload_id
                        else item.media_type
                    ),
                    content_text=(
                        item.text
                        if item.text is not None
                        else f"[用户上传文件：{upload_metadata[item.upload_id].file_name}]"
                    ),
                )
            )
        await uow.turns.add_event(event)
        await uow.outbox.add(outbox)
        return self._receipt(turn)

    @staticmethod
    async def _focused_target_from_context_hint(
        *,
        uow,
        context_hint: TrustedContextHint | None,
    ) -> UUID | None:
        """只允许单一数据库资源把会话上下文收敛为执行 Target。"""
        if context_hint is None or len(context_hint.resource_ids) != 1:
            return None
        database = await uow.business_systems.get_database_profile(
            context_hint.resource_ids[0]
        )
        return database.target_id if database is not None else None

    async def _validate_context_hint(
        self,
        *,
        uow,
        domain_id: int,
        agent_version_id: UUID,
        context_hint,
    ) -> None:
        if context_hint is None:
            return
        allowed_system_ids = set(
            await uow.agents.version_business_system_ids(
                agent_version_id=agent_version_id
            )
        )
        if (
            context_hint.business_system_id is not None
            and context_hint.business_system_id not in allowed_system_ids
        ):
            raise self._error(
                "AIOPS_CONTEXT_SCOPE_DENIED",
                "上下文业务系统不在当前 Agent 授权范围内",
            )
        allowed_resources: set[UUID] = set()
        relevant_system_ids = (
            {context_hint.business_system_id}
            if context_hint.business_system_id is not None
            else allowed_system_ids
        )
        for business_system_id in relevant_system_ids:
            system = await uow.business_systems.get_scoped(
                business_system_id=business_system_id,
                domain_id=domain_id,
            )
            if system is None or system.status != "ENABLED":
                continue
            allowed_resources.update(
                resource.resource_id
                for link, resource, _database, _target in (
                    await uow.business_systems.list_system_resource_rows(
                        business_system_id
                    )
                )
                if link.status == "ACTIVE"
                and resource.lifecycle_status == "ACTIVE"
            )
        if not set(context_hint.resource_ids).issubset(allowed_resources):
            raise self._error(
                "AIOPS_CONTEXT_SCOPE_DENIED",
                "上下文资源不在当前 Agent 授权范围内",
            )

    @staticmethod
    async def _context_hint_for_target(
        *,
        uow,
        domain_id: int,
        target_id: UUID,
        source: str,
        source_resource_id: UUID,
    ) -> dict[str, Any]:
        resource = await uow.business_systems.get_resource_for_target(
            target_id=target_id,
            domain_id=domain_id,
        )
        system = await uow.business_systems.find_system_for_target(
            target_id=target_id,
            domain_id=domain_id,
        )
        if resource is None or system is None:
            raise resource_not_found("Business System Resource")
        return {
            "source_kind": source,
            "business_system_id": system.business_system_id,
            "resource_ids": (resource.resource_id,),
            "source_resource_id": source_resource_id,
        }

    async def _require_existing_target(
        self,
        *,
        uow,
        domain_id: int,
        target_id: UUID,
    ):
        target = await uow.targets.get_scoped(
            target_id=target_id,
            domain_id=domain_id,
        )
        if target is None or target.status != "ENABLED":
            raise self._error(
                "AIOPS_TARGET_NOT_ENABLED",
                "Target 不存在或未启用，不能开始或继续诊断",
            )
        return target

    async def _require_conversation_runnable(
        self,
        *,
        uow,
        domain_id: int,
        conversation: OpsConversationEntity,
    ) -> None:
        """继续聊天前重新校验 Agent 状态和冻结版本。"""

        agent = await uow.agents.get(
            domain_id=domain_id,
            agent_id=conversation.agent_id,
        )
        if (
            agent is None
            or agent.status != "ACTIVE"
            or await uow.agents.version(
                agent_id=conversation.agent_id,
                agent_version_id=conversation.agent_version_id,
            ) is None
        ):
            raise self._error(
                "AIOPS_CONVERSATION_AGENT_NOT_RUNNABLE",
                "当前 Agent 已停用或冻结版本不可用，不能继续聊天",
            )

    @staticmethod
    def _title(message: str) -> str:
        compact = " ".join(message.split())
        return compact if len(compact) <= 48 else f"{compact[:48].rstrip()}…"

    @staticmethod
    def _receipt(turn) -> dict[str, Any]:
        return {
            "conversation_id": str(turn.conversation_id),
            "turn_id": str(turn.turn_id),
            "turn_no": int(turn.turn_no),
            "status": turn.status,
            "event_cursor": int(turn.event_cursor),
            "created_at": turn.created_at.isoformat() if turn.created_at else None,
        }

    @staticmethod
    def _conversation_summary(row) -> dict[str, Any]:
        return {
            "conversation_id": str(row.conversation_id),
            "agent_id": str(row.agent_id),
            "agent_version_id": str(row.agent_version_id),
            "context_hint": dict(row.context_hint_json or {}) or None,
            "title": row.title,
            "status": row.status,
            "source_type": row.source_type,
            "source_situation_id": (
                str(row.source_situation_id) if row.source_situation_id else None
            ),
            "source_run_id": str(row.source_run_id) if row.source_run_id else None,
            "source_report_id": (
                str(row.source_report_id) if row.source_report_id else None
            ),
            "source_inspection_fire_id": (
                str(row.source_inspection_fire_id)
                if row.source_inspection_fire_id
                else None
            ),
            "last_turn_no": int(row.last_turn_no),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    @staticmethod
    def _turn_summary(row) -> dict[str, Any]:
        sufficiency = dict(row.sufficiency_json or {})
        return {
            "turn_id": str(row.turn_id),
            "conversation_id": str(row.conversation_id),
            "turn_no": int(row.turn_no),
            "status": row.status,
            "capability_plan": dict(
                getattr(row, "capability_plan_json", None) or {}
            ) or None,
            "scope_resolution": dict(
                getattr(row, "scope_resolution_json", None) or {}
            ) or None,
            "clarification": dict(
                getattr(row, "clarification_json", None) or {}
            ) or None,
            "current_plan_revision": int(row.current_plan_revision or 0),
            "investigation_round": int(row.investigation_round or 0),
            "tool_call_count": int(row.tool_call_count or 0),
            "sufficiency_status": row.sufficiency_status,
            "evidence_gaps": [
                {
                    "source_id": item.get("source_id"),
                    "step_id": item.get("step_id"),
                    "code": item.get("code"),
                    "detail": item.get("detail"),
                    "retryable": bool(item.get("retryable", False)),
                }
                for item in sufficiency.get("gaps", [])
                if isinstance(item, dict)
            ],
            "event_cursor": int(row.event_cursor),
            "error_domain": row.error_domain,
            "error_code": row.error_code,
            "error_message": row.error_message,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "completed_at": (
                row.completed_at.isoformat() if row.completed_at else None
            ),
        }

    @staticmethod
    def _message_view(row) -> dict[str, Any]:
        return {
            "message_id": str(row.message_id),
            "sequence_no": int(row.sequence_no),
            "role": row.role,
            "message_type": row.message_type,
            "payload_schema": row.payload_schema,
            "payload": dict(row.payload_json),
            "artifact_id": str(row.artifact_id) if row.artifact_id else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    @staticmethod
    def _block_view(row, citations=()) -> dict[str, Any]:
        return {
            "answer_block_id": str(row.answer_block_id),
            "block_no": int(row.block_no),
            "block_type": row.block_type,
            "schema_version": row.schema_version,
            "payload": dict(row.payload_json),
            "content_hash": row.content_hash,
            "citations": [
                {
                    "citation_no": int(item.citation_no),
                    "turn_evidence_id": str(item.turn_evidence_id),
                    "label": item.label,
                }
                for item in citations
            ],
        }

    @staticmethod
    def _error(code: str, message: str) -> AIOpsApplicationError:
        return AIOpsApplicationError(
            code=code,
            message=message,
            status_code=422,
        )
