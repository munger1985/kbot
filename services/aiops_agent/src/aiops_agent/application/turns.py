"""专业 DBA Conversation 与 Turn 的命令和查询用例。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from aiops_agent.application.errors import (
    AIOpsApplicationError,
    resource_not_found,
    state_conflict,
)
from aiops_agent.entities import (
    OpsConversationEntity,
    OpsConversationMessageEntity,
    OpsConversationTurnEntity,
    OpsTurnEventEntity,
    OutboxEntity,
)
from platform_core.contracts.aiops import (
    ConversationCreate,
    ConversationSourceType,
    TurnCreate,
)
from platform_core.identity import uuid7


class ConversationTurnService:
    """在单一 UoW 内接收一轮用户问题并可靠投递规划任务。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

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
            source = conversation_create.source
            source_run = None
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
            conversation = OpsConversationEntity(
                domain_id=domain_id,
                agent_id=agent.agent_id,
                agent_version_id=version.agent_version_id,
                title=conversation_create.title,
                status="ACTIVE",
                source_type=str(source.source_type),
                source_situation_id=source.situation_id,
                source_run_id=source.run_id,
                source_report_id=source.report_id,
                last_turn_no=0,
                last_message_no=0,
                created_by=actor_id,
                updated_by=actor_id,
            )
            await uow.conversations.add_conversation(conversation)
            receipt = await self._create_turn(
                uow=uow,
                conversation=conversation,
                version=version,
                command=first_turn,
                actor_id=actor_id,
                trace_id=trace_id,
                source_run=source_run,
            )
            await uow.commit()
            return receipt

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
            existing = await uow.turns.get_by_idempotency(
                conversation_id=conversation_id,
                idempotency_key=command.idempotency_key,
            )
            if existing is not None:
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
            receipt = await self._create_turn(
                uow=uow,
                conversation=conversation,
                version=version,
                command=command,
                actor_id=actor_id,
                trace_id=trace_id,
                source_run=source_run,
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
            return {
                **self._turn_summary(turn),
                "messages": [self._message_view(row) for row in messages],
                "answer_blocks": [
                    self._block_view(
                        row,
                        citations_by_block.get(row.answer_block_id, []),
                    )
                    for row in blocks
                ],
            }

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
    ) -> dict[str, Any]:
        message = command.message.strip()
        if not message:
            raise self._error("AIOPS_TURN_MESSAGE_REQUIRED", "诊断问题不能为空")
        target_id = await self._resolve_target(
            uow=uow,
            domain_id=int(conversation.domain_id),
            version=version,
            requested_target_id=command.target_id,
            source_run=source_run,
        )
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
            resolved_target_id=target_id,
            event_cursor=1,
            created_by=actor_id,
        )
        user_message = OpsConversationMessageEntity(
            conversation_id=conversation.conversation_id,
            turn_id=turn.turn_id,
            sequence_no=conversation.last_message_no,
            role="USER",
            message_type="USER_MESSAGE",
            payload_schema="AIOPS_USER_MESSAGE.v1",
            payload_json={"text": message},
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
            "schema_version": "aiops.turn-command.v1",
            "domain_id": int(conversation.domain_id),
            "conversation_id": str(conversation.conversation_id),
            "turn_id": str(turn.turn_id),
            "target_id": str(target_id),
            "agent_id": str(conversation.agent_id),
            "agent_version_id": str(conversation.agent_version_id),
            "source_run_id": str(source_run.ops_run_id) if source_run else None,
            "trace_id": trace_id,
        }
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
        await uow.turns.add_event(event)
        await uow.outbox.add(outbox)
        return self._receipt(turn)

    async def _resolve_target(
        self,
        *,
        uow,
        domain_id: int,
        version,
        requested_target_id: UUID | None,
        source_run,
    ) -> UUID:
        fixed_target_id = version.target_id
        source_target_id = source_run.target_id if source_run is not None else None
        source_ids = await uow.agents.version_source_ids(
            agent_version_id=version.agent_version_id
        )
        candidates = await uow.targets.target_ids_shared_by_sources(
            domain_id=domain_id,
            source_ids=source_ids,
        )
        allowed = set(candidates)
        if fixed_target_id is not None:
            allowed.add(fixed_target_id)
        if source_target_id is not None:
            if source_target_id not in allowed:
                raise self._error(
                    "AIOPS_SOURCE_TARGET_FORBIDDEN",
                    "来源 Run 的 Target 不在当前 Agent 允许范围内",
                )
            if requested_target_id not in (None, source_target_id):
                raise self._error(
                    "AIOPS_TARGET_SCOPE_CONFLICT",
                    "显式 Target 与来源 Run 的 Target 不一致",
                )
            await self._require_available_target(
                uow=uow,
                domain_id=domain_id,
                target_id=source_target_id,
            )
            return source_target_id
        if requested_target_id is not None:
            if requested_target_id not in allowed:
                raise self._error(
                    "AIOPS_TARGET_SCOPE_FORBIDDEN",
                    "所选 Target 不在当前 Agent 允许范围内",
                )
            await self._require_available_target(
                uow=uow,
                domain_id=domain_id,
                target_id=requested_target_id,
            )
            return requested_target_id
        if fixed_target_id is not None:
            await self._require_available_target(
                uow=uow,
                domain_id=domain_id,
                target_id=fixed_target_id,
            )
            return fixed_target_id
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise self._error(
                "AIOPS_TARGET_SELECTION_REQUIRED",
                "当前 Agent 可诊断多个 Target，请先明确选择数据库",
            )
        raise self._error(
            "AIOPS_TARGET_SCOPE_UNRESOLVED",
            "当前监控源未映射到可诊断 Target，请先配置 Target Source Binding",
        )

    async def _require_available_target(
        self,
        *,
        uow,
        domain_id: int,
        target_id: UUID,
    ) -> None:
        target = await uow.targets.get_scoped(
            target_id=target_id,
            domain_id=domain_id,
        )
        if target is None or target.status != "ENABLED":
            raise self._error(
                "AIOPS_TARGET_NOT_AVAILABLE",
                "所选 Target 不存在或未启用",
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
            "resolved_target_id": (
                str(row.resolved_target_id) if row.resolved_target_id else None
            ),
            "primary_intent": row.primary_intent,
            "primary_domain": row.primary_domain,
            "subject": row.subject,
            "sufficiency_status": row.sufficiency_status,
            "evidence_gaps": [
                {
                    "skill_id": item.get("skill_id"),
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
