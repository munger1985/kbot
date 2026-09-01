"""Conversation Turn 的规划事务骨架。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict
from aiops_agent.entities import (
    OpsAnswerBlockEntity,
    OpsArtifactEntity,
    OpsConversationMessageEntity,
    OpsRunEntity,
    OpsTurnEventEntity,
    OpsTurnRunEntity,
)
from platform_core.identity import uuid7


class TurnPlannerService:
    """建立唯一 Primary Run，并为输入理解与调查规划冻结上下文。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def begin(self, payload: dict) -> dict:
        """把已接收 Turn 推进到 UNDERSTANDING，不调用任何外部依赖。"""
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
            existing = await uow.turns.get_run_link(
                turn_id=turn_id,
                purpose="PRIMARY",
            )
            if existing is not None:
                return {
                    "turn_id": str(turn_id),
                    "ops_run_id": str(existing.ops_run_id),
                    "status": turn.status,
                }
            if turn.status == "CANCELLED":
                return {
                    "turn_id": str(turn_id),
                    "ops_run_id": None,
                    "status": turn.status,
                }
            if turn.status != "ACCEPTED":
                raise state_conflict(
                    f"只有 ACCEPTED Turn 可以进入规划，当前状态为 {turn.status}"
                )
            if turn.cancel_requested_at is not None:
                turn.status = "CANCELLED"
                turn.completed_at = datetime.now(UTC)
                await self._append_event(
                    uow,
                    turn,
                    event_type="turn.status",
                    payload={
                        "status": "CANCELLED",
                        "public_summary": "诊断已取消",
                    },
                )
                await uow.commit()
                return {
                    "turn_id": str(turn_id),
                    "ops_run_id": None,
                    "status": turn.status,
                }

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
            now = datetime.now(UTC)
            ops_run_id = uuid7()
            run = OpsRunEntity(
                ops_run_id=ops_run_id,
                domain_id=domain_id,
                target_id=turn.resolved_target_id,
                agent_id=UUID(str(payload["agent_id"])),
                agent_version_id=UUID(str(payload["agent_version_id"])),
                trigger_type="CHAT",
                interaction_mode="INTERACTIVE",
                workflow_kind="CHAT_TURN",
                actor_id=turn.created_by,
                original_request=str(user_message.payload_json["text"]),
                idempotency_key=f"turn:{turn_id}:primary",
                status="RUNNING",
                plan_snapshot_json={
                    "schema_version": "aiops.turn-planning-context.v1",
                    "conversation_id": str(turn.conversation_id),
                    "turn_id": str(turn_id),
                    "source_run_id": payload.get("source_run_id"),
                },
                policy_snapshot_json={
                    "agent_version_id": str(payload["agent_version_id"]),
                },
                trace_id=str(payload["trace_id"]),
                started_at=now,
            )
            await uow.runs.add_run(run)
            await uow.turns.add_run(
                OpsTurnRunEntity(
                    turn_run_id=uuid7(),
                    turn_id=turn_id,
                    ops_run_id=ops_run_id,
                    purpose="PRIMARY",
                    sequence_no=1,
                )
            )
            turn.status = "UNDERSTANDING"
            turn.started_at = turn.started_at or now
            await self._append_event(
                uow,
                turn,
                event_type="planning.started",
                payload={
                    "public_summary": (
                        "已锁定 Agent、Target 和模型配置，正在生成调查计划"
                    ),
                },
            )
            await self._append_event(
                uow,
                turn,
                event_type="turn.status",
                payload={
                    "status": "UNDERSTANDING",
                    "public_summary": "正在识别输入材料并理解要解决的问题",
                },
            )
            await uow.commit()
            return {
                "turn_id": str(turn_id),
                "ops_run_id": str(ops_run_id),
                "status": turn.status,
            }

    async def cancel(self, payload: dict) -> dict:
        """幂等取消 Turn 及其已经建立的 Primary Run。"""
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
                "COMPLETED", "PARTIAL", "FAILED", "CANCELLED"
            }:
                return {"turn_id": str(turn_id), "status": turn.status}
            now = datetime.now(UTC)
            link = await uow.turns.get_run_link(
                turn_id=turn_id,
                purpose="PRIMARY",
            )
            if link is not None:
                run = await uow.runs.get_run(
                    ops_run_id=link.ops_run_id,
                    lock=True,
                )
                if run is not None and run.status not in {
                    "COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED"
                }:
                    run.cancel_requested_at = now
                    run.cancel_requested_by = str(payload["requested_by"])
                    run.status = "CANCELLED"
                    run.completed_at = now
            turn.status = "CANCELLED"
            turn.completed_at = now
            await self._append_event(
                uow,
                turn,
                event_type="turn.status",
                payload={
                    "status": "CANCELLED",
                    "public_summary": "诊断已取消",
                },
            )
            await uow.commit()
            return {"turn_id": str(turn_id), "status": turn.status}

    async def complete_empty(
        self,
        *,
        domain_id: int,
        turn_id: UUID,
        ops_run_id: UUID,
    ) -> dict:
        """完成无外部依赖且没有调查Tool的验收流。"""
        markdown = "诊断调度链路已经建立，但本轮没有需要执行的调查Tool。"
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id,
                turn_id=turn_id,
                lock=True,
            )
            run = await uow.runs.get_run(
                ops_run_id=ops_run_id,
                lock=True,
            )
            link = await uow.turns.get_run_link(
                turn_id=turn_id,
                purpose="PRIMARY",
            )
            if (
                turn is None
                or run is None
                or link is None
                or link.ops_run_id != ops_run_id
            ):
                raise resource_not_found("Turn Primary Run")
            if turn.status == "COMPLETED" and run.status == "COMPLETED":
                return {
                    "turn_id": str(turn_id),
                    "ops_run_id": str(ops_run_id),
                    "status": turn.status,
                }
            if turn.status not in {"UNDERSTANDING", "PLANNING"} or run.status != "RUNNING":
                raise state_conflict("只有规划中的 Turn 可以完成空流程")

            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id,
                conversation_id=turn.conversation_id,
                lock=True,
            )
            if conversation is None:
                raise resource_not_found("Conversation")
            conversation.last_message_no = int(conversation.last_message_no) + 1
            conversation.updated_by = turn.created_by
            now = datetime.now(UTC)
            conversation.updated_at = now
            message_id = uuid7()
            artifact_id = uuid7()
            payload = {
                "status": "COMPLETED",
                "sufficiency_status": "CAPABILITY_UNAVAILABLE",
                "markdown": markdown,
            }
            encoded = json.dumps(
                payload, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            await uow.runs.add_artifact(
                OpsArtifactEntity(
                    artifact_id=artifact_id,
                    ops_run_id=ops_run_id,
                    artifact_key="turn-empty-result",
                    artifact_type="TURN_RESULT",
                    schema_version="AIOPS_TURN_RESULT.v1",
                    payload_json=payload,
                    content_hash=digest,
                    byte_size=len(encoded),
                    provenance_json={"producer": "aiops.turn-planner"},
                    trust_level="SOURCE_VERIFIED",
                    security_level=1,
                )
            )
            await uow.turns.add_message(
                OpsConversationMessageEntity(
                    message_id=message_id,
                    conversation_id=turn.conversation_id,
                    turn_id=turn_id,
                    sequence_no=conversation.last_message_no,
                    role="AGENT",
                    message_type="ASSISTANT_MESSAGE",
                    payload_schema="AIOPS_ASSISTANT_MESSAGE.v1",
                    payload_json={"text": markdown},
                    artifact_id=artifact_id,
                    created_by="aiops.turn-planner",
                )
            )
            await uow.turns.add_answer_block(
                OpsAnswerBlockEntity(
                    answer_block_id=uuid7(),
                    turn_id=turn_id,
                    message_id=message_id,
                    block_no=1,
                    block_type="MARKDOWN",
                    schema_version="AIOPS_MARKDOWN_BLOCK.v1",
                    payload_json={"markdown": markdown},
                    content_hash=hashlib.sha256(
                        markdown.encode("utf-8")
                    ).hexdigest(),
                )
            )
            run.status = "COMPLETED"
            run.final_artifact_id = artifact_id
            run.completed_at = now
            turn.status = "COMPLETED"
            turn.sufficiency_status = "CAPABILITY_UNAVAILABLE"
            turn.completed_at = now
            await self._append_event(
                uow,
                turn,
                event_type="answer.delta",
                payload={"delta": markdown},
            )
            await self._append_event(
                uow,
                turn,
                event_type="answer.completed",
                payload={"answer_block_count": 1},
            )
            await self._append_event(
                uow,
                turn,
                event_type="turn.status",
                payload={"status": "COMPLETED"},
            )
            await uow.commit()
            return {
                "turn_id": str(turn_id),
                "ops_run_id": str(ops_run_id),
                "status": turn.status,
            }

    @staticmethod
    async def _append_event(
        uow,
        turn,
        *,
        event_type: str,
        payload: dict,
    ) -> None:
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
