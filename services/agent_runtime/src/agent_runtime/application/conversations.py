"""Conversation、Turn、历史投影和长期记忆应用服务。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from loguru import logger

from agent_runtime.application.commands import CreateRunCommand
from agent_runtime.application.runtime_service import (
    AgentDefinitionNotFound,
    AgentRuntimeConflict,
)
from agent_runtime.entities import (
    AgentConversationEntity,
    AgentConversationItemEntity,
    AgentConversationTurnEntity,
)
from platform_core.contracts import (
    ConversationItemView,
    ConversationTurnPage,
    ConversationTurnReceipt,
    ConversationTurnView,
    ConversationView,
    MemoryItemView,
    PublicTraceEvent,
)
from platform_core.identity import uuid7


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ConversationNotFound(AgentRuntimeConflict):
    def __init__(self):
        super().__init__(
            "CONVERSATION_NOT_FOUND_OR_DENIED",
            "Conversation 不存在或当前身份不可访问",
        )


class ConversationTurnNotFound(AgentRuntimeConflict):
    def __init__(self):
        super().__init__(
            "CONVERSATION_TURN_NOT_FOUND_OR_DENIED",
            "Conversation Turn 不存在或当前身份不可访问",
        )


@dataclass(frozen=True)
class MemoryRecallQuery:
    index_profile_id: UUID
    embedding: tuple[float, ...]


class MemoryRecallService:
    """在进入会话写事务前生成记忆查询向量。"""

    def __init__(self, *, uow_factory, model_client):
        self._uow_factory = uow_factory
        self._model_client = model_client

    async def prepare(
        self,
        *,
        conversation_id: UUID,
        domain_id: int,
        actor_id: str,
        query: str,
    ) -> MemoryRecallQuery | None:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_scoped(
                conversation_id=conversation_id,
                domain_id=domain_id,
                actor_id=actor_id,
            )
            if conversation is None:
                return None
            profile = await uow.memory_index_profiles.get(
                domain_id=domain_id,
                agent_id=conversation.agent_id,
            )
            if profile is None:
                return None
            profile_id = profile.index_profile_id
            model_name = profile.embedding_model_name
            dimension = int(profile.embedding_dimension)
        rows = await self._model_client.call_embedding_model(
            served_model_name=model_name,
            texts=[query],
            batch_size=1,
            is_query=True,
        )
        if len(rows) != 1 or len(rows[0].embedding) != dimension:
            raise RuntimeError("记忆查询向量数量或维度错误")
        vector = [float(item) for item in rows[0].embedding]
        magnitude = sum(item * item for item in vector) ** 0.5
        if magnitude <= 0:
            raise RuntimeError("记忆查询向量不能是零向量")
        return MemoryRecallQuery(
            index_profile_id=profile_id,
            embedding=tuple(item / magnitude for item in vector),
        )


class ConversationService:
    """管理会话事实源；Run 的执行仍交给 AgentRuntimeService。"""

    def __init__(
        self, *,
        uow_factory,
        runtime_service,
        memory_recall_service=None,
        attachment_store=None,
    ):
        self._uow_factory = uow_factory
        self._runtime_service = runtime_service
        self._memory_recall_service = memory_recall_service
        self._attachment_store = attachment_store

    async def create(
        self,
        *,
        domain_id: int,
        actor_id: str,
        agent_id: UUID,
        title: str | None,
        retention_policy: str,
    ) -> ConversationView:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get_active(
                agent_id=agent_id,
                domain_id=domain_id,
            )
            if agent is None:
                raise AgentDefinitionNotFound()
            conversation = await uow.conversations.add(
                AgentConversationEntity(
                    conversation_id=uuid7(),
                    domain_id=domain_id,
                    actor_id=actor_id,
                    agent_id=agent_id,
                    title=title.strip() if title else None,
                    status="ACTIVE",
                    retention_policy=retention_policy,
                    purge_after=None,
                    last_turn_sequence=0,
                    last_item_sequence=0,
                    last_active_at=_now(),
                )
            )
            await uow.commit()
            return self._conversation_view(conversation)

    async def list(
        self,
        *,
        domain_id: int,
        actor_id: str,
        limit: int,
    ) -> list[ConversationView]:
        async with self._uow_factory() as uow:
            rows = await uow.conversations.list_scoped(
                domain_id=domain_id,
                actor_id=actor_id,
                limit=limit,
            )
            return [self._conversation_view(row) for row in rows]

    async def get(
        self,
        *,
        conversation_id: UUID,
        domain_id: int,
        actor_id: str,
    ) -> ConversationView:
        async with self._uow_factory() as uow:
            row = await uow.conversations.get_scoped(
                conversation_id=conversation_id,
                domain_id=domain_id,
                actor_id=actor_id,
            )
            if row is None:
                raise ConversationNotFound()
            return self._conversation_view(row)

    async def update(
        self,
        *,
        conversation_id: UUID,
        domain_id: int,
        actor_id: str,
        expected_row_version: int,
        title: str | None,
        status: str | None,
        retention_policy: str | None,
    ) -> ConversationView:
        async with self._uow_factory() as uow:
            row = await uow.conversations.get_scoped(
                conversation_id=conversation_id,
                domain_id=domain_id,
                actor_id=actor_id,
                lock=True,
            )
            if row is None:
                raise ConversationNotFound()
            if int(row.row_version) != expected_row_version:
                raise AgentRuntimeConflict(
                    "CONVERSATION_VERSION_CHANGED",
                    "Conversation 已被其他请求更新，请刷新后重试",
                )
            if title is not None:
                row.title = title.strip()
            if status == "ARCHIVED":
                active = await uow.turns.find_active(
                    conversation_id=conversation_id
                )
                if active is not None:
                    raise AgentRuntimeConflict(
                        "CONVERSATION_TURN_IN_PROGRESS",
                        "Conversation 有执行中的 Turn，不能归档",
                    )
            if retention_policy is not None:
                row.retention_policy = retention_policy
            if status is not None:
                row.status = status
            if row.status == "ARCHIVED":
                row.purge_after = self._retention_deadline(
                    row.retention_policy, now=_now()
                )
            else:
                row.purge_after = None
            row.row_version = int(row.row_version) + 1
            row.last_active_at = _now()
            await uow.commit()
            return self._conversation_view(row)

    async def delete(
        self,
        *,
        conversation_id: UUID,
        domain_id: int,
        actor_id: str,
        expected_row_version: int,
    ) -> None:
        async with self._uow_factory() as uow:
            row = await uow.conversations.get_scoped(
                conversation_id=conversation_id,
                domain_id=domain_id,
                actor_id=actor_id,
                lock=True,
            )
            if row is None:
                raise ConversationNotFound()
            if int(row.row_version) != expected_row_version:
                raise AgentRuntimeConflict(
                    "CONVERSATION_VERSION_CHANGED",
                    "Conversation 已被其他请求更新，请刷新后重试",
                )
            active = await uow.turns.find_active(
                conversation_id=conversation_id
            )
            if active is not None:
                raise AgentRuntimeConflict(
                    "CONVERSATION_TURN_IN_PROGRESS",
                    "Conversation 有执行中的 Turn，不能删除",
                )
            if await uow.memory_jobs.has_processing(
                conversation_id=conversation_id
            ):
                raise AgentRuntimeConflict(
                    "CONVERSATION_MEMORY_IN_PROGRESS",
                    "Conversation 正在整理记忆，请稍后再删除",
                )
            await self._purge_locked(uow, row)
            await uow.commit()
        await self._delete_attachments(conversation_id)

    async def purge_one_due(self) -> bool:
        """清理一条到期归档会话，供保留策略 Worker 调用。"""
        now = _now()
        async with self._uow_factory() as uow:
            row = await uow.conversations.claim_due_purge(now=now)
            if row is None:
                return False
            if await uow.memory_jobs.has_processing(
                conversation_id=row.conversation_id
            ):
                row.purge_after = now + timedelta(hours=1)
                await uow.commit()
                return True
            purged_conversation_id = row.conversation_id
            await self._purge_locked(uow, row)
            await uow.commit()
        await self._delete_attachments(purged_conversation_id)
        return True

    async def _delete_attachments(self, conversation_id: UUID) -> None:
        if self._attachment_store is None:
            return
        try:
            await self._attachment_store.delete_conversation(
                conversation_id
            )
        except Exception:
            logger.exception("清理 Conversation 查询图片失败")

    @staticmethod
    async def _purge_locked(uow, row) -> None:
        memory_ids = await uow.memory_sources.memory_ids_for_conversation(
            conversation_id=row.conversation_id
        )
        await uow.memory_sources.delete_by_conversation(
            conversation_id=row.conversation_id
        )
        for memory_id in memory_ids:
            memory = await uow.memory_items.get_scoped(
                memory_id=memory_id,
                domain_id=int(row.domain_id),
                actor_id=row.actor_id,
                lock=True,
            )
            if memory is None:
                continue
            if (
                await uow.memory_sources.count_by_memory(
                    memory_id=memory_id
                )
                == 0
            ):
                memory.status = "DELETED"
                memory.valid_to = _now()
                memory.row_version = int(memory.row_version) + 1
        await uow.memory_jobs.delete_by_conversation(
            conversation_id=row.conversation_id
        )
        await uow.memory_snapshots.delete_by_conversation(
            conversation_id=row.conversation_id
        )
        await uow.conversation_items.delete_by_conversation(
            conversation_id=row.conversation_id
        )
        await uow.turns.delete_by_conversation(
            conversation_id=row.conversation_id
        )
        await uow.conversations.remove(row)

    @staticmethod
    def _retention_deadline(
        policy: str, *, now: datetime
    ) -> datetime | None:
        days_by_policy = {
            "DEFAULT": 90,
            "DAYS_30": 30,
            "DAYS_90": 90,
            "DAYS_365": 365,
        }
        days = days_by_policy.get(policy)
        return now + timedelta(days=days) if days is not None else None

    async def create_turn(
        self,
        *,
        conversation_id: UUID,
        domain_id: int,
        actor_id: str,
        request_id: str,
        trace_id: str,
        auth_context: dict[str, Any],
        idempotency_key: str,
        raw_input: str,
        expected_conversation_version: int,
        collection_ids: tuple[UUID, ...],
        security_level: int,
        client_metadata: dict[str, Any],
        budget: dict[str, Any],
        query_images: tuple[Any, ...] = (),
    ) -> ConversationTurnReceipt:
        image_descriptors: tuple[dict[str, Any], ...] = ()
        if query_images:
            if self._attachment_store is None:
                raise RuntimeError("Conversation 查询附件存储未初始化")
            image_descriptors = await self._attachment_store.put_images(
                conversation_id=conversation_id, images=query_images
            )
        raw_hash = _hash(
            {
                "input": raw_input,
                "query_image_hashes": [
                    item["content_sha256"] for item in image_descriptors
                ],
            }
        )
        recall_query = None
        if self._memory_recall_service is not None:
            try:
                recall_query = await self._memory_recall_service.prepare(
                    conversation_id=conversation_id,
                    domain_id=domain_id,
                    actor_id=actor_id,
                    query=raw_input,
                )
            except Exception:
                logger.exception("记忆向量召回准备失败，本轮降级为词法召回")
        accepted = await self._accept_turn(
            conversation_id=conversation_id,
            domain_id=domain_id,
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            raw_input=raw_input,
            raw_hash=raw_hash,
            expected_conversation_version=expected_conversation_version,
            recall_query=recall_query,
            image_descriptors=image_descriptors,
        )
        turn, conversation = accepted
        if turn.raw_input_hash != raw_hash:
            raise AgentRuntimeConflict(
                "IDEMPOTENCY_CONFLICT",
                "相同 Idempotency-Key 对应的 Turn 输入不同",
            )
        if turn.root_run_id is not None:
            run = await self._runtime_service.get_run(
                run_id=turn.root_run_id,
                domain_id=domain_id,
            )
            return self._turn_receipt(turn, run)

        try:
            run_receipt = await self._runtime_service.create_run(
                CreateRunCommand(
                    domain_id=domain_id,
                    agent_id=conversation.agent_id,
                    actor_id=actor_id,
                    request_id=request_id,
                    trace_id=trace_id,
                    idempotency_key=f"turn:{turn.turn_id}",
                    original_input=raw_input,
                    collection_ids=collection_ids,
                    security_level=security_level,
                    client_metadata={
                        **client_metadata,
                        "query_images": list(image_descriptors),
                    },
                    policy_snapshot={"auth_context": auth_context},
                    budget=budget,
                    conversation_id=conversation_id,
                    turn_id=turn.turn_id,
                    conversation_context=dict(turn.context_snapshot_json),
                )
            )
        except Exception:
            await self._mark_turn_failed(turn_id=turn.turn_id)
            raise
        async with self._uow_factory() as uow:
            locked = await uow.turns.get(turn_id=turn.turn_id, lock=True)
            if locked is None:
                raise ConversationTurnNotFound()
            if (
                locked.root_run_id is not None
                and locked.root_run_id != run_receipt.run_id
            ):
                raise AgentRuntimeConflict(
                    "CONVERSATION_RUN_CONFLICT",
                    "Turn 已绑定不同 Root Run",
                )
            locked.root_run_id = run_receipt.run_id
            locked.status = "RUNNING"
            locked.started_at = _now()
            await uow.commit()
            return ConversationTurnReceipt(
                conversation_id=conversation_id,
                turn_id=locked.turn_id,
                turn_sequence=int(locked.turn_sequence),
                turn_status=locked.status,
                run_id=run_receipt.run_id,
                run_status=run_receipt.status,
                event_cursor=run_receipt.event_cursor,
                events_url=run_receipt.events_url,
            )

    async def _mark_turn_failed(self, *, turn_id: UUID) -> None:
        async with self._uow_factory() as uow:
            turn = await uow.turns.get(turn_id=turn_id, lock=True)
            if turn is None or turn.root_run_id is not None:
                return
            if turn.status in {"ACCEPTED", "RUNNING"}:
                turn.status = "FAILED"
                turn.completed_at = _now()
            await uow.commit()

    async def _accept_turn(
        self,
        *,
        conversation_id: UUID,
        domain_id: int,
        actor_id: str,
        idempotency_key: str,
        raw_input: str,
        raw_hash: str,
        expected_conversation_version: int,
        recall_query: MemoryRecallQuery | None = None,
        image_descriptors: tuple[dict[str, Any], ...] = (),
    ) -> tuple[AgentConversationTurnEntity, AgentConversationEntity]:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_scoped(
                conversation_id=conversation_id,
                domain_id=domain_id,
                actor_id=actor_id,
                lock=True,
            )
            if conversation is None:
                raise ConversationNotFound()
            existing = await uow.turns.get_by_idempotency(
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
            )
            if existing is not None:
                return existing, conversation
            if int(conversation.row_version) != expected_conversation_version:
                raise AgentRuntimeConflict(
                    "CONVERSATION_VERSION_CHANGED",
                    "Conversation 已被其他请求更新，请刷新后重试",
                )
            if conversation.status != "ACTIVE":
                raise AgentRuntimeConflict(
                    "CONVERSATION_NOT_ACTIVE",
                    "Conversation 当前不能创建新 Turn",
                )
            active = await uow.turns.find_active(
                conversation_id=conversation_id
            )
            if active is not None:
                raise AgentRuntimeConflict(
                    "CONVERSATION_TURN_IN_PROGRESS",
                    "Conversation 已有尚未结束的 Turn",
                )
            recent = await uow.conversation_items.list_recent(
                conversation_id=conversation_id, limit=12
            )
            snapshot = await uow.memory_snapshots.get_active(
                conversation_id=conversation_id
            )
            memory_candidates = await uow.memory_items.list_active(
                domain_id=domain_id,
                actor_id=actor_id,
                agent_id=conversation.agent_id,
                now=_now(),
                limit=100,
            )
            memories = self._select_memories(
                raw_input,
                memory_candidates,
                limit=20,
                recall_query=recall_query,
            )
            context_snapshot = {
                "summary": (
                    dict(snapshot.summary_json) if snapshot else {}
                ),
                "summary_ref": (
                    {
                        "snapshot_id": str(snapshot.snapshot_id),
                        "covered_turn_sequence": int(
                            snapshot.covered_turn_sequence
                        ),
                    }
                    if snapshot
                    else None
                ),
                "recent_items": [
                    {
                        "item_id": str(item.item_id),
                        "role": item.role,
                        "content": item.content_json,
                        "item_sequence": int(item.item_sequence),
                    }
                    for item in recent
                ],
                "memories": [
                    {
                        "memory_id": str(item.memory_id),
                        "memory_type": item.memory_type,
                        "canonical_key": item.canonical_key,
                        "value": item.value_json,
                        "confidence": float(item.confidence),
                        "salience": float(item.salience),
                        "scope_type": item.scope_type,
                    }
                    for item in memories
                ],
                "query_images": list(image_descriptors),
            }
            turn_sequence = int(conversation.last_turn_sequence) + 1
            item_sequence = int(conversation.last_item_sequence) + 1
            turn = await uow.turns.add(
                AgentConversationTurnEntity(
                    turn_id=uuid7(),
                    conversation_id=conversation_id,
                    turn_sequence=turn_sequence,
                    status="ACCEPTED",
                    raw_input_hash=raw_hash,
                    context_snapshot_json=context_snapshot,
                    idempotency_key=idempotency_key,
                )
            )
            user_item = await uow.conversation_items.add(
                AgentConversationItemEntity(
                    item_id=uuid7(),
                    conversation_id=conversation_id,
                    item_sequence=item_sequence,
                    turn_id=turn.turn_id,
                    item_type="MESSAGE",
                    role="USER",
                    content_json={
                        "text": raw_input,
                        "images": list(image_descriptors),
                    },
                    content_hash=_hash(
                        {
                            "text": raw_input,
                            "images": list(image_descriptors),
                        }
                    ),
                    visibility="USER",
                )
            )
            turn.user_item_id = user_item.item_id
            conversation.last_turn_sequence = turn_sequence
            conversation.last_item_sequence = item_sequence
            conversation.last_active_at = _now()
            conversation.row_version = int(conversation.row_version) + 1
            if not conversation.title:
                conversation.title = raw_input.strip()[:100]
            await uow.commit()
            return turn, conversation

    @classmethod
    def _select_memories(
        cls,
        query: str,
        candidates: list,
        *,
        limit: int,
        recall_query: MemoryRecallQuery | None = None,
    ) -> list:
        """以字词重合和显著性选择与本轮相关的结构化记忆。"""
        query_terms = cls._memory_terms(query)
        ranked: list[tuple[float, Any]] = []
        for item in candidates:
            searchable = " ".join(
                (
                    item.canonical_key,
                    str(item.search_text or ""),
                    json.dumps(
                        item.value_json,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                )
            )
            terms = cls._memory_terms(searchable)
            overlap = (
                len(query_terms & terms) / max(1, len(query_terms))
            )
            vector_score = None
            if (
                recall_query is not None
                and item.index_profile_id == recall_query.index_profile_id
                and item.embedding is not None
            ):
                candidate_vector = [
                    float(value) for value in item.embedding
                ]
                if len(candidate_vector) == len(recall_query.embedding):
                    vector_score = sum(
                        left * right
                        for left, right in zip(
                            recall_query.embedding,
                            candidate_vector,
                            strict=True,
                        )
                    )
                    vector_score = max(0.0, min(1.0, vector_score))
            if vector_score is None:
                score = overlap * 0.75 + float(item.salience) * 0.25
            else:
                score = (
                    vector_score * 0.60
                    + overlap * 0.25
                    + float(item.salience) * 0.15
                )
            ranked.append((score, item))
        ranked.sort(
            key=lambda value: (
                value[0],
                float(value[1].salience),
                str(value[1].updated_at or ""),
            ),
            reverse=True,
        )
        relevant = [
            item
            for score, item in ranked
            if score > 0.15 or len(ranked) <= limit
        ]
        return relevant[:limit]

    @staticmethod
    def _memory_terms(value: str) -> set[str]:
        normalized = value.casefold()
        words = set(re.findall(r"[a-z0-9_.-]{2,}", normalized))
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        words.update(
            chinese[index : index + 2]
            for index in range(max(0, len(chinese) - 1))
        )
        return words

    async def list_turns(
        self,
        *,
        conversation_id: UUID,
        domain_id: int,
        actor_id: str,
        after_sequence: int,
        limit: int,
    ) -> ConversationTurnPage:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_scoped(
                conversation_id=conversation_id,
                domain_id=domain_id,
                actor_id=actor_id,
            )
            if conversation is None:
                raise ConversationNotFound()
            turns = await uow.turns.list_by_conversation(
                conversation_id=conversation_id,
                after_sequence=after_sequence,
                limit=limit,
            )
            items = await uow.conversation_items.list_by_turn_ids(
                turn_ids=[item.turn_id for item in turns]
            )
            jobs = await uow.memory_jobs.list_by_turn_ids(
                turn_ids=[item.turn_id for item in turns]
            )
            item_by_id = {item.item_id: item for item in items}
            job_by_turn = {item.turn_id: item for item in jobs}
            views: list[ConversationTurnView] = []
            for turn in turns:
                traces = await self._trace_for_turn(uow, turn, limit=20)
                views.append(
                    ConversationTurnView(
                        conversation_id=conversation_id,
                        turn_id=turn.turn_id,
                        turn_sequence=int(turn.turn_sequence),
                        status=turn.status,
                        run_id=turn.root_run_id,
                        user_item=self._item_view(
                            item_by_id.get(turn.user_item_id)
                        ),
                        assistant_item=self._item_view(
                            item_by_id.get(turn.assistant_item_id)
                        ),
                        trace_summary=tuple(traces),
                        created_at=turn.created_at,
                        completed_at=turn.completed_at,
                        memory_status=(
                            job_by_turn[turn.turn_id].status
                            if turn.turn_id in job_by_turn
                            else None
                        ),
                    )
                )
            next_sequence = (
                int(turns[-1].turn_sequence)
                if turns
                else after_sequence
            )
            return ConversationTurnPage(
                conversation_id=conversation_id,
                turns=tuple(views),
                next_sequence=next_sequence,
            )

    async def list_trace(
        self,
        *,
        conversation_id: UUID,
        turn_id: UUID,
        domain_id: int,
        actor_id: str,
        after_sequence: int,
        limit: int,
    ) -> list[PublicTraceEvent]:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_scoped(
                conversation_id=conversation_id,
                domain_id=domain_id,
                actor_id=actor_id,
            )
            turn = await uow.turns.get(turn_id=turn_id)
            if (
                conversation is None
                or turn is None
                or turn.conversation_id != conversation_id
            ):
                raise ConversationTurnNotFound()
            return await self._trace_for_turn(
                uow,
                turn,
                after_sequence=after_sequence,
                limit=limit,
            )

    async def list_memories(
        self,
        *,
        domain_id: int,
        actor_id: str,
        agent_id: UUID,
        limit: int,
    ) -> list[MemoryItemView]:
        async with self._uow_factory() as uow:
            rows = await uow.memory_items.list_active(
                domain_id=domain_id,
                actor_id=actor_id,
                agent_id=agent_id,
                now=_now(),
                limit=limit,
            )
            return [self._memory_view(row) for row in rows]

    async def forget_memory(
        self,
        *,
        memory_id: UUID,
        domain_id: int,
        actor_id: str,
    ) -> None:
        async with self._uow_factory() as uow:
            row = await uow.memory_items.get_scoped(
                memory_id=memory_id,
                domain_id=domain_id,
                actor_id=actor_id,
                lock=True,
            )
            if row is None:
                raise ConversationNotFound()
            row.status = "DELETED"
            row.valid_to = _now()
            row.row_version = int(row.row_version) + 1
            await uow.memory_sources.delete_by_memory(
                memory_id=memory_id
            )
            await uow.commit()

    async def _trace_for_turn(
        self,
        uow,
        turn: AgentConversationTurnEntity,
        *,
        after_sequence: int = 0,
        limit: int,
    ) -> list[PublicTraceEvent]:
        if turn.root_run_id is None:
            return []
        rows = await uow.events.list_after(
            run_id=turn.root_run_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        result: list[PublicTraceEvent] = []
        for row in rows:
            projected = self._public_trace(turn.turn_id, row)
            if projected is not None:
                result.append(projected)
        return result

    @staticmethod
    def _public_trace(turn_id, row) -> PublicTraceEvent | None:
        mapping = {
            "RUN_CREATED": ("planning", "已接收请求", "RUNNING"),
            "RUN_STARTED": ("planning", "已生成执行计划", "COMPLETED"),
            "TASK_READY": ("planning", "执行步骤已就绪", "READY"),
            "TASK_STARTED": ("skill", "开始执行 Skill", "RUNNING"),
            "TASK_COMPLETED": ("skill", "Skill 执行完成", "COMPLETED"),
            "TASK_RETRYING": ("skill", "Skill 准备重试", "RETRYING"),
            "TASK_FAILED": ("skill", "Skill 执行失败", "FAILED"),
            "query.rewritten": (
                "context",
                "已完成上下文理解",
                "COMPLETED",
            ),
            "memory.context_loaded": (
                "memory",
                "已加载会话上下文",
                "COMPLETED",
            ),
            "skill.started": (
                "skill",
                "开始执行 Skill",
                "RUNNING",
            ),
            "retrieval.completed": (
                "retrieval",
                "已完成知识检索",
                "COMPLETED",
            ),
            "data.query.completed": (
                "data",
                "已完成结构化数据查询",
                "COMPLETED",
            ),
            "chart.completed": (
                "visualization",
                "已生成数据图表",
                "COMPLETED",
            ),
            "thinking.delta": (
                "thinking",
                "正在组织回答",
                "RUNNING",
            ),
            "answer.completed": (
                "answer",
                "回答与引用已生成",
                "COMPLETED",
            ),
            "RUN_COMPLETED": ("answer", "回答生成完成", "COMPLETED"),
            "RUN_FAILED": ("answer", "本轮处理失败", "FAILED"),
            "RUN_CANCELLED": ("answer", "本轮处理已取消", "CANCELLED"),
            "delegation.submitting": (
                "delegation",
                "正在提交子 Agent",
                "RUNNING",
            ),
            "delegation.started": (
                "delegation",
                "子 Agent 已开始",
                "RUNNING",
            ),
            "delegation.completed": (
                "delegation",
                "子 Agent 已完成",
                "COMPLETED",
            ),
        }
        selected = mapping.get(row.event_type)
        if selected is None:
            return None
        stage, title, status = selected
        payload = dict(row.event_payload_json or {})
        summary = str(
            payload.get("public_summary")
            or payload.get("task_key")
            or title
        )[:1000]
        return PublicTraceEvent(
            run_id=row.run_id,
            turn_id=turn_id,
            task_id=row.task_id,
            sequence_no=int(row.sequence_no),
            stage=stage,
            title=title,
            summary=summary,
            status=status,
            resource_refs=(),
            occurred_at=row.created_at,
        )

    @staticmethod
    def _conversation_view(row) -> ConversationView:
        return ConversationView(
            conversation_id=row.conversation_id,
            agent_id=row.agent_id,
            title=row.title,
            status=row.status,
            row_version=int(row.row_version),
            last_turn_sequence=int(row.last_turn_sequence),
            last_active_at=row.last_active_at,
            created_at=row.created_at,
            retention_policy=row.retention_policy,
            purge_after=row.purge_after,
        )

    @staticmethod
    def _item_view(row) -> ConversationItemView | None:
        if row is None:
            return None
        return ConversationItemView(
            item_id=row.item_id,
            item_sequence=int(row.item_sequence),
            item_type=row.item_type,
            role=row.role,
            content=dict(row.content_json),
            run_id=row.run_id,
            artifact_id=row.artifact_id,
            created_at=row.created_at,
        )

    @staticmethod
    def _memory_view(row) -> MemoryItemView:
        return MemoryItemView(
            memory_id=row.memory_id,
            agent_id=row.agent_id,
            memory_type=row.memory_type,
            scope_type=row.scope_type,
            canonical_key=row.canonical_key,
            value=dict(row.value_json),
            confidence=float(row.confidence),
            salience=float(row.salience),
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _turn_receipt(turn, run) -> ConversationTurnReceipt:
        return ConversationTurnReceipt(
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
            turn_sequence=int(turn.turn_sequence),
            turn_status=turn.status,
            run_id=turn.root_run_id,
            run_status=run.status,
            event_cursor=run.event_cursor,
            events_url=f"/api/v1/runs/{turn.root_run_id}/events",
        )
