"""异步会话摘要与长期记忆归并 Worker。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypeVar
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_runtime.entities import (
    AgentMemoryIndexProfileEntity,
    AgentMemoryItemEntity,
    AgentMemorySnapshotEntity,
    AgentMemorySourceEntity,
)
from platform_core.identity import uuid7
from platform_core.prompts import StrictPromptRenderer


_ModelOutput = TypeVar("_ModelOutput", bound=BaseModel)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ConversationSnapshotOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_topic: str | None = Field(default=None, max_length=512)
    user_goal: str | None = Field(default=None, max_length=2000)
    entities: tuple[dict[str, Any], ...] = ()
    corrections: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_type: Literal["USER_FACT", "USER_PREFERENCE"]
    canonical_key: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[a-z][a-z0-9_.-]*$",
    )
    value: dict[str, Any]
    search_text: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0, le=1)
    salience: float = Field(ge=0, le=1)


class MemoryCandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: tuple[MemoryCandidate, ...] = Field(
        default=(), max_length=20
    )
    forget_keys: tuple[
        str, ...
    ] = Field(default=(), max_length=20)


class MemoryConflictDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "ADD", "CONFIRM", "SUPERSEDE", "DISPUTE", "IGNORE"
    ]
    reason: str = Field(min_length=1, max_length=1000)


@dataclass(frozen=True)
class ResolvedMemoryDecision:
    candidate: MemoryCandidate
    action: Literal[
        "ADD", "CONFIRM", "SUPERSEDE", "DISPUTE", "IGNORE"
    ]
    reason: str
    existing_memory_id: UUID | None
    prompt_ref: dict[str, Any] | None
    scope_type: Literal["USER_AGENT", "USER_SHARED"]
    embedding: list[float] | None = None


@dataclass(frozen=True)
class MemoryRuntimeConfig:
    llm_model_name: str
    embedding_enabled: bool
    embedding_model_name: str | None
    shared_keys: frozenset[str]
    episodic_enabled: bool


@dataclass(frozen=True)
class MemoryIndexProfile:
    index_profile_id: UUID
    model_name: str
    dimension: int


@dataclass(frozen=True)
class MemoryJobLease:
    job_id: UUID
    lease_token: UUID
    attempt_count: int
    max_attempts: int
    conversation_id: UUID
    turn_id: UUID
    turn_sequence: int
    domain_id: int
    actor_id: str
    agent_id: UUID
    user_item_id: UUID
    user_message: str
    assistant_message: dict[str, Any]
    previous_summary: dict[str, Any]
    existing_memories: tuple[dict[str, Any], ...]


class MemoryConsolidationWorker:
    """以数据库租约异步生成摘要和可溯源长期记忆。"""

    def __init__(
        self,
        *,
        uow_factory,
        model_client,
        prompt_resolver,
        worker_id: str,
        poll_interval_seconds: float,
        lease_seconds: int = 180,
        embedding_dimension: int = 1536,
        model_resolver=None,
    ):
        self._uow_factory = uow_factory
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver
        self._worker_id = worker_id
        self._poll_interval_seconds = poll_interval_seconds
        self._lease_seconds = lease_seconds
        self._embedding_dimension = embedding_dimension
        self._model_resolver = model_resolver
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    async def run_forever(self) -> None:
        logger.info("长期记忆归并 Worker 已启动：{}", self._worker_id)
        while not self._stop_event.is_set():
            worked = await self.run_once()
            if not worked:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._poll_interval_seconds,
                    )
                except TimeoutError:
                    pass
        logger.info("长期记忆归并 Worker 已停止：{}", self._worker_id)

    async def run_once(self) -> bool:
        try:
            lease = await self._claim()
        except Exception:
            logger.exception("领取长期记忆任务失败")
            return False
        if lease is None:
            return False
        try:
            snapshot_prompt = await self._prompt_resolver.resolve(
                "agent_runtime.conversation_snapshot"
            )
            memory_prompt = await self._prompt_resolver.resolve(
                "agent_runtime.memory_extract"
            )
            runtime_config = await self._runtime_config(lease)
            model_name = runtime_config.llm_model_name
            snapshot_request = StrictPromptRenderer.render(
                snapshot_prompt,
                {
                    "previous_summary": lease.previous_summary,
                    "new_turns": {
                        "user": lease.user_message,
                        "assistant": lease.assistant_message,
                    },
                },
            )
            memory_request = StrictPromptRenderer.render(
                memory_prompt,
                {
                    "user_message": lease.user_message,
                    "existing_memories": lease.existing_memories,
                },
            )
            snapshot_raw, memory_raw = await asyncio.gather(
                self._model_client.get_llm_json(
                    served_model_name=model_name,
                    prompt=snapshot_request,
                    max_tokens=2048,
                ),
                self._model_client.get_llm_json(
                    served_model_name=model_name,
                    prompt=memory_request,
                    max_tokens=2048,
                ),
            )
            snapshot = await self._validate_model_output(
                model_type=ConversationSnapshotOutput,
                response=snapshot_raw,
                model_name=model_name,
                prompt_version=snapshot_prompt.version,
                rendered_prompt=snapshot_request,
                output_name="会话摘要",
                correction_instruction=(
                    "active_topic 和 user_goal 必须是字符串或 null；"
                    "entities、corrections、unresolved_questions 必须是数组。"
                ),
            )
            candidates = await self._validate_model_output(
                model_type=MemoryCandidateBatch,
                response=memory_raw,
                model_name=model_name,
                prompt_version=memory_prompt.version,
                rendered_prompt=memory_request,
                output_name="长期记忆候选",
                correction_instruction=(
                    "顶层只能包含 candidates 和 forget_keys，且两者必须是数组；"
                    "每个 candidate 必须完整满足已声明字段及类型。"
                ),
            )
            safe_candidates = self._safe_candidates(candidates)
            forget_keys = self._safe_forget_keys(candidates.forget_keys)
            safe_candidates = tuple(
                item
                for item in safe_candidates
                if item.canonical_key not in forget_keys
            )
            decisions = await self._decide_candidates(
                lease,
                candidates=safe_candidates,
                model_name=model_name,
                shared_keys=runtime_config.shared_keys,
            )
            profile = await self._ensure_index_profile(
                lease, runtime_config=runtime_config
            )
            decisions = await self._embed_decisions(
                decisions, profile=profile
            )
            episode_text = (
                self._episode_search_text(lease)
                if runtime_config.episodic_enabled
                else None
            )
            episode_embedding = (
                await self._embedding_for_text(
                    episode_text, profile=profile
                )
                if episode_text is not None
                else None
            )
            await self._complete(
                lease,
                snapshot=snapshot,
                extracted_candidate_count=len(candidates.candidates),
                decisions=decisions,
                forget_keys=forget_keys,
                snapshot_prompt=snapshot_prompt,
                memory_prompt=memory_prompt,
                model_name=model_name,
                index_profile=profile,
                episode_text=episode_text,
                episode_embedding=episode_embedding,
            )
        except Exception as exc:
            logger.exception(
                "长期记忆归并失败：job_id={} error={}",
                lease.job_id,
                type(exc).__name__,
            )
            await self._fail(lease, exc)
        return True

    async def _validate_model_output(
        self,
        *,
        model_type: type[_ModelOutput],
        response: Any,
        model_name: str,
        prompt_version: str,
        rendered_prompt: str,
        output_name: str,
        correction_instruction: str,
    ) -> _ModelOutput:
        """严格校验模型结构，失败时仅允许模型修正一次格式。"""
        try:
            return model_type.model_validate(response)
        except ValidationError as exc:
            logger.warning(
                "{}模型输出不符合契约，准备执行一次格式修正 "
                "| model={} | prompt_version={} | shape={} | errors={}",
                output_name,
                model_name,
                prompt_version,
                self._response_shape(response),
                self._validation_summary(exc),
            )
        corrected = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=(
                rendered_prompt
                + "\n\n上一份输出未通过字段校验。"
                "请仅重新输出满足原 Schema 的 JSON 对象；"
                + correction_instruction
            ),
            max_tokens=2048,
        )
        try:
            return model_type.model_validate(corrected)
        except ValidationError as exc:
            logger.error(
                "{}模型格式修正后仍不符合契约 "
                "| model={} | prompt_version={} | shape={} | errors={}",
                output_name,
                model_name,
                prompt_version,
                self._response_shape(corrected),
                self._validation_summary(exc),
            )
            raise

    @staticmethod
    def _response_shape(response: Any) -> dict[str, Any]:
        """只记录字段类型，避免把用户内容写入运行日志。"""
        if not isinstance(response, dict):
            return {"response_type": type(response).__name__}
        return {
            "response_type": "dict",
            "fields": {
                str(key): type(value).__name__
                for key, value in response.items()
            },
        }

    @staticmethod
    def _validation_summary(exc: ValidationError) -> list[str]:
        """提取不含模型原始值的 Pydantic 错误摘要。"""
        return [
            "{}:{}".format(
                ".".join(str(part) for part in error["loc"]),
                error["type"],
            )
            for error in exc.errors(include_input=False)
        ]

    async def _claim(self) -> MemoryJobLease | None:
        now = _now()
        token = uuid7()
        async with self._uow_factory() as uow:
            job = await uow.memory_jobs.claim(
                worker_id=self._worker_id,
                lease_token=token,
                now=now,
                lease_until=now + timedelta(seconds=self._lease_seconds),
            )
            if job is None:
                return None
            conversation = await uow.conversations.get(
                conversation_id=job.conversation_id
            )
            turn = await uow.turns.get(turn_id=job.turn_id)
            if (
                conversation is None
                or turn is None
                or turn.user_item_id is None
            ):
                raise RuntimeError("记忆任务关联的会话事实不完整")
            items = await uow.conversation_items.list_by_turn_ids(
                turn_ids=[turn.turn_id]
            )
            user_item = next(
                (item for item in items if item.item_id == turn.user_item_id),
                None,
            )
            assistant_item = next(
                (
                    item
                    for item in items
                    if item.item_id == turn.assistant_item_id
                ),
                None,
            )
            if user_item is None or assistant_item is None:
                raise RuntimeError("记忆任务缺少完整的用户或助手消息")
            snapshot = await uow.memory_snapshots.get_active(
                conversation_id=conversation.conversation_id
            )
            memories = await uow.memory_items.list_active(
                domain_id=int(conversation.domain_id),
                actor_id=conversation.actor_id,
                agent_id=conversation.agent_id,
                now=now,
                limit=100,
            )
            lease = MemoryJobLease(
                job_id=job.memory_job_id,
                lease_token=token,
                attempt_count=int(job.attempt_count),
                max_attempts=int(job.max_attempts),
                conversation_id=conversation.conversation_id,
                turn_id=turn.turn_id,
                turn_sequence=int(turn.turn_sequence),
                domain_id=int(conversation.domain_id),
                actor_id=conversation.actor_id,
                agent_id=conversation.agent_id,
                user_item_id=user_item.item_id,
                user_message=str(user_item.content_json.get("text") or ""),
                assistant_message=dict(assistant_item.content_json),
                previous_summary=(
                    dict(snapshot.summary_json) if snapshot else {}
                ),
                existing_memories=tuple(
                    {
                        "memory_id": str(item.memory_id),
                        "memory_type": item.memory_type,
                        "canonical_key": item.canonical_key,
                        "value": item.value_json,
                        "scope_type": item.scope_type,
                        "agent_id": (
                            str(item.agent_id) if item.agent_id else None
                        ),
                    }
                    for item in memories
                ),
            )
            await uow.commit()
            return lease

    async def _runtime_config(
        self, lease: MemoryJobLease
    ) -> MemoryRuntimeConfig:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get_active(
                agent_id=lease.agent_id,
                domain_id=lease.domain_id,
            )
            if agent is None:
                raise RuntimeError("记忆任务对应 Agent 已不可用")
            models = dict(agent.models_json or {})
            raw = dict(agent.config_json or {}).get("memory") or {}
            if not isinstance(raw, dict):
                raise RuntimeError("Agent memory 配置必须是对象")
            shared_keys = frozenset(
                str(item).strip()
                for item in raw.get("shared_keys", [])
                if str(item).strip()
            )
            invalid = [
                item
                for item in shared_keys
                if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,255}", item)
            ]
            if invalid:
                raise RuntimeError("memory.shared_keys 包含非法键")
            episodic_enabled = bool(raw.get("episodic_enabled", True))
        if self._model_resolver is None:
            raise RuntimeError("Agent 模型目录解析器尚未初始化")
        resolved = await self._model_resolver.resolve(
            models, roles={"memory_llm", "memory_embedding"}
        )
        llm_model_name = str(
            resolved.get("memory_llm", {}).get("served_model_name") or ""
        ).strip()
        embedding_model_name = str(
            resolved.get("memory_embedding", {}).get(
                "served_model_name"
            )
            or ""
        ).strip()
        if not llm_model_name or not embedding_model_name:
            raise RuntimeError("Agent 记忆模型配置不完整")
        return MemoryRuntimeConfig(
            llm_model_name=llm_model_name,
            embedding_enabled=True,
            embedding_model_name=embedding_model_name,
            shared_keys=shared_keys,
            episodic_enabled=episodic_enabled,
        )

    async def _ensure_index_profile(
        self,
        lease: MemoryJobLease,
        *,
        runtime_config: MemoryRuntimeConfig,
    ) -> MemoryIndexProfile | None:
        """建立不可变索引画像；配置漂移时拒绝混用旧向量。"""
        if not runtime_config.embedding_enabled:
            return None
        model_name = runtime_config.embedding_model_name
        if model_name is None:
            raise RuntimeError("记忆向量模型配置不完整")
        config_hash = _hash(
            {
                "embedding_model_name": model_name,
                "embedding_dimension": self._embedding_dimension,
                "normalization": "L2",
            }
        )
        async with self._uow_factory() as uow:
            active = await uow.memory_index_profiles.get(
                domain_id=lease.domain_id,
                agent_id=lease.agent_id,
                lock=True,
            )
            if active is not None:
                if active.config_sha256 != config_hash:
                    raise RuntimeError(
                        "记忆 Embedding 模型或维度一经设定禁止更换"
                    )
                return MemoryIndexProfile(
                    index_profile_id=active.index_profile_id,
                    model_name=active.embedding_model_name,
                    dimension=int(active.embedding_dimension),
                )
            active = await uow.memory_index_profiles.add(
                AgentMemoryIndexProfileEntity(
                    index_profile_id=uuid7(),
                    domain_id=lease.domain_id,
                    agent_id=lease.agent_id,
                    embedding_model_name=model_name,
                    embedding_dimension=self._embedding_dimension,
                    normalization="L2",
                    config_sha256=config_hash,
                )
            )
            await uow.commit()
            return MemoryIndexProfile(
                index_profile_id=active.index_profile_id,
                model_name=active.embedding_model_name,
                dimension=int(active.embedding_dimension),
            )

    async def _embed_decisions(
        self,
        decisions: tuple[ResolvedMemoryDecision, ...],
        *,
        profile: MemoryIndexProfile | None,
    ) -> tuple[ResolvedMemoryDecision, ...]:
        if profile is None:
            return decisions
        targets = [
            item
            for item in decisions
            if item.action in {"ADD", "SUPERSEDE", "DISPUTE"}
        ]
        if not targets:
            return decisions
        vectors = await self._model_client.call_embedding_model(
            served_model_name=profile.model_name,
            texts=[item.candidate.search_text for item in targets],
            batch_size=len(targets),
            is_query=False,
        )
        if len(vectors) != len(targets):
            raise RuntimeError("记忆向量服务返回数量与请求不一致")
        by_candidate = {
            id(item): self._validate_embedding(
                vector.embedding, dimension=profile.dimension
            )
            for item, vector in zip(targets, vectors, strict=True)
        }
        return tuple(
            replace(item, embedding=by_candidate.get(id(item)))
            for item in decisions
        )

    async def _embedding_for_text(
        self,
        text: str,
        *,
        profile: MemoryIndexProfile | None,
    ) -> list[float] | None:
        if profile is None:
            return None
        rows = await self._model_client.call_embedding_model(
            served_model_name=profile.model_name,
            texts=[text],
            batch_size=1,
            is_query=False,
        )
        if len(rows) != 1:
            raise RuntimeError("情景记忆向量服务必须返回一个结果")
        return self._validate_embedding(
            rows[0].embedding, dimension=profile.dimension
        )

    @staticmethod
    def _validate_embedding(
        value: list[float], *, dimension: int
    ) -> list[float]:
        vector = [float(item) for item in value]
        if len(vector) != dimension:
            raise RuntimeError(
                f"记忆向量维度错误：期望 {dimension}，实际 {len(vector)}"
            )
        magnitude = sum(item * item for item in vector) ** 0.5
        if magnitude <= 0:
            raise RuntimeError("记忆向量不能是零向量")
        return [item / magnitude for item in vector]

    @staticmethod
    def _episode_search_text(lease: MemoryJobLease) -> str:
        assistant = str(
            lease.assistant_message.get("text")
            or lease.assistant_message.get("answer")
            or ""
        )
        return (
            f"用户：{lease.user_message}\n助手：{assistant}"
        )[:4000]

    async def _complete(
        self,
        lease: MemoryJobLease,
        *,
        snapshot: ConversationSnapshotOutput,
        extracted_candidate_count: int,
        decisions: tuple[ResolvedMemoryDecision, ...],
        forget_keys: tuple[str, ...],
        snapshot_prompt,
        memory_prompt,
        model_name: str,
        index_profile: MemoryIndexProfile | None,
        episode_text: str | None,
        episode_embedding: list[float] | None,
    ) -> None:
        now = _now()
        async with self._uow_factory() as uow:
            job = await uow.memory_jobs.get(
                memory_job_id=lease.job_id, lock=True
            )
            if (
                job is None
                or job.status != "PROCESSING"
                or job.lease_token != lease.lease_token
                or job.lease_until is None
                or _utc(job.lease_until) <= now
            ):
                raise RuntimeError("长期记忆任务租约已失效")
            await uow.memory_snapshots.supersede_active(
                conversation_id=lease.conversation_id
            )
            summary_json = snapshot.model_dump(mode="json")
            await uow.memory_snapshots.add(
                AgentMemorySnapshotEntity(
                    snapshot_id=uuid7(),
                    conversation_id=lease.conversation_id,
                    covered_turn_sequence=lease.turn_sequence,
                    summary_json=summary_json,
                    source_hash=_hash(
                        {
                            "previous": lease.previous_summary,
                            "user": lease.user_message,
                            "assistant": lease.assistant_message,
                        }
                    ),
                    model_name=model_name,
                    prompt_key=snapshot_prompt.prompt_key,
                    prompt_version=snapshot_prompt.version,
                    prompt_sha256=snapshot_prompt.sha256,
                    status="ACTIVE",
                )
            )
            changed = 0
            confirmed = 0
            disputed = 0
            ignored = 0
            forgotten = 0
            decision_results: list[dict[str, Any]] = []
            for canonical_key in forget_keys:
                rows = []
                for scoped_agent_id in (lease.agent_id, None):
                    rows.extend(
                        await uow.memory_items.list_active_by_canonical_key(
                            domain_id=lease.domain_id,
                            actor_id=lease.actor_id,
                            agent_id=scoped_agent_id,
                            canonical_key=canonical_key,
                            lock=True,
                        )
                    )
                for memory in rows:
                    memory.status = "DELETED"
                    memory.valid_to = now
                    memory.row_version = int(memory.row_version) + 1
                    await uow.memory_sources.delete_by_memory(
                        memory_id=memory.memory_id
                    )
                    forgotten += 1
            for decision in decisions:
                candidate = decision.candidate
                scoped_agent_id = (
                    None
                    if decision.scope_type == "USER_SHARED"
                    else lease.agent_id
                )
                memory = await uow.memory_items.get_active_by_key(
                    domain_id=lease.domain_id,
                    actor_id=lease.actor_id,
                    agent_id=scoped_agent_id,
                    memory_type=candidate.memory_type,
                    canonical_key=candidate.canonical_key,
                    lock=True,
                )
                actual_memory_id = memory.memory_id if memory else None
                if actual_memory_id != decision.existing_memory_id:
                    raise RuntimeError(
                        "记忆冲突判定后有效版本发生变化，必须重新计算"
                    )
                target_memory = None
                if decision.action == "IGNORE":
                    ignored += 1
                elif decision.action == "CONFIRM":
                    if memory is None or _hash(
                        memory.value_json
                    ) != _hash(candidate.value):
                        raise RuntimeError("CONFIRM 决策与当前记忆不一致")
                    memory.confidence = max(
                        float(memory.confidence), candidate.confidence
                    )
                    memory.salience = max(
                        float(memory.salience), candidate.salience
                    )
                    memory.row_version = int(memory.row_version) + 1
                    target_memory = memory
                    confirmed += 1
                elif decision.action == "ADD":
                    if memory is not None:
                        raise RuntimeError("ADD 决策遇到并发新增的有效记忆")
                    target_memory = await uow.memory_items.add(
                        AgentMemoryItemEntity(
                            memory_id=uuid7(),
                            domain_id=lease.domain_id,
                            actor_id=lease.actor_id,
                            agent_id=scoped_agent_id,
                            memory_type=candidate.memory_type,
                            scope_type=decision.scope_type,
                            canonical_key=candidate.canonical_key,
                            value_json=candidate.value,
                            search_text=candidate.search_text,
                            confidence=candidate.confidence,
                            salience=candidate.salience,
                            valid_from=now,
                            status="ACTIVE",
                            sensitivity_level=0,
                            index_profile_id=(
                                index_profile.index_profile_id
                                if decision.embedding is not None
                                and index_profile is not None
                                else None
                            ),
                            embedding=decision.embedding,
                        )
                    )
                    changed += 1
                elif decision.action in {"SUPERSEDE", "DISPUTE"}:
                    if memory is None:
                        raise RuntimeError(
                            "冲突决策对应的旧记忆已不存在"
                        )
                    memory.status = (
                        "SUPERSEDED"
                        if decision.action == "SUPERSEDE"
                        else "DISPUTED"
                    )
                    memory.valid_to = now
                    memory.row_version = int(memory.row_version) + 1
                    target_memory = await uow.memory_items.add(
                        AgentMemoryItemEntity(
                            memory_id=uuid7(),
                            domain_id=lease.domain_id,
                            actor_id=lease.actor_id,
                            agent_id=scoped_agent_id,
                            memory_type=candidate.memory_type,
                            scope_type=decision.scope_type,
                            canonical_key=candidate.canonical_key,
                            value_json=candidate.value,
                            search_text=candidate.search_text,
                            confidence=candidate.confidence,
                            salience=candidate.salience,
                            valid_from=now,
                            valid_to=(
                                now
                                if decision.action == "DISPUTE"
                                else None
                            ),
                            status=(
                                "DISPUTED"
                                if decision.action == "DISPUTE"
                                else "ACTIVE"
                            ),
                            sensitivity_level=0,
                            index_profile_id=(
                                index_profile.index_profile_id
                                if decision.embedding is not None
                                and index_profile is not None
                                else None
                            ),
                            embedding=decision.embedding,
                        )
                    )
                    changed += 1
                    if decision.action == "DISPUTE":
                        disputed += 1
                if target_memory is not None:
                    await uow.memory_sources.add(
                        AgentMemorySourceEntity(
                            memory_source_id=uuid7(),
                            memory_id=target_memory.memory_id,
                            conversation_id=lease.conversation_id,
                            turn_id=lease.turn_id,
                            item_id=lease.user_item_id,
                            excerpt_hash=_hash(lease.user_message),
                            extractor=(
                                f"{memory_prompt.prompt_key}@"
                                f"{memory_prompt.version}"
                            ),
                        )
                    )
                decision_results.append(
                    {
                        "canonical_key": candidate.canonical_key,
                        "memory_type": candidate.memory_type,
                        "scope_type": decision.scope_type,
                        "candidate_hash": _hash(candidate.value),
                        "action": decision.action,
                        "reason": decision.reason,
                        "existing_memory_id": (
                            str(decision.existing_memory_id)
                            if decision.existing_memory_id
                            else None
                        ),
                        "result_memory_id": (
                            str(target_memory.memory_id)
                            if target_memory is not None
                            else None
                        ),
                        "conflict_prompt": decision.prompt_ref,
                    }
                )
            episodic_memory_id = None
            if episode_text is not None:
                episode = await uow.memory_items.add(
                    AgentMemoryItemEntity(
                        memory_id=uuid7(),
                        domain_id=lease.domain_id,
                        actor_id=lease.actor_id,
                        agent_id=lease.agent_id,
                        memory_type="EPISODIC",
                        scope_type="USER_AGENT",
                        canonical_key=f"episode.{lease.turn_id}",
                        value_json={
                            "conversation_id": str(lease.conversation_id),
                            "turn_id": str(lease.turn_id),
                            "turn_sequence": lease.turn_sequence,
                        },
                        search_text=episode_text,
                        confidence=1,
                        salience=0.5,
                        valid_from=now,
                        status="ACTIVE",
                        sensitivity_level=0,
                        index_profile_id=(
                            index_profile.index_profile_id
                            if episode_embedding is not None
                            and index_profile is not None
                            else None
                        ),
                        embedding=episode_embedding,
                    )
                )
                episodic_memory_id = episode.memory_id
                await uow.memory_sources.add(
                    AgentMemorySourceEntity(
                        memory_source_id=uuid7(),
                        memory_id=episode.memory_id,
                        conversation_id=lease.conversation_id,
                        turn_id=lease.turn_id,
                        item_id=lease.user_item_id,
                        excerpt_hash=_hash(lease.user_message),
                        extractor="agent_runtime.episodic@1",
                    )
                )
            job.status = "COMPLETED"
            job.lease_owner = None
            job.lease_token = None
            job.lease_until = None
            job.result_json = {
                "schema_version": "MemoryConsolidationResult.v1",
                "extracted_candidate_count": extracted_candidate_count,
                "accepted_candidate_count": len(decisions),
                "changed_count": changed,
                "confirmed_count": confirmed,
                "disputed_count": disputed,
                "ignored_count": ignored,
                "forgotten_count": forgotten,
                "forget_keys": list(forget_keys),
                "snapshot_prompt": snapshot_prompt.ref(),
                "memory_prompt": memory_prompt.ref(),
                "decisions": decision_results,
                "index_profile_id": (
                    str(index_profile.index_profile_id)
                    if index_profile
                    else None
                ),
                "episodic_memory_id": (
                    str(episodic_memory_id)
                    if episodic_memory_id
                    else None
                ),
            }
            await uow.commit()

    async def _decide_candidates(
        self,
        lease: MemoryJobLease,
        *,
        candidates: tuple[MemoryCandidate, ...],
        model_name: str,
        shared_keys: frozenset[str] = frozenset(),
    ) -> tuple[ResolvedMemoryDecision, ...]:
        existing_by_key = {
            (
                item["memory_type"],
                item["canonical_key"],
                item.get("scope_type", "USER_AGENT"),
            ): item
            for item in lease.existing_memories
        }
        conflict_prompt = None
        decisions: list[ResolvedMemoryDecision] = []
        for candidate in candidates:
            scope_type: Literal["USER_AGENT", "USER_SHARED"] = (
                "USER_SHARED"
                if candidate.canonical_key in shared_keys
                else "USER_AGENT"
            )
            existing = existing_by_key.get(
                (
                    candidate.memory_type,
                    candidate.canonical_key,
                    scope_type,
                )
            )
            if existing is None:
                decisions.append(
                    ResolvedMemoryDecision(
                        candidate=candidate,
                        action="ADD",
                        reason="当前作用域不存在同键有效记忆",
                        existing_memory_id=None,
                        prompt_ref=None,
                        scope_type=scope_type,
                    )
                )
                continue
            existing_id = UUID(str(existing["memory_id"]))
            if _hash(existing.get("value")) == _hash(candidate.value):
                decisions.append(
                    ResolvedMemoryDecision(
                        candidate=candidate,
                        action="CONFIRM",
                        reason="候选值与当前有效记忆一致",
                        existing_memory_id=existing_id,
                        prompt_ref=None,
                        scope_type=scope_type,
                    )
                )
                continue
            if conflict_prompt is None:
                conflict_prompt = await self._prompt_resolver.resolve(
                    "agent_runtime.memory_conflict_assess"
                )
            rendered = StrictPromptRenderer.render(
                conflict_prompt,
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "existing_memory": existing,
                },
            )
            raw = await self._model_client.get_llm_json(
                served_model_name=model_name,
                prompt=rendered,
                max_tokens=1024,
            )
            assessed = MemoryConflictDecision.model_validate(raw)
            if assessed.action not in {
                "SUPERSEDE",
                "DISPUTE",
                "IGNORE",
            }:
                raise ValueError(
                    "同键异值冲突只能选择 SUPERSEDE、DISPUTE 或 IGNORE"
                )
            decisions.append(
                ResolvedMemoryDecision(
                    candidate=candidate,
                    action=assessed.action,
                    reason=assessed.reason,
                    existing_memory_id=existing_id,
                    prompt_ref=conflict_prompt.ref(),
                    scope_type=scope_type,
                )
            )
        return tuple(decisions)

    @staticmethod
    def _safe_candidates(
        batch: MemoryCandidateBatch,
    ) -> tuple[MemoryCandidate, ...]:
        """对模型候选执行确定性敏感信息拦截。"""
        forbidden = re.compile(
            r"(?i)(password|passwd|pwd|token|secret|api[_-]?key|"
            r"private[_-]?key|credential|jdbc:|oracle://|mysql://|"
            r"postgres(?:ql)?://|-----BEGIN [A-Z ]+PRIVATE KEY-----)"
        )
        accepted: list[MemoryCandidate] = []
        for candidate in batch.candidates:
            rendered = json.dumps(
                {
                    "canonical_key": candidate.canonical_key,
                    "value": candidate.value,
                    "search_text": candidate.search_text,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if forbidden.search(rendered):
                logger.warning(
                    "长期记忆候选因敏感字段被拒绝：canonical_key={}",
                    candidate.canonical_key,
                )
                continue
            accepted.append(candidate)
        return tuple(accepted)

    @staticmethod
    def _safe_forget_keys(values: tuple[str, ...]) -> tuple[str, ...]:
        pattern = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
        result: list[str] = []
        for value in values:
            normalized = value.strip()
            if not pattern.fullmatch(normalized):
                raise ValueError("forget_keys 包含无效 canonical_key")
            if normalized not in result:
                result.append(normalized)
        return tuple(result)

    async def _fail(
        self, lease: MemoryJobLease, exc: Exception
    ) -> None:
        async with self._uow_factory() as uow:
            job = await uow.memory_jobs.get(
                memory_job_id=lease.job_id, lock=True
            )
            if (
                job is None
                or job.status != "PROCESSING"
                or job.lease_token != lease.lease_token
            ):
                return
            exhausted = int(job.attempt_count) >= int(job.max_attempts)
            job.status = "FAILED" if exhausted else "RETRY_WAIT"
            job.next_attempt_at = _now() + timedelta(
                seconds=min(2 ** int(job.attempt_count), 60)
            )
            job.error_code = type(exc).__name__.upper()[:128]
            job.error_message = (str(exc) or "记忆归并失败")[:1000]
            job.lease_owner = None
            job.lease_token = None
            job.lease_until = None
            await uow.commit()
