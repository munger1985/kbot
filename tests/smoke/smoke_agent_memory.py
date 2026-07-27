"""在开发 Oracle 上验收 Conversation Memory 完整链路。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import time_ns
from types import SimpleNamespace

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.application import (  # noqa: E402
    ConversationRetentionWorker,
    ConversationService,
    MemoryConsolidationWorker,
)
from agent_runtime.entities import (  # noqa: E402
    AgentConversationEntity,
    AgentConversationItemEntity,
    AgentConversationTurnEntity,
    AgentDefinitionEntity,
    AgentMemoryJobEntity,
)
from agent_runtime.persistence import create_agent_runtime_uow  # noqa: E402
from main_api.entities import PlatformDomainEntity  # noqa: E402
from platform_core.database.oracle import create_database_runtime  # noqa: E402
from platform_core.identity import uuid7  # noqa: E402
from platform_core.prompts import PromptResolver, load_prompt_catalog  # noqa: E402


class _MemoryModel:
    async def get_llm_json(self, *, prompt, **kwargs):
        rendered = str(prompt)
        if "会话摘要器" in rendered:
            return {
                "active_topic": "回答语言偏好",
                "user_goal": "使用中文交流",
                "entities": [],
                "corrections": [],
                "unresolved_questions": [],
            }
        if "长期记忆候选提取器" in rendered:
            return {
                "candidates": [
                    {
                        "memory_type": "USER_PREFERENCE",
                        "canonical_key": "response.language",
                        "value": {"language": "zh-CN"},
                        "search_text": "用户偏好中文回答",
                        "confidence": 1,
                        "salience": 0.9,
                    }
                ]
            }
        raise RuntimeError("Smoke Model 收到未知 Prompt")

    async def call_embedding_model(
        self, *, texts, is_query, **kwargs
    ):
        """返回固定维度的非零向量，验证读写模式参数。"""
        if is_query:
            raise RuntimeError("归并 Worker 写向量时不应使用查询模式")
        return [
            SimpleNamespace(embedding=[1.0, 0.5, 0.25])
            for _ in texts
        ]


class _ModelResolver:
    async def resolve(self, models, *, roles=None):
        names = {
            "memory_llm": "memory-smoke-model",
            "memory_embedding": "memory-smoke-embedding",
        }
        return {
            role: {
                "model_id": str(model_id),
                "served_model_name": names[role],
                "category": 2 if role == "memory_embedding" else 1,
                "config_fingerprint": "a" * 64,
            }
            for role, model_id in models.items()
            if role in names and (roles is None or role in roles)
        }


async def _cleanup(
    runtime,
    *,
    conversation_id,
    actor_id: str,
    agent_id,
    domain_id: int,
    app_id: int,
) -> None:
    """只清理由本脚本生成且拥有随机标识的数据。"""
    async with runtime.engine.begin() as connection:
        values = {
            "conversation_id": conversation_id.bytes,
            "actor_id": actor_id,
            "agent_id": agent_id.bytes,
            "domain_id": domain_id,
            "app_id": app_id,
        }
        statements = (
            "DELETE FROM KBOT_AGENT_MEMORY_SOURCE "
            "WHERE CONVERSATION_ID = :conversation_id",
            "DELETE FROM KBOT_AGENT_MEMORY_JOB "
            "WHERE CONVERSATION_ID = :conversation_id",
            "DELETE FROM KBOT_AGENT_MEMORY_SNAPSHOT "
            "WHERE CONVERSATION_ID = :conversation_id",
            "DELETE FROM KBOT_AGENT_CONVERSATION_ITEM "
            "WHERE CONVERSATION_ID = :conversation_id",
            "DELETE FROM KBOT_AGENT_CONVERSATION_TURN "
            "WHERE CONVERSATION_ID = :conversation_id",
            "DELETE FROM KBOT_AGENT_CONVERSATION "
            "WHERE CONVERSATION_ID = :conversation_id",
            "DELETE FROM KBOT_AGENT_MEMORY_ITEM "
            "WHERE ACTOR_ID = :actor_id AND AGENT_ID = :agent_id",
            "DELETE FROM KBOT_AGENT_MEMORY_INDEX_PROFILE "
            "WHERE AGENT_ID = :agent_id",
            "DELETE FROM KBOT_AGENT_DEFINITION "
            "WHERE AGENT_ID = :agent_id",
            "DELETE FROM KBOT_PLATFORM_DOMAIN "
            "WHERE DOMAIN_ID = :domain_id AND APP_ID = :app_id",
        )
        for statement in statements:
            await connection.execute(text(statement), values)


async def main() -> None:
    runtime = create_database_runtime()
    app_id = 9901
    domain_id = 900_000_000 + time_ns() % 90_000_000
    agent_id = uuid7()
    conversation_id = uuid7()
    turn_id = uuid7()
    user_item_id = uuid7()
    assistant_item_id = uuid7()
    llm_model_id = uuid7()
    embedding_model_id = uuid7()
    actor_id = f"memory-smoke-{uuid7()}"
    now = datetime.now(timezone.utc)
    uow_factory = create_agent_runtime_uow(runtime.session_factory)
    try:
        async with runtime.session_factory() as session:
            session.add(
                PlatformDomainEntity(
                    domain_id=domain_id,
                    app_id=app_id,
                    name=f"memory-smoke-{domain_id}",
                    status="ACTIVE",
                    row_version=1,
                    created_by="memory-smoke",
                    updated_by="memory-smoke",
                )
            )
            await session.flush()
            session.add(
                AgentDefinitionEntity(
                    agent_id=agent_id,
                    app_id=app_id,
                    domain_id=domain_id,
                    agent_key=f"memory-smoke-{domain_id}",
                    display_name="记忆实库验收",
                    status="ACTIVE",
                    enabled_capabilities_json=["document"],
                    models_json={
                        "router_llm": str(llm_model_id),
                        "context_llm": str(llm_model_id),
                        "composer_llm": str(llm_model_id),
                        "memory_llm": str(llm_model_id),
                        "memory_embedding": str(embedding_model_id),
                    },
                    config_json={
                        "memory": {
                            "episodic_enabled": True,
                        }
                    },
                    row_version=1,
                    created_by="memory-smoke",
                    updated_by="memory-smoke",
                )
            )
            await session.flush()
            session.add(
                AgentConversationEntity(
                    conversation_id=conversation_id,
                    app_id=app_id,
                    domain_id=domain_id,
                    actor_id=actor_id,
                    agent_id=agent_id,
                    title="记忆验收",
                    status="ACTIVE",
                    row_version=1,
                    last_turn_sequence=1,
                    last_item_sequence=2,
                    retention_policy="DEFAULT",
                    last_active_at=now,
                )
            )
            await session.flush()
            turn = AgentConversationTurnEntity(
                turn_id=turn_id,
                conversation_id=conversation_id,
                turn_sequence=1,
                status="COMPLETED",
                raw_input_hash="a" * 64,
                context_snapshot_json={},
                idempotency_key="memory-smoke-turn",
                completed_at=now,
            )
            session.add(turn)
            await session.flush()
            session.add_all(
                [
                    AgentConversationItemEntity(
                        item_id=user_item_id,
                        conversation_id=conversation_id,
                        item_sequence=1,
                        turn_id=turn_id,
                        item_type="MESSAGE",
                        role="USER",
                        content_json={"text": "以后请使用中文回答"},
                        content_hash="b" * 64,
                        visibility="USER",
                    ),
                    AgentConversationItemEntity(
                        item_id=assistant_item_id,
                        conversation_id=conversation_id,
                        item_sequence=2,
                        turn_id=turn_id,
                        item_type="MESSAGE",
                        role="ASSISTANT",
                        content_json={"text": "好的"},
                        content_hash="c" * 64,
                        visibility="USER",
                    ),
                ]
            )
            await session.flush()
            turn.user_item_id = user_item_id
            turn.assistant_item_id = assistant_item_id
            session.add(
                AgentMemoryJobEntity(
                    memory_job_id=uuid7(),
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    status="PENDING",
                    attempt_count=0,
                    max_attempts=3,
                    next_attempt_at=now,
                )
            )
            await session.commit()

        resolver = PromptResolver(
            session_factory=runtime.session_factory,
            catalog=load_prompt_catalog(),
        )
        worker = MemoryConsolidationWorker(
            uow_factory=uow_factory,
            model_client=_MemoryModel(),
            prompt_resolver=resolver,
            worker_id="memory-smoke",
            poll_interval_seconds=1,
            embedding_dimension=3,
            model_resolver=_ModelResolver(),
        )
        if not await worker.run_once():
            raise RuntimeError("Memory Worker 未领取到验收任务")

        async with runtime.session_factory() as session:
            job = (
                await session.execute(
                    text(
                        """
                        SELECT status, result_json
                        FROM KBOT_AGENT_MEMORY_JOB
                        WHERE conversation_id = :conversation_id
                        """
                    ),
                    {"conversation_id": conversation_id.bytes},
                )
            ).one()
            memory_count = (
                await session.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM KBOT_AGENT_MEMORY_ITEM
                        WHERE actor_id = :actor_id AND status = 'ACTIVE'
                        """
                    ),
                    {"actor_id": actor_id},
                )
            ).scalar_one()
            snapshot_count = (
                await session.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM KBOT_AGENT_MEMORY_SNAPSHOT
                        WHERE conversation_id = :conversation_id
                          AND status = 'ACTIVE'
                        """
                    ),
                    {"conversation_id": conversation_id.bytes},
                )
            ).scalar_one()
            profile_count = (
                await session.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM KBOT_AGENT_MEMORY_INDEX_PROFILE
                        WHERE agent_id = :agent_id
                        """
                    ),
                    {"agent_id": agent_id.bytes},
                )
            ).scalar_one()
            embedded_count = (
                await session.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM KBOT_AGENT_MEMORY_ITEM
                        WHERE actor_id = :actor_id
                          AND status = 'ACTIVE'
                          AND index_profile_id IS NOT NULL
                          AND embedding IS NOT NULL
                        """
                    ),
                    {"actor_id": actor_id},
                )
            ).scalar_one()
            memory_types = {
                row.memory_type
                for row in (
                    await session.execute(
                        text(
                            """
                            SELECT memory_type
                            FROM KBOT_AGENT_MEMORY_ITEM
                            WHERE actor_id = :actor_id
                              AND status = 'ACTIVE'
                            """
                        ),
                        {"actor_id": actor_id},
                    )
                ).all()
            }
            if job.status != "COMPLETED" or not job.result_json:
                raise RuntimeError("Memory Job 未保存完成状态和决策结果")
            if (
                int(memory_count) != 2
                or int(snapshot_count) != 1
                or int(profile_count) != 1
                or int(embedded_count) != 2
                or memory_types != {"USER_PREFERENCE", "EPISODIC"}
            ):
                raise RuntimeError("Snapshot 或长期记忆写入数量错误")

        conversation_service = ConversationService(
            uow_factory=uow_factory,
            runtime_service=None,
        )
        view = await conversation_service.get(
            conversation_id=conversation_id,
            app_id=app_id,
            domain_id=domain_id,
            actor_id=actor_id,
        )
        archived = await conversation_service.update(
            conversation_id=conversation_id,
            app_id=app_id,
            domain_id=domain_id,
            actor_id=actor_id,
            expected_row_version=view.row_version,
            title=None,
            status="ARCHIVED",
            retention_policy="DAYS_30",
        )
        if archived.purge_after is None:
            raise RuntimeError("归档 Conversation 未计算保留期限")
        async with runtime.engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    UPDATE KBOT_AGENT_CONVERSATION
                    SET purge_after = CURRENT_TIMESTAMP - INTERVAL '1' SECOND
                    WHERE conversation_id = :conversation_id
                    """
                ),
                {"conversation_id": conversation_id.bytes},
            )
        retention_worker = ConversationRetentionWorker(
            conversation_service=conversation_service,
            poll_interval_seconds=60,
        )
        if not await retention_worker.run_once():
            raise RuntimeError("Retention Worker 未清理到期 Conversation")
        print("Conversation Memory 实库归并、决策结果和隐私清理验收通过")
    finally:
        await _cleanup(
            runtime,
            conversation_id=conversation_id,
            actor_id=actor_id,
            agent_id=agent_id,
            domain_id=domain_id,
            app_id=app_id,
        )
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
