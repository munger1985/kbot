"""验收 Main API、Model、KC 与 Agent Runtime 的 Oracle 持久化边界。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from uuid import UUID

from sqlalchemy import delete, select


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.entities import AgentDefinitionEntity  # noqa: E402
from agent_runtime.persistence import create_agent_runtime_uow  # noqa: E402
from knowledge_core.entities import KcCollectionEntity  # noqa: E402
from knowledge_core.persistence import create_kc_uow  # noqa: E402
from main_api.entities import PlatformDomainEntity  # noqa: E402
from main_api.persistence import create_main_api_uow  # noqa: E402
from model_serving.common.entities import AIModelEntity  # noqa: E402
from model_serving.common.model_registry import ModelRegistryService  # noqa: E402
from platform_core.config import get_settings  # noqa: E402
from platform_core.database.oracle import create_database_runtime  # noqa: E402
from platform_core.identity import uuid7  # noqa: E402


async def _cleanup(
    runtime,
    *,
    agent_ids: tuple,
    collection_ids: tuple,
    model_id,
    domain_id: int | None,
) -> None:
    """严格按外键逆序清理本次 Smoke 数据。"""
    async with runtime.session_factory() as session:
        if agent_ids:
            await session.execute(
                delete(AgentDefinitionEntity).where(
                    AgentDefinitionEntity.agent_id.in_(agent_ids)
                )
            )
        if collection_ids:
            await session.execute(
                delete(KcCollectionEntity).where(
                    KcCollectionEntity.collection_id.in_(collection_ids)
                )
            )
        if model_id is not None:
            await session.execute(
                delete(AIModelEntity).where(
                    AIModelEntity.model_id == model_id
                )
            )
        if domain_id is not None:
            await session.execute(
                delete(PlatformDomainEntity).where(
                    PlatformDomainEntity.domain_id == domain_id
                )
            )
        await session.commit()


async def smoke() -> None:
    settings = get_settings()
    runtime = create_database_runtime(settings)
    marker = str(uuid7())
    model_id = None
    domain_id: int | None = None
    collection_id = uuid7()
    rollback_collection_id = uuid7()
    agent_id = uuid7()
    rollback_agent_id = uuid7()
    try:
        async with runtime.session_factory() as session:
            domain = PlatformDomainEntity(
                name=f"oracle-smoke-{marker}",
                status="ACTIVE",
                created_by="oracle-smoke",
                updated_by="oracle-smoke",
            )
            session.add(domain)
            await session.flush()
            domain_id = int(domain.domain_id)
            await session.commit()

        main_uow_factory = create_main_api_uow(runtime.session_factory)
        async with main_uow_factory() as uow:
            assert await uow.domains.exists_active(
                domain_id=domain_id,
            )

        model_service = ModelRegistryService(
            session_factory=runtime.session_factory,
        )
        model = await model_service.create(
            {
                "served_model_name": f"oracle-smoke-{marker}",
                "display_name": "Oracle Smoke Embedding",
                "provider_model_name": "oracle-smoke",
                "category": 2,
                "provider": "smoke",
                "status": 1,
                "embedding_dimension": settings.vector.dimensions,
                "model_params": {
                    "temperature": 0.2,
                    "runtime": {"device": "cpu"},
                    "features": ["json", "中文"],
                },
            },
            actor_id="oracle-smoke",
        )
        model_id = UUID(model["model_id"])
        loaded_model = await model_service.get(model_id, category=2)
        assert loaded_model["served_model_name"] == model["served_model_name"]
        expected_model_params = {
            "temperature": 0.2,
            "runtime": {"device": "cpu"},
            "features": ["json", "中文"],
        }
        assert (
            loaded_model["model_params"] == expected_model_params
        ), repr(loaded_model["model_params"])

        async with create_kc_uow(runtime.session_factory) as uow:
            await uow.collections.add(
                KcCollectionEntity(
                    collection_id=collection_id,
                    domain_id=domain_id,
                    display_name="Oracle Smoke Collection",
                    models_json={
                        "parser_llm": str(model_id),
                        "retrieval_llm": str(model_id),
                        "embedding": str(model_id),
                    },
                    status="ACTIVE",
                    default_security_level=1,
                    metadata_json={"source": "oracle-smoke"},
                    created_by="oracle-smoke",
                    updated_by="oracle-smoke",
                )
            )
            await uow.commit()
        async with create_kc_uow(runtime.session_factory) as uow:
            assert await uow.collections.get_by_id(
                collection_id=collection_id
            )
            await uow.collections.add(
                KcCollectionEntity(
                    collection_id=rollback_collection_id,
                    domain_id=domain_id,
                    display_name="Oracle Smoke Rollback Collection",
                    models_json={
                        "parser_llm": str(model_id),
                        "retrieval_llm": str(model_id),
                        "embedding": str(model_id),
                    },
                    status="ACTIVE",
                    default_security_level=1,
                    metadata_json={},
                )
            )
        async with runtime.session_factory() as session:
            assert (
                await session.execute(
                    select(KcCollectionEntity).where(
                        KcCollectionEntity.collection_id
                        == rollback_collection_id
                    )
                )
            ).scalar_one_or_none() is None

        agent_uow_factory = create_agent_runtime_uow(
            runtime.session_factory
        )
        async with agent_uow_factory() as uow:
            await uow.agents.add(
                AgentDefinitionEntity(
                    agent_id=agent_id,
                    domain_id=domain_id,
                    display_name="Oracle Smoke Agent",
                    status="ACTIVE",
                    enabled_capabilities_json=["document"],
                    models_json={
                        "composer_llm": str(model_id),
                        "context_llm": str(model_id),
                        "memory_llm": str(model_id),
                        "memory_embedding": str(model_id),
                    },
                    config_json={},
                    created_by="oracle-smoke",
                    updated_by="oracle-smoke",
                )
            )
            await uow.commit()
        async with agent_uow_factory() as uow:
            assert await uow.agents.get_active(
                agent_id=agent_id,
                domain_id=domain_id,
            )
            await uow.agents.add(
                AgentDefinitionEntity(
                    agent_id=rollback_agent_id,
                    domain_id=domain_id,
                    display_name="Oracle Smoke Rollback Agent",
                    status="DRAFT",
                    enabled_capabilities_json=[],
                    models_json={
                        "composer_llm": str(model_id),
                        "context_llm": str(model_id),
                        "memory_llm": str(model_id),
                        "memory_embedding": str(model_id),
                    },
                    config_json={},
                    created_by="oracle-smoke",
                    updated_by="oracle-smoke",
                )
            )
        async with runtime.session_factory() as session:
            assert (
                await session.execute(
                    select(AgentDefinitionEntity).where(
                        AgentDefinitionEntity.agent_id == rollback_agent_id
                    )
                )
            ).scalar_one_or_none() is None

        print(
            "跨服务 Oracle UoW Smoke 通过："
            "Domain 作用域、Model Service 提交、KC/Agent 显式提交与"
            "漏提交回滚均正常"
        )
    finally:
        await _cleanup(
            runtime,
            agent_ids=(agent_id, rollback_agent_id),
            collection_ids=(collection_id, rollback_collection_id),
            model_id=model_id,
            domain_id=domain_id,
        )
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(smoke())
