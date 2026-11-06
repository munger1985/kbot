"""验收 Main API、Model、KC 与知识检索 App 的 Oracle 持久化边界。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from uuid import UUID

from sqlalchemy import delete, select


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_retrieval_app.entities.knowledge_library import KnowledgeLibraryEntity
from knowledge_retrieval_app.domain.knowledge_ingestion import IngestionScope
from knowledge_retrieval_app.entities import (  # noqa: E402
    KnowledgeRetrievalAgentEntity,
)
from knowledge_retrieval_app.persistence import (  # noqa: E402
    create_knowledge_retrieval_app_uow,
)
from main_api.entities import PlatformDomainEntity  # noqa: E402
from main_api.persistence import create_main_api_uow  # noqa: E402
from model_serving.common.entities import AIModelEntity  # noqa: E402
from model_serving.common.model_registry import ModelRegistryService  # noqa: E402
from model_serving.persistence import create_model_serving_uow_factory  # noqa: E402
from platform_core.config import get_settings  # noqa: E402
from platform_core.database.oracle import create_database_runtime  # noqa: E402
from platform_core.identity import uuid7  # noqa: E402


async def _cleanup(
    runtime,
    *,
    agent_ids: tuple,
    library_ids: tuple,
    model_id,
    domain_id: int | None,
) -> None:
    """严格按外键逆序清理本次 Smoke 数据。"""
    async with runtime.session_factory() as session:
        if agent_ids:
            await session.execute(
                delete(KnowledgeRetrievalAgentEntity).where(
                    KnowledgeRetrievalAgentEntity.agent_id.in_(agent_ids)
                )
            )
        if library_ids:
            await session.execute(
                delete(KnowledgeLibraryEntity).where(
                    KnowledgeLibraryEntity.library_id.in_(library_ids)
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
    library_id = uuid7()
    rollback_library_id = uuid7()
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
            uow_factory=create_model_serving_uow_factory(
                runtime.session_factory
            ),
        )
        model = await model_service.create(
            {
                "served_model_name": f"oracle-smoke-{marker}",
                "display_name": "Oracle Smoke Embedding",
                "provider_model_name": "oracle-smoke",
                "category": 2,
                "provider": "local_bge",
                "status": "DRAFT",
                "model_params": {
                    "embedding_dimension": settings.vector.dimensions,
                    "model_path": "/tmp/kbot-oracle-smoke-model",
                    "batch_size": 8,
                    "normalize": True,
                },
            },
            actor_id="oracle-smoke",
        )
        model_id = UUID(model["model_id"])
        loaded_model = await model_service.get(model_id, category=2)
        assert loaded_model["served_model_name"] == model["served_model_name"]
        expected_model_params = {
            "embedding_dimension": settings.vector.dimensions,
            "model_path": "/tmp/kbot-oracle-smoke-model",
            "batch_size": 8,
            "normalize": True,
        }
        assert (
            loaded_model["model_params"] == expected_model_params
        ), repr(loaded_model["model_params"])

        scope = IngestionScope(domain_id, None, "oracle-smoke")
        libraries = create_knowledge_retrieval_app_uow(runtime.session_factory)
        async with libraries() as uow:
            await uow.library(scope).create(library_id=library_id, name="Oracle Smoke Library",
                description=None, configuration={"models":{"embedding":str(model_id)}, "profile_ref":"smoke"})
            await uow.commit()
        async with libraries() as uow:
            assert await uow.library(scope).detail(library_id)
            await uow.library(scope).create(library_id=rollback_library_id, name="Rollback Library",
                description=None, configuration={"profile_ref":"smoke"})
        async with runtime.session_factory() as session:
            assert (await session.execute(select(KnowledgeLibraryEntity).where(
                KnowledgeLibraryEntity.library_id == rollback_library_id))).scalar_one_or_none() is None

        agent_uow_factory = create_knowledge_retrieval_app_uow(
            runtime.session_factory
        )
        async with agent_uow_factory() as uow:
            await uow.agents.add_agent(
                KnowledgeRetrievalAgentEntity(
                    agent_id=agent_id,
                    domain_id=domain_id,
                    display_name="Oracle Smoke Knowledge Agent",
                    status="DRAFT",
                    created_by="oracle-smoke",
                    updated_by="oracle-smoke",
                )
            )
            await uow.commit()
        async with agent_uow_factory() as uow:
            assert await uow.agents.get(
                agent_id=agent_id,
                domain_id=domain_id,
            )
            await uow.agents.add_agent(
                KnowledgeRetrievalAgentEntity(
                    agent_id=rollback_agent_id,
                    domain_id=domain_id,
                    display_name="Oracle Smoke Rollback Knowledge Agent",
                    status="DRAFT",
                    created_by="oracle-smoke",
                    updated_by="oracle-smoke",
                )
            )
        async with runtime.session_factory() as session:
            assert (
                await session.execute(
                    select(KnowledgeRetrievalAgentEntity).where(
                        KnowledgeRetrievalAgentEntity.agent_id
                        == rollback_agent_id
                    )
                )
            ).scalar_one_or_none() is None

        print(
            "跨服务 Oracle UoW Smoke 通过："
            "Domain 作用域、Model Service 提交、资料库/知识 Agent 显式提交与"
            "漏提交回滚均正常"
        )
    finally:
        await _cleanup(
            runtime,
            agent_ids=(agent_id, rollback_agent_id),
            library_ids=(library_id, rollback_library_id),
            model_id=model_id,
            domain_id=domain_id,
        )
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(smoke())
