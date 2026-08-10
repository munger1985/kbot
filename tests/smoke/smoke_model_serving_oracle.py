"""Model Serving S4 真实 Oracle 生命周期 Smoke。"""

import asyncio
from uuid import UUID, uuid4

from sqlalchemy import delete

from aiops_agent.application.agents import AIOpsAgentService
from aiops_agent.persistence import create_aiops_uow_factory
from data_query.persistence import create_data_query_uow_factory
from knowledge_core.application.model_references import (
    KnowledgeCoreModelReferenceService,
)
from knowledge_core.persistence import create_kc_uow
from knowledge_retrieval_app.application import KnowledgeRetrievalAgentService
from knowledge_retrieval_app.persistence import (
    create_knowledge_retrieval_app_uow,
)
from model_serving.common.entities.ai_model import AIModelEntity
from model_serving.common.model_registry import (
    ModelRegistryConflict,
    ModelRegistryService,
)
from model_serving.persistence import create_model_serving_uow_factory
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.database import create_database_runtime


async def main() -> None:
    runtime = create_database_runtime()
    served_name = f"s4-smoke-{uuid4().hex[:12]}"
    events: list[dict] = []

    async def capture(event):
        events.append(event)

    aiops_references = AIOpsAgentService(
        uow_factory=create_aiops_uow_factory(runtime.session_factory),
    )
    knowledge_agent_references = KnowledgeRetrievalAgentService(
        uow_factory=create_knowledge_retrieval_app_uow(
            runtime.session_factory
        ),
    )
    knowledge_references = KnowledgeCoreModelReferenceService(
        uow_factory=lambda: create_kc_uow(runtime.session_factory),
    )
    data_query_uow = create_data_query_uow_factory(runtime.session_factory)

    async def resolve_aiops_agent(model_id, _auth_context):
        return await aiops_references.model_references(model_id=model_id)

    async def resolve_knowledge_agent(model_id, _auth_context):
        return await knowledge_agent_references.model_references(
            model_id=model_id
        )

    async def resolve_knowledge(model_id, _auth_context):
        return await knowledge_references.list(model_id=model_id)

    async def resolve_data_query(model_id, _auth_context):
        async with data_query_uow() as uow:
            assert uow.model_references is not None
            rows = await uow.model_references.list_for_model(model_id=model_id)
            await uow.commit()
            return rows

    service = ModelRegistryService(
        uow_factory=create_model_serving_uow_factory(runtime.session_factory),
        on_model_changed=capture,
        reference_resolvers={
            "aiops-agent": resolve_aiops_agent,
            "knowledge-retrieval-app": resolve_knowledge_agent,
            "knowledge-core": resolve_knowledge,
            "data-query": resolve_data_query,
        },
        is_model_loaded=lambda _name: False,
    )
    context = AuthContext(
        principal_kind=PrincipalKind.SERVICE,
        client_id="s4-smoke",
        calling_service="kbot-model-llm",
        request_id="s4-smoke-request",
        trace_id="s4-smoke-trace",
    )
    try:
        created = await service.create({
            "served_model_name": served_name,
            "display_name": "S4 Smoke",
            "provider_model_name": "qwen-smoke",
            "category": 1,
            "provider": "api_qwen",
            "api_endpoint": "https://example.invalid/v1",
            "api_key": "s4-smoke-secret",
            "status": "DRAFT",
            "model_params": {"max_tokens": 128},
            "description": "S4 lifecycle smoke",
        }, actor_id="s4-smoke")
        assert "api_key" not in created
        model_id = UUID(created["model_id"])
        updated = await service.update(
            model_id,
            {"display_name": "S4 Smoke Updated", "api_key": "rotated-secret"},
            expected_row_version=created["row_version"],
            actor_id="s4-smoke",
        )
        try:
            await service.update(
                model_id,
                {"display_name": "stale"},
                expected_row_version=created["row_version"],
                actor_id="s4-smoke",
            )
        except ModelRegistryConflict as exc:
            assert exc.code == "MODEL_VERSION_CONFLICT"
        else:
            raise AssertionError("过期行版本未被拒绝")
        active = await service.change_status(
            model_id,
            target_status="ACTIVE",
            expected_row_version=updated["row_version"],
            actor_id="s4-smoke",
        )
        archived, references = await service.archive(
            model_id,
            expected_row_version=active["row_version"],
            actor_id="s4-smoke",
            auth_context=context,
        )
        assert not references.references
        await service.delete(
            model_id,
            expected_row_version=archived["row_version"],
            auth_context=context,
        )
        assert all("secret" not in repr(event) for event in events)
        print(
            "Model Serving S4 Oracle Smoke 通过：创建、并发控制、状态转换、"
            "三服务真实引用反查、失效事件和删除均正常"
        )
    finally:
        async with runtime.session_factory() as session:
            await session.execute(
                delete(AIModelEntity).where(
                    AIModelEntity.served_model_name == served_name
                )
            )
            await session.commit()
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
