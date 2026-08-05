"""四个推理进程共享的模型目录装配。"""

from model_serving.common.model_registry import ModelRegistryService
from model_serving.common.notifications import ModelCatalogNotificationPublisher
from model_serving.persistence import create_model_serving_uow_factory
from platform_clients import AgentRuntimeClient, DataQueryClient, KnowledgeCoreClient


def create_model_registry(
    *, session_factory, runtime_service, service_name: str, settings,
) -> ModelRegistryService:
    """装配 UoW、引用客户端和模型粒度缓存失效。"""

    async def agent_references(model_id, auth_context):
        client = AgentRuntimeClient(
            base_url=settings.agent_runtime.base_url,
            caller_service=service_name,
            audience=settings.agent_runtime.audience,
            timeout_seconds=settings.agent_runtime.timeout_seconds,
        )
        return await client.list_model_references(
            model_id=model_id, auth_context=auth_context,
        )

    async def knowledge_references(model_id, auth_context):
        client = KnowledgeCoreClient(
            base_url=settings.knowledge_core.base_url,
            caller_service=service_name,
            audience=settings.knowledge_core.audience,
            timeout_seconds=settings.knowledge_core.timeout_seconds,
        )
        return await client.list_model_references(
            model_id=model_id, auth_context=auth_context,
        )

    async def data_query_references(model_id, auth_context):
        client = DataQueryClient(
            base_url=settings.data_query.base_url,
            caller_service=service_name,
            audience=settings.data_query.audience,
            timeout_seconds=settings.data_query.timeout_seconds,
        )
        try:
            return await client.list_model_references(
                model_id=model_id, auth_context=auth_context,
            )
        finally:
            await client.close()

    async def invalidate(event):
        await runtime_service.invalidate_model(event["served_model_name"])

    uow_factory = create_model_serving_uow_factory(session_factory)
    return ModelRegistryService(
        uow_factory=uow_factory,
        on_model_changed=invalidate,
        reference_resolvers={
            "agent-runtime": agent_references,
            "knowledge-core": knowledge_references,
            "data-query": data_query_references,
        },
        is_model_loaded=runtime_service.is_model_loaded,
        notification_publisher=ModelCatalogNotificationPublisher(
            uow_factory=uow_factory,
        ),
    )
