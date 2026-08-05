"""使用真实 Oracle 验证 S7 Receipt、幂等重放与不确定命令恢复。"""

from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy import text

from knowledge_core.application.collections import (
    CollectionNotFoundError,
    CreateCollectionCommand,
    KnowledgeCoreCollectionService,
)
from knowledge_core.persistence import create_kc_uow
from main_api.application.resource_composition import CompositionError, ResourceCompositionService
from main_api.persistence import create_main_api_uow
from model_serving.common.model_registry import (
    ModelDefinitionNotFound,
    ModelRegistryService,
)
from model_serving.persistence import create_model_serving_uow_factory
from platform_core.contracts import CollectionCompositionCreate
from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7


class _ModelCatalog:
    def __init__(self, service: ModelRegistryService):
        self._service = service

    async def get_model(self, model_id):
        try:
            return await self._service.get(model_id)
        except ModelDefinitionNotFound as exc:
            raise LookupError(str(exc)) from exc


class _KnowledgeCore:
    def __init__(self, service: KnowledgeCoreCollectionService, *, actor_id: str):
        self._service = service
        self._actor_id = actor_id
        self.commands = 0
        self.store_before_timeout = True

    async def create_collection(self, *, domain_id, payload, auth_context):
        del auth_context
        self.commands += 1
        if self.store_before_timeout:
            await self.persist_collection(domain_id=domain_id, payload=payload)
        raise TimeoutError("注入下游响应超时")

    async def get_collection(self, *, domain_id, collection_id, auth_context):
        del auth_context
        try:
            row = await self._service.get(
                domain_id=domain_id,
                collection_id=collection_id,
            )
        except CollectionNotFoundError as exc:
            raise LookupError("注入资源尚不可见") from exc
        return {
            "collection_id": str(row.collection_id),
            "domain_id": row.domain_id,
            "display_name": row.display_name,
            "models": row.models_json,
            "status": row.status,
            "row_version": row.row_version,
        }

    async def persist_collection(self, *, domain_id, payload):
        await self._service.create(CreateCollectionCommand(
            collection_id=UUID(str(payload["collection_id"])),
            domain_id=domain_id,
            display_name=payload["display_name"],
            models=payload["models"],
            description=payload.get("description"),
            default_security_level=payload.get("default_security_level", 1),
            metadata=payload.get("metadata") or {},
            actor_id=self._actor_id,
        ))


class _Notifications:
    async def resource_notifications(self, **kwargs):
        return []


async def main() -> None:
    runtime = create_database_runtime()
    prefix = f"s7-oracle-{uuid7()}"
    actor_id = f"smoke:{prefix}"
    model_id: UUID | None = None
    try:
        async with runtime.session_factory() as session:
            domain_id = (await session.execute(text(
                "SELECT DOMAIN_ID FROM KBOT_PLATFORM_DOMAIN "
                "WHERE STATUS = 'ACTIVE' FETCH FIRST 1 ROWS ONLY"
            ))).scalar_one_or_none()
        if domain_id is None:
            raise RuntimeError("真实 Oracle 中没有可用于 S7 Smoke 的 ACTIVE Domain")

        model_registry = ModelRegistryService(
            uow_factory=create_model_serving_uow_factory(runtime.session_factory),
        )
        model = await model_registry.create({
            "served_model_name": prefix,
            "display_name": "S7 Oracle Smoke",
            "provider_model_name": "s7-smoke-model",
            "category": 1,
            "provider": "api_qwen",
            "api_endpoint": "https://example.invalid/v1",
            "api_key": "s7-smoke-secret",
            "status": "ACTIVE",
            "model_params": {"max_tokens": 128},
            "description": "S7 组合编排真实下游 Smoke",
        }, actor_id=actor_id)
        model_id = UUID(model["model_id"])
        collection_service = KnowledgeCoreCollectionService(
            uow_factory=lambda: create_kc_uow(runtime.session_factory),
        )
        knowledge = _KnowledgeCore(collection_service, actor_id=actor_id)
        service = ResourceCompositionService(
            uow_factory=create_main_api_uow(runtime.session_factory),
            agent_client=object(), knowledge_client=knowledge,
            data_query_client=object(),
            model_clients=(_ModelCatalog(model_registry),),
            notification_center=_Notifications(),
        )
        body = CollectionCompositionCreate.model_validate({
            "collection": {
                "display_name": "S7 Oracle Smoke",
                "models": {
                    "parser_llm": str(model_id),
                    "retrieval_llm": str(model_id),
                    "embedding": str(model_id),
                },
            }
        })
        succeeded = await service.create_collection(
            body=body, domain_id=int(domain_id), actor_id=actor_id,
            idempotency_key=f"{prefix}-verified", context=None,
        )
        if succeeded.status != "SUCCEEDED" or knowledge.commands != 1:
            raise RuntimeError("提交后超时的验证恢复未成功")
        replay = await service.create_collection(
            body=body, domain_id=int(domain_id), actor_id=actor_id,
            idempotency_key=f"{prefix}-verified", context=None,
        )
        if not replay.idempotent_replay or knowledge.commands != 1:
            raise RuntimeError("成功 Receipt 重放产生了重复命令")

        knowledge.store_before_timeout = False
        recovery_key = f"{prefix}-recovery"
        try:
            await service.create_collection(
                body=body, domain_id=int(domain_id), actor_id=actor_id,
                idempotency_key=recovery_key, context=None,
            )
        except CompositionError as exc:
            if exc.code != "COMPOSITION_COMPENSATION_REQUIRED":
                raise
        else:
            raise RuntimeError("不可确认命令没有进入恢复状态")
        async with runtime.session_factory() as session:
            status, resource_id = (await session.execute(text(
                "SELECT STATUS, RESOURCE_ID FROM KBOT_COMPOSITION_RECEIPT "
                "WHERE DOMAIN_ID = :domain_id AND ACTOR_ID = :actor_id "
                "AND IDEMPOTENCY_KEY = :idempotency_key"
            ), {
                "domain_id": domain_id, "actor_id": actor_id,
                "idempotency_key": recovery_key,
            })).one()
        if status != "COMPENSATION_REQUIRED":
            raise RuntimeError("Oracle 未持久化恢复状态")
        await knowledge.persist_collection(
            domain_id=int(domain_id),
            payload={
                "collection_id": str(resource_id),
                **body.collection.model_dump(mode="json"),
            },
        )
        recovered = await service.create_collection(
            body=body, domain_id=int(domain_id), actor_id=actor_id,
            idempotency_key=recovery_key, context=None,
        )
        if recovered.status != "SUCCEEDED" or knowledge.commands != 2:
            raise RuntimeError("恢复重放重复发送了下游命令")
        print(
            "S7 真实 Oracle Smoke 通过：模型目录、Knowledge Core、Receipt、"
            "幂等与恢复均有效"
        )
    finally:
        async with runtime.engine.begin() as connection:
            await connection.execute(text(
                "DELETE FROM KBOT_COMPOSITION_RECEIPT WHERE ACTOR_ID = :actor_id"
            ), {"actor_id": actor_id})
            await connection.execute(text(
                "DELETE FROM KBOT_KC_COLLECTION WHERE CREATED_BY = :actor_id"
            ), {"actor_id": actor_id})
            if model_id is not None:
                await connection.execute(text(
                    "DELETE FROM KBOT_AI_MODEL WHERE MODEL_ID = :model_id"
                ), {"model_id": model_id.bytes})
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
