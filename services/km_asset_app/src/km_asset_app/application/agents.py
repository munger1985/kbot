"""KM Asset App 专属问文问数 Agent。"""

from typing import Any
from uuid import UUID

from km_asset_app.application.assets import KmAssetApplicationError
from km_asset_app.entities import KmAgentEntity, KmAgentVersionEntity
from platform_core.identity import uuid7
from platform_core.contracts import AuthContext, PrincipalKind
from platform_clients import DataQueryClientError


KM_AGENT_CAPABILITIES = ("document", "data_query")


class KmAgentService:
    def __init__(self, *, uow_factory, data_query_client, knowledge_core_client):
        self._uow_factory = uow_factory
        self._data_query = data_query_client
        self._knowledge_core = knowledge_core_client

    async def create(self, *, domain_id: int, source_id: UUID, display_name: str, description: str | None, models: dict[str, UUID], do_rerank: bool, instruction: str | None, actor_id: str, status: str):
        async with self._uow_factory() as uow:
            source = await uow.assets.get_source(domain_id=domain_id, source_id=source_id)
            if source is None:
                raise KmAssetApplicationError(status_code=404, code="KM_SOURCE_NOT_FOUND", message="KM Asset 来源不存在")
            if source.model_status != "READY" or source.semantic_model_id is None or source.policy_binding_id is None:
                raise KmAssetApplicationError(status_code=409, code="MANAGED_MODEL_NOT_READY", message="系统托管问数模型尚未就绪")
            if status == "ACTIVE":
                self._validate_models(models)
            agent_id, version_id = uuid7(), uuid7()
            agent = KmAgentEntity(agent_id=agent_id, domain_id=domain_id, display_name=display_name.strip(), description=description, status="DRAFT", current_version_id=None, created_by=actor_id, updated_by=actor_id)
            await uow.agents.add(agent)
            version = KmAgentVersionEntity(agent_version_id=version_id, agent_id=agent_id, version_no=1, source_id=source_id, collection_id=source.collection_id, semantic_model_id=source.semantic_model_id, policy_binding_id=source.policy_binding_id, models_json={role: str(value) for role, value in models.items()}, do_rerank=do_rerank, instruction=instruction, config_json={"resource_mode": "managed_resources", "data_query_mode": "SEMANTIC", "managed_model": True}, created_by=actor_id)
            await uow.agents.add(version)
            agent.current_version_id = version_id
            await uow.commit()
            collection_id = source.collection_id
            semantic_model_id = source.semantic_model_id
            policy_binding_id = source.policy_binding_id
        await self._data_query.management_create(
            resource="agent-bindings",
            payload={"consumer_app_id": "km_asset", "agent_id": str(agent_id), "agent_version_id": str(version_id), "semantic_model_id": str(semantic_model_id), "policy_binding_id": str(policy_binding_id)},
            auth_context=self._auth_context(domain_id=domain_id, actor_id=actor_id),
        )
        await self._ensure_collection_binding(
            domain_id=domain_id,
            agent_id=agent_id,
            collection_id=collection_id,
            actor_id=actor_id,
        )
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            version = await uow.agents.version(agent_id=agent_id, version_id=version_id)
            if agent is None or version is None:
                raise KmAssetApplicationError(status_code=409, code="KM_AGENT_PERSISTENCE_CONFLICT", message="KM Asset Agent 创建状态不完整")
            agent.status = status
            agent.updated_by = actor_id
            await uow.commit()
            return self._view(agent, version)

    async def activate(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        expected_row_version: int,
        actor_id: str,
    ) -> dict[str, Any]:
        """激活 Agent，并以幂等方式修复可能缺失的问数绑定。"""
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if agent is None:
                raise KmAssetApplicationError(status_code=404, code="KM_AGENT_NOT_FOUND", message="KM Asset Agent 不存在")
            if int(agent.row_version) != expected_row_version:
                raise KmAssetApplicationError(status_code=409, code="ROW_VERSION_CONFLICT", message="Agent 已被其他请求修改")
            version = await self._version(uow, agent)
            self._validate_models(version.models_json)
            binding = {
                "consumer_app_id": "km_asset",
                "agent_id": str(agent.agent_id),
                "agent_version_id": str(version.agent_version_id),
                "semantic_model_id": str(version.semantic_model_id),
                "policy_binding_id": str(version.policy_binding_id),
            }
        try:
            await self._data_query.management_create(
                resource="agent-bindings",
                payload=binding,
                auth_context=self._auth_context(domain_id=domain_id, actor_id=actor_id),
            )
        except DataQueryClientError as exc:
            if exc.code != "AGENT_BINDING_CONFLICT":
                raise
        await self._ensure_collection_binding(
            domain_id=domain_id,
            agent_id=agent_id,
            collection_id=version.collection_id,
            actor_id=actor_id,
        )
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if agent is None or int(agent.row_version) != expected_row_version:
                raise KmAssetApplicationError(status_code=409, code="ROW_VERSION_CONFLICT", message="Agent 已被其他请求修改")
            version = await self._version(uow, agent)
            agent.status = "ACTIVE"
            agent.row_version += 1
            agent.updated_by = actor_id
            await uow.commit()
            return self._view(agent, version)

    async def list(self, *, domain_id: int):
        async with self._uow_factory() as uow:
            result = []
            for agent in await uow.agents.list(domain_id=domain_id):
                result.append(self._view(agent, await self._version(uow, agent)))
            return result

    async def get(self, *, domain_id: int, agent_id: UUID):
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if agent is None:
                raise KmAssetApplicationError(status_code=404, code="KM_AGENT_NOT_FOUND", message="KM Asset Agent 不存在")
            return self._view(agent, await self._version(uow, agent))

    async def execution_spec(
        self, *, domain_id: int, agent_id: UUID, actor_id: str
    ):
        row = await self.get(domain_id=domain_id, agent_id=agent_id)
        if row["status"] != "ACTIVE":
            raise KmAssetApplicationError(status_code=422, code="KM_AGENT_NOT_ACTIVE", message="KM Asset Agent 未激活")
        await self._ensure_collection_binding(
            domain_id=domain_id,
            agent_id=agent_id,
            collection_id=UUID(row["collection_id"]),
            actor_id=actor_id,
        )
        return {"schema_version": "1.0", "owner_app_id": "km_asset", "domain_id": domain_id, "consumer_agent_id": row["agent_id"], "consumer_agent_version_id": row["agent_version_id"], "agent_kind": "KNOWLEDGE_RETRIEVAL", "display_name": row["display_name"], "enabled_capabilities": list(KM_AGENT_CAPABILITIES), "models": row["models"], "do_rerank": row["do_rerank"], "instruction": row["instruction"], "resource_context": {**row["config"], "collection_ids": [row["collection_id"]], "semantic_model_id": row["semantic_model_id"], "policy_binding_id": row["policy_binding_id"], "source_id": row["source_id"]}, "runtime_policy": {"routing": "document_and_managed_data", "allow_general_conversation": False}}

    async def _ensure_collection_binding(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        collection_id: UUID,
        actor_id: str,
    ) -> None:
        """幂等建立 KC Agent 与固定 Collection 的检索授权。"""
        await self._knowledge_core.bind_collection(
            domain_id=domain_id,
            agent_id=agent_id,
            collection_id=collection_id,
            note="KM Asset App 系统托管绑定",
            auth_context=self._auth_context(
                domain_id=domain_id,
                actor_id=actor_id,
            ),
        )

    @staticmethod
    async def _version(uow, agent):
        version = await uow.agents.version(agent_id=agent.agent_id, version_id=agent.current_version_id)
        if version is None:
            raise KmAssetApplicationError(status_code=409, code="KM_AGENT_VERSION_MISSING", message="KM Asset Agent 当前版本不存在")
        return version

    @staticmethod
    def _view(agent, version) -> dict[str, Any]:
        return {"agent_id": str(agent.agent_id), "agent_version_id": str(version.agent_version_id), "domain_id": str(agent.domain_id), "display_name": agent.display_name, "description": agent.description, "status": agent.status, "source_id": str(version.source_id), "collection_id": str(version.collection_id), "semantic_model_id": str(version.semantic_model_id), "policy_binding_id": str(version.policy_binding_id), "models": dict(version.models_json), "do_rerank": bool(version.do_rerank), "instruction": version.instruction, "config": dict(version.config_json), "row_version": int(agent.row_version)}

    @staticmethod
    def _validate_models(models: dict[str, Any]) -> None:
        if not models:
            raise KmAssetApplicationError(status_code=422, code="AGENT_MODELS_REQUIRED", message="激活 Agent 前必须配置模型")
        if "router_llm" not in models:
            raise KmAssetApplicationError(status_code=422, code="AGENT_ROUTER_MODEL_REQUIRED", message="KM 问文问数 Agent 必须配置 router_llm")

    @staticmethod
    def _auth_context(*, domain_id: int, actor_id: str) -> AuthContext:
        token = str(uuid7())
        return AuthContext(principal_kind=PrincipalKind.SERVICE, client_id="kbot-km-asset-app-api", calling_service="kbot-km-asset-app-api", request_id=token, trace_id=token, domain_id=str(domain_id), asserted_user_id=actor_id)
