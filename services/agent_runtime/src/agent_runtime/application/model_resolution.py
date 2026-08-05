"""把 Agent 模型 UUID 解析为可冻结的模型调用快照。"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from typing import Any
from uuid import UUID

from platform_core.dictionary import ModelCategory


AGENT_MODEL_ROLE_CATEGORIES = {
    "router_llm": ModelCategory.LLM,
    "context_llm": ModelCategory.LLM,
    "composer_llm": ModelCategory.LLM,
    "memory_llm": ModelCategory.LLM,
    "chart_llm": ModelCategory.LLM,
    "diagnosis_llm": ModelCategory.LLM,
    "query_vlm": ModelCategory.VLM,
    "memory_embedding": ModelCategory.TXT_EMBEDDING,
}


class AgentModelCatalogResolver:
    """通过模型归属服务解析并校验 Agent 的角色绑定。"""

    def __init__(self, clients: Mapping[ModelCategory, Any]):
        self._clients = dict(clients)

    async def resolve(
        self,
        models: Mapping[str, UUID | str],
        *,
        roles: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        selected = {
            role: UUID(str(model_id))
            for role, model_id in models.items()
            if roles is None or role in roles
        }
        resolved: dict[str, dict[str, Any]] = {}
        definitions: dict[
            tuple[ModelCategory | None, UUID], dict[str, Any]
        ] = {}
        for role, model_id in selected.items():
            expected = AGENT_MODEL_ROLE_CATEGORIES.get(role)
            cache_key = (expected, model_id)
            definition = definitions.get(cache_key)
            if definition is None:
                definition = await self._get_definition(role, model_id)
                definitions[cache_key] = definition
            category = ModelCategory(int(definition.get("category") or 0))
            if expected is not None and category != expected:
                raise ValueError(
                    f"模型角色 {role} 必须绑定 {expected.name} 模型"
                )
            if definition.get("status") != "ACTIVE":
                raise ValueError(f"模型角色 {role} 绑定的模型未启用")
            served_name = str(
                definition.get("served_model_name") or ""
            ).strip()
            if not served_name:
                raise ValueError(
                    f"模型角色 {role} 缺少 served_model_name"
                )
            fingerprint_source = {
                "model_id": str(model_id),
                "served_model_name": served_name,
                "category": int(category),
                "model_params": definition.get("model_params") or {},
            }
            fingerprint = sha256(
                json.dumps(
                    fingerprint_source,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            resolved[role] = {
                "model_id": str(model_id),
                "served_model_name": served_name,
                "category": int(category),
                "config_fingerprint": fingerprint,
            }
        return resolved

    async def _get_definition(
        self, role: str, model_id: UUID
    ) -> dict[str, Any]:
        expected = AGENT_MODEL_ROLE_CATEGORIES.get(role)
        if expected is not None:
            client = self._clients.get(expected)
            if client is None:
                raise RuntimeError(
                    f"模型角色 {role} 的目录客户端未配置"
                )
            return await client.get_model(model_id)
        for client in dict.fromkeys(self._clients.values()):
            try:
                return await client.get_model(model_id)
            except LookupError:
                continue
        raise LookupError(f"模型 {model_id} 不存在")
