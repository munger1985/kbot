"""固定 App 知识端口的纯 HTTP 客户端；不含算法、ORM 或跨 App 路由。"""

import re
from contextlib import asynccontextmanager
from uuid import UUID

from .knowledge_computation import KnowledgeComputationClient, KnowledgeComputationClientError


class AppKnowledgeClient(KnowledgeComputationClient):
    def __init__(self, *, owner, delegation_codec=None, delegation_factory=None, management=False, **kwargs):
        paths = {"knowledge_retrieval": "/internal/v1/knowledge-retrieval/knowledge",
                 "aiops": "/internal/v1/aiops/knowledge"}
        if owner not in paths:
            raise ValueError("未登记知识 App")
        kwargs["service_scopes"] = ("knowledge.scope.create", "knowledge.retrieve", "knowledge.content.read") + (
            ("knowledge.publish", "knowledge.withdraw") if management else ())
        super().__init__(**kwargs)
        self.owner, self.prefix = owner, paths[owner]
        self.delegation_codec, self.delegation_factory = delegation_codec, delegation_factory

    def for_delegation(self, claims, *, clock=None):
        """每个 Run/Scope 使用独立不可变委托，不修改共享 HTTP 会话或其他 Run。"""
        import copy
        import time
        from types import MappingProxyType
        if self.delegation_codec is None or claims.get("owner_binding_ref") != self.owner:
            raise ValueError("知识 Run 委托与固定 App 不匹配")
        frozen = MappingProxyType(dict(claims))
        current_time = clock or time.time
        def factory(auth_context, operation, audience):
            if operation not in frozen["operations"] or frozen["audience"] != audience:
                raise ValueError("知识 Run 委托没有授权当前操作或受众")
            now = int(current_time())
            expires = min(frozen["expires_at"], now + self.delegation_codec.max_ttl)
            return {**frozen, "expires_at": expires}
        result = copy.copy(self)
        result.delegation_factory = factory
        return result

    @asynccontextmanager
    async def response(self, method, path, *, auth_context, **kwargs):
        operation = ("knowledge.publish" if path.endswith((":publish", "/citation-targets"))
                     else "knowledge.withdraw" if path.endswith(":withdraw")
                     else "knowledge.scope.create" if path.endswith("/scopes")
                     else "knowledge.content.read" if "/content/" in path or "/evidence:" in path
                     else "knowledge.retrieve")
        headers = None
        if self.delegation_factory is not None:
            claims = self.delegation_factory(auth_context, operation, self.audience)
            if self.delegation_codec is None or claims["owner_binding_ref"] != self.owner:
                raise ValueError("知识委托签名或固定 App 不匹配")
            headers = {"X-Knowledge-Delegation": self.delegation_codec.encode(claims)}
        async with super().response(method, path, auth_context=auth_context,
                                    extra_headers=headers, **kwargs) as response:
            yield response

    async def publish(self, version_id, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/versions/" + str(UUID(version_id)) + ":publish",
                               auth_context=auth_context, payload=payload)

    async def withdraw(self, version_id, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/versions/" + str(UUID(version_id)) + ":withdraw",
                               auth_context=auth_context, payload=payload)

    async def register_citation_target(self, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/citation-targets", auth_context=auth_context, payload=payload)

    async def create_scope(self, resource_ref, *, auth_context):
        return await self.json("POST", self.prefix + "/scopes",
                               auth_context=auth_context, payload={"resource_ref": resource_ref})

    async def retrieve(self, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/retrieve", auth_context=auth_context, payload=payload)

    async def continue_retrieval(self, retrieval_ref, cursor, *, auth_context):
        identity = str(UUID(retrieval_ref))
        return await self.json("POST", self.prefix + "/retrievals/" + identity + ":continue",
                               auth_context=auth_context, payload={"cursor": cursor})

    async def read_manifest(self, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/evidence:read", auth_context=auth_context, payload=payload)

    async def register_query_image(self, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/query-images", auth_context=auth_context, payload=payload)

    async def continue_relations(self, retrieval_ref, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/retrievals/" + str(UUID(retrieval_ref)) + "/relations:continue",
                               auth_context=auth_context, payload=payload)

    async def retain(self, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/evidence:retain", auth_context=auth_context, payload=payload)

    async def release(self, pin_ids, *, auth_context):
        return await self.json("POST", self.prefix + "/evidence:release", auth_context=auth_context, payload={"pin_ids": pin_ids})

    async def content(self, retrieval_ref, stable_key, *, auth_context, max_bytes):
        identity = str(UUID(retrieval_ref))
        if type(stable_key) is not str or not re.fullmatch(r"[0-9a-f]{64}", stable_key):
            raise ValueError("知识内容句柄无效")
        path = self.prefix + "/retrievals/" + identity + "/content/" + stable_key
        async with self.response("GET", path, auth_context=auth_context) as response:
            count = 0
            async for block in response.content.iter_chunked(65536):
                count += len(block)
                if count > max_bytes:
                    raise KnowledgeComputationClientError(502, "RESOURCE_LIMIT")
                yield block

    async def open_historical_content(self, retrieval_ref, stable_key, *, auth_context, max_bytes=8388608):
        """在下游发出响应头前完成内部历史读取授权，并显式持有流资源。"""
        from .knowledge_library import KnowledgeLibraryClient
        identity = str(UUID(str(retrieval_ref)))
        if type(stable_key) is not str or not re.fullmatch(r"[0-9a-f]{64}", stable_key):
            raise ValueError("历史证据身份无效")
        return await KnowledgeLibraryClient.open_download(self,
            self.prefix + "/history/" + identity + "/content/" + stable_key,
            auth_context=auth_context, max_bytes=max_bytes)
    async def release_run(self, scope, *, auth_context):
        return await self.json("POST", self.prefix + "/evidence:release-run",
            payload={"scope": scope}, auth_context=auth_context)
