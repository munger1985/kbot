"""正式资料库业务客户端；固定知识检索 App，不再调用公共 KC 存储 API。"""

from contextlib import AsyncExitStack
from dataclasses import dataclass
from uuid import UUID


@dataclass
class KnowledgeDownload:
    mime_type: str
    headers: dict[str, str]
    blocks: object
    stack: AsyncExitStack

    async def close(self):
        await self.stack.aclose()
from .knowledge_computation import KnowledgeComputationClient, KnowledgeComputationClientError


class KnowledgeLibraryClient(KnowledgeComputationClient):
    prefix = "/internal/v1/knowledge-retrieval/knowledge"

    def __init__(self, **kwargs):
        super().__init__(service_scopes=("knowledge_retrieval.manage",), **kwargs)

    @staticmethod
    def library_path(library_id):
        return "/libraries/" + str(UUID(str(library_id)))

    async def libraries(self, *, auth_context):
        return await self.json("GET", self.prefix + "/libraries", auth_context=auth_context)

    async def create_library(self, payload, *, auth_context):
        return await self.json("POST", self.prefix + "/libraries", payload=payload, auth_context=auth_context)

    async def library(self, library_id, *, auth_context):
        return await self.json("GET", self.prefix + self.library_path(library_id), auth_context=auth_context)

    async def update_library(self, library_id, payload, *, auth_context):
        return await self.json("PATCH", self.prefix + self.library_path(library_id),
                               payload=payload, auth_context=auth_context)

    async def files(self, library_id, *, page=1, page_size=20, auth_context):
        return await self.json("GET", self.prefix + self.library_path(library_id) + "/documents",
                               params={"page": page, "page_size": page_size}, auth_context=auth_context)

    async def upload(self, library_id, *, name, mime_type, body, idempotency_key, auth_context):
        import json
        async with self.response("POST", self.prefix + self.library_path(library_id) + "/documents",
            params={"name": name}, body=body, content_type=mime_type, idempotency_key=idempotency_key,
            auth_context=auth_context) as response:
            raw = await response.content.read(65537)
            if len(raw) > 65536:
                raise KnowledgeComputationClientError(502, "RESOURCE_LIMIT")
            return json.loads(raw)

    async def version_action(self, library_id, version_id, action, *, payload=None, auth_context):
        if action not in {"review", "publish", "withdraw", "cancel"}:
            raise ValueError("未登记版本操作")
        path = self.prefix + self.library_path(library_id) + "/versions/" + str(UUID(str(version_id))) + ":" + action
        return await self.json("POST", path, payload=payload, auth_context=auth_context)

    async def search(self, library_id, payload, *, auth_context):
        return await self.json("POST", self.prefix + self.library_path(library_id) + "/search",
                               payload=payload, auth_context=auth_context)

    async def continue_results(self, retrieval_ref, payload, *, relations=False, auth_context):
        path = self.prefix + "/results/" + str(UUID(str(retrieval_ref))) + (
            "/relations:continue" if relations else ":continue")
        return await self.json("POST", path, payload=payload, auth_context=auth_context)

    async def open_download(self, path, *, auth_context, max_bytes, attachment=False):
        stack = AsyncExitStack()
        try:
            response = await stack.enter_async_context(self.response("GET", path, auth_context=auth_context))
            content_type = response.headers.get("Content-Type", "application/octet-stream").split(";")[0].lower()
            safe_types = {"text/plain", "text/markdown", "image/png", "image/jpeg", "image/webp", "application/pdf"}
            mime_type = content_type if content_type in safe_types and not attachment else "application/octet-stream"
            headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                       "Content-Security-Policy": "default-src 'none'; sandbox"}
            if attachment or mime_type == "application/octet-stream":
                headers["Content-Disposition"] = "attachment"
            size = response.headers.get("Content-Length")
            if size is not None:
                if not size.isdecimal() or int(size) > max_bytes:
                    raise KnowledgeComputationClientError(502, "RESOURCE_LIMIT")
                headers["Content-Length"] = size
            async def blocks():
                try:
                    count = 0
                    async for block in response.content.iter_chunked(65536):
                        count += len(block)
                        if count > max_bytes:
                            raise KnowledgeComputationClientError(502, "RESOURCE_LIMIT")
                        yield block
                finally:
                    await stack.aclose()
            return KnowledgeDownload(mime_type, headers, blocks(), stack)
        except BaseException:
            await stack.aclose()
            raise

    async def open_content(self, retrieval_ref, key, *, auth_context):
        import re
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("知识内容句柄无效")
        path = self.prefix + "/results/" + str(UUID(str(retrieval_ref))) + "/content/" + key
        return await self.open_download(path, auth_context=auth_context, max_bytes=33554432)

    async def open_source(self, library_id, version_id, *, auth_context):
        path = self.prefix + self.library_path(library_id) + "/versions/" + str(UUID(str(version_id))) + "/source"
        return await self.open_download(path, auth_context=auth_context, max_bytes=268435456, attachment=True)
