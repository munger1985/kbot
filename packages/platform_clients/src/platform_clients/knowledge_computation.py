"""App 知识纯计算 HTTP 传输；每个请求签发新内部身份，不携带存储回调地址。"""

from contextlib import asynccontextmanager
import json
import re

import aiohttp
from platform_core.security import build_scoped_internal_auth_headers


class KnowledgeComputationClientError(RuntimeError):
    def __init__(self, status_code, code):
        super().__init__("知识计算请求失败")
        self.status_code, self.code = status_code, code


class KnowledgeComputationClient:
    def __init__(self, *, base_url, caller_service, audience, timeout_seconds=600, session=None,
                 service_scopes=("knowledge.parse.compute",)):
        self.base_url, self.caller_service, self.audience = base_url.rstrip("/"), caller_service, audience
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.session = session
        self.service_scopes = service_scopes

    def _auth_headers(self, auth_context):
        """纯计算服务使用带操作范围的短期服务身份。"""
        return build_scoped_internal_auth_headers(audience=self.audience, caller_service=self.caller_service,
                                                 scopes=self.service_scopes, context=auth_context)

    @asynccontextmanager
    async def response(self, method, path, *, auth_context, payload=None, params=None, extra_headers=None,
                       body=None, content_type=None, idempotency_key=None):
        # base_url 与 path 仅由组合根和本模块静态方法决定。
        headers = self._auth_headers(auth_context)
        if extra_headers:
            if set(extra_headers) != {"X-Knowledge-Delegation"}:
                raise ValueError("不允许覆盖内部认证头")
            headers.update(extra_headers)
        if body is not None:
            if payload is not None or not content_type or not idempotency_key:
                raise ValueError("知识上传缺少类型、幂等键或混入 JSON")
            headers.update({"Content-Type": content_type, "Idempotency-Key": idempotency_key})
        owned = self.session is None
        session = self.session or aiohttp.ClientSession(timeout=self.timeout)
        try:
            async with session.request(method, self.base_url + path, headers=headers, json=payload, data=body, params=params,
                                       timeout=self.timeout, allow_redirects=False) as response:
                if response.status >= 300:
                    code = "ARTIFACT_EXPIRED" if response.status == 410 else "COMPUTE_REQUEST_REJECTED"
                    raw = await response.content.read(4097)
                    if len(raw) <= 4096:
                        try:
                            problem = json.loads(raw)
                            detail = problem.get("detail", problem)
                            reported = detail.get("code") if isinstance(detail, dict) else None
                            if isinstance(reported, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reported):
                                code = reported
                        except (ValueError, TypeError, AttributeError):
                            pass
                    raise KnowledgeComputationClientError(response.status, code)
                yield response
        finally:
            if owned:
                await session.close()

    async def json(self, method, path, *, auth_context, payload=None, params=None, max_bytes=8388608):
        async with self.response(method, path, auth_context=auth_context, payload=payload, params=params) as response:
            value = bytearray()
            async for block in response.content.iter_chunked(65536):
                if len(value) + len(block) > max_bytes:
                    raise KnowledgeComputationClientError(502, "RESOURCE_LIMIT")
                value.extend(block)
            try:
                return json.loads(value)
            except (ValueError, TypeError) as exc:
                raise KnowledgeComputationClientError(502, "INVALID_CONTRACT") from exc

    @staticmethod
    def job_path(job_ref):
        if not isinstance(job_ref, str) or re.fullmatch(r"[0-9a-f]{64}", job_ref) is None:
            raise ValueError("计算任务句柄无效")
        return "/internal/v1/knowledge/parse-jobs/" + job_ref

    async def submit(self, payload, *, auth_context):
        return await self.json("POST", "/internal/v1/knowledge/parse-jobs", payload=payload, auth_context=auth_context)

    async def events(self, job_ref, *, after_sequence, limit, auth_context):
        return await self.json("GET", self.job_path(job_ref) + "/events", auth_context=auth_context,
                               params={"after_sequence":after_sequence, "limit":limit})

    async def acknowledge(self, job_ref, *, sequence, event_digest, auth_context):
        return await self.json("POST", self.job_path(job_ref) + "/acknowledgements", auth_context=auth_context,
                               payload={"sequence":sequence, "event_digest":event_digest})

    async def cancel(self, job_ref, *, auth_context):
        return await self.json("POST", self.job_path(job_ref) + ":cancel", auth_context=auth_context)

    async def artifact(self, job_ref, artifact_ref, *, auth_context, max_bytes=67108864):
        if not isinstance(artifact_ref,str) or re.fullmatch(r"[A-Za-z0-9_-]{1,256}",artifact_ref) is None:
            raise ValueError("制品句柄无效")
        async with self.response("GET", self.job_path(job_ref) + "/artifacts/" + artifact_ref,
                                 auth_context=auth_context) as response:
            count = 0
            async for block in response.content.iter_chunked(65536):
                count += len(block)
                if count > max_bytes:
                    raise KnowledgeComputationClientError(502,"RESOURCE_LIMIT")
                yield block


class ControlledAppSourceClient(KnowledgeComputationClient):
    def __init__(self, *, owner, **kwargs):
        kwargs["service_scopes"] = ("knowledge.source.read",)
        super().__init__(**kwargs)
        paths = {"knowledge_retrieval":"/internal/v1/knowledge-retrieval/knowledge/sources:read",
                 "aiops":"/internal/v1/aiops/knowledge/sources:read"}
        if owner not in paths:
            raise ValueError("未登记 App 源读取端点")
        self.path = paths[owner]

    async def source(self, payload, *, auth_context, max_bytes):
        async with self.response("POST", self.path, payload=payload, auth_context=auth_context) as response:
            count = 0
            async for block in response.content.iter_chunked(65536):
                count += len(block)
                if count > max_bytes:
                    raise KnowledgeComputationClientError(502,"RESOURCE_LIMIT")
                yield block
