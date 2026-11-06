"""解析计算内部协议；可信访问范围只由认证依赖提供。"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.wire import decode
from knowledge_algorithms.parsing.contracts import ParseProfile, SourceDescriptor
from knowledge_algorithms.parsing.events import ParseRunIdentity
from knowledge_algorithms.parsing.wire import event_to_wire
from knowledge_core.domain.parse_jobs import ParseJob


async def bounded_json(request, *, max_bytes=1048576):
    value = bytearray()
    async for block in request.stream():
        if len(value) + len(block) > max_bytes:
            raise HTTPException(413, detail={"code": "RESOURCE_LIMIT", "message": "请求体超过预算"})
        value.extend(block)
    try:
        result = json.loads(value)
        require(isinstance(result, dict), "请求必须为 JSON 对象")
        return result
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail={"code": "INVALID_CONTRACT", "message": "请求 JSON 无效"}) from exc


def failure(exc):
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, detail={"code": "NOT_FOUND", "message": "计算资源不存在"})
    code = exc.code if isinstance(exc, AlgorithmError) else ErrorCode.INVALID_CONTRACT
    return HTTPException(410 if code == ErrorCode.ARTIFACT_EXPIRED else 422,
        detail={"code": code.value, "message": "计算请求被拒绝"})


def create_parse_jobs_router(*, service, access_dependency, source_resolver):
    """组合根必须固定依赖和源读取注册表，不接受请求内回调 URL。"""
    router = APIRouter(prefix="/internal/v1/knowledge/parse-jobs", tags=["knowledge-computation"])

    @router.post("", status_code=202)
    async def submit(request: Request, access=Depends(access_dependency)):
        value = await bounded_json(request)
        try:
            require(set(value) == {"run", "source", "profile", "expires_at"}, "计算请求字段不匹配")
            job = ParseJob(access, decode(ParseRunIdentity, value["run"]), decode(SourceDescriptor, value["source"]),
                           decode(ParseProfile, value["profile"]), value["expires_at"])
            reader = await source_resolver(job, request)
            ref = await service.submit(job, reader)
            return {"job_ref": ref}
        except (AlgorithmError, ValueError, TypeError, KeyError, FileNotFoundError) as exc:
            raise failure(exc) from exc

    @router.get("/{job_ref}/events")
    async def events(job_ref: str, after_sequence: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
                     access=Depends(access_dependency)):
        try:
            page = service.events(access, job_ref, after_sequence=after_sequence, limit=limit)
            _, _, checkpoint = service.store.load(access, job_ref, service.clock())
            return {"events": [event_to_wire(event) for event in page],
                    "next_cursor": page[-1].sequence if page else after_sequence,
                    "status": checkpoint["status"], "event_digests": [event.event_digest for event in page]}
        except (AlgorithmError, FileNotFoundError) as exc:
            raise failure(exc) from exc

    @router.post("/{job_ref}/acknowledgements")
    async def ack(job_ref: str, request: Request, access=Depends(access_dependency)):
        value = await bounded_json(request, max_bytes=4096)
        try:
            require(set(value) == {"sequence", "event_digest"}, "确认请求字段不匹配")
            service.acknowledge(access, job_ref, sequence=value["sequence"], event_digest=value["event_digest"])
            return {"acknowledged_sequence": value["sequence"]}
        except (AlgorithmError, FileNotFoundError, TypeError) as exc:
            raise failure(exc) from exc

    @router.post("/{job_ref}:cancel")
    async def cancel(job_ref: str, access=Depends(access_dependency)):
        try:
            await service.cancel(access, job_ref)
            return {"status": "CANCELLED"}
        except (AlgorithmError, FileNotFoundError) as exc:
            raise failure(exc) from exc

    @router.get("/{job_ref}/artifacts/{artifact_ref}")
    async def artifact(job_ref: str, artifact_ref: str, access=Depends(access_dependency)):
        iterator = service.artifact(access, job_ref, artifact_ref, max_bytes=67108864)
        try:
            # 在 HTTP 头发送前完成身份、期限、句柄和文件打开验证。
            first = await anext(iterator)
        except (AlgorithmError, FileNotFoundError) as exc:
            await iterator.aclose()
            raise failure(exc) from exc
        except StopAsyncIteration:
            raise HTTPException(404, detail={"code": "NOT_FOUND"})
        async def body():
            try:
                yield first
                async for block in iterator:
                    yield block
            finally:
                await iterator.aclose()
        return StreamingResponse(body(), media_type="application/octet-stream",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    return router
