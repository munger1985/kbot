"""App 自有源数据端点；服务及 AuthContext 校验由固定依赖完成。"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from knowledge_algorithms.contracts.errors import AlgorithmError, require
from knowledge_algorithms.contracts.wire import decode
from knowledge_algorithms.parsing.events import ParseRunIdentity
from knowledge_retrieval_app.api.knowledge_http import bounded_request


def create_knowledge_source_router(*, ingestion=None, scope_dependency):
    fixed_ingestion = ingestion
    def resolve_ingestion(request: Request):
        return fixed_ingestion or request.app.state.knowledge_ingestion_service
    router = APIRouter(prefix="/internal/v1/knowledge-retrieval/knowledge", tags=["app-knowledge-source"])

    @router.post("/sources:read")
    async def read(request: Request, scope=Depends(scope_dependency), ingestion=Depends(resolve_ingestion)):
        value = await bounded_request(request)
        try:
            require(set(value) == {"job_ref", "identity"}, "源读取请求字段不匹配")
            identity = decode(ParseRunIdentity, value["identity"])
            reader = await ingestion.source_reader(scope, UUID(value["job_ref"]), identity=identity)
        except (AlgorithmError, ValueError, TypeError) as exc:
            raise HTTPException(422, detail={"code": "SOURCE_ACCESS_REJECTED", "message": "源读取被拒绝"}) from exc
        return StreamingResponse(reader.chunks(), media_type=reader.descriptor.mime_type,
            headers={"Content-Length": str(reader.descriptor.byte_count), "Cache-Control": "no-store",
                     "X-Content-Type-Options": "nosniff"})

    return router
