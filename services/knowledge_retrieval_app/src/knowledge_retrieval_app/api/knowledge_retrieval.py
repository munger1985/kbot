"""App 知识检索内部端点；不提供任意后端、表名、模型策略或文件路径。"""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from knowledge_algorithms.contracts.errors import AlgorithmError, require
from knowledge_algorithms.contracts.identity import AuthorizedScope, EvidenceKey
from knowledge_algorithms.contracts.query import QueryInput
from knowledge_algorithms.contracts.wire import decode, encode
from ..domain.knowledge_ingestion import OWNER
from .knowledge_http import bounded_request


def create_knowledge_retrieval_router(*, service=None, access_dependency):
    fixed_service = service
    def resolve_service(request: Request):
        return fixed_service or request.app.state.knowledge_retrieval_service
    prefix = {"knowledge_retrieval": "/internal/v1/knowledge-retrieval/knowledge",
              "aiops": "/internal/v1/aiops/knowledge"}[OWNER]
    router = APIRouter(prefix=prefix, tags=["app-owned-knowledge"])

    async def body(request, required, max_bytes=65536):
        value = await bounded_request(request, max_bytes=max_bytes)
        require(type(value) is dict and set(value) == set(required), "知识请求字段不匹配")
        return value

    def identifier(value):
        require(type(value) is str and len(value) <= 36, "知识版本标识无效")
        try:
            return UUID(value)
        except ValueError:
            require(False, "知识版本标识无效")

    @router.post("/versions/{version_id}:publish")
    async def publish(version_id: UUID, request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("resource_ref", "generation_id", "expected_generation_id"))
        await service.publish(access, resource_ref=value["resource_ref"], version_id=version_id,
            generation_id=identifier(value["generation_id"]),
            expected_generation_id=identifier(value["expected_generation_id"])
                if value["expected_generation_id"] is not None else None)
        return {"status": "PUBLISHED"}

    @router.post("/versions/{version_id}:withdraw")
    async def withdraw(version_id: UUID, request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("resource_ref", "expected_revision"))
        require(type(value["expected_revision"]) is int and value["expected_revision"] >= 0, "撤回版本修订无效")
        await service.withdraw(access, resource_ref=value["resource_ref"], version_id=version_id,
                               expected_revision=value["expected_revision"])
        return {"status": "WITHDRAWN"}

    @router.post("/citation-targets")
    async def citation_target(request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("resource_ref", "target", "identifiers"))
        require(type(value["identifiers"]) is list, "文献标识清单无效")
        await service.register_citation_target(access, resource_ref=value["resource_ref"],
            target=decode(EvidenceKey, value["target"]), identifiers=tuple(value["identifiers"]))
        return {"status": "REGISTERED"}

    @router.post("/scopes")
    async def scope(request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("resource_ref",))
        return encode(await service.create_scope(access, resource_ref=value["resource_ref"]))

    @router.post("/retrieve")
    async def retrieve(request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("scope", "query", "profile_ref"))
        receipt = await service.retrieve(access, scope=decode(AuthorizedScope, value["scope"]),
            query=decode(QueryInput, value["query"]), profile_ref=value["profile_ref"])
        return {"retrieval_ref": receipt.retrieval_ref, "evidence_set": encode(receipt.evidence_set)}

    @router.post("/retrievals/{retrieval_ref}:continue")
    async def continuation(retrieval_ref: UUID, request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("cursor",))
        receipt = await service.continue_retrieval(access, retrieval_ref=str(retrieval_ref), cursor=value["cursor"])
        return {"retrieval_ref": receipt.retrieval_ref, "evidence_set": encode(receipt.evidence_set)}

    @router.post("/query-images")
    async def query_image(request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        import base64
        import binascii
        value = await body(request, ("scope", "image_base64", "mime_type"), max_bytes=12000000)
        require(type(value["image_base64"]) is str, "图片编码无效")
        try:
            raw = base64.b64decode(value["image_base64"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(422, detail={"code": "INVALID_QUERY_IMAGE"}) from exc
        handle = await service.register_query_image(access, scope=decode(AuthorizedScope, value["scope"]),
            raw=raw, mime_type=value["mime_type"])
        return {"image_handle": handle}

    @router.post("/retrievals/{retrieval_ref}/relations:continue")
    async def relations(retrieval_ref: UUID, request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("anchor_key", "cursor"))
        receipt = await service.continue_relations(access, retrieval_ref=str(retrieval_ref),
            anchor_key=decode(EvidenceKey, value["anchor_key"]), cursor=value["cursor"])
        return {"retrieval_ref": receipt.retrieval_ref, "evidence_set": encode(receipt.evidence_set)}

    @router.post("/evidence:retain")
    async def retain(request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("retrieval_ref", "evidence_keys", "expires_at"))
        require(type(value["evidence_keys"]) is list, "保留证据清单无效")
        return {"pin_ids": await service.retain_evidence(access, retrieval_ref=value["retrieval_ref"],
            keys=tuple(decode(EvidenceKey, key) for key in value["evidence_keys"]), expires_at=value["expires_at"])}

    @router.post("/evidence:release-run")
    async def release_run(request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("scope",))
        await service.release_run(access, scope=decode(AuthorizedScope, value["scope"]))
        return {"status": "RELEASED"}

    @router.post("/evidence:release")
    async def release(request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("pin_ids",))
        require(type(value["pin_ids"]) is list, "保留引用清单无效")
        await service.release_evidence(access, pin_ids=[str(identifier(value)) for value in value["pin_ids"]])
        return {"status": "RELEASED"}

    @router.post("/evidence:read")
    async def read(request: Request, access=Depends(access_dependency), service=Depends(resolve_service)):
        value = await body(request, ("retrieval_ref", "evidence_keys"))
        require(type(value["evidence_keys"]) is list, "证据清单无效")
        keys = tuple(decode(EvidenceKey, raw) for raw in value["evidence_keys"])
        return {"items": await service.read_manifest(access, retrieval_ref=value["retrieval_ref"], keys=keys)}

    @router.get("/retrievals/{retrieval_ref}/content/{stable_key}")
    async def content(retrieval_ref: UUID, stable_key: str, access=Depends(access_dependency), service=Depends(resolve_service)):
        require(len(stable_key) == 64 and all(c in "0123456789abcdef" for c in stable_key), "证据句柄无效")
        item, chunks = await service.stream_content(access, retrieval_ref=str(retrieval_ref), stable_key=stable_key)
        return StreamingResponse(chunks, media_type=item.mime_type,
            headers={"Cache-Control": "no-store", "Content-Length": str(item.byte_count),
                     "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; sandbox"})

    @router.get("/history/{retrieval_ref}/content/{stable_key}")
    async def historical_content(retrieval_ref: UUID, stable_key: str,
            access=Depends(access_dependency), service=Depends(resolve_service)):
        require(len(stable_key) == 64 and all(c in "0123456789abcdef" for c in stable_key), "历史证据句柄无效")
        item, chunks = await service.stream_historical_content(access,
            retrieval_ref=str(retrieval_ref), stable_key=stable_key)
        return StreamingResponse(chunks, media_type=item.mime_type,
            headers={"Cache-Control": "no-store", "Content-Length": str(item.byte_count),
                     "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; sandbox"})

    return router


def install_knowledge_errors(app):
    from fastapi.responses import JSONResponse
    @app.exception_handler(AlgorithmError)
    async def algorithm_error(request, exc):
        del request
        status = 409 if exc.code.value in {"SNAPSHOT_STALE", "SCOPE_CHANGED"} else 422
        return JSONResponse(status_code=status, content={"code": exc.code.value, "stage": exc.stage})
