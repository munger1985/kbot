"""LLM 进程的 Responses 内部路由。"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from model_serving.common.model_registry import ModelDefinitionNotFound, ModelRegistryConflict
from model_serving.llm.responses.errors import GenerativeAdapterError
from model_serving.llm.responses.service import ResponsesService
from platform_core.contracts import (
    INTERNAL_API_V1,
    ImageGenerationRequest,
    ImageGenerationResult,
    ModelCapabilityVerifyRequest,
    ModelCatalogItem,
    ResearchRequest,
    ResearchResult,
)
from platform_core.security import get_actor_id


def _service(request: Request) -> ResponsesService:
    service = getattr(request.app.state, "responses_service", None)
    if service is None:
        raise HTTPException(503, {"code": "RESPONSES_NOT_READY", "message": "Responses 服务尚未初始化"})
    return service


def _auth_context(request: Request):
    context = getattr(request.state, "auth_context", None)
    if context is None:
        raise HTTPException(401, "缺少内部身份上下文")
    return context


def _raise(exc: Exception) -> None:
    if isinstance(exc, ModelDefinitionNotFound):
        raise HTTPException(404, {"code": exc.code, "message": str(exc)}) from exc
    if isinstance(exc, ModelRegistryConflict):
        status = 409
        raise HTTPException(status, {"code": exc.code, "message": str(exc)}) from exc
    if isinstance(exc, GenerativeAdapterError):
        raise HTTPException(exc.status_code, {"code": exc.code, "message": exc.message}) from exc
    raise exc


def create_responses_router() -> APIRouter:
    router = APIRouter(tags=["LLM Responses"])

    @router.post(f"{INTERNAL_API_V1}/responses/research", response_model=ResearchResult)
    async def research(payload: ResearchRequest, request: Request) -> ResearchResult:
        try:
            return await _service(request).research(payload)
        except Exception as exc:
            _raise(exc)

    @router.post(
        f"{INTERNAL_API_V1}/responses/image-generations",
        response_model=ImageGenerationResult,
    )
    async def generate_image(
        payload: ImageGenerationRequest, request: Request,
    ) -> ImageGenerationResult:
        try:
            return await _service(request).generate_image(payload)
        except Exception as exc:
            _raise(exc)

    @router.post(
        f"{INTERNAL_API_V1}/models/{{model_id}}/capabilities/image-generation:verify",
        response_model=ModelCatalogItem,
    )
    async def verify_image_generation(
        model_id: UUID, payload: ModelCapabilityVerifyRequest, request: Request,
    ):
        try:
            context = _auth_context(request)
            return await _service(request).verify_image_generation(
                model_id,
                expected_row_version=payload.expected_row_version,
                actor_id=get_actor_id(request),
                trace_id=context.trace_id,
                auth_context=context,
            )
        except Exception as exc:
            _raise(exc)

    return router
