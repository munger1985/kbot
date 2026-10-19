"""图像生成与编辑内部路由。"""

from fastapi import APIRouter, HTTPException, Request

from model_serving.common.model_registry import (
    ModelDefinitionNotFound,
    ModelRegistryConflict,
)
from model_serving.llm.responses.errors import GenerativeAdapterError
from platform_core.contracts import (
    INTERNAL_API_V1,
    ImageGenerationRequest,
    ImageGenerationResult,
)

from .service import ImageService


def _service(request: Request) -> ImageService:
    service = getattr(request.app.state, "image_service", None)
    if service is None:
        raise HTTPException(
            503,
            {"code": "IMAGE_SERVICE_NOT_READY", "message": "图像生成服务尚未初始化"},
        )
    return service


def _raise(exc: Exception) -> None:
    if isinstance(exc, ModelDefinitionNotFound):
        raise HTTPException(404, {"code": exc.code, "message": str(exc)}) from exc
    if isinstance(exc, ModelRegistryConflict):
        raise HTTPException(409, {"code": exc.code, "message": str(exc)}) from exc
    if isinstance(exc, GenerativeAdapterError):
        raise HTTPException(
            exc.status_code, {"code": exc.code, "message": exc.message},
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(
            422,
            {"code": "IMAGE_PROVIDER_UNSUPPORTED", "message": str(exc)},
        ) from exc
    raise exc


def create_image_router() -> APIRouter:
    router = APIRouter(tags=["Image Generation"])

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

    return router
