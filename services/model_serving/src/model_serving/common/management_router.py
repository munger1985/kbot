"""各类模型进程共享的目录管理 HTTP 适配器。"""

import asyncio
from collections.abc import Callable
from time import perf_counter
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status
from loguru import logger

from model_serving.common.connection_probe import MODEL_PROBE_TIMEOUT_SECONDS, probe_model
from model_serving.common.model_registry import (
    ModelDefinitionNotFound,
    ModelRegistryConflict,
    ModelRegistryService,
)
from model_serving.common.provider_catalog import list_provider_options
from platform_core.contracts import (
    INTERNAL_API_V1,
    ModelArchiveRequest,
    ModelCatalogItem,
    ModelConnectionTestResult,
    ModelCreateRequest,
    ModelDeleteRequest,
    ModelProviderOption,
    ModelReferenceSummary,
    ModelStatusRequest,
    ModelUpdateRequest,
)
from platform_core.security import get_actor_id


def _service(request: Request) -> ModelRegistryService:
    return request.app.state.model_registry


def _auth_context(request: Request):
    context = getattr(request.state, "auth_context", None)
    if context is None:
        raise HTTPException(status_code=401, detail="缺少内部身份上下文")
    return context


def _raise_registry_error(exc: Exception) -> None:
    if isinstance(exc, ModelDefinitionNotFound):
        code = 404
    elif isinstance(exc, ModelRegistryConflict):
        code = 503 if exc.code == "MODEL_REFERENCE_CHECK_UNAVAILABLE" else 409
    elif isinstance(exc, ValueError):
        code = 422
    else:
        raise exc
    error_code = getattr(exc, "code", "MODEL_CONFIGURATION_INVALID")
    detail = {"code": error_code, "message": str(exc)}
    if isinstance(exc, ModelRegistryConflict) and exc.details:
        detail["context"] = exc.details
    raise HTTPException(
        status_code=code,
        detail=detail,
    ) from exc


def create_model_management_router(
    *, category: int, probe_service: Callable[[Request], object] | None = None,
) -> APIRouter:
    router = APIRouter(
        prefix=f"{INTERNAL_API_V1}/models",
        tags=["Model Configuration"],
    )

    @router.get("/provider-options", response_model=list[ModelProviderOption])
    async def provider_options() -> list[ModelProviderOption]:
        return list_provider_options(category=category)

    @router.get("", response_model=list[ModelCatalogItem])
    async def list_models(request: Request):
        return await _service(request).list(category=category)

    @router.get("/{model_id}", response_model=ModelCatalogItem)
    async def get_model(model_id: UUID, request: Request):
        try:
            return await _service(request).get(model_id, category=category)
        except Exception as exc:
            _raise_registry_error(exc)

    @router.get(
        "/{model_id}/references", response_model=ModelReferenceSummary,
    )
    async def get_model_references(model_id: UUID, request: Request):
        try:
            await _service(request).get(model_id, category=category)
            return await _service(request).references(
                model_id, auth_context=_auth_context(request),
            )
        except Exception as exc:
            _raise_registry_error(exc)

    @router.post("/{model_id}:test-connection", response_model=ModelConnectionTestResult)
    async def test_model_connection(model_id: UUID, request: Request):
        context = _auth_context(request)
        try:
            model = await _service(request).get(model_id, category=category)
        except ModelDefinitionNotFound as exc:
            _raise_registry_error(exc)
        if model["status"] != "ACTIVE":
            raise HTTPException(409, {
                "code": "MODEL_NOT_ACTIVE", "message": "请先启用模型，再测试连通性。",
            })
        if probe_service is None:
            raise HTTPException(501, {
                "code": "MODEL_CONNECTION_TEST_UNAVAILABLE", "message": "当前模型服务未提供连接测试。",
            })
        started = perf_counter()
        try:
            await asyncio.wait_for(probe_model(
                category=category, service=probe_service(request), model=model,
                model_id=model_id, trace_id=context.trace_id,
            ), timeout=MODEL_PROBE_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            raise HTTPException(504, {
                "code": "MODEL_CONNECTION_TEST_TIMEOUT", "message": "模型测试超时，请检查模型地址或稍后重试。",
            }) from exc
        except Exception as exc:
            logger.warning(
                "模型测试失败 | model_id={} | category={} | request_id={} | error_type={}",
                model_id, category, context.request_id, type(exc).__name__,
            )
            raise HTTPException(502, {
                "code": "MODEL_CONNECTION_TEST_FAILED",
                "message": "模型未能完成测试推理，请检查模型地址、厂商模型名称、凭据或本地模型资源。",
            }) from exc
        return ModelConnectionTestResult(latency_ms=round((perf_counter() - started) * 1000))

    @router.post("", status_code=201, response_model=ModelCatalogItem)
    async def create_model(payload: ModelCreateRequest, request: Request):
        if payload.category != category:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "MODEL_CATEGORY_MISMATCH",
                    "message": "模型类别不属于当前推理进程",
                },
            )
        try:
            return await _service(request).create(
                payload.model_dump(), actor_id=get_actor_id(request),
                auth_context=_auth_context(request),
            )
        except Exception as exc:
            _raise_registry_error(exc)

    @router.patch("/{model_id}", response_model=ModelCatalogItem)
    async def update_model(
        model_id: UUID, payload: ModelUpdateRequest, request: Request,
    ):
        values = payload.model_dump(
            exclude={"expected_row_version"}, exclude_unset=True,
        )
        try:
            return await _service(request).update(
                model_id, values,
                expected_row_version=payload.expected_row_version,
                actor_id=get_actor_id(request),
                auth_context=_auth_context(request),
            )
        except Exception as exc:
            _raise_registry_error(exc)

    @router.patch("/{model_id}/status", response_model=ModelCatalogItem)
    async def change_model_status(
        model_id: UUID, payload: ModelStatusRequest, request: Request,
    ):
        try:
            return await _service(request).change_status(
                model_id, target_status=payload.status,
                expected_row_version=payload.expected_row_version,
                actor_id=get_actor_id(request),
                auth_context=_auth_context(request),
            )
        except Exception as exc:
            _raise_registry_error(exc)

    @router.post("/{model_id}/archive")
    async def archive_model(
        model_id: UUID, payload: ModelArchiveRequest, request: Request,
    ):
        try:
            model, references = await _service(request).archive(
                model_id,
                expected_row_version=payload.expected_row_version,
                actor_id=get_actor_id(request),
                auth_context=_auth_context(request),
            )
            return {"model": model, "reference_summary": references}
        except Exception as exc:
            _raise_registry_error(exc)

    @router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_model(
        model_id: UUID, payload: ModelDeleteRequest, request: Request,
    ) -> Response:
        try:
            await _service(request).delete(
                model_id,
                expected_row_version=payload.expected_row_version,
                auth_context=_auth_context(request),
            )
            return Response(status_code=204)
        except Exception as exc:
            _raise_registry_error(exc)

    return router


__all__ = ["create_model_management_router"]
