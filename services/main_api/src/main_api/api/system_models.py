"""系统管理的全局模型配置公开接口。"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from main_api.api.models import (
    load_managed_model_catalog,
    load_model_provider_options,
    model_config_client,
)
from main_api.application.access_control import AccessDeniedError, MODEL_MANAGEMENT_PERMISSION
from platform_clients import AIModelConfigClientError
from platform_core.contracts import (
    ModelArchiveRequest,
    ModelCatalogItem as ManagedModelCatalogItem,
    ModelConnectionTestResult,
    ModelCreateRequest,
    ModelDeleteRequest,
    ModelProviderOption,
    ModelReferenceSummary,
    ModelStatusRequest,
    ModelUpdateRequest,
    PUBLIC_API_V1,
)
from platform_core.dictionary import ModelCategory


async def require_model_manager(request: Request) -> None:
    """仅接受有效平台用户会话，每次重新核对全局模型权限。"""
    claims = getattr(request.state, "user_token_claims", None)
    if claims is None or claims.entry_kind != "PLATFORM":
        raise HTTPException(403, {
            "code": "PLATFORM_SESSION_REQUIRED", "message": "请使用平台账号登录系统管理",
        })
    try:
        await request.app.state.access_control_service.require_platform(
            user_id=claims.user_id, permission_code=MODEL_MANAGEMENT_PERMISSION,
        )
    except AccessDeniedError as exc:
        raise HTTPException(403, {
            "code": "PLATFORM_PERMISSION_DENIED", "message": "当前账号没有管理 AI 模型的权限",
            "permission": MODEL_MANAGEMENT_PERMISSION,
        }) from exc


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/platform",
    tags=["System Management"],
    dependencies=[Depends(require_model_manager)],
)


def _raise_model_configuration_error(exc: AIModelConfigClientError) -> None:
    detail: dict[str, Any] = {
        "code": exc.code,
        "message": exc.message,
    }
    if exc.context:
        detail["context"] = exc.context
    raise HTTPException(status_code=exc.status_code, detail=detail) from exc


@router.get(
    "/models/provider-options", response_model=list[ModelProviderOption],
)
async def list_model_provider_options(request: Request):
    return await load_model_provider_options(
        request, auth_context=request.state.auth_context,
    )


@router.get("/models", response_model=list[ManagedModelCatalogItem])
async def list_models(request: Request):
    return await load_managed_model_catalog(
        request, auth_context=request.state.auth_context,
    )


@router.post(
    "/models", status_code=status.HTTP_201_CREATED,
    response_model=ManagedModelCatalogItem,
    openapi_extra={"x-sensitive-input": True},
)
async def create_model(payload: ModelCreateRequest, request: Request):
    client = model_config_client(request, category=payload.category)
    try:
        return await client.create_model(
            payload.model_dump(mode="json"),
            auth_context=request.state.auth_context,
        )
    except AIModelConfigClientError as exc:
        _raise_model_configuration_error(exc)


@router.patch(
    "/models/{category}/{model_id}", response_model=ManagedModelCatalogItem,
    openapi_extra={"x-sensitive-input": True},
)
async def update_model(
    category: ModelCategory,
    model_id: UUID,
    payload: ModelUpdateRequest,
    request: Request,
):
    client = model_config_client(request, category=category)
    try:
        return await client.update_model(
            model_id,
            payload.model_dump(mode="json", exclude_unset=True),
            auth_context=request.state.auth_context,
        )
    except AIModelConfigClientError as exc:
        _raise_model_configuration_error(exc)


@router.patch(
    "/models/{category}/{model_id}/status",
    response_model=ManagedModelCatalogItem,
)
async def change_model_status(
    category: ModelCategory,
    model_id: UUID,
    payload: ModelStatusRequest,
    request: Request,
):
    client = model_config_client(request, category=category)
    try:
        return await client.change_model_status(
            model_id,
            payload.model_dump(mode="json"),
            auth_context=request.state.auth_context,
        )
    except AIModelConfigClientError as exc:
        _raise_model_configuration_error(exc)


@router.get(
    "/models/{category}/{model_id}/references",
    response_model=ModelReferenceSummary,
)
async def get_model_references(
    category: ModelCategory, model_id: UUID, request: Request,
):
    client = model_config_client(request, category=category)
    try:
        return await client.get_model_references(
            model_id, auth_context=request.state.auth_context,
        )
    except AIModelConfigClientError as exc:
        _raise_model_configuration_error(exc)


@router.post(
    "/models/{category}/{model_id}:test-connection",
    response_model=ModelConnectionTestResult,
)
async def test_model_connection(category: ModelCategory, model_id: UUID, request: Request):
    client = model_config_client(request, category=category)
    try:
        return await client.test_model_connection(
            model_id, auth_context=request.state.auth_context,
        )
    except AIModelConfigClientError as exc:
        _raise_model_configuration_error(exc)


@router.post("/models/{category}/{model_id}/archive")
async def archive_model(
    category: ModelCategory,
    model_id: UUID,
    payload: ModelArchiveRequest,
    request: Request,
):
    client = model_config_client(request, category=category)
    try:
        return await client.archive_model(
            model_id,
            payload.model_dump(mode="json"),
            auth_context=request.state.auth_context,
        )
    except AIModelConfigClientError as exc:
        _raise_model_configuration_error(exc)


@router.delete(
    "/models/{category}/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_model(
    category: ModelCategory,
    model_id: UUID,
    payload: ModelDeleteRequest,
    request: Request,
):
    client = model_config_client(request, category=category)
    try:
        await client.delete_model(
            model_id,
            payload.model_dump(mode="json"),
            auth_context=request.state.auth_context,
        )
    except AIModelConfigClientError as exc:
        _raise_model_configuration_error(exc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
