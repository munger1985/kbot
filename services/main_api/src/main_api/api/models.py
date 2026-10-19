"""Main API 对外提供的安全模型目录。"""

import asyncio
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from platform_clients import AIModelConfigClient
from platform_core.contracts import AuthContext, PUBLIC_API_V1
from platform_core.dictionary import coerce_model_category, is_enabled_model_status


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/model-catalog",
    tags=["Model Catalog"],
)


class PublicModelCatalogItem(BaseModel):
    model_id: UUID
    served_model_name: str
    display_name: str
    provider_model_name: str
    category: int
    provider: str
    status: str
    model_params: dict[str, Any] = Field(default_factory=dict)
    supports_image_generation: bool = False


def _clients(request: Request) -> tuple[AIModelConfigClient, ...]:
    clients = getattr(request.app.state, "model_config_clients", None)
    if not clients:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_CATALOG_UNAVAILABLE",
                "message": "模型目录客户端尚未初始化",
            },
        )
    return tuple(clients)


def model_config_client(
    request: Request, *, category: int,
) -> AIModelConfigClient:
    """按模型类别选择唯一归属服务。"""
    matches = tuple(
        client
        for client in _clients(request)
        if getattr(client, "category", None) == category
    )
    if len(matches) != 1:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_CONFIGURATION_SERVICE_UNAVAILABLE",
                "message": f"模型类别 {category} 的配置服务不可用",
            },
        )
    return matches[0]


async def load_managed_model_catalog(
    request: Request, *, auth_context: AuthContext,
) -> list[dict[str, Any]]:
    """严格聚合全部生命周期模型；任一归属服务失败即拒绝伪造完整目录。"""
    results = await asyncio.gather(
        *(
            client.list_models(auth_context=auth_context)
            for client in _clients(request)
        ),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        logger.warning(
            "完整模型配置目录读取失败：failed_services={}", len(failures),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_CONFIGURATION_CATALOG_UNAVAILABLE",
                "message": "部分模型配置服务不可用，无法返回完整目录",
            },
        )
    rows = [
        row
        for result in results
        for row in result
        if isinstance(row, dict)
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("category") or 0),
            str(row.get("display_name") or ""),
            str(row.get("served_model_name") or ""),
        )
    )
    return rows


async def load_model_provider_options(
    request: Request, *, auth_context: AuthContext,
) -> list[dict[str, Any]]:
    """返回所有模型归属服务声明的 Provider 表单合同。"""
    results = await asyncio.gather(
        *(
            client.list_provider_options(auth_context=auth_context)
            for client in _clients(request)
        ),
        return_exceptions=True,
    )
    if any(isinstance(result, BaseException) for result in results):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_PROVIDER_CATALOG_UNAVAILABLE",
                "message": "部分模型 Provider 目录不可用",
            },
        )
    rows = [
        row
        for result in results
        for row in result
        if isinstance(row, dict)
    ]
    rows.sort(key=lambda row: (int(row.get("category") or 0), row.get("provider") or ""))
    return rows


def _normalize_catalog_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """只保留已启用模型，并把类别/状态规范成公开目录契约。"""
    category = coerce_model_category(row.get("category"))
    if category is None or not is_enabled_model_status(row.get("status")):
        return None
    return {
        **row,
        "category": int(category),
        "status": "ACTIVE",
        "model_params": {
            key: value
            for key, value in (row.get("model_params") or {}).items()
            if key != "config_file"
        },
    }


async def load_model_catalog(request: Request) -> list[dict[str, Any]]:
    """聚合各模型进程中的已启用模型，供各公开 App 安全复用。"""
    results = await asyncio.gather(
        *(client.list_models() for client in _clients(request)),
        return_exceptions=True,
    )
    batches = []
    for result in results:
        if isinstance(result, BaseException):
            logger.warning(
                "模型目录子服务暂时不可用，已跳过：{}",
                type(result).__name__,
            )
            continue
        batches.append(result)
    if not batches:
        raise HTTPException(
            status_code=503,
            detail="所有模型目录服务当前均不可用",
        )
    rows = [
        normalized
        for batch in batches
        for row in batch
        if (normalized := _normalize_catalog_row(row)) is not None
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("category") or 0),
            str(row.get("display_name") or ""),
            str(row.get("served_model_name") or ""),
        )
    )
    return rows


@router.get("", response_model=list[PublicModelCatalogItem])
async def list_model_catalog(request: Request) -> list[dict[str, Any]]:
    """返回全局模型目录，供已认证管理页面配置业务对象。"""
    return await load_model_catalog(request)
