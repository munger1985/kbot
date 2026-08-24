"""Main API 对外提供的安全模型目录。"""

import asyncio
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from platform_clients import AIModelConfigClient
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/model-catalog",
    tags=["Model Catalog"],
)


class ModelCatalogItem(BaseModel):
    model_id: UUID
    served_model_name: str
    display_name: str
    category: int
    provider: str
    status: str
    model_params: dict[str, Any] = Field(default_factory=dict)


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
        {
            **row,
            "model_params": {
                key: value
                for key, value in (row.get("model_params") or {}).items()
                if key != "config_file"
            },
        }
        for batch in batches
        for row in batch
        if row.get("status") == "ACTIVE"
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("category") or 0),
            str(row.get("display_name") or ""),
            str(row.get("served_model_name") or ""),
        )
    )
    return rows


@router.get("", response_model=list[ModelCatalogItem])
async def list_model_catalog(request: Request) -> list[dict[str, Any]]:
    """返回全局模型目录，供已认证管理页面配置业务对象。"""
    return await load_model_catalog(request)
