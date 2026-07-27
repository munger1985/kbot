"""仅在 development 环境注册的本地日志浏览 API。"""

from pathlib import Path

from fastapi import APIRouter, Query, Request
from typing import Literal

from main_api.developer_tools import LocalLogSearchService
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/development/logs",
    tags=["Development Log Console"],
)


def _service(request: Request) -> LocalLogSearchService:
    return LocalLogSearchService(
        log_root=Path(request.app.state.development_log_root)
    )


@router.get("/services")
async def list_log_services(request: Request):
    return {"services": _service(request).services()}


@router.get("/events")
async def list_log_events(
    request: Request,
    service_name: str = Query(min_length=1, max_length=64),
    log_type: Literal["RUNTIME", "ACCESS"] = "RUNTIME",
    level: list[str] = Query(default=[]),
    keyword: str | None = Query(default=None, max_length=256),
    limit: int = Query(default=200, ge=1, le=500),
):
    events = _service(request).search(
        service_name=service_name,
        log_type=log_type,
        levels=set(level),
        keyword=keyword,
        limit=limit,
    )
    return {"events": events, "count": len(events)}
