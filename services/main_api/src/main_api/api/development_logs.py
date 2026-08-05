"""仅在 development 环境注册的受控日志浏览 API。"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from main_api.log_reader import LocalLogSearchService, LogQueryError
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/development/logs",
    tags=["Development Log Console"],
)


def _service(request: Request) -> LocalLogSearchService:
    """取得应用启动时构造的受控日志服务。"""
    return request.app.state.development_log_search_service


def _query_error(exc: LogQueryError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"code": "DEVELOPMENT_LOG_QUERY_INVALID", "message": str(exc)},
    )


@router.get("/services")
async def list_log_services(request: Request):
    return {"services": _service(request).services()}


@router.get("/events")
async def list_log_events(
    request: Request,
    service_name: str | None = Query(default=None, min_length=1, max_length=128),
    stream: list[Literal["RUNTIME", "ACCESS"]] = Query(default=[]),
    level: list[str] = Query(default=[]),
    filter_by_level: bool = Query(default=False),
    keyword: str | None = Query(default=None, max_length=256),
    request_id: str | None = Query(default=None, max_length=256),
    trace_id: str | None = Query(default=None, max_length=256),
    error_id: str | None = Query(default=None, max_length=256),
    run_id: str | None = Query(default=None, max_length=256),
    job_id: str | None = Query(default=None, max_length=256),
    http_status: int | None = Query(default=None, ge=100, le=599),
    started_at: datetime | None = Query(default=None),
    ended_at: datetime | None = Query(default=None),
    cursor: str | None = Query(default=None, max_length=4096),
    limit: int = Query(default=200, ge=1, le=2000),
):
    try:
        events, next_cursor, total = _service(request).search(
            service_name=service_name,
            streams=set(stream) or None,
            levels=set(level),
            filter_by_level=filter_by_level,
            keyword=keyword,
            request_id=request_id,
            trace_id=trace_id,
            error_id=error_id,
            run_id=run_id,
            job_id=job_id,
            http_status=http_status,
            started_at=started_at,
            ended_at=ended_at,
            cursor=cursor,
            limit=limit,
        )
    except LogQueryError as exc:
        raise _query_error(exc) from exc
    return {
        "events": events,
        "count": len(events),
        "total": total,
        "next_cursor": next_cursor,
    }


@router.get("/events/{event_id}")
async def get_log_event(event_id: str, request: Request):
    try:
        event = _service(request).event_detail(event_id=event_id)
    except LogQueryError as exc:
        raise _query_error(exc) from exc
    if event is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "DEVELOPMENT_LOG_EVENT_NOT_FOUND", "message": "日志事件不存在或已轮转"},
        )
    return event
