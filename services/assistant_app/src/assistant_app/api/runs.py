"""智能工作台运行记录内部 API。"""

from fastapi import APIRouter, Query, Request

from assistant_app.api.context import require_actor
from assistant_app.application import AssistantRunQueryService


router = APIRouter(prefix="/internal/v1/assistant/runs", tags=["Assistant Runs"])


def _service(request: Request) -> AssistantRunQueryService:
    return request.app.state.run_query_service


@router.get("")
async def list_runs(
    domain_id: int,
    request: Request,
    kind: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    require_actor(request, domain_id)
    return await _service(request).list(
        domain_id=domain_id, kind=kind, status=status, limit=limit,
    )
