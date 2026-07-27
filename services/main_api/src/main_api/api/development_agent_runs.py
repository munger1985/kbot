"""仅在 development 环境注册的 Agent Run 调试聚合 API。"""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Query, Request

from main_api.developer_tools import LocalLogSearchService
from platform_clients import AgentRuntimeClient
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/development/agent-runs",
    tags=["Development Agent Run Console"],
)


def _client(request: Request) -> AgentRuntimeClient:
    return request.app.state.agent_runtime_client


def _logs(request: Request) -> LocalLogSearchService:
    return LocalLogSearchService(
        log_root=Path(request.app.state.development_log_root)
    )


@router.get("")
async def list_agent_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
):
    rows = await _client(request).list_debug_runs(
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return {"runs": rows, "count": len(rows)}


@router.get("/{run_id}")
async def get_agent_run_debug(
    run_id: UUID,
    request: Request,
    log_limit: int = Query(default=500, ge=1, le=1000),
):
    projection = await _client(request).get_debug_run(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    run = projection.get("run") or {}
    identifiers = {
        str(run_id),
        str(run.get("trace_id") or ""),
        str(run.get("request_id") or ""),
    }
    identifiers.update(
        str(task.get("task_id") or "")
        for task in projection.get("tasks") or []
    )
    projection["logs"] = _logs(request).search_correlated(
        identifiers=identifiers,
        limit=log_limit,
    )
    return projection
