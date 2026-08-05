"""仅在 development 环境注册的 Agent Run 调试聚合 API。"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, Request

from main_api.log_reader import LocalLogSearchService, redact_recursive
from platform_clients import AgentRuntimeClient
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/development/agent-runs",
    tags=["Development Agent Run Console"],
)

_CORRELATION_FIELDS = {
    "run_id": "agent_run_ids",
    "task_id": "task_ids",
    "job_id": "kc_job_ids",
    "kc_job_id": "kc_job_ids",
    "data_query_run_id": "data_query_run_ids",
    "model_call_id": "model_call_ids",
    "model_id": "model_ids",
    "trace_id": "trace_ids",
    "request_id": "request_ids",
}
_RUN_LIST_FIELDS = (
    "run_id", "agent_id", "status", "error_code", "request_id", "trace_id",
    "created_at", "started_at", "completed_at", "duration_ms",
)
_ARTIFACT_FIELDS = (
    "artifact_id", "task_id", "artifact_type", "schema_version", "producer",
    "producer_version", "content_hash", "created_at", "provenance",
)


def _client(request: Request) -> AgentRuntimeClient:
    return request.app.state.agent_runtime_client


def _logs(request: Request) -> LocalLogSearchService:
    return request.app.state.development_log_search_service


def _collect_correlations(value: Any) -> dict[str, list[str]]:
    """递归收集跨服务事实标识，不复制业务载荷。"""
    result: dict[str, set[str]] = {
        output: set() for output in set(_CORRELATION_FIELDS.values())
    }

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                output = _CORRELATION_FIELDS.get(str(key))
                if output and child not in (None, ""):
                    result[output].add(str(child))
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {key: sorted(values) for key, values in sorted(result.items())}


def _run_list_projection(row: dict[str, Any]) -> dict[str, Any]:
    return redact_recursive({key: row.get(key) for key in _RUN_LIST_FIELDS})


def _artifact_projection(row: dict[str, Any]) -> dict[str, Any]:
    return redact_recursive({key: row.get(key) for key in _ARTIFACT_FIELDS})


@router.get("")
async def list_agent_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
):
    rows = await _client(request).list_debug_runs(
        limit=limit,
        auth_context=request.state.auth_context,
    )
    runs = [_run_list_projection(row) for row in rows]
    return {"runs": runs, "count": len(runs)}


@router.get("/{run_id}")
async def get_agent_run_debug(
    run_id: UUID,
    request: Request,
    log_limit: int = Query(default=500, ge=1, le=1000),
):
    source = await _client(request).get_debug_run(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    correlations = _collect_correlations(source)
    correlations["agent_run_ids"] = sorted(
        set(correlations["agent_run_ids"]) | {str(run_id)}
    )
    log_identifiers = {
        item
        for values in correlations.values()
        for item in values
    }
    return {
        "run": redact_recursive(source.get("run") or {}),
        "tasks": redact_recursive(source.get("tasks") or []),
        "events": redact_recursive(source.get("events") or []),
        "artifacts": [
            _artifact_projection(row) for row in source.get("artifacts") or []
        ],
        "correlations": correlations,
        "logs": _logs(request).search_correlated(
            identifiers=log_identifiers,
            limit=log_limit,
        ),
    }
