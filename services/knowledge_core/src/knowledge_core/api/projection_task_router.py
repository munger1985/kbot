"""KC Projection Worker 的统一内部抢占协议。"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from knowledge_core.domain.parse_tasks import ParseTaskClaim
from platform_core.contracts import INTERNAL_API_V1


router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/projection-tasks",
    tags=["Knowledge Core Projection"],
)


class ProjectionClaimRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    max_tasks: int = Field(default=1, ge=1, le=32)
    lease_seconds: int = Field(default=120, ge=30, le=3600)


@router.post("/claim")
async def claim_projection_tasks(
    payload: ProjectionClaimRequest,
    request: Request,
):
    try:
        tasks = await request.app.state.kc_projection_task_service.claim(
            ParseTaskClaim(
                payload.worker_id,
                payload.max_tasks,
                payload.lease_seconds,
            )
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_CLAIM", "message": str(exc)},
        ) from exc
    return {
        "tasks": [
            {
                **task.__dict__,
                "lease_until": task.lease_until.isoformat(),
            }
            for task in tasks
        ]
    }
