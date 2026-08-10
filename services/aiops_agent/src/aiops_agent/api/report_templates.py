"""巡检报告模板内部 API。"""

from typing import Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from aiops_agent.api.dependencies import get_aiops_auth_context, require_service_scope
from platform_core.contracts import AuthContext

router = APIRouter(prefix="/internal/v1/aiops/report-templates", tags=["AIOps Report Templates"])
class CreateTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=256)
    definition: dict[str, Any]
class CreateVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)
    definition: dict[str, Any]
def _scope(request, context):
    require_service_scope(request, "aiops.manage")
    if context.domain_id is None or int(context.domain_id) < 1:
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"})
    return int(context.domain_id), context.asserted_user_id or context.client_id
@router.get("")
async def list_templates(request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, _ = _scope(request, context); return await request.app.state.report_template_service.list(domain_id=domain_id)
@router.get("/{template_id}")
async def get_template(template_id: UUID, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, _ = _scope(request, context); return await request.app.state.report_template_service.get(domain_id=domain_id, template_id=template_id)
@router.post("", status_code=201)
async def create_template(body: CreateTemplate, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context); return await request.app.state.report_template_service.create(domain_id=domain_id, actor_id=actor_id, **body.model_dump())
@router.post("/{template_id}/versions", status_code=201)
async def create_version(template_id: UUID, body: CreateVersion, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context); return await request.app.state.report_template_service.create_version(domain_id=domain_id, actor_id=actor_id, template_id=template_id, **body.model_dump())
