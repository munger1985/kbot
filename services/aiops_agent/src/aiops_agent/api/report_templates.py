"""巡检模板、会话报告模板和只读系统报告版式内部 API。"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from aiops_agent.application.reporting import list_system_templates
from platform_core.contracts import AuthContext


router = APIRouter(prefix="/internal/v1/aiops", tags=["AIOps Templates"])


class CreateInspectionTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=256)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)


class CreateInspectionTemplateVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)


class CreateSessionReportTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=256)
    definition: dict[str, Any]


class CreateSessionReportTemplateVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)
    definition: dict[str, Any]


def _scope(request, context):
    require_service_scope(request, "aiops.manage")
    if context.domain_id is None or int(context.domain_id) < 1:
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"})
    return int(context.domain_id), context.asserted_user_id or context.client_id


@router.get("/inspection-templates")
async def list_inspection_templates(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.inspection_template_service.list(
        domain_id=domain_id
    )


@router.get("/inspection-templates/{template_id}")
async def get_inspection_template(
    template_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.inspection_template_service.get(
        domain_id=domain_id, template_id=template_id
    )


@router.post("/inspection-templates", status_code=201)
async def create_inspection_template(
    body: CreateInspectionTemplate,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.create(
        domain_id=domain_id, actor_id=actor_id, **body.model_dump()
    )


@router.post("/inspection-templates/{template_id}/versions", status_code=201)
async def create_inspection_template_version(
    template_id: UUID,
    body: CreateInspectionTemplateVersion,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.create_version(
        domain_id=domain_id,
        actor_id=actor_id,
        template_id=template_id,
        **body.model_dump(),
    )


@router.get("/session-report-templates")
async def list_session_report_templates(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.session_report_template_service.list(
        domain_id=domain_id
    )


@router.get("/session-report-templates/{template_id}")
async def get_session_report_template(
    template_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.session_report_template_service.get(
        domain_id=domain_id, template_id=template_id
    )


@router.post("/session-report-templates", status_code=201)
async def create_session_report_template(
    body: CreateSessionReportTemplate,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.session_report_template_service.create(
        domain_id=domain_id, actor_id=actor_id, **body.model_dump()
    )


@router.post(
    "/session-report-templates/{template_id}/versions", status_code=201
)
async def create_session_report_template_version(
    template_id: UUID,
    body: CreateSessionReportTemplateVersion,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.session_report_template_service.create_version(
        domain_id=domain_id,
        actor_id=actor_id,
        template_id=template_id,
        **body.model_dump(),
    )


@router.get("/report-layouts")
async def list_report_layouts(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    _scope(request, context)
    return list_system_templates()
