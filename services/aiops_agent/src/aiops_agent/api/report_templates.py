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


from platform_core.contracts.aiops.daily_report import DailyReportLayout
from platform_core.contracts.aiops.public import ReportPresentation


class CreateInspectionTemplate(BaseModel):
    daily_report_layout: DailyReportLayout | None = None
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=256)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)


class SaveInspectionTemplateDraft(CreateInspectionTemplate):
    expected_row_version: int = Field(ge=1)


class InspectionTemplateRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)


class CreateSessionReportTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=256)
    definition: dict[str, Any]


class PreviewDailyTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    daily_report_layout: DailyReportLayout = Field(default_factory=DailyReportLayout)
    selected_check_ids: tuple[str, ...] = ()
    database_type: str = Field(default="ORACLE", pattern=r"^[A-Z][A-Z0-9_]{0,31}$")
    scenario: str = Field(default="NORMAL", pattern="^(NORMAL|ATTENTION|INCOMPLETE)$")


class CreateSessionReportTemplateVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)
    definition: dict[str, Any]


class UpdateSessionReportTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)
    status: str = Field(pattern="^(ACTIVE|INACTIVE)$")


class SetDefaultSessionReportTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template_ref: str = Field(min_length=1, max_length=128)


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
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.list(
        domain_id=domain_id, actor_id=actor_id
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


@router.put("/inspection-templates/{template_id}/draft")
async def draft_inspection_template(template_id: UUID, request: Request, body: SaveInspectionTemplateDraft, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.save_draft(
        domain_id=domain_id, actor_id=actor_id, template_id=template_id, **body.model_dump(mode="json"))


@router.post("/inspection-templates/{template_id}:publish", status_code=201)
async def publish_inspection_template(template_id: UUID, request: Request, body: InspectionTemplateRevision, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.publish(
        domain_id=domain_id, actor_id=actor_id, template_id=template_id, **body.model_dump(mode="json"))


@router.post("/inspection-templates/{template_id}:default")
async def default_inspection_template(template_id: UUID, request: Request, body: InspectionTemplateRevision, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.set_default(
        domain_id=domain_id, actor_id=actor_id, template_id=template_id, **body.model_dump(mode="json"))


@router.delete("/inspection-templates/{template_id}/draft")
async def discard_inspection_template(template_id: UUID, request: Request, body: InspectionTemplateRevision, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.discard_draft(
        domain_id=domain_id, actor_id=actor_id, template_id=template_id, **body.model_dump(mode="json"))


@router.delete("/inspection-templates/{template_id}")
async def delete_inspection_template(template_id: UUID, request: Request, body: InspectionTemplateRevision, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.delete(
        domain_id=domain_id, actor_id=actor_id, template_id=template_id, **body.model_dump(mode="json"))


@router.get("/inspection-templates/{template_id}/plans")
async def plans_inspection_template(template_id: UUID, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.list_plans(
        domain_id=domain_id, template_id=template_id)


@router.post("/inspection-templates:preview", response_model=ReportPresentation)
async def preview_daily_template(body: PreviewDailyTemplate, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    _scope(request, context)
    from aiops_agent.application.inspections.daily_preview import daily_preview
    return daily_preview(**body.model_dump(mode="json"))


@router.get("/inspection-templates/{template_id}/versions")
async def inspection_template_versions(template_id: UUID, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, _ = _scope(request, context)
    return await request.app.state.inspection_template_service.list_versions(domain_id=domain_id, template_id=template_id)


@router.patch("/inspection-templates/{template_id}")
async def update_inspection_template(template_id: UUID, body: UpdateSessionReportTemplate, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.inspection_template_service.update(domain_id=domain_id, actor_id=actor_id, template_id=template_id, **body.model_dump())


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


@router.get("/session-report-templates/{template_id}/versions")
async def list_session_report_template_versions(
    template_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.session_report_template_service.list_versions(
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
        **body.model_dump(mode="json"),
    )


@router.patch("/session-report-templates/{template_id}")
async def update_session_report_template(
    template_id: UUID,
    body: UpdateSessionReportTemplate,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.session_report_template_service.update(
        domain_id=domain_id,
        actor_id=actor_id,
        template_id=template_id,
        **body.model_dump(),
    )


@router.put("/session-report-templates/default")
async def set_default_session_report_template(
    body: SetDefaultSessionReportTemplate,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.session_report_template_service.set_default(
        domain_id=domain_id,
        actor_id=actor_id,
        template_ref=body.template_ref,
    )


@router.get("/report-layouts")
async def list_report_layouts(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    _scope(request, context)
    return list_system_templates()
