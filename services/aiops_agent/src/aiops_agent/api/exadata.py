"""Exadata 专用 API，硬件诊断不进入数据库任务 API。"""
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends, Header, Request
from aiops_agent.api.dependencies import get_aiops_auth_context, require_service_scope
from aiops_agent.application.configuration import ConfigurationScope
from aiops_agent.application.exadata import ExadataOEMService
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.errors import AIOpsApplicationError
from platform_core.contracts.aiops.exadata import ExadataDiscovery, ExadataQuery, ExadataView, ExadataReport, ExadataBaselineQuery, ExadataBaseline

router = APIRouter(prefix="/internal/v1/aiops/exadata", tags=["Exadata OEM"])


def service(request: Request):
    return request.app.state.exadata_service


def run_scope(request: Request):
    require_service_scope(request, "aiops.run")
    return ConfigurationScope.from_auth(auth_context=get_aiops_auth_context(request))


def manage_scope(request: Request):
    require_service_scope(request, "aiops.manage")
    return ConfigurationScope.from_auth(auth_context=get_aiops_auth_context(request))


Service = Annotated[ExadataOEMService, Depends(service)]
RunScope = Annotated[ConfigurationScope, Depends(run_scope)]
ManageScope = Annotated[ConfigurationScope, Depends(manage_scope)]


async def read(operation):
    try:
        return await operation
    except DiagnosticSourceAdapterError as exc:
        raise AIOpsApplicationError(code=exc.code, message="OEM Exadata 身份或指标接口无法核验",
            status_code=502, retryable=exc.retryable) from exc
    except TimeoutError as exc:
        raise AIOpsApplicationError(code="EXADATA_SOURCE_TIMEOUT", message="OEM Exadata 查询超时",
            status_code=504, retryable=True) from exc


@router.post("/sources/{source_id}/discoveries", response_model=ExadataDiscovery)
async def discover(source_id: UUID, service: Service, scope: ManageScope):
    return await read(service.discover(scope=scope, source_id=source_id))


@router.post("/systems/{system_id}/resources/{resource_id}/sources/{source_id}/view", response_model=ExadataView)
async def view(system_id: UUID, resource_id: UUID, source_id: UUID, body: ExadataQuery, service: Service, scope: RunScope):
    return await read(service.collect(scope=scope, system_id=system_id, resource_id=resource_id, source_id=source_id, query=body))


@router.post("/systems/{system_id}/resources/{resource_id}/sources/{source_id}/diagnoses", response_model=ExadataReport)
async def diagnose(system_id: UUID, resource_id: UUID, source_id: UUID, body: ExadataQuery, service: Service,
                   scope: RunScope, idempotency_key: Annotated[str, Header(alias="Idempotency-Key")]):
    return await read(service.diagnose(scope=scope, system_id=system_id, resource_id=resource_id,
        source_id=source_id, query=body, idempotency_key=idempotency_key))


@router.post("/systems/{system_id}/resources/{resource_id}/sources/{source_id}/baselines", response_model=ExadataBaseline)
async def baseline(system_id: UUID, resource_id: UUID, source_id: UUID, body: ExadataBaselineQuery, service: Service, scope: RunScope):
    return await read(service.baseline(scope=scope, system_id=system_id, resource_id=resource_id, source_id=source_id, query=body))


@router.get("/systems/{system_id}/resources/{resource_id}/sources/{source_id}/reports/{report_id}", response_model=ExadataReport)
async def report(system_id: UUID, resource_id: UUID, source_id: UUID, report_id: UUID, service: Service, scope: RunScope):
    return await read(service.get_report(scope=scope, system_id=system_id, resource_id=resource_id,
        source_id=source_id, report_id=report_id))
