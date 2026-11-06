"""外发详情定位的父子关系、租户、版本与具体处理人回归。"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aiops_agent.application.notification_projection import WorkflowProjectionService
from platform_core.contracts.aiops.notification_projection import NotificationResourceRequest


def source(tenant_scoped):
    now=datetime.now(timezone.utc)
    tenant=uuid4() if tenant_scoped else None
    domain=str(uuid4()) if tenant_scoped else "12"
    run=NS(ops_run_id=uuid4(),tenant_id=tenant,actor_id="A",row_version=2)
    proposal=NS(proposal_id=uuid4(),ops_run_id=run.ops_run_id,ops_task_id=uuid4(),
        row_version=3,status="PENDING_APPROVAL",expires_at=now+timedelta(minutes=5))
    hitl=NS(hitl_id=uuid4(),ops_run_id=run.ops_run_id,proposal_id=proposal.proposal_id,
        row_version=4,status="PENDING",expires_at=now+timedelta(minutes=5),
        requested_by="A",assignee_user_id="B",request_type="CHANGE_APPROVAL")
    execution=NS(ops_run_id=run.ops_run_id,proposal_id=proposal.proposal_id,row_version=5)
    report=NS(report_id=uuid4(),ops_run_id=run.ops_run_id,report_version=2,row_version=6)
    job=NS(job_type="REPORT_GENERATION",status="SUCCEEDED",tenant_id=tenant,
        result_json={"report_id":str(report.report_id),"report_version":"2"})
    uow=NS(runs=NS(get_run_scoped=AsyncMock(return_value=run)),
        changes=NS(get_proposal_scoped=AsyncMock(return_value=proposal),
            get_hitl_scoped=AsyncMock(return_value=hitl),get_execution=AsyncMock(return_value=execution),
            get_pending_hitl=AsyncMock(return_value=hitl)),
        inspections=NS(get_report_scoped=AsyncMock(return_value=report)),
        output_jobs=NS(get=AsyncMock(return_value=job)))
    @asynccontextmanager
    async def factory(): yield uow
    service=WorkflowProjectionService(uow_factory=factory,tenant_scoped=tenant_scoped)
    return service,uow,run,proposal,hitl,execution,report,job,domain,tenant


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_scoped",[False,True])
@pytest.mark.parametrize("case",["valid","hitl_parent","execution_parent","tenant","closed"])
async def test_specific_workflow_resource_scope(tenant_scoped,case):
    service,uow,run,proposal,hitl,execution,report,job,domain,tenant=source(tenant_scoped)
    params=dict(domain_id=domain,run_id=str(run.ops_run_id),proposal_id=str(proposal.proposal_id),
        hitl_id=str(hitl.hitl_id))
    if case=="execution_parent": params["execution_id"]=str(uuid4())
    request=NotificationResourceRequest(event_type="aiops.proposal.review_required",domain_id=domain,
        tenant_id=tenant,route_key="aiops.run.detail",route_params=params)
    if case=="hitl_parent": hitl.ops_run_id=uuid4()
    if case=="execution_parent": execution.proposal_id=uuid4()
    if case=="tenant":
        if tenant_scoped: run.tenant_id=uuid4()
        else: uow.runs.get_run_scoped.return_value=None
    if case=="closed": hitl.status="ANSWERED"
    if case in {"hitl_parent","execution_parent","tenant"}:
        with pytest.raises(Exception): await service.resolve_notification_resource(request)
    else:
        result=await service.resolve_notification_resource(request)
        assert result.actionable==(case!="closed")
        assert result.assignee_actor_id=="B" and result.run_actor_id=="A"


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_scoped",[False,True])
@pytest.mark.parametrize("case",["valid","report_version","parent_tenant","job_report","job_tenant","job_status"])
async def test_report_and_output_job_cannot_be_substituted(tenant_scoped,case):
    service,uow,run,proposal,hitl,execution,report,job,domain,tenant=source(tenant_scoped)
    request=NotificationResourceRequest(event_type="aiops.report.ready",domain_id=domain,tenant_id=tenant,
        route_key="aiops.report.detail",route_params=dict(domain_id=domain,report_id=str(report.report_id),
            report_version="2",job_id=str(uuid4())))
    if case=="report_version": report.report_version=3
    if case=="parent_tenant":
        if tenant_scoped: run.tenant_id=uuid4()
        else: uow.runs.get_run_scoped.return_value=None
    if case=="job_report": job.result_json["report_id"]=str(uuid4())
    if case=="job_tenant":
        if tenant_scoped: job.tenant_id=uuid4()
        else: uow.output_jobs.get.return_value=None
    if case=="job_status": job.status="RUNNING"
    if case=="valid":
        assert (await service.resolve_notification_resource(request)).source_version==6
    else:
        with pytest.raises(Exception): await service.resolve_notification_resource(request)


@pytest.mark.parametrize("event,params",[
    ("unknown.event",{}),
    ("aiops.execution.succeeded",{}),
    ("aiops.diagnostic.input_required",{}),
    ("aiops.proposal.review_required",{"url":"https://evil.test"}),
    ("aiops.situation.detected",{}),
])
def test_unknown_event_missing_specific_resource_and_arbitrary_url_rejected(event,params):
    with pytest.raises(ValueError):
        NotificationResourceRequest(event_type=event,domain_id="12",route_key="aiops.run.detail",
            route_params={"domain_id":"12","run_id":str(uuid4()),**params})
