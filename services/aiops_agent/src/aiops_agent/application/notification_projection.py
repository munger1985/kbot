"""只读取指定空间的工作流状态；服务身份不冒充用户，也不返回正文。"""

from uuid import UUID

from aiops_agent.application.errors import resource_not_found
from platform_core.contracts.aiops.notification_projection import WorkflowProjectionSnapshot


class WorkflowProjectionService:

    async def resolve_notification_resource(self, request):
        """重新核对父子身份、空间及待处理状态，不返回任务正文。"""
        from datetime import datetime, timezone
        from platform_core.contracts.aiops.notification_projection import NotificationResourceSnapshot
        try:
            domain = UUID(request.domain_id) if self._tenant_scoped else int(request.domain_id)
        except ValueError as exc:
            raise resource_not_found("NotificationResource") from exc
        if (self._tenant_scoped and request.tenant_id is None) or (not self._tenant_scoped and request.tenant_id is not None):
            raise resource_not_found("NotificationResource")
        params, now = request.route_params, datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            initiator, run_actor, assignee, actionable = None, None, None, False
            if request.route_key == "aiops.situation.detail":
                entity = await uow.situations.get_situation_scoped(situation_id=UUID(params["situation_id"]), domain_id=domain)
                if entity is None or (self._tenant_scoped and entity.tenant_id != request.tenant_id):
                    raise resource_not_found("NotificationResource")
            elif request.route_key == "aiops.report.detail":
                entity = await uow.inspections.get_report_scoped(report_id=UUID(params["report_id"]), domain_id=domain)
                if entity is None:
                    raise resource_not_found("NotificationResource")
                parent_run = await uow.runs.get_run_scoped(ops_run_id=entity.ops_run_id, domain_id=domain)
                if parent_run is None or (self._tenant_scoped and parent_run.tenant_id != request.tenant_id):
                    raise resource_not_found("NotificationResource")
                initiator = run_actor = str(parent_run.actor_id)
                if params.get("report_version") and int(params["report_version"]) != int(entity.report_version):
                    raise resource_not_found("NotificationResource")
                if params.get("job_id"):
                    job = await uow.output_jobs.get(domain_id=domain, job_id=UUID(params["job_id"]))
                    result = job.result_json if job is not None else None
                    if (job is None or job.job_type != "REPORT_GENERATION" or job.status != "SUCCEEDED"
                        or (self._tenant_scoped and job.tenant_id != request.tenant_id)
                        or not isinstance(result, dict) or result.get("report_id") != str(entity.report_id)
                        or str(result.get("report_version")) != str(entity.report_version)):
                        raise resource_not_found("NotificationResource")
            else:
                run = await uow.runs.get_run_scoped(ops_run_id=UUID(params["run_id"]), domain_id=domain)
                if run is None or (self._tenant_scoped and run.tenant_id != request.tenant_id):
                    raise resource_not_found("NotificationResource")
                entity, initiator = run, str(run.actor_id)
                run_actor = str(run.actor_id)
                proposal, hitl = None, None
                if params.get("proposal_id"):
                    proposal = await uow.changes.get_proposal_scoped(proposal_id=UUID(params["proposal_id"]), domain_id=domain)
                    if proposal is None or proposal.ops_run_id != run.ops_run_id:
                        raise resource_not_found("NotificationResource")
                    entity = proposal
                    actionable = proposal.status == "PENDING_APPROVAL" and (proposal.expires_at is None or proposal.expires_at > now)
                    if request.event_type == "aiops.proposal.review_required" and not params.get("hitl_id"):
                        hitl = await uow.changes.get_pending_hitl(
                            ops_task_id=proposal.ops_task_id, request_type="CHANGE_APPROVAL")
                        if hitl is None or hitl.ops_run_id != run.ops_run_id or hitl.proposal_id != proposal.proposal_id:
                            raise resource_not_found("NotificationResource")
                        initiator = str(hitl.requested_by)
                        assignee = str(hitl.assignee_user_id) if hitl.assignee_user_id is not None else None
                        actionable = actionable and hitl.status == "PENDING" and hitl.expires_at > now
                if params.get("hitl_id"):
                    hitl = await uow.changes.get_hitl_scoped(hitl_id=UUID(params["hitl_id"]), domain_id=domain)
                    if hitl is None or hitl.ops_run_id != run.ops_run_id or (proposal and hitl.proposal_id != proposal.proposal_id):
                        raise resource_not_found("NotificationResource")
                    initiator = str(hitl.requested_by)
                    assignee = str(hitl.assignee_user_id) if hitl.assignee_user_id is not None else None
                    if request.event_type.startswith("aiops.diagnostic.input_"):
                        if hitl.request_type == "CHANGE_APPROVAL":
                            raise resource_not_found("NotificationResource")
                    elif request.event_type == "aiops.proposal.review_required" and hitl.request_type != "CHANGE_APPROVAL":
                        raise resource_not_found("NotificationResource")
                    entity = hitl
                    actionable = hitl.status == "PENDING" and hitl.expires_at > now
                    if proposal:
                        actionable = actionable and proposal.status == "PENDING_APPROVAL" and (proposal.expires_at is None or proposal.expires_at > now)
                if params.get("execution_id"):
                    execution = await uow.changes.get_execution(execution_id=UUID(params["execution_id"]))
                    if execution is None or execution.ops_run_id != run.ops_run_id or not proposal or execution.proposal_id != proposal.proposal_id:
                        raise resource_not_found("NotificationResource")
                    entity, actionable = execution, False
            return NotificationResourceSnapshot(request=request, actionable=actionable,
                source_version=int(entity.row_version), initiator_actor_id=initiator,
                run_actor_id=run_actor, assignee_actor_id=assignee)

    def __init__(self, *, uow_factory, tenant_scoped: bool):
        self._uow_factory, self._tenant_scoped = uow_factory, tenant_scoped

    async def read(self, request):
        try:
            domain = UUID(request.domain_id) if self._tenant_scoped else int(request.domain_id)
        except ValueError as exc:
            raise resource_not_found("Workflow") from exc
        if ((self._tenant_scoped and request.tenant_id is None)
            or (not self._tenant_scoped and (request.tenant_id is not None or domain < 1))):
            raise resource_not_found("Workflow")
        async with self._uow_factory() as uow:
            kind = request.resource_type
            if kind == "aiops_hitl":
                entity = await uow.changes.get_hitl_scoped(
                    hitl_id=request.resource_id, domain_id=domain)
            elif kind == "aiops_proposal":
                entity = await uow.changes.get_proposal_scoped(
                    proposal_id=request.resource_id, domain_id=domain)
            else:
                entity = await uow.runs.get_run_scoped(
                    ops_run_id=request.resource_id, domain_id=domain)
            if entity is None:
                raise resource_not_found("Workflow")
            run = (entity if kind == "aiops_run" else await uow.runs.get_run_scoped(
                ops_run_id=entity.ops_run_id, domain_id=domain))
            if run is None or (self._tenant_scoped and run.tenant_id != request.tenant_id):
                raise resource_not_found("Workflow")
            status = str(entity.status)
            if kind == "aiops_hitl":
                resolution = (None if status == "PENDING" else
                    "CANCELLED" if status in {"EXPIRED", "CANCELLED"} else "COMPLETED")
            elif kind == "aiops_proposal":
                resolution = (None if status == "PENDING_APPROVAL" else
                    "CANCELLED" if status in {"EXPIRED", "SUPERSEDED"} else "COMPLETED")
            else:
                pending = await uow.changes.list_pending_hitls(ops_run_id=run.ops_run_id)
                # 仅供旧 Run 级输入待办对账；报告生成或 Run 终态不能替代具体交互状态。
                resolution = (None if any(item.request_type != "CHANGE_APPROVAL" for item in pending)
                    else "CANCELLED" if status in {"CANCELLED", "EXPIRED"} else "COMPLETED")
            return WorkflowProjectionSnapshot(request=request, source_status=status,
                source_version=int(entity.row_version), work_item_status=resolution)
