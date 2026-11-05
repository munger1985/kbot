"""只读取指定空间的工作流状态；服务身份不冒充用户，也不返回正文。"""

from uuid import UUID

from aiops_agent.application.errors import resource_not_found
from platform_core.contracts.aiops.notification_projection import WorkflowProjectionSnapshot


class WorkflowProjectionService:
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
