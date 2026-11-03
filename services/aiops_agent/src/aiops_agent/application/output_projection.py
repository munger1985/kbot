"""为通知 Worker 读取确切任务版本；当前人类授权由 Main API 对账用例重建。"""

from uuid import UUID

from aiops_agent.application.errors import resource_not_found
from aiops_agent.application.output_jobs import OutputJobScope
from platform_core.contracts.aiops.output_authority import OutputAuthorityRequest
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput
from platform_core.contracts.aiops.output_projection import OutputProjectionSnapshot


class OutputProjectionService:
    def __init__(self, *, uow_factory, tenant_scoped: bool):
        self._uow_factory, self._tenant_scoped = uow_factory, tenant_scoped

    async def read(self, request):
        try:
            domain = UUID(request.domain_id) if self._tenant_scoped else int(request.domain_id)
            actor = UUID(request.actor_id) if self._tenant_scoped else request.actor_id
        except ValueError as exc:
            raise resource_not_found("OutputJob") from exc
        if ((self._tenant_scoped and request.tenant_id is None)
            or (not self._tenant_scoped and (request.tenant_id is not None or domain < 1))):
            raise resource_not_found("OutputJob")
        scope = OutputJobScope(domain_id=domain, actor_id=actor, trace_id="notification-projection",
            tenant_id=request.tenant_id)
        async with self._uow_factory() as uow:
            job = await uow.output_jobs.get(domain_id=domain, actor_id=actor, job_id=request.job_id)
            if (job is None or job.job_type != request.job_type
                or (self._tenant_scoped and job.tenant_id != scope.tenant_id)):
                raise resource_not_found("OutputJob")
            frozen = FrozenOutputInput.model_validate(job.input_snapshot_json)
            metadata = frozen.parameters["authority"]
            authority = OutputAuthorityRequest(
                job_id=job.job_id, job_type=job.job_type, source=frozen.source,
                actor_id=str(actor), domain_id=str(domain), tenant_id=request.tenant_id,
                origin=metadata["origin"], source_agent_ids=metadata["source_agent_ids"])
            return OutputProjectionSnapshot(request=request, authority=authority,
                status=job.status, stage=job.stage, source_version=int(job.row_version),
                source_updated_at=job.updated_at,
                error_code=job.error_code,
                source_security_level=max(item.security_level for item in frozen.artifacts))
