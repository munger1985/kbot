"""执行阶段从 Main API 重建当前授权；不把返回快照缓存为长期凭据。"""

from datetime import datetime, timezone

from pydantic import ValidationError

from aiops_agent.application.errors import AIOpsApplicationError
from platform_clients.output_authority import OutputAuthorityClientError
from platform_core.contracts.aiops.output_authority import OutputAuthorityRequest, output_required_permissions


class OutputJobAuthority:
    def __init__(self, *, client):
        self._client = client

    async def resolve(self, execution):
        try:
            metadata = execution.frozen.parameters["authority"]
            request = OutputAuthorityRequest(
                job_id=execution.job_id, job_type=execution.job_type, source=execution.frozen.source,
                origin=metadata["origin"], source_agent_ids=metadata["source_agent_ids"],
                actor_id=str(execution.scope.actor_id), domain_id=str(execution.scope.domain_id),
                tenant_id=execution.scope.tenant_id,
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_AUTHORITY_INPUT_INVALID",
                message="任务缺少可信发起来源与 Agent 清单", status_code=409, retryable=False) from exc
        try:
            snapshot = await self._client.resolve(request, trace_id=execution.scope.trace_id)
        except OutputAuthorityClientError as exc:
            raise AIOpsApplicationError(code=("AIOPS_OUTPUT_AUTHORITY_UNAVAILABLE" if exc.retryable
                else "AIOPS_OUTPUT_AUTHORITY_REVOKED"),
                message="当前输出授权不可用", status_code=exc.status_code, retryable=exc.retryable) from exc
        age = (datetime.now(timezone.utc) - snapshot.checked_at).total_seconds()
        maximum = snapshot.max_security_level
        if (snapshot.request != request or not -5 <= age <= 30
            or not output_required_permissions(request) <= set(snapshot.permissions)
            or (maximum is not None and any(item.security_level > maximum for item in execution.frozen.artifacts))):
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_AUTHORITY_REVOKED",
                message="当前输出权限、安全级或业务空间已变化", status_code=403, retryable=False)
        return snapshot
