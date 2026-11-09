"""以真实服务身份读取工作流权威状态；只允许相同作用域的响应写回。"""

from platform_core.contracts.aiops.notification_projection import WorkflowProjectionSnapshot
from platform_core.security import create_service_auth_context
from platform_clients.aiops import AIOpsClientError


class WorkflowSourceUnavailable(LookupError):
    """权威服务确认该作用域内已无来源对象，不代表原执行成功或失败。"""

    def __init__(self, request):
        super().__init__("NOTIFICATION_SOURCE_UNAVAILABLE")
        self.request = request


class AIOpsWorkflowReconciliation:
    def __init__(self, *, client, caller_service):
        self._client, self._caller = client, caller_service

    async def resolve(self, request):
        try:
            payload = await self._client.get_workflow_projection(
                payload=request.model_dump(mode="json"),
                auth_context=create_service_auth_context(caller_service=self._caller,
                    trace_id=f"notification-workflow-{request.resource_id}"))
        except AIOpsClientError as exc:
            if exc.status_code == 404 and exc.code == "OPS_RESOURCE_NOT_FOUND" and not exc.retryable:
                raise WorkflowSourceUnavailable(request) from exc
            raise
        snapshot = WorkflowProjectionSnapshot.model_validate(payload)
        if snapshot.request != request:
            raise ValueError("工作流对账响应作用域不匹配")
        return snapshot
