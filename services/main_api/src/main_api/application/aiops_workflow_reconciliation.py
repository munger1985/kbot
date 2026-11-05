"""以真实服务身份读取工作流权威状态；只允许相同作用域的响应写回。"""

from platform_core.contracts.aiops.notification_projection import WorkflowProjectionSnapshot
from platform_core.security import create_service_auth_context


class AIOpsWorkflowReconciliation:
    def __init__(self, *, client, caller_service):
        self._client, self._caller = client, caller_service

    async def resolve(self, request):
        payload = await self._client.get_workflow_projection(
            payload=request.model_dump(mode="json"),
            auth_context=create_service_auth_context(caller_service=self._caller,
                trace_id=f"notification-workflow-{request.resource_id}"))
        snapshot = WorkflowProjectionSnapshot.model_validate(payload)
        if snapshot.request != request:
            raise ValueError("工作流对账响应作用域不匹配")
        return snapshot
