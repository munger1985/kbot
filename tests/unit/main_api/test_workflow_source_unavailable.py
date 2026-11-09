"""来源不可用与临时故障必须分开处理，不能猜测原执行结果。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from main_api.application.aiops_workflow_reconciliation import (
    AIOpsWorkflowReconciliation, WorkflowSourceUnavailable,
)
from platform_clients.aiops import AIOpsClientError
from platform_core.contracts.aiops.notification_projection import WorkflowProjectionRequest
from platform_core.identity import uuid7


@pytest.mark.parametrize("status,code,retryable,missing", [
    (404, "OPS_RESOURCE_NOT_FOUND", False, True),
    (404, "OPS_RESOURCE_NOT_FOUND", True, False),
    (404, "ROUTE_NOT_FOUND", False, False),
    (403, "OPS_RESOURCE_NOT_FOUND", False, False),
    (503, "OPS_RESOURCE_NOT_FOUND", True, False),
])
def test_only_authoritative_nonretryable_missing_source_expires_projection(status, code, retryable, missing):
    async def run():
        request = WorkflowProjectionRequest(resource_type="aiops_run", resource_id=uuid7(), domain_id="73")
        error = AIOpsClientError(status_code=status, code=code, retryable=retryable, message="测试")
        client = SimpleNamespace(get_workflow_projection=AsyncMock(side_effect=error))
        resolver = AIOpsWorkflowReconciliation(client=client, caller_service="main-api")
        with pytest.raises(WorkflowSourceUnavailable if missing else AIOpsClientError) as caught:
            await resolver.resolve(request)
        if missing:
            assert caught.value.request == request
        else:
            assert caught.value is error
    asyncio.run(run())
