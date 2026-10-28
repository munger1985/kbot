"""正式租约合同的测试构造器，不在生产路径兼容缺失的执行授权。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.diagnostics.grants import DiagnosticGrantError
from platform_core.contracts.aiops.executor import CredentialIssueResponse, ResourceLeaseReceipt


def resource_receipt(grant):
    return ResourceLeaseReceipt(lease_id=grant.grant_id, release_token="r" * 43,
        execution_deadline_at=grant.expires_at, held_until=grant.expires_at + timedelta(seconds=30))


def bind_resource_control(control, codec):
    async def issue(token, *, trace_id):
        try:
            grant = codec.verify(token)
        except DiagnosticGrantError:
            grant = codec.verify_dynamic(token)
        values = control.issue_credential.return_value
        return CredentialIssueResponse(username=values.username, password=values.password,
            resource_lease=resource_receipt(grant))
    control.issue_credential.side_effect = issue
    control.release_resource = AsyncMock()
    return control


def resource_repository(now=None):
    stamp = now or datetime.now(timezone.utc)
    return SimpleNamespace(
        lock_pool=AsyncMock(return_value=SimpleNamespace(global_limit=8, target_limit=2,
            source_limit=4, cleanup_grace_seconds=30)),
        database_now=AsyncMock(return_value=stamp), active_counts=AsyncMock(return_value=(0, 0, 0)),
        get_lease=AsyncMock(return_value=None), add_lease=AsyncMock(),
    )
