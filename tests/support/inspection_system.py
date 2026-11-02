"""构造业务系统范围，用真实范围解析器测试，不模拟计划范围决策。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from platform_core.identity import uuid7


def configure_inspection_system(uow, targets, *, domain_id, business_system_id, tenant_id=None, resources=None):
    rows = []
    for target in targets:
        if tenant_id is not None:
            target.tenant_id = tenant_id
        target.domain_id = domain_id
        target.status = "ENABLED"
        resource = (resources or {}).get(target.target_id) or SimpleNamespace(resource_id=uuid7())
        resource.display_name = target.display_name
        resource.resource_type_key = "DATABASE"
        resource.lifecycle_status = "ACTIVE"
        resource.tenant_id = tenant_id
        resource.domain_id = domain_id
        rows.append((SimpleNamespace(status="ACTIVE"), resource,
                     SimpleNamespace(target_id=target.target_id), target))
    repository = getattr(uow, "business_systems", None) or SimpleNamespace()
    repository.get_scoped = AsyncMock(return_value=SimpleNamespace(
        business_system_id=business_system_id, display_name="ERP 生产系统", status="ENABLED",
    ))
    repository.list_system_resource_rows = AsyncMock(return_value=rows)
    repository.get_resource_scoped = AsyncMock(side_effect=lambda *, resource_id, **_: next(
        row[1] for row in rows if row[1].resource_id == resource_id
    ))
    uow.business_systems = repository
    uow.agents.version_has_business_system = AsyncMock(return_value=True)
    uow.agents.version_capabilities = AsyncMock(return_value=["ACTIVE_DIAGNOSIS"])
    return rows
