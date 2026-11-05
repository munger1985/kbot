"""按可信管理范围读取共享主机；不修改系统关系或 Agent 授权。"""

from collections import deque
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, validation_failed
from .common import ConfigurationScope
from platform_core.contracts.aiops.host_topology import (
    HostTopologyParticipant, HostTopologyRelation, HostTopologyResource,
    HostTopologySystem, HostTopologyView,
)


async def read_host_topology(service, *, scope: ConfigurationScope, business_system_id: UUID, resource_id: UUID, max_nodes: int = 200, max_depth: int = 4) -> HostTopologyView:
    owned = {"domain_id": scope.domain_id}
    async with service._uow_factory() as uow:
        repository = uow.business_systems
        system = await repository.get_scoped(business_system_id=business_system_id, **owned)
        if system is None:
            raise resource_not_found("Business System")
        root = await repository.get_resource_scoped(resource_id=resource_id, **owned)
        rows = await repository.list_system_resource_rows(business_system_id)
        linked = any(link.resource_id == resource_id and link.status == "ACTIVE" for link, *_ in rows)
        if root is None or root.lifecycle_status != "ACTIVE" or not linked:
            raise resource_not_found("Managed Resource")
        host_types = {"PHYSICAL_SERVER", "VIRTUAL_MACHINE", "CONTAINER", "POD", "KUBERNETES_CLUSTER"}
        if root.resource_type_key not in host_types:
            raise validation_failed("只能从主机或容器资源查看承载拓扑")
        resources = {root.resource_id: root}
        memberships = {root.resource_id: {system.business_system_id: system}}
        relations = {}
        queue = deque([(root.resource_id, 0)])
        visited = set()
        truncated = False
        remaining_rows = max_nodes * 4
        while queue:
            host_id, depth = queue.popleft()
            if host_id in visited:
                continue
            visited.add(host_id)
            if remaining_rows <= 0:
                truncated = True
                break
            # 限制读取规模；达到深度边界仍探测一行，以诚实标记不完整结果。
            row_limit = min(max_nodes, remaining_rows)
            related = await repository.list_host_workload_rows(resource_id=host_id, limit=row_limit + 1 if depth < max_depth else 1, **owned)
            if depth >= max_depth:
                truncated = truncated or bool(related)
                continue
            if len(related) > row_limit:
                truncated = True
            remaining_rows -= min(len(related), row_limit)
            for workload, owner, relation in related[:row_limit]:
                if workload.resource_id not in resources and len(resources) >= max_nodes:
                    truncated = True
                    continue
                resources[workload.resource_id] = workload
                memberships.setdefault(workload.resource_id, {})[owner.business_system_id] = owner
                memberships.setdefault(host_id, {})[owner.business_system_id] = owner
                relations[relation.relation_id] = HostTopologyRelation(
                    relation_id=relation.relation_id, business_system_id=owner.business_system_id,
                    relation_type_key="RUNS_ON", display_name=relation.display_name,
                    participants=(
                        HostTopologyParticipant(resource_id=workload.resource_id, participant_role="WORKLOAD", display_order=0),
                        HostTopologyParticipant(resource_id=host_id, participant_role="HOST", display_order=1),
                    ),
                )
                if workload.resource_type_key in host_types and workload.resource_id not in visited:
                    queue.append((workload.resource_id, depth + 1))
        return HostTopologyView(
            host_resource_id=root.resource_id, max_nodes=max_nodes, truncated=truncated,
            resources=tuple(HostTopologyResource(
                resource_id=item.resource_id, resource_type_key=item.resource_type_key,
                display_name=item.display_name, location_name=item.location_name,
                business_systems=tuple(HostTopologySystem(
                    business_system_id=owner.business_system_id, display_name=owner.display_name, status=owner.status,
                ) for owner in memberships.get(item.resource_id, {}).values()),
            ) for item in resources.values()),
            relations=tuple(relations.values()),
        )
