"""按 Domain、Agent 版本和可信业务上下文返回可用功能入口。"""

from __future__ import annotations

from uuid import UUID

from aiops_agent.application.errors import resource_not_found
from platform_core.contracts.aiops import TrustedContextHint

from .catalog import ConversationStarterCatalog


class ConversationStarterService:
    def __init__(self, *, uow_factory, catalog: ConversationStarterCatalog) -> None:
        self._uow_factory = uow_factory
        self._catalog = catalog

    async def list(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        context_hint: TrustedContextHint | None = None,
    ) -> dict:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if (
                agent is None
                or agent.status != "ACTIVE"
                or agent.current_version_id is None
            ):
                raise resource_not_found("Active AIOps Agent")
            version = await uow.agents.version(
                agent_id=agent.agent_id,
                agent_version_id=agent.current_version_id,
            )
            if version is None:
                raise resource_not_found("Agent Version")
            business_system_ids = tuple(
                await uow.agents.version_business_system_ids(
                    agent_version_id=version.agent_version_id,
                )
            )
            target_ids = set(
                await uow.agents.active_version_target_ids(
                    domain_id=domain_id,
                    agent_version_id=version.agent_version_id,
                )
            )
            if context_hint is not None:
                allowed_system_ids = set(business_system_ids)
                if (
                    context_hint.business_system_id is not None
                    and context_hint.business_system_id not in allowed_system_ids
                ):
                    raise resource_not_found("Agent Business System")
                if (
                    context_hint.business_system_id is not None
                    and context_hint.resource_ids
                ):
                    system_resource_ids = {
                        resource.resource_id
                        for link, resource, _database, _target in (
                            await uow.business_systems.list_system_resource_rows(
                                context_hint.business_system_id
                            )
                        )
                        if link.status == "ACTIVE"
                        and resource.lifecycle_status == "ACTIVE"
                    }
                    if not set(context_hint.resource_ids) <= system_resource_ids:
                        raise resource_not_found("Agent Business Resource")
                hinted_target_ids: set[UUID] = set()
                for resource_id in context_hint.resource_ids:
                    database = await uow.business_systems.get_database_profile(
                        resource_id
                    )
                    if database is not None:
                        hinted_target_ids.add(database.target_id)
                if context_hint.resource_ids:
                    if not hinted_target_ids <= target_ids:
                        raise resource_not_found("Agent Business Resource")
                    target_ids = hinted_target_ids
                elif context_hint.business_system_id is not None:
                    system_target_ids = {
                        database.target_id
                        for link, resource, database, _target in (
                            await uow.business_systems.list_system_resource_rows(
                                context_hint.business_system_id
                            )
                        )
                        if link.status == "ACTIVE"
                        and resource.lifecycle_status == "ACTIVE"
                        and database is not None
                    }
                    target_ids &= system_target_ids
            target_by_id = {}
            for target_id in sorted(target_ids, key=str):
                target = await uow.targets.get_scoped(
                    target_id=target_id,
                    domain_id=domain_id,
                )
                if target is not None:
                    target_by_id[target_id] = target
            selected_resource_ids = set(
                context_hint.resource_ids if context_hint is not None else ()
            )
            business_systems = []
            for business_system_id in business_system_ids:
                if (
                    context_hint is not None
                    and context_hint.business_system_id is not None
                    and context_hint.business_system_id != business_system_id
                ):
                    continue
                system = await uow.business_systems.get_scoped(
                    business_system_id=business_system_id,
                    domain_id=domain_id,
                )
                if system is None or system.status != "ENABLED":
                    continue
                options = []
                for link, resource, database, _target in (
                    await uow.business_systems.list_system_resource_rows(
                        business_system_id
                    )
                ):
                    if (
                        link.status != "ACTIVE"
                        or resource.lifecycle_status != "ACTIVE"
                        or database is None
                        or database.target_id not in target_by_id
                        or (
                            selected_resource_ids
                            and resource.resource_id not in selected_resource_ids
                        )
                    ):
                        continue
                    target = target_by_id[database.target_id]
                    options.append({
                        "target_id": str(target.target_id),
                        "resource_id": str(resource.resource_id),
                        "target_name": str(target.display_name),
                        "resource_name": str(resource.display_name),
                        "db_type": str(target.db_type),
                        "status": str(target.status),
                        "connectivity_status": str(target.connectivity_status),
                        "readonly_connection_enabled": bool(
                            target.readonly_connection_enabled
                        ),
                    })
                if options:
                    business_systems.append({
                        "business_system_id": str(system.business_system_id),
                        "display_name": str(system.display_name),
                        "environment": str(system.environment),
                        "targets": sorted(
                            options,
                            key=lambda item: (
                                item["target_name"], item["resource_name"]
                            ),
                        ),
                    })
            payload = self._catalog.list_for_agent(
                agent_id=agent_id,
                capabilities=set(version.capabilities_json or ()),
                targets=tuple(target_by_id.values()),
                context_hint=context_hint,
            )
            payload["business_systems"] = sorted(
                business_systems,
                key=lambda item: item["display_name"],
            )
            return payload
