"""按 Domain、Agent 版本和 Target 返回可用功能入口。"""

from __future__ import annotations

from uuid import UUID

from aiops_agent.application.errors import resource_not_found

from .catalog import ConversationStarterCatalog


class ConversationStarterService:
    def __init__(self, *, uow_factory, catalog: ConversationStarterCatalog) -> None:
        self._uow_factory = uow_factory
        self._catalog = catalog

    async def list(
        self, *, domain_id: int, agent_id: UUID, target_id: UUID
    ) -> dict:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            if (
                agent is None
                or agent.status != "ACTIVE"
                or agent.current_version_id is None
            ):
                raise resource_not_found("Active AIOps Agent")
            if not await uow.agents.version_has_target(
                agent_version_id=agent.current_version_id,
                target_id=target_id,
            ):
                raise resource_not_found("Agent Target Binding")
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=domain_id,
            )
            if target is None:
                raise resource_not_found("Target")
            return self._catalog.list_for_target(target)
