"""Agent Runtime Client 到配置校验 Port 的适配器。"""

from uuid import UUID

from aiops_agent.application.errors import (
    dependency_unavailable,
    validation_failed,
)
from platform_clients.agent_runtime import (
    AgentRuntimeClient,
    AgentRuntimeClientError,
)
from platform_core.contracts import AuthContext


class AgentRuntimeValidator:
    def __init__(self, client: AgentRuntimeClient):
        self._client = client

    async def validate_aiops_agent(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        auth_context: AuthContext,
    ) -> None:
        try:
            agent = await self._client.get_agent(
                agent_id=agent_id,
                auth_context=auth_context,
            )
        except AgentRuntimeClientError as exc:
            if exc.status_code == 404:
                raise validation_failed("Agent 不存在或不属于当前 Domain") from exc
            raise dependency_unavailable("Agent Runtime 暂时不可用") from exc
        if (
            str(agent.get("domain_id")) != str(domain_id)
            or agent.get("status") != "ACTIVE"
            or "aiops" not in set(agent.get("enabled_capabilities", ()))
        ):
            raise validation_failed(
                "Agent 必须属于当前 Domain、处于 ACTIVE 且声明 aiops 能力"
            )
