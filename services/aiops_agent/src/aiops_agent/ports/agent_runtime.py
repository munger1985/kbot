"""Agent Runtime 配置校验 Port。"""

from typing import Protocol
from uuid import UUID

from platform_core.contracts import AuthContext


class AgentRuntimePort(Protocol):
    async def validate_aiops_agent(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        auth_context: AuthContext,
    ) -> None: ...

    async def resolve_diagnosis_model(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        trace_id: str,
    ) -> dict[str, str]: ...
