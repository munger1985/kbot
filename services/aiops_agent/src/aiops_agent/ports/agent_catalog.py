"""AIOps 私有 Agent 配置校验 Port。"""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from platform_core.contracts import AuthContext


@dataclass(frozen=True, slots=True)
class AgentRuntimeBinding:
    binding_id: UUID
    agent_id: UUID
    target_id: UUID
    policy_id: UUID
    status: str
    row_version: int
    allow_mutation: bool
    allowed_actions_json: tuple[str, ...]


class AgentCatalogPort(Protocol):
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

    async def resolve_runtime_binding(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        target_id: UUID,
    ) -> AgentRuntimeBinding: ...
