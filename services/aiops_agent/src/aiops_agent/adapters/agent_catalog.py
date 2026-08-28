"""AIOps 私有 Agent 到配置校验 Port 的适配器。"""

from uuid import UUID

from aiops_agent.application.agents import AIOpsAgentError, AIOpsAgentService
from aiops_agent.application.errors import (
    dependency_unavailable,
    validation_failed,
)
from platform_clients.model import AIModelConfigClient
from platform_core.contracts import AuthContext
from aiops_agent.ports.agent_catalog import AgentRuntimeBinding


class AIOpsAgentValidator:
    """校验本应用拥有的 Agent，并解析诊断模型目录信息。"""

    def __init__(
        self,
        service: AIOpsAgentService,
        *,
        model_client: AIModelConfigClient | None = None,
    ) -> None:
        self._service = service
        self._model_client = model_client

    async def validate_aiops_agent(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        auth_context: AuthContext,
    ) -> None:
        del auth_context
        agent = await self._get_active(agent_id=agent_id, domain_id=domain_id)
        if agent.get("status") != "ACTIVE":
            raise validation_failed(
                "AIOps Agent 必须属于当前 Domain 且处于 ACTIVE"
            )

    async def resolve_diagnosis_model(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        trace_id: str,
    ) -> dict[str, str]:
        del trace_id
        if self._model_client is None:
            raise dependency_unavailable("模型目录暂时不可用")
        agent = await self._get_active(agent_id=agent_id, domain_id=domain_id)
        model_id = dict(agent.get("models") or {}).get("diagnosis_llm")
        if not model_id:
            raise validation_failed("AIOps Agent 必须配置 models.diagnosis_llm")
        try:
            definition = await self._model_client.get_model(UUID(str(model_id)))
        except (LookupError, RuntimeError, ValueError) as exc:
            raise dependency_unavailable("诊断模型目录暂时不可用") from exc
        served_name = str(definition.get("served_model_name") or "").strip()
        if not served_name:
            raise validation_failed("诊断模型缺少 served_model_name")
        return {"technical_name": served_name, "revision": str(model_id)}

    async def resolve_runtime_binding(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        target_id: UUID,
    ) -> AgentRuntimeBinding:
        """将私有 Agent 当前版本转换为 Run 的唯一配置绑定。"""
        agent = await self._get_active(
            agent_id=agent_id, domain_id=domain_id
        )
        target_ids = {str(item) for item in agent.get("target_ids") or ()}
        policy_id = agent.get("policy_id")
        version_id = agent.get("agent_version_id")
        if str(target_id) not in target_ids:
            raise validation_failed("AIOps Agent 未绑定当前诊断目标")
        target = next(
            (
                item
                for item in agent.get("target_candidates") or ()
                if str(item.get("target_id")) == str(target_id)
            ),
            None,
        )
        if target is None:
            raise validation_failed("AIOps Agent 当前诊断目标配置不完整")
        if not policy_id or not version_id:
            raise validation_failed("AIOps Agent 当前版本配置不完整")
        return AgentRuntimeBinding(
            binding_id=UUID(str(version_id)),
            agent_id=agent_id,
            target_id=UUID(str(target_id)),
            policy_id=UUID(str(policy_id)),
            status="ACTIVE",
            row_version=int(agent.get("row_version") or 1),
            allow_mutation=(
                bool(agent.get("allow_change_execution", False))
                and bool(target.get("controlled_change_enabled", False))
            ),
            allowed_actions_json=tuple(
                str(item) for item in agent.get("allowed_action_types", [])
            ),
        )

    async def _get_active(self, *, agent_id: UUID, domain_id: int) -> dict:
        try:
            agent = await self._service.get(
                domain_id=domain_id,
                agent_id=agent_id,
            )
        except AIOpsAgentError as exc:
            if exc.status_code == 404:
                raise validation_failed(
                    "AIOps Agent 不存在或不属于当前 Domain"
                ) from exc
            raise
        if agent.get("status") != "ACTIVE":
            raise validation_failed(
                "AIOps Agent 必须属于当前 Domain 且处于 ACTIVE"
            )
        return agent
