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
from platform_clients.model import AIModelConfigClient
from platform_core.contracts import AuthContext
from platform_core.security import create_service_auth_context


class AgentRuntimeValidator:
    def __init__(
        self,
        client: AgentRuntimeClient,
        *,
        model_client: AIModelConfigClient | None = None,
        caller_service: str = "kbot-aiops-api",
    ):
        self._client = client
        self._model_client = model_client
        self._caller_service = caller_service

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

    async def resolve_diagnosis_model(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        trace_id: str,
    ) -> dict[str, str]:
        """解析 Agent 的诊断模型 UUID，并冻结实际调用名称。"""
        if self._model_client is None:
            raise dependency_unavailable("模型目录暂时不可用")
        context = create_service_auth_context(
            caller_service=self._caller_service,
            trace_id=trace_id,
        ).model_copy(
            update={
                "domain_id": str(domain_id),
                "authorized_agent_ids": (agent_id,),
            }
        )
        try:
            agent = await self._client.get_agent(
                agent_id=agent_id,
                auth_context=context,
            )
        except AgentRuntimeClientError as exc:
            if exc.status_code == 404:
                raise validation_failed(
                    "Agent 不存在或不属于当前 Domain"
                ) from exc
            raise dependency_unavailable(
                "Agent Runtime 暂时不可用"
            ) from exc
        if (
            str(agent.get("domain_id")) != str(domain_id)
            or agent.get("status") != "ACTIVE"
            or "aiops" not in set(agent.get("enabled_capabilities", ()))
        ):
            raise validation_failed(
                "Agent 必须属于当前 Domain、处于 ACTIVE 且声明 aiops 能力"
            )
        model_id = dict(agent.get("models") or {}).get("diagnosis_llm")
        if not model_id:
            raise validation_failed(
                "AIOps Agent 必须配置 models.diagnosis_llm"
            )
        try:
            definition = await self._model_client.get_model(UUID(str(model_id)))
        except (LookupError, RuntimeError, ValueError) as exc:
            raise dependency_unavailable("诊断模型目录暂时不可用") from exc
        served_name = str(
            definition.get("served_model_name") or ""
        ).strip()
        if not served_name:
            raise validation_failed("诊断模型缺少 served_model_name")
        return {
            "technical_name": served_name,
            "revision": str(model_id),
        }
