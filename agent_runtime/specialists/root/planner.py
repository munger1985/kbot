"""只生成结构化 PlanDraft 的 Root Agent 规划器。"""

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.domain.planning import (
    ExecutionKind,
    ExecutionMode,
    PlanDraft,
    TaskSpec,
)


class RouteType(StrEnum):
    DOCUMENT = "DOCUMENT"
    AIOPS = "AIOPS"
    CLARIFY = "CLARIFY"


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    route_type: RouteType
    confidence: float = Field(ge=0, le=1)
    reason: str
    clarification_question: str | None = None
    classifier_version: str = "deterministic-document-v1"


class RootAgentPlanner:
    """按冻结 Agent 配置确定单一领域路由并生成可校验计划。"""

    def decide(
        self,
        *,
        agent_snapshot: dict[str, Any],
    ) -> RouteDecision:
        capabilities = set(
            agent_snapshot.get("enabled_capabilities") or []
        )
        default_route = str(
            (agent_snapshot.get("config") or {}).get(
                "default_route", ""
            )
        ).upper()
        if capabilities == {"document"} or (
            "document" in capabilities and default_route == "DOCUMENT"
        ):
            return RouteDecision(
                route_type=RouteType.DOCUMENT,
                confidence=1.0,
                reason="Agent 配置明确限定为 Document 路由",
            )
        if capabilities == {"aiops"} or (
            "aiops" in capabilities and default_route == "AIOPS"
        ):
            config = agent_snapshot.get("config") or {}
            try:
                agent_id = UUID(str(config["aiops_agent_id"]))
                target_id = UUID(str(config["aiops_target_id"]))
                if agent_id.version != 7 or target_id.version != 7:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return RouteDecision(
                    route_type=RouteType.CLARIFY,
                    confidence=0.0,
                    reason="AIOps 路由缺少有效的 UUIDv7 冻结配置",
                    clarification_question=(
                        "请先为 Agent 配置 aiops_agent_id 和 "
                        "aiops_target_id。"
                    ),
                )
            return RouteDecision(
                route_type=RouteType.AIOPS,
                confidence=1.0,
                reason="Agent 配置明确限定为 AIOps 路由",
                classifier_version="deterministic-aiops-v1",
            )
        return RouteDecision(
            route_type=RouteType.CLARIFY,
            confidence=0.0,
            reason="当前首版不能确定唯一执行路由",
            clarification_question=(
                "该 Agent 需要在配置中指定 default_route=DOCUMENT "
                "或 default_route=AIOPS；"
                "多领域自然语言 Router 将在后续阶段启用。"
            ),
        )

    def build_plan(
        self,
        *,
        objective: str,
        decision: RouteDecision,
        ttl_seconds: int = 300,
    ) -> PlanDraft:
        if decision.route_type == RouteType.AIOPS:
            return self._build_aiops_plan(
                objective=objective, ttl_seconds=ttl_seconds
            )
        if decision.route_type != RouteType.DOCUMENT:
            raise ValueError("CLARIFY 决策不能生成执行计划")
        return PlanDraft(
            plan_version="document-plan-v1",
            objective=objective,
            tasks=(
                TaskSpec(
                    task_key="knowledge_retrieval",
                    task_type="RETRIEVE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="document",
                    skill_id="knowledge-retrieval",
                    skill_version="1.0.0",
                    input_refs=("RUN_INPUT",),
                    expected_outputs=("CITATION_PACK",),
                    required_scopes=(
                        "knowledge.discovery.read",
                        "knowledge.evidence.read",
                    ),
                    timeout_seconds=120,
                    max_retries=2,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
                TaskSpec(
                    task_key="response_compose",
                    task_type="COMPOSE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="response_composer",
                    skill_id="response-composer",
                    skill_version="1.0.0",
                    depends_on=("knowledge_retrieval",),
                    input_refs=(
                        "task_output:knowledge_retrieval",
                    ),
                    expected_outputs=("GROUNDED_ANSWER",),
                    timeout_seconds=120,
                    max_retries=1,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
            ),
            final_task_key="response_compose",
            expires_at=(
                datetime.now(timezone.utc)
                + timedelta(seconds=ttl_seconds)
            ),
        )

    @staticmethod
    def _build_aiops_plan(
        *, objective: str, ttl_seconds: int
    ) -> PlanDraft:
        return PlanDraft(
            plan_version="aiops-delegation-plan-v1",
            objective=objective,
            tasks=(
                TaskSpec(
                    task_key="aiops_diagnosis",
                    task_type="DELEGATE",
                    execution_kind=ExecutionKind.DELEGATION,
                    specialist="aiops",
                    delegate_service="aiops_agent",
                    delegate_capability="diagnosis",
                    input_refs=("RUN_INPUT",),
                    expected_outputs=("DELEGATED_AIOPS_RESULT",),
                    required_scopes=("aiops.delegate",),
                    timeout_seconds=600,
                    max_retries=2,
                    execution_mode=ExecutionMode.DELEGATED,
                ),
                TaskSpec(
                    task_key="response_compose",
                    task_type="COMPOSE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="response_composer",
                    skill_id="response-composer",
                    skill_version="1.0.0",
                    depends_on=("aiops_diagnosis",),
                    input_refs=("task_output:aiops_diagnosis",),
                    expected_outputs=("GROUNDED_ANSWER",),
                    timeout_seconds=120,
                    max_retries=1,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
            ),
            final_task_key="response_compose",
            expires_at=(
                datetime.now(timezone.utc)
                + timedelta(seconds=ttl_seconds)
            ),
        )
