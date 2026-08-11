"""只生成结构化 PlanDraft 的 Root Agent 规划器。"""

from datetime import datetime, timedelta, timezone
from enum import StrEnum
import json
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.domain.planning import (
    ExecutionKind,
    ExecutionMode,
    PlanDraft,
    TaskSpec,
)


class RouteType(StrEnum):
    CONVERSATION = "CONVERSATION"
    DOCUMENT = "DOCUMENT"
    DATA_QUERY = "DATA_QUERY"
    HYBRID_PARALLEL = "HYBRID_PARALLEL"
    HYBRID_DOCUMENT_FIRST = "HYBRID_DOCUMENT_FIRST"
    HYBRID_DATA_FIRST = "HYBRID_DATA_FIRST"
    AIOPS = "AIOPS"
    CLARIFY = "CLARIFY"


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    route_type: RouteType
    confidence: float = Field(ge=0, le=1)
    reason: str
    clarification_question: str | None = None
    requires_chart: bool = False
    classifier_version: str = "deterministic-document-v1"


class RootAgentPlanner:
    """按冻结 Agent 配置确定单一领域路由并生成可校验计划。"""

    def __init__(self, *, model_client=None, prompt_resolver=None):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

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
        if capabilities == {"conversation"}:
            return RouteDecision(
                route_type=RouteType.CONVERSATION,
                confidence=1.0,
                reason="Agent 配置明确限定为 Conversation 路由",
                classifier_version="deterministic-conversation-v1",
            )
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
                target_id = UUID(str(config["aiops_target_id"]))
                if target_id.version != 7:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return RouteDecision(
                    route_type=RouteType.CLARIFY,
                    confidence=0.0,
                    reason="AIOps 路由尚未选择有效的默认 Target",
                    clarification_question=(
                        "请先将 Agent 绑定到一个可用 Target，"
                        "并将该绑定选为聊天默认目标。"
                    ),
                )
            return RouteDecision(
                route_type=RouteType.AIOPS,
                confidence=1.0,
                reason="Agent 配置明确限定为 AIOps 路由",
                classifier_version="deterministic-aiops-v1",
            )
        if capabilities == {"data_query"}:
            return RouteDecision(
                route_type=RouteType.DATA_QUERY,
                confidence=1.0,
                reason="Agent 配置明确限定为 Data Query 路由",
                classifier_version="deterministic-data-query-v1",
            )
        return RouteDecision(
            route_type=RouteType.CLARIFY,
            confidence=0.0,
            reason="当前首版不能确定唯一执行路由",
            clarification_question=(
                "当前 Agent 启用了多个聊天能力，但尚未完成本轮意图分类。"
            ),
        )

    async def decide_for_input(
        self,
        *,
        agent_snapshot: dict[str, Any],
        objective: str,
        conversation_context: dict[str, Any] | None = None,
        client_metadata: dict[str, Any] | None = None,
    ) -> RouteDecision:
        """单能力确定性路由；多能力使用冻结 Router 模型选择唯一分支。"""
        capabilities = set(
            agent_snapshot.get("enabled_capabilities") or []
        )
        if (
            "document" in capabilities
            and (client_metadata or {}).get("query_images")
        ):
            return RouteDecision(
                route_type=RouteType.DOCUMENT,
                confidence=1,
                reason="请求包含查询图片，进入 Document 多模态检索",
                classifier_version="deterministic-query-image-v1",
            )
        if capabilities == {"data_query"}:
            decision = self.decide(agent_snapshot=agent_snapshot)
            return decision.model_copy(
                update={"requires_chart": self._requests_chart(objective)}
            )
        if len(capabilities) == 1 or "aiops" in capabilities:
            return self.decide(agent_snapshot=agent_snapshot)
        enabled_routes = [
            route
            for capability, route in (
                ("conversation", "CONVERSATION"),
                ("document", "DOCUMENT"),
                ("data_query", "DATA_QUERY"),
            )
            if capability in capabilities
        ]
        if {"document", "data_query"}.issubset(capabilities):
            enabled_routes.extend(
                (
                    "HYBRID_PARALLEL",
                    "HYBRID_DOCUMENT_FIRST",
                    "HYBRID_DATA_FIRST",
                )
            )
        if not enabled_routes:
            return self.decide(agent_snapshot=agent_snapshot)
        model_name = str(
            agent_model_name(agent_snapshot, "router_llm") or ""
        ).strip()
        if (
            not model_name
            or self._model_client is None
            or self._prompt_resolver is None
        ):
            return RouteDecision(
                route_type=RouteType.CLARIFY,
                confidence=0,
                reason="多能力 Agent 未配置可用的 Router 模型",
                clarification_question=(
                    "请先为该 Agent 配置 models.router_llm。"
                ),
                classifier_version="router-unavailable-v1",
            )
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.intent_route"
        )
        context = conversation_context or {}
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=[
                {"role": "system", "content": prompt.content},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "enabled_routes": enabled_routes,
                            "current_input": objective,
                            "conversation_summary": (
                                context.get("summary") or {}
                            ),
                            "recent_items": (
                                context.get("recent_items") or []
                            ),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            max_tokens=1024,
        )
        route_value = str(
            response.get("route_type") or "CLARIFY"
        ).upper()
        document_fallback = (
            "document" in capabilities
            and str(
                (agent_snapshot.get("config") or {}).get(
                    "resource_mode", ""
                )
            ).lower()
            == "managed_resources"
        )
        if route_value not in {*enabled_routes, "CLARIFY"}:
            route_value = "DOCUMENT" if document_fallback else "CLARIFY"
            response["reason"] = (
                "模型选择了未启用的路由，按托管资源配置进入文档检索"
                if document_fallback
                else "模型选择了未启用的路由"
            )
        confidence = max(
            0.0, min(float(response.get("confidence", 0)), 1.0)
        )
        threshold = max(
            0.0,
            min(
                float(
                    (agent_snapshot.get("config") or {}).get(
                        "router_confidence_threshold", 0.6
                    )
                ),
                1.0,
            ),
        )
        if route_value == "CLARIFY" and document_fallback:
            route_value = "DOCUMENT"
            response["reason"] = (
                "托管资源 Agent 对不明确的企业知识问题优先检索文档"
            )
        elif route_value != "CLARIFY" and confidence < threshold:
            route_value = "DOCUMENT" if document_fallback else "CLARIFY"
            response["reason"] = (
                "自然语言路由置信度低，按托管资源配置进入文档检索"
                if document_fallback
                else "自然语言路由置信度低于执行阈值"
            )
        if route_value == "CLARIFY":
            response["clarification_question"] = (
                str(response.get("clarification_question") or "").strip()
                or "请说明这是通用问题、文档查询还是业务数据查询。"
            )
        else:
            response["clarification_question"] = None
        requires_chart = bool(response.get("requires_chart", False))
        if route_value != "DATA_QUERY":
            requires_chart = False
        return RouteDecision(
            route_type=RouteType(route_value),
            confidence=confidence,
            reason=str(response.get("reason") or "自然语言意图分类"),
            clarification_question=(
                str(response.get("clarification_question") or "").strip()
                or None
            ),
            requires_chart=requires_chart,
            classifier_version="llm-single-route-v2",
        )

    @staticmethod
    def _requests_chart(value: str) -> bool:
        normalized = value.casefold()
        return any(
            keyword in normalized
            for keyword in (
                "图表",
                "可视化",
                "趋势图",
                "折线图",
                "柱状图",
                "饼图",
                "散点图",
                "echarts",
                " chart",
                "plot ",
                "visualize",
            )
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
        if decision.route_type == RouteType.CONVERSATION:
            return self._build_conversation_plan(
                objective=objective, ttl_seconds=ttl_seconds
            )
        if decision.route_type == RouteType.CLARIFY:
            return self._build_conversation_plan(
                objective=objective, ttl_seconds=ttl_seconds
            )
        if decision.route_type == RouteType.DATA_QUERY:
            return self._build_data_query_plan(
                objective=objective,
                requires_chart=decision.requires_chart,
                ttl_seconds=ttl_seconds,
            )
        if decision.route_type == RouteType.HYBRID_PARALLEL:
            return self._build_hybrid_parallel_plan(
                objective=objective,
                ttl_seconds=ttl_seconds,
            )
        if decision.route_type == RouteType.HYBRID_DOCUMENT_FIRST:
            return self._build_hybrid_document_first_plan(
                objective=objective,
                ttl_seconds=ttl_seconds,
            )
        if decision.route_type == RouteType.HYBRID_DATA_FIRST:
            return self._build_hybrid_data_first_plan(
                objective=objective,
                ttl_seconds=ttl_seconds,
            )
        if decision.route_type != RouteType.DOCUMENT:
            raise ValueError("不支持的路由不能生成执行计划")
        return PlanDraft(
            plan_version="document-plan-v1",
            objective=objective,
            tasks=(
                TaskSpec(
                    task_key="context_rewrite",
                    task_type="CONTEXT_REWRITE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="conversation",
                    skill_id="context-rewrite",
                    skill_version="1.0.0",
                    input_refs=("RUN_INPUT", "CONVERSATION_CONTEXT"),
                    expected_outputs=("CONTEXT_REWRITE",),
                    timeout_seconds=60,
                    max_retries=1,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
                TaskSpec(
                    task_key="knowledge_retrieval",
                    task_type="RETRIEVE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="document",
                    skill_id="knowledge-retrieval",
                    skill_version="1.0.0",
                    depends_on=("context_rewrite",),
                    input_refs=("task_output:context_rewrite",),
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
                    depends_on=(
                        "context_rewrite",
                        "knowledge_retrieval",
                    ),
                    input_refs=(
                        "task_output:context_rewrite",
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
    def _build_conversation_plan(
        *, objective: str, ttl_seconds: int
    ) -> PlanDraft:
        return PlanDraft(
            plan_version="conversation-plan-v1",
            objective=objective,
            tasks=(
                TaskSpec(
                    task_key="context_rewrite",
                    task_type="CONTEXT_REWRITE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="conversation",
                    skill_id="context-rewrite",
                    skill_version="1.0.0",
                    input_refs=("RUN_INPUT", "CONVERSATION_CONTEXT"),
                    expected_outputs=("CONTEXT_REWRITE",),
                    timeout_seconds=60,
                    max_retries=1,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
                TaskSpec(
                    task_key="conversation_response",
                    task_type="COMPOSE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="conversation",
                    skill_id="conversation-response",
                    skill_version="1.0.0",
                    depends_on=("context_rewrite",),
                    input_refs=("task_output:context_rewrite",),
                    expected_outputs=("GROUNDED_ANSWER",),
                    timeout_seconds=120,
                    max_retries=1,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
            ),
            final_task_key="conversation_response",
            expires_at=(
                datetime.now(timezone.utc)
                + timedelta(seconds=ttl_seconds)
            ),
        )

    @staticmethod
    def _build_data_query_plan(
        *,
        objective: str,
        requires_chart: bool,
        ttl_seconds: int,
    ) -> PlanDraft:
        tasks = [
            TaskSpec(
                task_key="context_rewrite",
                task_type="CONTEXT_REWRITE",
                execution_kind=ExecutionKind.LOCAL_SKILL,
                specialist="conversation",
                skill_id="context-rewrite",
                skill_version="1.0.0",
                input_refs=("RUN_INPUT", "CONVERSATION_CONTEXT"),
                expected_outputs=("CONTEXT_REWRITE",),
                timeout_seconds=60,
                max_retries=1,
                execution_mode=ExecutionMode.READ_ONLY,
            ),
            TaskSpec(
                task_key="data_query",
                task_type="DATA_QUERY",
                execution_kind=ExecutionKind.LOCAL_SKILL,
                specialist="data_query",
                skill_id="data-query",
                skill_version="1.0.0",
                depends_on=("context_rewrite",),
                input_refs=("task_output:context_rewrite",),
                expected_outputs=("QUERY_RESULT",),
                timeout_seconds=180,
                max_retries=2,
                execution_mode=ExecutionMode.READ_ONLY,
            ),
        ]
        compose_dependencies = ["context_rewrite", "data_query"]
        compose_inputs = [
            "task_output:context_rewrite",
            "task_output:data_query",
        ]
        if requires_chart:
            tasks.append(
                TaskSpec(
                    task_key="echarts",
                    task_type="VISUALIZE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="visualization",
                    skill_id="echarts",
                    skill_version="1.0.0",
                    depends_on=("data_query",),
                    input_refs=("task_output:data_query",),
                    expected_outputs=("ECHARTS_CONFIG",),
                    timeout_seconds=120,
                    max_retries=1,
                    execution_mode=ExecutionMode.READ_ONLY,
                )
            )
            compose_dependencies.append("echarts")
            compose_inputs.append("task_output:echarts")
        tasks.append(
            TaskSpec(
                task_key="response_compose",
                task_type="COMPOSE",
                execution_kind=ExecutionKind.LOCAL_SKILL,
                specialist="response_composer",
                skill_id="response-composer",
                skill_version="1.0.0",
                depends_on=tuple(compose_dependencies),
                input_refs=tuple(compose_inputs),
                expected_outputs=("GROUNDED_ANSWER",),
                timeout_seconds=120,
                max_retries=1,
                execution_mode=ExecutionMode.READ_ONLY,
            )
        )
        return PlanDraft(
            plan_version="data-query-plan-v1",
            objective=objective,
            tasks=tuple(tasks),
            final_task_key="response_compose",
            expires_at=(
                datetime.now(timezone.utc)
                + timedelta(seconds=ttl_seconds)
            ),
        )

    @staticmethod
    def _context_task() -> TaskSpec:
        return TaskSpec(
            task_key="context_rewrite",
            task_type="CONTEXT_REWRITE",
            execution_kind=ExecutionKind.LOCAL_SKILL,
            specialist="conversation",
            skill_id="context-rewrite",
            skill_version="1.0.0",
            input_refs=("RUN_INPUT", "CONVERSATION_CONTEXT"),
            expected_outputs=("CONTEXT_REWRITE",),
            timeout_seconds=60,
            max_retries=1,
            execution_mode=ExecutionMode.READ_ONLY,
        )

    @staticmethod
    def _data_query_task(*, dependencies: tuple[str, ...]) -> TaskSpec:
        return TaskSpec(
            task_key="data_query",
            task_type="DATA_QUERY",
            execution_kind=ExecutionKind.LOCAL_SKILL,
            specialist="data_query",
            skill_id="data-query",
            skill_version="1.0.0",
            depends_on=dependencies,
            input_refs=tuple(f"task_output:{item}" for item in dependencies),
            expected_outputs=("QUERY_RESULT",),
            timeout_seconds=180,
            max_retries=2,
            execution_mode=ExecutionMode.READ_ONLY,
        )

    @staticmethod
    def _document_task(*, dependencies: tuple[str, ...]) -> TaskSpec:
        return TaskSpec(
            task_key="knowledge_retrieval",
            task_type="RETRIEVE",
            execution_kind=ExecutionKind.LOCAL_SKILL,
            specialist="document",
            skill_id="knowledge-retrieval",
            skill_version="1.0.0",
            depends_on=dependencies,
            input_refs=tuple(f"task_output:{item}" for item in dependencies),
            expected_outputs=("CITATION_PACK",),
            required_scopes=(
                "knowledge.discovery.read",
                "knowledge.evidence.read",
            ),
            timeout_seconds=120,
            max_retries=2,
            execution_mode=ExecutionMode.READ_ONLY,
        )

    @staticmethod
    def _compose_task(*, dependencies: tuple[str, ...]) -> TaskSpec:
        return TaskSpec(
            task_key="response_compose",
            task_type="COMPOSE",
            execution_kind=ExecutionKind.LOCAL_SKILL,
            specialist="response_composer",
            skill_id="response-composer",
            skill_version="1.0.0",
            depends_on=dependencies,
            input_refs=tuple(f"task_output:{item}" for item in dependencies),
            expected_outputs=("GROUNDED_ANSWER",),
            timeout_seconds=120,
            max_retries=1,
            execution_mode=ExecutionMode.READ_ONLY,
        )

    @classmethod
    def _build_hybrid_parallel_plan(
        cls, *, objective: str, ttl_seconds: int
    ) -> PlanDraft:
        tasks = (
            cls._context_task(),
            cls._document_task(dependencies=("context_rewrite",)),
            cls._data_query_task(dependencies=("context_rewrite",)),
            cls._compose_task(dependencies=(
                "context_rewrite",
                "knowledge_retrieval",
                "data_query",
            )),
        )
        return cls._hybrid_plan(
            "hybrid-parallel-plan-v1", objective, tasks, ttl_seconds
        )

    @classmethod
    def _build_hybrid_document_first_plan(
        cls, *, objective: str, ttl_seconds: int
    ) -> PlanDraft:
        tasks = (
            cls._context_task(),
            cls._document_task(dependencies=("context_rewrite",)),
            TaskSpec(
                task_key="data_constraints",
                task_type="EXTRACT",
                execution_kind=ExecutionKind.LOCAL_SKILL,
                specialist="hybrid",
                skill_id="data-constraint-extract",
                skill_version="1.0.0",
                depends_on=("knowledge_retrieval",),
                input_refs=("task_output:knowledge_retrieval",),
                expected_outputs=("DATA_QUERY_CONSTRAINTS",),
                timeout_seconds=90,
                max_retries=1,
                execution_mode=ExecutionMode.READ_ONLY,
            ),
            cls._data_query_task(
                dependencies=("context_rewrite", "data_constraints")
            ),
            cls._compose_task(dependencies=(
                "context_rewrite",
                "knowledge_retrieval",
                "data_query",
            )),
        )
        return cls._hybrid_plan(
            "hybrid-document-first-plan-v1",
            objective,
            tasks,
            ttl_seconds,
        )

    @classmethod
    def _build_hybrid_data_first_plan(
        cls, *, objective: str, ttl_seconds: int
    ) -> PlanDraft:
        tasks = (
            cls._context_task(),
            cls._data_query_task(dependencies=("context_rewrite",)),
            TaskSpec(
                task_key="document_scope",
                task_type="EXTRACT",
                execution_kind=ExecutionKind.LOCAL_SKILL,
                specialist="hybrid",
                skill_id="document-scope-extract",
                skill_version="1.0.0",
                depends_on=("data_query",),
                input_refs=("task_output:data_query",),
                expected_outputs=("DOCUMENT_SCOPE",),
                timeout_seconds=90,
                max_retries=1,
                execution_mode=ExecutionMode.READ_ONLY,
            ),
            cls._document_task(
                dependencies=("context_rewrite", "document_scope")
            ),
            cls._compose_task(dependencies=(
                "context_rewrite",
                "data_query",
                "knowledge_retrieval",
            )),
        )
        return cls._hybrid_plan(
            "hybrid-data-first-plan-v1", objective, tasks, ttl_seconds
        )

    @staticmethod
    def _hybrid_plan(
        version: str,
        objective: str,
        tasks: tuple[TaskSpec, ...],
        ttl_seconds: int,
    ) -> PlanDraft:
        return PlanDraft(
            plan_version=version,
            objective=objective,
            tasks=tasks,
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
                    task_key="context_rewrite",
                    task_type="CONTEXT_REWRITE",
                    execution_kind=ExecutionKind.LOCAL_SKILL,
                    specialist="conversation",
                    skill_id="context-rewrite",
                    skill_version="1.0.0",
                    input_refs=("RUN_INPUT", "CONVERSATION_CONTEXT"),
                    expected_outputs=("CONTEXT_REWRITE",),
                    timeout_seconds=60,
                    max_retries=1,
                    execution_mode=ExecutionMode.READ_ONLY,
                ),
                TaskSpec(
                    task_key="aiops_diagnosis",
                    task_type="DELEGATE",
                    execution_kind=ExecutionKind.DELEGATION,
                    specialist="aiops",
                    delegate_service="aiops_agent",
                    delegate_capability="diagnosis",
                    depends_on=("context_rewrite",),
                    input_refs=("task_output:context_rewrite",),
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
                    depends_on=("context_rewrite", "aiops_diagnosis"),
                    input_refs=(
                        "task_output:context_rewrite",
                        "task_output:aiops_diagnosis",
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
