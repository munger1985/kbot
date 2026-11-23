"""验证资料授权范围内的知识问答必须先取证。"""

import asyncio
from types import SimpleNamespace
import pytest

from agent_runtime.specialists.root import RootAgentPlanner, RouteType


@pytest.mark.parametrize(
    ("app_id", "scope_hash", "classification", "confidence", "expected"),
    [
        ("knowledge_retrieval", "scope", "CONVERSATION", 0.99, RouteType.DOCUMENT),
        ("knowledge_retrieval", "scope", "CONVERSATION", 0.20, RouteType.DOCUMENT),
        ("knowledge_retrieval", "scope", "DATA_QUERY", 0.99, RouteType.DATA_QUERY),
        (
            "knowledge_retrieval", "scope", "HYBRID_DOCUMENT_FIRST", 0.99,
            RouteType.HYBRID_DOCUMENT_FIRST,
        ),
        ("knowledge_retrieval", None, "CONVERSATION", 0.99, RouteType.CONVERSATION),
        ("another_app", "scope", "CONVERSATION", 0.99, RouteType.CONVERSATION),
    ],
)
def test_bound_knowledge_questions_do_not_skip_retrieval(
    app_id, scope_hash, classification, confidence, expected
):
    class Router:
        async def get_llm_json(self, **kwargs):
            return {"route_type": classification, "confidence": confidence}

    class Prompts:
        async def resolve(self, prompt_key):
            return SimpleNamespace(content="选择路由")

    planner = RootAgentPlanner(model_client=Router(), prompt_resolver=Prompts())
    snapshot = {
        "owner_app_id": app_id,
        "knowledge_scope_hash": scope_hash,
        "enabled_capabilities": ["document", "conversation", "data_query"],
        "models": {"router_llm": {"model_id": "01a0046d-7e89-7c33-bf1b-ba878f274209"}},
    }
    question = "解释一下审计工作管理"
    decision = asyncio.run(
        planner.decide_for_input(
            agent_snapshot=snapshot,
            objective=question,
        )
    )
    assert decision.route_type == expected
    if expected == RouteType.DOCUMENT:
        plan = planner.build_plan(objective=question, decision=decision)
        assert [task.skill_id for task in plan.tasks] == [
            "context-rewrite", "knowledge-retrieval", "response-composer",
        ]
