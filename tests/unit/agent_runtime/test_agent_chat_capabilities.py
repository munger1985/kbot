"""4.0 通用对话、问数、图表与 Dify Adapter 的聚焦测试。"""

import json
import unittest
from types import SimpleNamespace

from agent_runtime.application.commands import LeasedArtifact
from agent_runtime.runtime import ExecutionContext, SkillProgress, SkillResult
from agent_runtime.specialists.conversation_response import (
    ConversationResponseSkill,
)
from agent_runtime.specialists.data_query import (
    DataQuerySkill,
    MCPDataQueryExecutor,
    QueryResult,
    SemanticDataQueryExecutor,
)
from agent_runtime.specialists.response_composer import ResponseComposerSkill
from agent_runtime.specialists.visualization import EChartsSkill
from agent_runtime.specialists.root import (
    KMAnswerBasis,
    RootAgentPlanner,
    RouteType,
)
from main_api.api.dify import _records
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.contracts.data_query import DataQueryPlanV1
from platform_core.identity import uuid7


class _PromptResolver:
    async def resolve(self, key):
        return SimpleNamespace(
            content=f"prompt:{key}",
            version="1.0.0",
            ref=lambda: {"prompt_key": key, "version": "1.0.0"},
        )


class _ModelClient:
    def __init__(self, *, response=None, responses=(), chunks=()):
        self.response = response or {}
        self.responses = list(responses)
        self.chunks = chunks
        self.last_json_request = None
        self.json_requests = []

    async def get_llm_json(self, **kwargs):
        self.last_json_request = kwargs
        self.json_requests.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return self.response

    async def stream_llm_chunks(self, **kwargs):
        for content in self.chunks:
            yield SimpleNamespace(content=content, reasoning_content=None)


class _DataClient:
    def __init__(self):
        self.request = None

    async def query(self, **kwargs):
        self.request = kwargs
        return {
            "rows": [{"month": "2026-07", "sales": 10}],
            "upstream_row_count": 1,
            "truncated": False,
        }


class _UnavailableSemanticExecutor:
    async def execute(self, **kwargs):
        raise AssertionError("MCP 模式不应调用语义 Provider")


def _artifact(artifact_type, payload):
    return LeasedArtifact(
        artifact_id=uuid7(),
        task_id=uuid7(),
        artifact_type=artifact_type,
        schema_version=f"{artifact_type}.v1",
        producer="test",
        producer_version="1.0.0",
        payload=payload,
        content_hash="a" * 64,
        security_level=0,
    )


def _context(
    *,
    artifacts=(),
    agent=None,
    route=None,
    original_input="你好",
    policy_snapshot=None,
    language=None,
):
    config_snapshot = {"agent": agent or {}, "route": route}
    if language is not None:
        config_snapshot["language"] = language
    return ExecutionContext(
        domain_id=20,
        agent_id=uuid7(),
        run_id=uuid7(),
        task_id=uuid7(),
        task_key="test",
        actor_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
        original_input=original_input,
        policy_snapshot=policy_snapshot or {},
        config_snapshot=config_snapshot,
        input_artifacts=tuple(artifacts),
    )


class AgentChatCapabilitiesTest(unittest.IsolatedAsyncioTestCase):
    def test_semantic_plan_removes_only_known_input_echo_fields(self):
        model_id = uuid7()
        models = [{
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "datasets": [{"name": "assets"}],
            "dimensions": [{"name": "author"}],
            "measures": [{
                "name": "asset_count",
                "aggregation": "COUNT",
            }],
            "max_rows": 1000,
        }]
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["title"],
                "filters": [{
                    "field": "author",
                    "operator": "EQ",
                    "values": ["madhumitha.k@oracle.com"],
                }],
                "limit": 100,
                "question": "any assets of madhumitha.k@oracle.com；",
                "models": models,
                "document_constraints": None,
            },
            models=models,
            question="any assets of madhumitha.k@oracle.com；",
            consumer_app_id="km_asset",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual(
            ("madhumitha.k@oracle.com",),
            plan.filters[0].values,
        )
        for field in ("question", "models", "document_constraints"):
            self.assertNotIn(field, normalized)

    def test_semantic_plan_keeps_unknown_fields_for_strict_validation(self):
        model_id = uuid7()
        models = [{
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "datasets": [{"name": "assets"}],
            "measures": [{
                "name": "asset_count",
                "aggregation": "COUNT",
            }],
            "max_rows": 1000,
        }]
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "unexpected_field": "must still fail",
            },
            models=models,
            question="any assets",
            consumer_app_id="km_asset",
        )

        with self.assertRaises(ValueError):
            DataQueryPlanV1.model_validate(normalized)

    def test_semantic_plan_normalizes_catalog_aggregation_and_limit(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "limit": "100",
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="有多少个 Asset",
            consumer_app_id="km_asset",
        )

        self.assertEqual(
            [{"name": "asset_count", "aggregation": "COUNT"}],
            normalized["measures"],
        )
        self.assertEqual(100, normalized["limit"])

    def test_km_semantic_plan_restores_empty_measure_from_managed_catalog(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [],
                "limit": None,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "measures": [
                    {"name": "asset_count", "aggregation": "COUNT"},
                    {
                        "name": "author_count",
                        "aggregation": "COUNT_DISTINCT",
                    },
                ],
                "max_rows": 50,
            }],
            question="AI 相关的 Asset 有哪些",
            consumer_app_id="km_asset",
        )

        self.assertEqual(
            [{"name": "asset_count", "aggregation": "COUNT"}],
            normalized["measures"],
        )
        self.assertEqual(50, normalized["limit"])

    def test_km_semantic_plan_normalizes_catalog_filter_shape(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["title"],
                "filters": [{
                    "dimension": "solution",
                    "operator": "contains",
                    "value": "chatbi",
                }],
                "limit": 100,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "dimensions": [
                    {"name": "title"},
                    {"name": "solution"},
                ],
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="查一下关于 chatbi 的 asset",
            consumer_app_id="km_asset",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual("solution", plan.filters[0].field)
        self.assertEqual("CONTAINS", plan.filters[0].operator)
        self.assertEqual(("chatbi",), plan.filters[0].values)

    def test_km_semantic_plan_uses_catalog_author_operator(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["title", "author"],
                "filters": [{
                    "field": "author",
                    "operator": "CONTAINS",
                    "values": ["lavkesh.singh"],
                }],
                "limit": 100,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "dimensions": [
                    {"name": "title"},
                    {
                        "name": "author",
                        "allowed_filter_operators": ["EQ", "IN"],
                    },
                ],
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="list all assets of lavkesh.singh",
            consumer_app_id="km_asset",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual("EQ", plan.filters[0].operator)

    async def test_km_topic_aggregate_plan_repairs_single_field_filter(self):
        model_id = uuid7()
        base_response = {
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "dataset": "assets",
            "dimensions": [],
            "measures": [{"name": "asset_count", "aggregation": "COUNT"}],
            "order_by": [],
            "limit": 100,
        }
        model_client = _ModelClient(responses=(
            {
                **base_response,
                "filters": [{
                    "field": "solution",
                    "operator": "CONTAINS",
                    "values": ["OAC"],
                }],
            },
            {
                **base_response,
                "filters": [{
                    "field": "topic",
                    "operator": "CONTAINS",
                    "values": ["OAC"],
                }],
            },
        ))
        executor = SemanticDataQueryExecutor(
            client=None,
            model_client=model_client,
            prompt_resolver=_PromptResolver(),
        )
        context = _context(
            original_input="how many asset about OAC",
            agent={
                "owner_app_id": "km_asset",
                "models": {
                    "data_planner_llm": {
                        "served_model_name": "planner-model"
                    }
                },
            },
            route={
                "route_type": "DATA_QUERY",
                "answer_basis": "SEMANTIC_RELEVANCE_AGGREGATE",
            },
        )
        models = [{
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "datasets": [{"name": "assets"}],
            "dimensions": [
                {
                    "name": "topic",
                    "groupable": False,
                    "allowed_filter_operators": ["CONTAINS"],
                },
                {"name": "solution"},
            ],
            "measures": [
                {"name": "asset_count", "aggregation": "COUNT"}
            ],
            "max_rows": 1000,
        }]

        plan = await executor._create_plan(
            context=context,
            question="how many asset about OAC",
            models=models,
        )

        self.assertEqual("topic", plan.filters[0].field)
        self.assertEqual("CONTAINS", plan.filters[0].operator)
        self.assertEqual(2, len(model_client.json_requests))

    async def test_km_topic_enumeration_freezes_complete_asset_scope(self):
        model_id = uuid7()
        model_client = _ModelClient(response={
            "semantic_model_id": str(model_id),
            "semantic_model_version": 2,
            "dataset": "assets",
            "measures": [{
                "name": "asset_count", "aggregation": "COUNT"
            }],
            "dimensions": ["title"],
            "filters": [{
                "field": "topic",
                "operator": "CONTAINS",
                "values": ["OAC"],
            }],
            "order_by": [],
            "limit": 3,
        })
        executor = SemanticDataQueryExecutor(
            client=None,
            model_client=model_client,
            prompt_resolver=_PromptResolver(),
        )
        context = _context(
            original_input="列出关于 OAC 的 asset",
            agent={
                "owner_app_id": "km_asset",
                "models": {
                    "data_planner_llm": {
                        "served_model_name": "planner-model"
                    }
                },
            },
            route={
                "route_type": "HYBRID_DATA_FIRST",
                "answer_basis": "SEMANTIC_RELEVANCE_ENUMERATION",
            },
        )
        dimensions = [
            {"name": name}
            for name in (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "topic",
            )
        ]
        dimensions[-1].update({
            "groupable": False,
            "allowed_filter_operators": ["CONTAINS"],
        })
        models = [{
            "semantic_model_id": str(model_id),
            "semantic_model_version": 2,
            "datasets": [{"name": "assets"}],
            "dimensions": dimensions,
            "measures": [{
                "name": "asset_count", "aggregation": "COUNT"
            }],
            "max_rows": 1000,
        }]

        plan = await executor._create_plan(
            context=context,
            question="列出关于 OAC 的 asset",
            models=models,
        )

        self.assertEqual(
            (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution",
            ),
            plan.dimensions,
        )
        self.assertEqual(10, plan.limit)
        self.assertEqual("topic", plan.filters[0].field)

    async def test_router_only_exposes_selected_knowledge_capabilities(self):
        model = _ModelClient(
            response={
                "route_type": "HYBRID_PARALLEL",
                "confidence": 0.9,
                "reason": "需要同时查询制度和业务数据",
                "clarification_question": None,
                "requires_chart": False,
                "context_required": False,
            }
        )
        planner = RootAgentPlanner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {
                        "served_model_name": "router-model"
                    }
                },
            },
            objective="根据员工套餐制度统计本月使用人数",
        )

        request = json.loads(
            model.last_json_request["prompt"][2]["content"]
        )
        self.assertNotIn("CONVERSATION", request["enabled_routes"])
        self.assertIn("DOCUMENT", request["enabled_routes"])
        self.assertIn("DATA_QUERY", request["enabled_routes"])
        self.assertEqual(decision.route_type, RouteType.HYBRID_PARALLEL)

    async def test_km_topic_enumeration_uses_data_first_hybrid(self):
        model = _ModelClient(
            response={
                "route_type": "HYBRID_DATA_FIRST",
                "confidence": 0.9,
                "reason": "先确定完整 Asset 集合，再读取对应正文",
                "clarification_question": None,
                "requires_chart": False,
                "context_required": False,
                "coverage_mode": "BALANCED",
                "answer_basis": "SEMANTIC_RELEVANCE_ENUMERATION",
            }
        )
        planner = RootAgentPlanner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="查一下关于 ChatBI 的 asset",
        )

        self.assertEqual(RouteType.HYBRID_DATA_FIRST, decision.route_type)
        self.assertEqual("BALANCED", decision.coverage_mode)
        self.assertEqual(
            "llm-km-asset-v1:1.0.0", decision.classifier_version
        )
        self.assertIsNotNone(model.last_json_request)

    async def test_km_aggregate_question_uses_data_query(self):
        planner = RootAgentPlanner(
            model_client=_ModelClient(response={
                "route_type": "DATA_QUERY",
                "confidence": 0.99,
                "reason": "需要按作者统计 Asset 数量",
                "clarification_question": None,
                "requires_chart": False,
                "context_required": False,
                "coverage_mode": "BALANCED",
                "answer_basis": "EXACT_METADATA",
            }),
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="THASNEEM.FATHIMA 发布了多少个 asset",
        )

        self.assertEqual(RouteType.DATA_QUERY, decision.route_type)

    async def test_km_colloquial_count_questions_use_data_query(self):
        model = _ModelClient(response={
            "route_type": "DATA_QUERY",
            "confidence": 0.98,
            "reason": "问题要求统计 Asset 数量",
            "clarification_question": None,
            "requires_chart": False,
            "context_required": False,
            "coverage_mode": "BALANCED",
            "answer_basis": "UNSCOPED_AGGREGATE",
        })
        planner = RootAgentPlanner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )
        agent = {
            "owner_app_id": "km_asset",
            "enabled_capabilities": ["document", "data_query"],
            "models": {
                "router_llm": {"served_model_name": "router-model"}
            },
        }

        for objective in (
            "现在有几个asset",
            "总共有几条 Asset",
            "知识库中的 Asset 总数",
        ):
            with self.subTest(objective=objective):
                decision = await planner.decide_for_input(
                    agent_snapshot=agent,
                    objective=objective,
                )
                self.assertEqual(RouteType.DATA_QUERY, decision.route_type)

    async def test_km_multilingual_intents_use_semantic_router(self):
        cases = (
            (
                "How many assets are available?",
                RouteType.DATA_QUERY,
                "en-US",
                "BALANCED",
                KMAnswerBasis.UNSCOPED_AGGREGATE,
            ),
            (
                "현재 자산은 몇 개입니까?",
                RouteType.DATA_QUERY,
                "ko-KR",
                "BALANCED",
                KMAnswerBasis.UNSCOPED_AGGREGATE,
            ),
            (
                "現在のアセット数はいくつですか？",
                RouteType.DATA_QUERY,
                "ja-JP",
                "BALANCED",
                KMAnswerBasis.UNSCOPED_AGGREGATE,
            ),
            (
                "how many asset about OAC",
                RouteType.DATA_QUERY,
                "en-US",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_AGGREGATE,
            ),
            (
                "有多少关于 OAC 的 asset",
                RouteType.DATA_QUERY,
                "zh-CN",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_AGGREGATE,
            ),
            (
                "Find assets related to ChatBI",
                RouteType.HYBRID_DATA_FIRST,
                "en-US",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            ),
            (
                "ChatBI 관련 자료를 찾아주세요",
                RouteType.HYBRID_DATA_FIRST,
                "ko-KR",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            ),
            (
                "ChatBIに関連する資料を探してください",
                RouteType.HYBRID_DATA_FIRST,
                "ja-JP",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            ),
        )
        agent = {
            "owner_app_id": "km_asset",
            "enabled_capabilities": ["document", "data_query"],
            "models": {
                "router_llm": {"served_model_name": "router-model"}
            },
        }
        for (
            objective,
            expected,
            expected_language,
            coverage_mode,
            answer_basis,
        ) in cases:
            with self.subTest(objective=objective):
                model = _ModelClient(response={
                    "route_type": expected.value,
                    "confidence": 0.97,
                    "reason": "语义分类结果",
                    "clarification_question": None,
                    "requires_chart": False,
                    "context_required": False,
                    "coverage_mode": coverage_mode,
                    "answer_basis": answer_basis.value,
                })
                planner = RootAgentPlanner(
                    model_client=model,
                    prompt_resolver=_PromptResolver(),
                )
                decision = await planner.decide_for_input(
                    agent_snapshot=agent,
                    objective=objective,
                )
                self.assertEqual(expected, decision.route_type)
                self.assertEqual(coverage_mode, decision.coverage_mode)
                self.assertEqual(answer_basis, decision.answer_basis)
                messages = model.last_json_request["prompt"]
                request = json.loads(messages[2]["content"])
                self.assertEqual(expected_language, request["language"])
                self.assertIn(
                    f"language={expected_language}",
                    messages[1]["content"],
                )

    async def test_km_follow_up_resolves_previous_count_scope_without_clarify(self):
        model = _ModelClient(responses=(
            {
                "route_type": "CLARIFY",
                "confidence": 0.7,
                "reason": "当前短语可能依赖上一轮问题",
                "clarification_question": "请说明要查找还是统计相关 Asset。",
                "requires_chart": False,
                "context_required": True,
                "coverage_mode": "BALANCED",
                "answer_basis": "AMBIGUOUS",
            },
            {
                "route_type": "DATA_QUERY",
                "confidence": 0.96,
                "reason": "用户已回答上一轮澄清，应统计 ChatBI 主题相关 Asset",
                "clarification_question": None,
                "requires_chart": False,
                "context_required": False,
                "coverage_mode": "BALANCED",
                "answer_basis": "SEMANTIC_RELEVANCE_AGGREGATE",
            },
        ))
        planner = RootAgentPlanner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )
        context = {
            "summary": {},
            "recent_items": [
                {
                    "role": "USER",
                    "content": {"text": "现在有几个asset"},
                    "item_sequence": 1,
                },
                {
                    "role": "ASSISTANT",
                    "content": {
                        "text": (
                            "您指的是知识库中所有 asset 的总数，"
                            "还是特指与 chatbi 相关的 asset 数量？"
                        )
                    },
                    "item_sequence": 2,
                },
            ],
        }

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="chatbi相关的",
            conversation_context=context,
        )

        self.assertEqual(RouteType.DATA_QUERY, decision.route_type)
        self.assertIsNone(decision.clarification_question)
        request = json.loads(model.last_json_request["prompt"][2]["content"])
        self.assertEqual("chatbi相关的", request["current_input"])
        self.assertEqual(context["recent_items"], request["recent_items"])
        self.assertEqual(2, len(model.json_requests))

    async def test_km_topic_count_rejects_document_route_and_repairs(self):
        model = _ModelClient(responses=(
            {
                "route_type": "DOCUMENT",
                "confidence": 0.91,
                "reason": "需要查找相关 Asset",
                "clarification_question": None,
                "requires_chart": False,
                "context_required": False,
                "coverage_mode": "BREADTH",
                "answer_basis": "SEMANTIC_RELEVANCE_AGGREGATE",
            },
            {
                "route_type": "DATA_QUERY",
                "confidence": 0.99,
                "reason": "主题相关 Asset 数量应由托管问数模型统计",
                "clarification_question": None,
                "requires_chart": False,
                "context_required": False,
                "coverage_mode": "BALANCED",
                "answer_basis": "SEMANTIC_RELEVANCE_AGGREGATE",
            },
        ))
        planner = RootAgentPlanner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="how many asset about OAC",
        )

        self.assertEqual(RouteType.DATA_QUERY, decision.route_type)
        self.assertEqual(
            KMAnswerBasis.SEMANTIC_RELEVANCE_AGGREGATE,
            decision.answer_basis,
        )
        self.assertEqual(2, len(model.json_requests))
        repair_message = model.json_requests[1]["prompt"][-1]["content"]
        self.assertIn("SEMANTIC_RELEVANCE_AGGREGATE", repair_message)
        self.assertIn("DATA_QUERY", repair_message)

    async def test_km_genuine_ambiguity_can_request_clarification(self):
        model = _ModelClient(response={
            "route_type": "CLARIFY",
            "confidence": 0.5,
            "reason": "缺少可解析的对象和操作",
            "clarification_question": "请说明您要查询哪个 Asset，以及需要内容还是统计数据。",
            "requires_chart": False,
            "context_required": True,
            "coverage_mode": "BALANCED",
            "answer_basis": "AMBIGUOUS",
        })
        planner = RootAgentPlanner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="那个呢",
        )

        self.assertEqual(RouteType.CLARIFY, decision.route_type)
        self.assertTrue(decision.clarification_question)

    async def test_multi_capability_router_selects_data_chart(self):
        planner = RootAgentPlanner(
            model_client=_ModelClient(
                response={
                    "route_type": "DATA_QUERY",
                    "confidence": 0.97,
                    "reason": "需要查询销售统计并绘图",
                    "clarification_question": None,
                    "requires_chart": True,
                }
            ),
            prompt_resolver=_PromptResolver(),
        )
        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {
                        "served_model_name": "router-model"
                    }
                },
            },
            objective="画出本月销售趋势图",
        )
        plan = planner.build_plan(
            objective="画出本月销售趋势图", decision=decision
        )

        self.assertEqual(decision.route_type, RouteType.DATA_QUERY)
        self.assertTrue(decision.requires_chart)
        self.assertEqual(
            [task.task_key for task in plan.tasks],
            [
                "context_rewrite",
                "data_query",
                "echarts",
                "response_compose",
            ],
        )

    async def test_query_image_routes_to_document(self):
        planner = RootAgentPlanner()
        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": ["conversation", "document"],
                "models": {
                    "router_llm": {
                        "served_model_name": "router-model"
                    }
                },
            },
            objective="找一下相似的案例",
            client_metadata={"query_images": [{"storage_uri": "/tmp/x"}]},
        )

        self.assertEqual(decision.route_type, RouteType.DOCUMENT)

    async def test_managed_resources_falls_back_to_document(self):
        planner = RootAgentPlanner(
            model_client=_ModelClient(
                response={
                    "route_type": "CLARIFY",
                    "confidence": 0.35,
                    "reason": "套餐含义不明确",
                    "clarification_question": (
                        "请说明这是通用问题、文档查询还是业务数据查询。"
                    ),
                    "requires_chart": False,
                }
            ),
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": [
                    "conversation",
                    "document",
                    "data_query",
                ],
                "models": {
                    "router_llm": {
                        "served_model_name": "router-model"
                    }
                },
                "config": {"resource_mode": "managed_resources"},
            },
            objective="员工有哪些套餐",
        )
        plan = planner.build_plan(
            objective="员工有哪些套餐",
            decision=decision,
        )

        self.assertEqual(RouteType.DOCUMENT, decision.route_type)
        self.assertIsNone(decision.clarification_question)
        self.assertIn(
            "knowledge_retrieval",
            [task.task_key for task in plan.tasks],
        )

    async def test_mcp_data_skill_preserves_profile_and_rows(self):
        client = _DataClient()
        result = await DataQuerySkill(
            mcp_executor=MCPDataQueryExecutor(client=client),
            semantic_executor=_UnavailableSemanticExecutor(),
        ).execute(
            _context(
                original_input="查询销售额",
                agent={
                    "config": {
                        "data_query_mode": "MCP",
                        "data_profile_name": "SALES_PROFILE",
                    },
                },
                artifacts=(
                    _artifact(
                        "CONTEXT_REWRITE",
                        {"standalone_query": "查询 2026 年 7 月销售额"},
                    ),
                ),
            )
        )

        self.assertEqual(result.artifact.artifact_type, "QUERY_RESULT")
        self.assertEqual(result.artifact.schema_version, "QUERY_RESULT.v1")
        self.assertEqual(result.artifact.payload["schema"], "QUERY_RESULT.v1")
        self.assertEqual(result.artifact.payload["provider"], "MCP")
        self.assertEqual(
            result.artifact.payload["rows"][0]["sales"], 10
        )
        self.assertEqual(client.request["profile"], "SALES_PROFILE")
        self.assertEqual(
            client.request["question"], "查询 2026 年 7 月销售额"
        )

    async def test_data_answer_prompt_uses_frozen_response_language(self):
        query = QueryResult.model_validate({
            "schema": "QUERY_RESULT.v1",
            "query_result_id": str(uuid7()),
            "provider": "MCP",
            "columns": [{"name": "asset_count"}],
            "rows": [{"asset_count": 3}],
            "row_count": 1,
            "truncated": False,
            "warnings": [],
            "provenance": {"profile": "KM_ASSET"},
        })
        skill = ResponseComposerSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        )

        _, messages = await skill._query_prompt(
            _context(
                original_input="현재 자산은 몇 개입니까?",
                language="ko-KR",
                agent={
                    "models": {
                        "composer_llm": {
                            "served_model_name": "composer-model"
                        }
                    }
                },
            ),
            query,
        )

        self.assertIn("language=ko-KR", messages[0]["content"])
        self.assertIn("language=ko-KR", messages[1]["content"])

    async def test_semantic_and_mcp_share_query_result_contract(self):
        semantic_model_id = uuid7()

        class _SemanticClient:
            async def get_planning_context(self, **kwargs):
                return {"models": [{
                    "semantic_model_id": str(semantic_model_id),
                    "semantic_model_version": 1,
                    "display_name": "销售",
                    "datasets": [{"name": "sales"}],
                    "dimensions": [],
                    "measures": [{"name": "amount"}],
                    "max_rows": 100,
                }]}

            async def create_run(self, **kwargs):
                return {"data_query_run_id": str(uuid7())}

            async def get_run(self, **kwargs):
                return {"status": "COMPLETED"}

            async def get_result(self, **kwargs):
                return {
                    "columns": [{"name": "amount"}],
                    "preview_rows": [{"amount": 10}],
                    "row_count": 1,
                    "observed_row_count": 1,
                    "truncated": False,
                    "provenance": {
                        "data_source_id": str(uuid7()),
                        "semantic_model_id": str(semantic_model_id),
                        "semantic_model_version": "1",
                        "query_plan_hash": "a" * 64,
                    },
                }

        model = _ModelClient(response={
            "contract_version": "DataQueryPlan.v1",
            "semantic_model_id": str(semantic_model_id),
            "semantic_model_version": 1,
            "dataset": "sales",
            "measures": [{"name": "amount", "aggregation": "SUM"}],
            "dimensions": [],
            "filters": [],
            "order_by": [],
            "limit": 100,
            "time_zone": "Asia/Shanghai",
        })
        auth = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id="test",
            calling_service="test",
            request_id="request-1",
            trace_id="trace-1",
            domain_id="20",
            asserted_user_id="user-1",
        )
        semantic = SemanticDataQueryExecutor(
            client=_SemanticClient(),
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )
        result = await DataQuerySkill(
            mcp_executor=MCPDataQueryExecutor(client=None),
            semantic_executor=semantic,
        ).execute(_context(
            original_input="查询销售额",
            agent={
                "config": {"data_query_mode": "SEMANTIC"},
                "agent_version_id": str(uuid7()),
                "models": {"data_planner_llm": {"served_model_name": "planner"}},
            },
            policy_snapshot={"auth_context": auth.model_dump(mode="json")},
        ))

        self.assertEqual(result.artifact.schema_version, "QUERY_RESULT.v1")
        self.assertEqual(result.artifact.payload["schema"], "QUERY_RESULT.v1")
        self.assertEqual(result.artifact.payload["provider"], "SEMANTIC")
        self.assertEqual(result.artifact.payload["rows"], [{"amount": 10}])

    async def test_data_query_stream_only_returns_final_artifact(self):
        skill = DataQuerySkill(
            mcp_executor=MCPDataQueryExecutor(client=_DataClient()),
            semantic_executor=_UnavailableSemanticExecutor(),
        )

        items = [
            item async for item in skill.execute_stream(_context(
                original_input="查询销售额",
                agent={
                    "config": {
                        "data_query_mode": "MCP",
                        "data_profile_name": "SALES_PROFILE",
                    },
                },
            ))
        ]

        self.assertEqual(1, len(items))
        self.assertIsInstance(items[0], SkillResult)

    def test_km_enumeration_threshold_message_is_exact(self):
        complete = ResponseComposerSkill._enumeration_prefix(
            language="zh-CN",
            total_count=5,
            shown_count=5,
            truncated=False,
            source_truncated=False,
        )
        clipped = ResponseComposerSkill._enumeration_prefix(
            language="zh-CN",
            total_count=12,
            shown_count=10,
            truncated=True,
            source_truncated=False,
        )

        self.assertIn("共命中 5 个", complete)
        self.assertIn("全部列出", complete)
        self.assertIn("共命中 12 个", clipped)
        self.assertIn("前 10 个", clipped)
        self.assertIn("已截断", clipped)

    async def test_conversation_response_streams_and_returns_answer(self):
        skill = ConversationResponseSkill(
            model_client=_ModelClient(chunks=("你", "好")),
            prompt_resolver=_PromptResolver(),
        )
        items = [
            item
            async for item in skill.execute_stream(
                _context(
                    agent={
                        "models": {
                            "composer_llm": {
                                "served_model_name": "chat-model"
                            }
                        }
                    }
                )
            )
        ]

        self.assertTrue(any(isinstance(item, SkillProgress) for item in items))
        result = next(
            item for item in items if isinstance(item, SkillResult)
        )
        self.assertEqual(result.artifact.payload["answer"], "你好")
        self.assertEqual(result.artifact.payload["references"], [])

    async def test_route_clarification_is_returned_as_chat_answer(self):
        skill = ConversationResponseSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        )
        items = [
            item
            async for item in skill.execute_stream(
                _context(
                    route={
                        "route_type": "CLARIFY",
                        "reason": "置信度不足",
                        "clarification_question": "请问要查文档还是业务数据？",
                    }
                )
            )
        ]
        result = next(
            item for item in items if isinstance(item, SkillResult)
        )

        self.assertEqual(
            result.artifact.payload["status"],
            "CLARIFICATION_REQUIRED",
        )
        self.assertIn("文档还是业务数据", result.artifact.payload["answer"])

    async def test_echarts_rejects_executable_formatter(self):
        skill = EChartsSkill(
            model_client=_ModelClient(
                response={
                    "chart_type": "bar",
                    "title": "销售",
                    "option": {
                        "tooltip": {
                            "formatter": "function(x){return x.value}"
                        }
                    },
                }
            ),
            prompt_resolver=_PromptResolver(),
        )
        query = {
            "schema": "QUERY_RESULT.v1",
            "query_result_id": str(uuid7()),
            "provider": "MCP",
            "columns": [{"name": "sales"}],
            "rows": [{"sales": 10}],
            "row_count": 1,
            "truncated": False,
            "warnings": [],
            "provenance": {"profile": "SALES_PROFILE"},
        }
        with self.assertRaisesRegex(ValueError, "可执行脚本"):
            await skill.execute(
                _context(
                    agent={
                        "models": {
                            "composer_llm": {
                                "served_model_name": "chart-model"
                            }
                        }
                    },
                    artifacts=(_artifact("QUERY_RESULT", query),),
                )
            )

    def test_dify_records_use_document_evidence(self):
        bundle_id = uuid7()
        records = _records(
            candidates=[
                {
                    "collection_id": str(uuid7()),
                    "bundle_id": str(bundle_id),
                    "bundle_revision_id": str(uuid7()),
                    "display_title": "案例文档",
                    "rrf_score": 0.02,
                }
            ],
            citations=[
                {
                    "bundle_id": str(bundle_id),
                    "items": [
                        {
                            "final_role": "PRIMARY",
                            "evidence": {
                                "evidence_id": str(uuid7()),
                                "document_id": str(uuid7()),
                                "document_version_id": str(uuid7()),
                                "document_name": "case.pdf",
                                "content_text": "这是可引用的案例正文。",
                                "locator": {"page": 3},
                            },
                        }
                    ],
                }
            ],
            limit=10,
            threshold=0.5,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["title"], "case.pdf")
        self.assertEqual(records[0]["score"], 1.0)
        self.assertIn("案例正文", records[0]["content"])


if __name__ == "__main__":
    unittest.main()
