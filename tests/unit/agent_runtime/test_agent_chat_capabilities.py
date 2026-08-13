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
    SemanticDataQueryExecutor,
)
from agent_runtime.specialists.visualization import EChartsSkill
from agent_runtime.specialists.root import RootAgentPlanner, RouteType
from main_api.api.dify import _records
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


class _PromptResolver:
    async def resolve(self, key):
        return SimpleNamespace(
            content=f"prompt:{key}",
            ref=lambda: {"prompt_key": key, "version": "1.0.0"},
        )


class _ModelClient:
    def __init__(self, *, response=None, chunks=()):
        self.response = response or {}
        self.chunks = chunks
        self.last_json_request = None

    async def get_llm_json(self, **kwargs):
        self.last_json_request = kwargs
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
):
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
        config_snapshot={"agent": agent or {}, "route": route},
        input_artifacts=tuple(artifacts),
    )


class AgentChatCapabilitiesTest(unittest.IsolatedAsyncioTestCase):
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

    async def test_router_only_exposes_selected_knowledge_capabilities(self):
        model = _ModelClient(
            response={
                "route_type": "HYBRID_PARALLEL",
                "confidence": 0.9,
                "reason": "需要同时查询制度和业务数据",
                "clarification_question": None,
                "requires_chart": False,
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
            model.last_json_request["prompt"][1]["content"]
        )
        self.assertNotIn("CONVERSATION", request["enabled_routes"])
        self.assertIn("DOCUMENT", request["enabled_routes"])
        self.assertIn("DATA_QUERY", request["enabled_routes"])
        self.assertEqual(decision.route_type, RouteType.HYBRID_PARALLEL)

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
