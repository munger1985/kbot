"""4.0 通用对话、问数、图表与 Dify Adapter 的聚焦测试。"""

import unittest
from types import SimpleNamespace

from agent_runtime.application.commands import LeasedArtifact
from agent_runtime.application.agent_definitions import (
    AgentDefinitionService,
)
from agent_runtime.application.runtime_service import AgentRuntimeConflict
from agent_runtime.runtime import ExecutionContext, SkillProgress, SkillResult
from agent_runtime.specialists.conversation_response import (
    ConversationResponseSkill,
)
from agent_runtime.specialists.mcp_data import (
    EChartsSkill,
    MCPDataQuerySkill,
)
from agent_runtime.specialists.root import RootAgentPlanner, RouteType
from main_api.api.dify import _records
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

    async def get_llm_json(self, **kwargs):
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
):
    return ExecutionContext(
        app_id=1,
        domain_id=20,
        agent_id=uuid7(),
        run_id=uuid7(),
        task_id=uuid7(),
        task_key="test",
        actor_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
        original_input=original_input,
        config_snapshot={"agent": agent or {}, "route": route},
        input_artifacts=tuple(artifacts),
    )


class AgentChatCapabilitiesTest(unittest.IsolatedAsyncioTestCase):
    def test_aiops_cannot_mix_with_chat_capabilities(self):
        with self.assertRaises(AgentRuntimeConflict):
            AgentDefinitionService._validate_capabilities(
                ("aiops", "document")
            )

    def test_active_data_agent_requires_profile(self):
        with self.assertRaises(AgentRuntimeConflict):
            AgentDefinitionService._validate_runtime_configuration(
                capabilities=("mcp_data",),
                status="ACTIVE",
                router_llm_model_name=None,
                data_profile_name=None,
            )

    async def test_multi_capability_router_selects_data_chart(self):
        planner = RootAgentPlanner(
            model_client=_ModelClient(
                response={
                    "route_type": "MCP_DATA",
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
                    "mcp_data",
                ],
                "router_llm_model_name": "router-model",
            },
            objective="画出本月销售趋势图",
        )
        plan = planner.build_plan(
            objective="画出本月销售趋势图", decision=decision
        )

        self.assertEqual(decision.route_type, RouteType.MCP_DATA)
        self.assertTrue(decision.requires_chart)
        self.assertEqual(
            [task.task_key for task in plan.tasks],
            [
                "context_rewrite",
                "mcp_data_query",
                "echarts",
                "response_compose",
            ],
        )

    async def test_query_image_routes_to_document(self):
        planner = RootAgentPlanner()
        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": ["conversation", "document"],
                "router_llm_model_name": "router-model",
            },
            objective="找一下相似的案例",
            client_metadata={"query_images": [{"storage_uri": "/tmp/x"}]},
        )

        self.assertEqual(decision.route_type, RouteType.DOCUMENT)

    async def test_mcp_data_skill_preserves_profile_and_rows(self):
        client = _DataClient()
        result = await MCPDataQuerySkill(data_client=client).execute(
            _context(
                original_input="查询销售额",
                agent={"data_profile_name": "SALES_PROFILE"},
                artifacts=(
                    _artifact(
                        "CONTEXT_REWRITE",
                        {"standalone_query": "查询 2026 年 7 月销售额"},
                    ),
                ),
            )
        )

        self.assertEqual(result.artifact.artifact_type, "QUERY_RESULT")
        self.assertEqual(
            result.artifact.payload["rows"][0]["sales"], 10
        )
        self.assertEqual(client.request["profile"], "SALES_PROFILE")
        self.assertEqual(
            client.request["question"], "查询 2026 年 7 月销售额"
        )

    async def test_conversation_response_streams_and_returns_answer(self):
        skill = ConversationResponseSkill(
            model_client=_ModelClient(chunks=("你", "好")),
            prompt_resolver=_PromptResolver(),
        )
        items = [
            item
            async for item in skill.execute_stream(
                _context(agent={"composer_llm_model_name": "chat-model"})
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
            "query_result_id": str(uuid7()),
            "profile": "SALES_PROFILE",
            "question": "查询销售",
            "rows": [{"sales": 10}],
            "row_count": 1,
            "upstream_row_count": 1,
            "truncated": False,
            "status": "READY",
        }
        with self.assertRaisesRegex(ValueError, "可执行脚本"):
            await skill.execute(
                _context(
                    agent={"composer_llm_model_name": "chart-model"},
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
