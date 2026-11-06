"""Agent Runtime 内置 Skill 的契约测试。"""

from datetime import datetime, timezone
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from pydantic import ValidationError
from agent_runtime.application import LeasedArtifact
from agent_runtime.runtime import ExecutionContext
from agent_runtime.specialists.conversation import ContextRewriteSkill
from agent_runtime.specialists.conversation.contracts import ContextRewriteOutput
from agent_runtime.specialists.conversation.skill import FOLLOWUP_FOCUS_RULE
from agent_runtime.specialists.response_composer import ResponseComposerSkill
from agent_runtime.specialists.response_composer.contracts import (
    GroundedAnswer,
    QueryResultReferenceCard,
    ReferenceCard,
)
from agent_runtime.specialists.root import RouteType, RootAgentPlanner
from platform_core.identity import uuid7
from platform_core.prompts import ResolvedPrompt


_COLLECTION_ID = uuid7()
_REVISION_ID = uuid7()
_AGENT_ID = uuid7()
_AGENT_VERSION_ID = uuid7()


class _ModelClient:
    async def get_llm_json(self, **kwargs):
        return {
            "answer": "该案例通过调整索引降低了查询延迟。[C1]",
            "used_citation_labels": ["C1"],
        }

    async def stream_llm_chunks(self, **kwargs):
        from platform_clients.model import LLMChunk

        yield LLMChunk(content="该案例通过调整索引")
        yield LLMChunk(content="降低了查询延迟。[C1]")

    async def get_vlm_answer(self, *args, **kwargs):
        return "图片中是数据库性能监控面板，显示查询延迟升高"


class _RecordingModelClient(_ModelClient):
    def __init__(self, *, response=None):
        self.last_json_request = None
        self.response = response

    async def get_llm_json(self, **kwargs):
        self.last_json_request = kwargs
        if self.response is not None:
            return self.response
        return await super().get_llm_json(**kwargs)


class _SinglePassLanguageModelClient(_ModelClient):
    def __init__(self):
        self.calls = 0
        self.prompts = []

    async def stream_llm_chunks(self, **kwargs):
        from platform_clients.model import LLMChunk

        self.calls += 1
        self.prompts.append(kwargs["prompt"])
        if self.calls == 1:
            yield LLMChunk(
                content=(
                    "根据证据，有一个相关资产："
                    "**Conversational Banking with Select AI Agents** [C1]"
                )
            )
            return
        yield LLMChunk(
            content=(
                "One related asset was found: "
                "**Conversational Banking with Select AI Agents** [C1]"
            )
        )


class _VariantCitationModelClient(_ModelClient):
    async def stream_llm_chunks(self, **kwargs):
        from platform_clients.model import LLMChunk

        yield LLMChunk(content="第一条事实使用全角引用【C1】。")
        yield LLMChunk(content="第二条事实使用上标引用<sup>C1</sup>。")


class _MissingCitationModelClient(_ModelClient):
    def __init__(self, *, repair_succeeds: bool):
        self.calls = 0
        self.repair_succeeds = repair_succeeds

    async def stream_llm_chunks(self, **kwargs):
        from platform_clients.model import LLMChunk

        self.calls += 1
        if self.calls == 2 and self.repair_succeeds:
            yield LLMChunk(content="证据支持的回答。[C1]")
            return
        yield LLMChunk(content="1+1=2。")


class _RepairingJsonCitationModelClient(_ModelClient):
    def __init__(self):
        self.calls = 0
        self.prompts = []

    async def get_llm_json(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["prompt"])
        if self.calls == 1:
            return {
                "answer": "证据支持的回答。",
                "used_citation_labels": [],
            }
        return {
            "answer": "证据支持的回答。[C1]",
            "used_citation_labels": ["C1"],
        }


class _QueryCitationModelClient(_ModelClient):
    def __init__(self):
        self.prompts = []

    async def get_llm_json(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return {
            "answer": "- Asset A [C1]\n\n- Asset B [C2]",
        }

    async def stream_llm_chunks(self, **kwargs):
        from platform_clients.model import LLMChunk

        self.prompts.append(kwargs["prompt"])
        yield LLMChunk(content="- Asset A [C1]\n\n")
        yield LLMChunk(content="- Asset B 【C2】")


class _HybridEvidenceModelClient(_ModelClient):
    def __init__(self):
        self.prompts = []

    async def get_llm_json(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return {
            "answer": (
                "金额最高的是 Asset A。[Q1] "
                "该案例通过调整索引降低了查询延迟。[C1]"
            ),
            "used_citation_labels": ["Q1", "C1"],
        }

    async def stream_llm_chunks(self, **kwargs):
        from platform_clients.model import LLMChunk

        self.prompts.append(kwargs["prompt"])
        yield LLMChunk(content="金额最高的是 Asset A。[Q1] ")
        yield LLMChunk(content="该案例通过调整索引降低了查询延迟。[C1]")


class _RewriteModelClient:
    def __init__(self):
        self.last_json_request = None

    async def get_llm_json(self, **kwargs):
        self.last_json_request = kwargs
        return {
            "raw_input": "它有什么优势？",
            "standalone_query": "数据库优化案例有什么优势？",
            "retrieval_queries": ["数据库优化案例有什么优势？"],
            "resolved_references": ["它=数据库优化案例"],
            "active_topic": "数据库优化案例",
            "ambiguity": False,
            "clarification_question": None,
            "memory_refs": [],
        }


class _EmptyObjectRewriteModelClient(_RewriteModelClient):
    async def get_llm_json(self, **kwargs):
        output = await super().get_llm_json(**kwargs)
        output["resolved_references"] = {}
        output["memory_refs"] = {}
        return output


class _ObjectReferenceRewriteModelClient(_RewriteModelClient):
    async def get_llm_json(self, **kwargs):
        output = await super().get_llm_json(**kwargs)
        output["resolved_references"] = [
            {
                "reference": "OAC",
                "resolved_to": "Oracle Analytics Cloud",
                "source": "会话摘要中的明确实体",
            }
        ]
        return output


class _MalformedRewriteModelClient(_RewriteModelClient):
    def __init__(self):
        self.call_count = 0

    async def get_llm_json(self, **kwargs):
        self.call_count += 1
        output = await super().get_llm_json(**kwargs)
        output["retrieval_queries"] = []
        output["ambiguity"] = "否"
        output["clarification_question"] = "不应展示的澄清问题"
        return output


class _BroadFollowupRewriteModelClient(_RewriteModelClient):
    async def get_llm_json(self, **kwargs):
        self.last_json_request = kwargs
        return {
            "raw_input": "根据会议纪要说明其风险原因。",
            "standalone_query": "说明当前所有风险客户的风险原因。",
            "retrieval_queries": [
                "说明甲客户、乙客户和北辰科技的风险原因。"
            ],
            "resolved_references": ["其=当前风险客户"],
            "active_topic": "当前风险客户",
            "ambiguity": False,
            "clarification_question": None,
            "memory_refs": [],
        }


class _PromptResolver:
    async def resolve(self, prompt_key):
        variables = {
            "agent_runtime.context_rewrite": (
                "raw_input",
                "conversation_summary",
                "recent_items",
                "recalled_memories",
            ),
            "agent_runtime.response_compose": (
                "agent_instruction",
                "raw_input",
                "standalone_query",
                "evidence",
            ),
            "agent_runtime.data_response_compose": (),
            "agent_runtime.query_image_description": (),
        }[prompt_key]
        content = "\n".join(
            f"{name}=${{{name}}}" for name in variables
        )
        return ResolvedPrompt(
            prompt_key=prompt_key,
            version="1.0.0",
            sha256="a" * 64,
            content=content,
            input_variables=variables,
            output_schema=None,
            source="TEST",
        )


def _context(
    *,
    input_artifacts=(),
    original_input="这个案例如何降低查询延迟？",
    language=None,
):
    run_id = uuid7()
    config_snapshot = {
        "agent": {
            "models": {
                "context_llm": {
                    "served_model_name": "context-model"
                },
                "composer_llm": {
                    "served_model_name": "composer-model"
                },
                "memory_llm": {
                    "served_model_name": "memory-model"
                },
                "memory_embedding": {
                    "served_model_name": "embedding-model"
                },
            },
            "instruction": "准确回答。",
            "config": {},
        },
    }
    if language is not None:
        config_snapshot["language"] = language
    return ExecutionContext(
        domain_id=10,
        agent_id=_AGENT_ID,
        run_id=run_id,
        task_id=uuid7(),
        task_key="knowledge_retrieval",
        actor_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
        original_input=original_input,
        config_snapshot=config_snapshot,
        deadline_at=datetime.now(timezone.utc),
        input_artifacts=input_artifacts,
    )


class AgentRuntimeSkillTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _query_result_artifact() -> LeasedArtifact:
        return LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type="QUERY_RESULT",
            schema_version="QUERY_RESULT.v1",
            producer="data-query",
            producer_version="1.0.0",
            payload={
                "schema": "QUERY_RESULT.v1",
                "query_result_id": str(uuid7()),
                "provider": "SEMANTIC",
                "columns": [
                    {"name": "title", "type": "STRING"},
                ],
                "rows": [
                    {"title": "Asset A"},
                    {"title": "Asset B"},
                ],
                "row_count": 2,
                "truncated": False,
                "warnings": [],
                "provenance": {},
            },
            content_hash="query-result-hash",
            provenance={},
            security_level=2,
        )

    @staticmethod
    def _ambiguous_rewrite_artifact() -> LeasedArtifact:
        return LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type="CONTEXT_REWRITE",
            schema_version="ContextRewriteOutput.v1",
            producer="context-rewrite",
            producer_version="1.0.0",
            payload={
                "raw_input": "移动套餐",
                "standalone_query": "员工可选的移动通信套餐有哪些？",
                "retrieval_queries": ["员工可选的移动通信套餐有哪些？"],
                "resolved_references": [],
                "active_topic": "员工套餐",
                "ambiguity": True,
                "clarification_question": "请说明移动套餐的具体类型。",
                "memory_refs": [],
            },
            content_hash="rewrite-hash",
            provenance={},
            security_level=2,
        )

    def test_root_planner_builds_fixed_document_dag(self):
        planner = RootAgentPlanner()
        decision = planner.decide(
            agent_snapshot={
                "enabled_capabilities": ["document"],
                "config": {},
            }
        )
        plan = planner.build_plan(
            objective="总结文档", decision=decision
        )

        self.assertEqual(decision.route_type, RouteType.DOCUMENT)
        self.assertEqual(plan.final_task_key, "response_compose")
        self.assertEqual(
            plan.tasks[1].depends_on, ("context_rewrite",)
        )
        self.assertEqual(
            plan.tasks[2].depends_on,
            ("context_rewrite", "knowledge_retrieval"),
        )

    def test_root_planner_rejects_aiops_agent_capability(self):
        planner = RootAgentPlanner()
        with self.assertRaisesRegex(ValueError, "不支持能力"):
            planner.decide(
                agent_snapshot={"enabled_capabilities": ["aiops"]}
            )

    async def test_context_rewrite_uses_frozen_conversation_context(self):
        context = _context()
        context = context.model_copy(
            update={
                "original_input": "它有什么优势？",
                "config_snapshot": {
                    **context.config_snapshot,
                    "conversation": {
                        "context": {
                            "summary": {
                                "active_topic": "数据库优化案例"
                            },
                            "recent_items": [],
                            "memories": [],
                        }
                    },
                },
            }
        )
        model = _RewriteModelClient()
        result = await ContextRewriteSkill(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        ).execute(context)

        self.assertEqual(
            result.artifact.payload["standalone_query"],
            "数据库优化案例有什么优势？",
        )
        self.assertIn(
            "response_language=zh-CN",
            model.last_json_request["prompt"][1]["content"],
        )
        self.assertEqual(
            model.last_json_request["prompt"][2]["content"],
            FOLLOWUP_FOCUS_RULE,
        )

    async def test_followup_uses_latest_unique_query_row_as_focus(self):
        context = _context().model_copy(
            update={
                "original_input": "根据会议纪要说明其风险原因。",
                "config_snapshot": {
                    **_context().config_snapshot,
                    "route": {
                        "route_type": "DOCUMENT",
                        "classifier_version": "llm-single-route-v2",
                        "context_required": True,
                    },
                    "conversation": {
                        "context": {
                            "summary": {
                                "active_topic": "客户开放商机"
                            },
                            "recent_items": [
                                {
                                    "role": "ASSISTANT",
                                    "content": {
                                        "text": (
                                            "北辰科技，开放商机金额为 480 万元。"
                                        ),
                                        "query_results": [{
                                            "columns": [
                                                {
                                                    "name": "customer_name",
                                                    "label": "客户名称",
                                                },
                                                {
                                                    "name": "open_amount",
                                                    "label": "开放商机金额",
                                                },
                                            ],
                                            "rows": [{
                                                "customer_name": "北辰科技",
                                                "open_amount": 4800000,
                                            }],
                                        }],
                                    },
                                }
                            ],
                            "memories": [],
                        }
                    },
                },
            }
        )

        result = await ContextRewriteSkill(
            model_client=_BroadFollowupRewriteModelClient(),
            prompt_resolver=_PromptResolver(),
        ).execute(context)

        payload = result.artifact.payload
        self.assertEqual(
            payload["standalone_query"],
            "上一轮唯一结构化结果：客户名称=北辰科技，开放商机金额=4800000。"
            "当前追问：根据会议纪要说明其风险原因。",
        )
        self.assertEqual(
            payload["retrieval_queries"],
            [payload["standalone_query"]],
        )
        self.assertNotIn("甲客户", payload["standalone_query"])
        self.assertIn(
            "上一轮唯一结构化结果=客户名称=北辰科技，开放商机金额=4800000",
            payload["resolved_references"],
        )

    def test_unique_query_row_does_not_override_complete_new_question(self):
        output = ContextRewriteOutput(
            raw_input="说明云峰连锁零售的风险原因。",
            standalone_query="说明云峰连锁零售的风险原因。",
            retrieval_queries=("说明云峰连锁零售的风险原因。",),
        )
        result = ContextRewriteSkill._apply_unique_query_focus(
            output,
            original_input=output.raw_input,
            route={"context_required": False},
            recent_items=[{
                "role": "ASSISTANT",
                "content": {
                    "query_results": [{
                        "columns": [{"name": "customer_name"}],
                        "rows": [{"customer_name": "北辰科技"}],
                    }]
                },
            }],
        )

        self.assertEqual(output, result)

    async def test_context_rewrite_normalizes_empty_object_sequences(self):
        context = _context().model_copy(
            update={
                "original_input": "它有什么优势？",
                "config_snapshot": {
                    **_context().config_snapshot,
                    "conversation": {
                        "context": {
                            "summary": {
                                "active_topic": "数据库优化案例"
                            },
                            "recent_items": [],
                            "memories": [],
                        }
                    },
                },
            }
        )

        result = await ContextRewriteSkill(
            model_client=_EmptyObjectRewriteModelClient(),
            prompt_resolver=_PromptResolver(),
        ).execute(context)

        self.assertEqual(
            result.artifact.payload["resolved_references"], []
        )
        self.assertEqual(result.artifact.payload["memory_refs"], [])

    async def test_context_rewrite_normalizes_object_reference_item(self):
        context = _context().model_copy(
            update={
                "original_input": "它有什么优势？",
                "config_snapshot": {
                    **_context().config_snapshot,
                    "conversation": {
                        "context": {
                            "summary": {"active_topic": "OAC"},
                            "recent_items": [],
                            "memories": [],
                        }
                    },
                },
            }
        )

        result = await ContextRewriteSkill(
            model_client=_ObjectReferenceRewriteModelClient(),
            prompt_resolver=_PromptResolver(),
        ).execute(context)

        self.assertEqual(
            ["OAC=Oracle Analytics Cloud"],
            result.artifact.payload["resolved_references"],
        )

    async def test_context_rewrite_rejects_invalid_output_after_one_correction_attempt(self):
        context = _context().model_copy(
            update={
                "config_snapshot": {
                    **_context().config_snapshot,
                    "conversation": {
                        "context": {
                            "summary": {"active_topic": "数据库优化案例"},
                            "recent_items": [],
                            "memories": [],
                        }
                    },
                },
            }
        )

        model_client = _MalformedRewriteModelClient()
        with self.assertRaises(ValidationError):
            await ContextRewriteSkill(
                model_client=model_client,
                prompt_resolver=_PromptResolver(),
            ).execute(context)

        self.assertEqual(2, model_client.call_count)

    async def test_clarification_relies_on_frozen_language_prompt(self):
        rewrite = self._ambiguous_rewrite_artifact()

        result = await ResponseComposerSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        ).execute(
            _context(
                input_artifacts=(rewrite,),
                original_input="Which asset do you mean?",
                language="en-US",
            )
        )

        self.assertEqual(
            result.artifact.payload["answer"],
            "请说明移动套餐的具体类型。",
        )

    async def test_query_composer_replaces_model_citations_with_query_label(self):
        model = _QueryCitationModelClient()
        context = _context(
            input_artifacts=(self._query_result_artifact(),),
            original_input="list all assets",
            language="en-US",
        )
        agent = deepcopy(context.config_snapshot["agent"])
        agent["instruction"] = "文档回答必须为每个 Asset 使用 [C1]、[C2]。"
        context = context.model_copy(
            update={
                "config_snapshot": {
                    **context.config_snapshot,
                    "agent": agent,
                }
            }
        )

        outputs = [
            item
            async for item in ResponseComposerSkill(
                model_client=model,
                prompt_resolver=_PromptResolver(),
            ).execute_stream(context)
        ]

        deltas = [
            item.payload["delta"]
            for item in outputs
            if getattr(item, "event_type", None) == "answer.delta"
        ]
        payload = outputs[-1].artifact.payload
        self.assertGreater(len(deltas), 1)
        self.assertEqual("".join(deltas), "- Asset A\n\n- Asset B [Q1]")
        self.assertEqual(payload["answer"], "".join(deltas))
        self.assertEqual(payload["used_citation_labels"], ["Q1"])
        self.assertEqual(
            [item["citation_label"] for item in payload["references"]],
            ["Q1"],
        )
        self.assertNotIn("文档回答必须", str(model.prompts))

    async def test_composer_projects_safe_aiops_result(self):
        delegation_id = uuid7()
        ops_run_id = uuid7()
        diagnosis_artifact_id = uuid7()
        artifact = LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type="DELEGATED_AIOPS_RESULT",
            schema_version="DELEGATED_AIOPS_RESULT.v1",
            producer="aiops-agent",
            producer_version="1",
            payload={
                "delegation_id": str(delegation_id),
                "ops_run_id": str(ops_run_id),
                "status": "COMPLETED",
                "safe_summary": "发现慢 SQL 与缺失索引相关。",
                "diagnosis": {
                    "root_cause_grade": "CONFIRMED",
                    "artifact": {
                        "artifact_id": str(diagnosis_artifact_id),
                        "content_hash": "a" * 64,
                    },
                },
            },
            content_hash="result-hash",
            provenance={},
            security_level=2,
        )

        result = await ResponseComposerSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        ).execute(_context(input_artifacts=(artifact,)))

        payload = result.artifact.payload
        self.assertEqual(payload["used_citation_labels"], ["O1"])
        self.assertIn("[O1]", payload["answer"])
        self.assertEqual(
            payload["references"][0]["ops_run_id"], str(ops_run_id)
        )
        self.assertEqual(
            payload["references"][0]["resource_url"],
            f"/api/v1/apps/aiops/runs/{ops_run_id}",
        )


if __name__ == "__main__":
    unittest.main()
