"""Agent Runtime 内置 Skill 的契约测试。"""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from pydantic import ValidationError
from agent_runtime.application import LeasedArtifact
from agent_runtime.runtime import ExecutionContext
from agent_runtime.specialists.conversation import ContextRewriteSkill
from agent_runtime.specialists.document import KnowledgeRetrievalSkill
from agent_runtime.specialists.response_composer import ResponseComposerSkill
from agent_runtime.specialists.response_composer.contracts import (
    GroundedAnswer,
    QueryResultReferenceCard,
)
from agent_runtime.specialists.root import RouteType, RootAgentPlanner
from platform_core.identity import uuid7
from platform_core.prompts import ResolvedPrompt


class _KnowledgeCoreClient:
    def __init__(self):
        self.collection_id = uuid7()
        self.bundle_id = uuid7()
        self.revision_id = uuid7()
        self.document_id = uuid7()
        self.document_version_id = uuid7()
        self.parse_view_id = uuid7()
        self.evidence_id = uuid7()
        self.last_discovery_query = ""
        self.last_discovery_request = {}
        self.last_evidence_request = {}
        self.visual_configured = False

    async def list_agent_bindings(self, **kwargs):
        return {
            "bindings": [
                {
                    "collection_id": str(self.collection_id),
                    "status": "ACTIVE",
                }
            ]
        }

    async def discover(self, **kwargs):
        self.last_discovery_request = kwargs
        self.last_discovery_query = kwargs["query"]
        return {
            "candidates": [
                {
                    "collection_id": str(self.collection_id),
                    "bundle_id": str(self.bundle_id),
                    "bundle_revision_id": str(self.revision_id),
                    "display_title": "数据库优化案例",
                    "member_count": 1,
                    "matched_members": [],
                    "match_signals": ["TEXT", "VECTOR"],
                    "local_rank": 1,
                    "rrf_score": 0.03,
                    "candidate_scope": "SINGLE_MEMBER",
                }
            ]
        }

    async def search_visual(self, **kwargs):
        return {
            "results": [],
            "searched_collection_ids": (
                [str(self.collection_id)]
                if self.visual_configured
                else []
            ),
            "skipped_collection_ids": (
                []
                if self.visual_configured
                else [str(self.collection_id)]
            ),
        }

    async def retrieve_evidence(self, **kwargs):
        self.last_evidence_request = kwargs
        evidence = {
            "evidence_id": str(self.evidence_id),
            "document_id": str(self.document_id),
            "content_text": "该案例通过调整索引降低了查询延迟。",
            "locator": {
                "pages": [{"page_no": 3, "bbox": [0.1, 0.2, 0.9, 0.4]}]
            },
            "locator_schema_version": "document/v1",
            "heading_path": ["优化方案"],
            "provenance": {"source_hash": "source-hash"},
        }
        return {
            "citations": [
                {
                    "citation_label": "C1",
                    "collection_id": str(self.collection_id),
                    "bundle_id": str(self.bundle_id),
                    "bundle_revision_id": str(self.revision_id),
                    "document_version_id": str(self.document_version_id),
                    "parse_view_id": str(self.parse_view_id),
                    "primary_evidence_ids": [str(self.evidence_id)],
                    "structural_context_ids": [],
                    "neighbor_evidence_ids": [],
                    "items": [
                        {
                            "item_label": "G1-A1",
                            "evidence": evidence,
                            "input_role": "ANCHOR",
                            "final_role": "PRIMARY",
                            "promoted_from_context": False,
                        }
                    ],
                }
            ]
        }


class _ModelClient:
    async def get_llm_json(self, **kwargs):
        return {
            "answer": "该案例通过调整索引降低了查询延迟。[C1]",
            "used_citation_labels": ["C1"],
            "presented_assets": [],
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


class _LanguageRepairModelClient(_ModelClient):
    def __init__(self):
        self.calls = 0
        self.prompts = []

    async def get_llm_json(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["prompt"])
        if self.calls == 1:
            return {
                "answer": (
                    "根据证据，有一个相关资产："
                    "**Conversational Banking with Select AI Agents** [C1]"
                ),
                "used_citation_labels": ["C1"],
                "presented_assets": [
                    {
                        "primary_citation_label": "C1",
                        "supporting_citation_labels": [],
                    }
                ],
            }
        return {
            "answer": (
                "One related asset was found: "
                "**Conversational Banking with Select AI Agents** [C1]"
            ),
            "used_citation_labels": ["C1"],
            "presented_assets": [
                {
                    "primary_citation_label": "C1",
                    "supporting_citation_labels": [],
                }
            ],
        }


class _VariantCitationModelClient(_ModelClient):
    async def get_llm_json(self, **kwargs):
        return {
            "answer": (
                "第一条事实使用全角引用【C1】。"
                "第二条事实使用上标引用<sup>C1</sup>。"
            ),
            "used_citation_labels": ["C1"],
            "presented_assets": [],
        }


class _MissingCitationModelClient(_ModelClient):
    def __init__(self, *, repair_succeeds: bool):
        self.calls = 0
        self.repair_succeeds = repair_succeeds

    async def get_llm_json(self, **kwargs):
        self.calls += 1
        if self.calls == 2 and self.repair_succeeds:
            return {
                "answer": "证据支持的回答。[C1]",
                "used_citation_labels": ["C1"],
                "presented_assets": [],
            }
        return {
            "answer": "1+1=2。",
            "used_citation_labels": [],
            "presented_assets": [],
        }


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
        "retrieval": {
            "collection_ids": [],
            "security_level": 2,
        },
    }
    if language is not None:
        config_snapshot["language"] = language
    return ExecutionContext(
        domain_id=10,
        agent_id=uuid7(),
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

    @staticmethod
    async def _retrieval_artifact() -> LeasedArtifact:
        retrieval = await KnowledgeRetrievalSkill(
            knowledge_core_client=_KnowledgeCoreClient(),
            service_name="agent-worker",
        ).execute(_context())
        return LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type=retrieval.artifact.artifact_type,
            schema_version=retrieval.artifact.schema_version,
            producer="knowledge-retrieval",
            producer_version="1.0.0",
            payload=retrieval.artifact.payload,
            content_hash="hash",
            provenance={},
            security_level=2,
        )

    @staticmethod
    def _image_context(
        path: Path, *, query_vlm_model_name: str | None
    ) -> ExecutionContext:
        context = _context()
        agent = dict(context.config_snapshot["agent"])
        agent["models"] = {
            **agent["models"],
            **(
                {
                    "query_vlm": {
                        "served_model_name": query_vlm_model_name
                    }
                }
                if query_vlm_model_name
                else {}
            ),
        }
        return context.model_copy(
            update={
                "config_snapshot": {
                    **context.config_snapshot,
                    "agent": agent,
                    "client_metadata": {
                        "query_images": [
                            {
                                "storage_uri": str(path),
                                "mime_type": "image/png",
                            }
                        ]
                    },
                }
            }
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
            "language=zh-CN",
            model.last_json_request["prompt"][1]["content"],
        )

    async def test_km_self_contained_route_does_not_inherit_old_topic(self):
        model = _MalformedRewriteModelClient()
        context = _context().model_copy(
            update={
                "original_input": "现在有几个asset",
                "config_snapshot": {
                    **_context().config_snapshot,
                    "route": {
                        "route_type": "DATA_QUERY",
                        "classifier_version": "llm-km-asset-v1:1.0.0",
                        "context_required": False,
                    },
                    "conversation": {
                        "context": {
                            "summary": {"active_topic": "ChatBI"},
                            "recent_items": [
                                {
                                    "role": "ASSISTANT",
                                    "content": {
                                        "text": "您是指 ChatBI 相关 Asset 吗？"
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
            model_client=model,
            prompt_resolver=_PromptResolver(),
        ).execute(context)

        self.assertEqual(
            "现在有几个asset",
            result.artifact.payload["standalone_query"],
        )
        self.assertEqual(0, model.call_count)

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

    async def test_document_skill_builds_document_level_citation(self):
        client = _KnowledgeCoreClient()
        result = await KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent-worker",
        ).execute(_context())

        payload = result.artifact.payload
        citation = payload["citation_pack"]["citations"][0]
        self.assertEqual(result.artifact.artifact_type, "CITATION_PACK")
        self.assertEqual(citation["title"], "数据库优化案例")
        self.assertEqual(citation["document_id"], str(client.document_id))
        self.assertEqual(citation["locator"]["pages"][0]["page_no"], 3)

    async def test_document_skill_retrieves_before_asking_for_clarification(self):
        client = _KnowledgeCoreClient()
        rewrite = self._ambiguous_rewrite_artifact()

        result = await KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent-worker",
        ).execute(_context(input_artifacts=(rewrite,)))

        self.assertEqual(
            client.last_discovery_query,
            "员工可选的移动通信套餐有哪些？",
        )
        self.assertEqual(
            len(result.artifact.payload["citation_pack"]["citations"]),
            1,
        )

    async def test_document_skill_propagates_agent_rerank_switch(self):
        client = _KnowledgeCoreClient()
        context = _context()
        snapshot = dict(context.config_snapshot)
        snapshot["agent"] = {
            **snapshot["agent"],
            "do_rerank": True,
        }
        snapshot["route"] = {
            "route_type": "DOCUMENT",
            "coverage_mode": "BREADTH",
        }

        result = await KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent-worker",
        ).execute(
            context.model_copy(update={"config_snapshot": snapshot})
        )

        self.assertTrue(client.last_discovery_request["do_rerank"])
        self.assertTrue(client.last_evidence_request["do_rerank"])
        self.assertEqual(
            "BREADTH", client.last_discovery_request["coverage_mode"]
        )
        self.assertEqual(
            "BREADTH", client.last_evidence_request["coverage_mode"]
        )
        self.assertEqual(
            result.artifact.payload["retrieval_report"]["selector"],
            "llm-object-and-evidence-group-v1",
        )

    async def test_document_skill_runs_visual_and_vlm_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.png"
            Image.new("RGB", (8, 8), "red").save(path)
            client = _KnowledgeCoreClient()
            client.visual_configured = True
            result = await KnowledgeRetrievalSkill(
                knowledge_core_client=client,
                model_client=_ModelClient(),
                prompt_resolver=_PromptResolver(),
                service_name="agent-worker",
            ).execute(
                self._image_context(
                    path, query_vlm_model_name="query-vlm"
                )
            )
        processing = result.artifact.payload["citation_pack"][
            "query_plan"
        ]["image_processing"]
        self.assertEqual(processing["visual_search"], "EXECUTED")
        self.assertEqual(processing["vlm_text_search"], "EXECUTED")
        self.assertIn("数据库性能监控面板", client.last_discovery_query)
        self.assertEqual(result.artifact.payload["warnings"], [])

    async def test_document_skill_skips_unconfigured_image_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.png"
            Image.new("RGB", (8, 8), "blue").save(path)
            client = _KnowledgeCoreClient()
            result = await KnowledgeRetrievalSkill(
                knowledge_core_client=client,
                service_name="agent-worker",
            ).execute(
                self._image_context(path, query_vlm_model_name=None)
            )
        processing = result.artifact.payload["citation_pack"][
            "query_plan"
        ]["image_processing"]
        self.assertEqual(
            processing["visual_search"], "SKIPPED_NOT_CONFIGURED"
        )
        self.assertEqual(
            processing["vlm_text_search"], "SKIPPED_NOT_CONFIGURED"
        )
        self.assertTrue(
            any(
                "已忽略上传图片" in warning
                for warning in result.artifact.payload["warnings"]
            )
        )

    async def test_document_skill_degrades_each_image_path_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.png"
            Image.new("RGB", (8, 8), "green").save(path)

            visual_client = _KnowledgeCoreClient()
            visual_client.visual_configured = True
            visual_only = await KnowledgeRetrievalSkill(
                knowledge_core_client=visual_client,
                service_name="agent-worker",
            ).execute(
                self._image_context(path, query_vlm_model_name=None)
            )

            vlm_client = _KnowledgeCoreClient()
            vlm_only = await KnowledgeRetrievalSkill(
                knowledge_core_client=vlm_client,
                model_client=_ModelClient(),
                prompt_resolver=_PromptResolver(),
                service_name="agent-worker",
            ).execute(
                self._image_context(
                    path, query_vlm_model_name="query-vlm"
                )
            )

        visual_status = visual_only.artifact.payload["citation_pack"][
            "query_plan"
        ]["image_processing"]
        self.assertEqual(visual_status["visual_search"], "EXECUTED")
        self.assertEqual(
            visual_status["vlm_text_search"], "SKIPPED_NOT_CONFIGURED"
        )
        vlm_status = vlm_only.artifact.payload["citation_pack"][
            "query_plan"
        ]["image_processing"]
        self.assertEqual(
            vlm_status["visual_search"], "SKIPPED_NOT_CONFIGURED"
        )
        self.assertEqual(vlm_status["vlm_text_search"], "EXECUTED")

    async def test_composer_returns_only_actually_used_references(self):
        client = _KnowledgeCoreClient()
        retrieval = await KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent-worker",
        ).execute(_context())
        artifact = LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type=retrieval.artifact.artifact_type,
            schema_version=retrieval.artifact.schema_version,
            producer="knowledge-retrieval",
            producer_version="1.0.0",
            payload=retrieval.artifact.payload,
            content_hash="hash",
            provenance=retrieval.artifact.provenance,
            security_level=2,
        )
        result = await ResponseComposerSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        ).execute(_context(input_artifacts=(artifact,)))

        payload = result.artifact.payload
        self.assertEqual(payload["used_citation_labels"], ["C1"])
        self.assertEqual(len(payload["references"]), 1)
        self.assertEqual(
            payload["references"][0]["document_id"],
            str(client.document_id),
        )
        self.assertEqual(
            payload["references"][0]["bundle_revision_id"],
            str(client.revision_id),
        )
        self.assertEqual(
            payload["references"][0]["locator_schema_version"],
            "document/v1",
        )

    async def test_document_answer_prompt_uses_frozen_response_language(self):
        retrieval = await self._retrieval_artifact()
        model = _RecordingModelClient(
            response={
                "answer": "この文書は索引最適化の事例です。[C1]",
                "used_citation_labels": ["C1"],
                "presented_assets": [],
            }
        )

        await ResponseComposerSkill(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        ).execute(
            _context(
                input_artifacts=(retrieval,),
                original_input="この文書の内容は何ですか？",
                language="ja-JP",
            )
        )

        messages = model.last_json_request["prompt"]
        self.assertTrue(
            any(
                "不得在回答中列出、讨论或对比不匹配"
                in message["content"]
                for message in messages
            )
        )
        self.assertIn("language=ja-JP", messages[-1]["content"])

    async def test_stream_retries_wrong_language_and_keeps_constraint_last(self):
        retrieval = await self._retrieval_artifact()
        model = _LanguageRepairModelClient()

        outputs = [
            item
            async for item in ResponseComposerSkill(
                model_client=model,
                prompt_resolver=_PromptResolver(),
            ).execute_stream(
                _context(
                    input_artifacts=(retrieval,),
                    original_input="Any asset relates to ChatBI?",
                    language="en-US",
                )
            )
        ]

        answer = "".join(
            item.payload["delta"]
            for item in outputs
            if getattr(item, "event_type", None) == "answer.delta"
        )
        self.assertEqual(model.calls, 2)
        self.assertTrue(answer.startswith("One related asset"))
        self.assertNotIn("根据证据", answer)
        for prompt in model.prompts:
            self.assertIn("language=en-US", prompt[-1]["content"])

    async def test_insufficient_evidence_uses_frozen_response_language(self):
        result = await ResponseComposerSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        ).execute(
            _context(
                original_input="관련 자료를 찾아주세요",
                language="ko-KR",
            )
        )

        self.assertEqual(
            result.artifact.payload["answer"],
            "현재 권한이 부여된 지식 범위에서 인용할 수 있는 "
            "충분한 근거를 찾지 못했습니다.",
        )

    async def test_composer_prefers_document_evidence_over_clarification(self):
        rewrite = self._ambiguous_rewrite_artifact()
        retrieval = await self._retrieval_artifact()

        result = await ResponseComposerSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        ).execute(_context(input_artifacts=(rewrite, retrieval)))

        self.assertEqual(result.artifact.payload["status"], "READY")
        self.assertEqual(
            result.artifact.payload["used_citation_labels"], ["C1"]
        )

    async def test_wrong_language_clarification_uses_localized_fallback(self):
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
            "Please specify the asset, topic, or statistical scope you mean.",
        )

    async def test_composer_streams_real_answer_deltas(self):
        client = _KnowledgeCoreClient()
        retrieval = await KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent-worker",
        ).execute(_context())
        artifact = LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type=retrieval.artifact.artifact_type,
            schema_version=retrieval.artifact.schema_version,
            producer="knowledge-retrieval",
            producer_version="1.0.0",
            payload=retrieval.artifact.payload,
            content_hash="hash",
            provenance={},
            security_level=2,
        )
        outputs = [
            item
            async for item in ResponseComposerSkill(
                model_client=_ModelClient(),
                prompt_resolver=_PromptResolver(),
            ).execute_stream(_context(input_artifacts=(artifact,)))
        ]

        deltas = [
            item.payload["delta"]
            for item in outputs
            if getattr(item, "event_type", None) == "answer.delta"
        ]
        self.assertEqual("".join(deltas), "该案例通过调整索引降低了查询延迟。[C1]")
        self.assertEqual(
            outputs[-1].artifact.payload["used_citation_labels"], ["C1"]
        )

    async def test_composer_normalizes_streamed_citation_variants(self):
        client = _KnowledgeCoreClient()
        retrieval = await KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent-worker",
        ).execute(_context())
        artifact = LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type=retrieval.artifact.artifact_type,
            schema_version=retrieval.artifact.schema_version,
            producer="knowledge-retrieval",
            producer_version="1.0.0",
            payload=retrieval.artifact.payload,
            content_hash="hash",
            provenance={},
            security_level=2,
        )
        outputs = [
            item
            async for item in ResponseComposerSkill(
                model_client=_VariantCitationModelClient(),
                prompt_resolver=_PromptResolver(),
            ).execute_stream(_context(input_artifacts=(artifact,)))
        ]

        deltas = [
            item.payload["delta"]
            for item in outputs
            if getattr(item, "event_type", None) == "answer.delta"
        ]
        self.assertEqual(
            "".join(deltas),
            "第一条事实使用全角引用[C1]。第二条事实使用上标引用[C1]。",
        )
        self.assertEqual(
            outputs[-1].artifact.payload["used_citation_labels"], ["C1"]
        )

    def test_composer_normalizes_non_streamed_citation_variants(self):
        answer, labels, assets = ResponseComposerSkill._validate_model_answer(
            {
                "answer": "第一条事实【C1】，第二条事实<sup>C1</sup>。",
                "used_citation_labels": ["C1"],
                "presented_assets": [],
            },
            {"C1": object()},
        )

        self.assertEqual(answer, "第一条事实[C1]，第二条事实[C1]。")
        self.assertEqual(labels, ("C1",))
        self.assertEqual(assets, ())

    def test_breadth_composer_allows_unused_retrieval_candidates(self):
        answer, labels, assets = ResponseComposerSkill._validate_model_answer(
            {
                "answer": "只有第一项与问题相关。[C1]",
                "used_citation_labels": ["C1"],
                "presented_assets": [],
            },
            {"C1": object(), "C2": object()},
        )

        self.assertEqual("只有第一项与问题相关。[C1]", answer)
        self.assertEqual(("C1",), labels)
        self.assertEqual(assets, ())

    def test_composer_preserves_presented_asset_order_and_groups_support(self):
        answer, labels, assets = ResponseComposerSkill._validate_model_answer(
            {
                "answer": (
                    "1. 第一个资产的完整说明。[C1][C2]\n"
                    "2. 第二个资产的完整说明。[C3]"
                ),
                "used_citation_labels": ["C1", "C2", "C3"],
                "presented_assets": [
                    {
                        "primary_citation_label": "C1",
                        "supporting_citation_labels": ["C2"],
                    },
                    {
                        "primary_citation_label": "C3",
                        "supporting_citation_labels": [],
                    },
                ],
            },
            {"C1": object(), "C2": object(), "C3": object()},
        )

        self.assertEqual(("C1", "C2", "C3"), labels)
        self.assertEqual(
            ["C1", "C3"],
            [item.primary_citation_label for item in assets],
        )
        self.assertIn("第二个资产", answer)

    def test_composer_rejects_reference_mapped_to_two_assets(self):
        with self.assertRaisesRegex(
            ValueError, "同一引用不得映射到多个 presented_assets"
        ):
            ResponseComposerSkill._validate_model_answer(
                {
                    "answer": "第一个资产。[C1] 第二个资产。[C2]",
                    "used_citation_labels": ["C1", "C2"],
                    "presented_assets": [
                        {
                            "primary_citation_label": "C1",
                            "supporting_citation_labels": ["C2"],
                        },
                        {
                            "primary_citation_label": "C2",
                            "supporting_citation_labels": [],
                        },
                    ],
                },
                {"C1": object(), "C2": object()},
            )

    def test_composer_requires_presented_assets_array(self):
        with self.assertRaisesRegex(
            ValueError, "模型未返回 presented_assets 数组"
        ):
            ResponseComposerSkill._validate_model_answer(
                {
                    "answer": "证据支持的回答。[C1]",
                    "used_citation_labels": ["C1"],
                },
                {"C1": object()},
            )

    def test_grounded_answer_rejects_unmentioned_reference(self):
        with self.assertRaisesRegex(
            ValidationError, "引用列表包含正文未使用的证据"
        ):
            GroundedAnswer(
                answer="只有第一项与问题相关。[Q1]",
                status="READY",
                used_citation_labels=("Q1",),
                references=(
                    QueryResultReferenceCard(
                        citation_label="Q1",
                        query_result_id=uuid7(),
                        provider="MCP",
                        row_count=1,
                    ),
                    QueryResultReferenceCard(
                        citation_label="Q2",
                        query_result_id=uuid7(),
                        provider="MCP",
                        row_count=1,
                    ),
                ),
            )

    async def test_composer_does_not_stream_answer_before_validation(self):
        artifact = await self._retrieval_artifact()
        model = _MissingCitationModelClient(repair_succeeds=True)
        outputs = [
            item
            async for item in ResponseComposerSkill(
                model_client=model,
                prompt_resolver=_PromptResolver(),
            ).execute_stream(_context(input_artifacts=(artifact,)))
        ]

        deltas = [
            item.payload["delta"]
            for item in outputs
            if getattr(item, "event_type", None) == "answer.delta"
        ]
        self.assertEqual(model.calls, 2)
        self.assertEqual(deltas, ["证据支持的回答。[C1]"])
        self.assertNotIn("1+1=2", "".join(deltas))

    async def test_composer_falls_back_to_verified_source_titles(self):
        artifact = await self._retrieval_artifact()
        model = _MissingCitationModelClient(repair_succeeds=False)
        outputs = [
            item
            async for item in ResponseComposerSkill(
                model_client=model,
                prompt_resolver=_PromptResolver(),
            ).execute_stream(_context(input_artifacts=(artifact,)))
        ]

        deltas = [
            item.payload["delta"]
            for item in outputs
            if getattr(item, "event_type", None) == "answer.delta"
        ]
        payload = outputs[-1].artifact.payload
        self.assertEqual(model.calls, 2)
        self.assertEqual(deltas, ["- 数据库优化案例 [C1]"])
        self.assertEqual(payload["status"], "READY")
        self.assertEqual(payload["used_citation_labels"], ["C1"])
        self.assertEqual(len(payload["references"]), 1)

    async def test_non_stream_composer_falls_back_to_verified_titles(self):
        artifact = await self._retrieval_artifact()
        model = _RecordingModelClient(
            response={
                "answer": "没有引用的回答。",
                "used_citation_labels": [],
                "presented_assets": [],
            }
        )

        result = await ResponseComposerSkill(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        ).execute(_context(input_artifacts=(artifact,)))

        payload = result.artifact.payload
        self.assertEqual(payload["answer"], "- 数据库优化案例 [C1]")
        self.assertEqual(payload["status"], "READY")
        self.assertEqual(payload["used_citation_labels"], ["C1"])
        self.assertEqual(len(payload["references"]), 1)

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
