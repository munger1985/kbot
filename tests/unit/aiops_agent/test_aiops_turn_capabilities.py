"""统一助手只读 specialist 的授权、模型与证据合同测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.turn_capabilities import (
    KnowledgeQuestionAdapter,
    MonitoringQueryAdapter,
    RuntimeQueryAdapter,
)
from platform_core.contracts.aiops import AIOpsCapability
from platform_core.identity import uuid7


def _internal(*target_ids, **overrides):
    runtime = {
        "turn_id": str(uuid7()),
        "domain_id": 7,
        "agent_id": str(uuid7()),
        "agent_version_id": str(uuid7()),
        "actor_id": "dba@example.com",
        "question": "MES 数据库连接数为什么升高？",
        "models": {"composer_llm": str(uuid7())},
        "knowledge_policy": {
            "enabled": True,
            "source_kinds": ["MANUAL", "DIAGNOSIS_CASE"],
            "knowledge_base_ids": [str(uuid7())],
            "tag_ids": [],
            "max_security_level": 2,
            "maximum_references": 4,
        },
        "agent_instruction": "回答应简洁",
        "caller_service": "aiops-worker",
        "request_id": "request-1",
        "trace_id": "trace-1",
    }
    runtime.update(overrides)
    return {
        "runtime_context": runtime,
        "execution_targets": [
            {"target_id": str(target_id), "readiness": "READY"}
            for target_id in target_ids
        ],
    }


class _KnowledgeService:
    def __init__(self) -> None:
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "READY",
            "results": [
                {
                    "asset_version_id": str(uuid7()),
                    "title": "Oracle 运维手册",
                    "citation_pack": [
                        {
                            "bundle_revision_id": str(uuid7()),
                            "document_version_id": str(uuid7()),
                            "quote": "连接数持续升高时检查连接池回收策略。",
                            "locator": {"page": 12},
                        }
                    ],
                }
            ],
        }


class _Model:
    def __init__(self, labels=("K1",)) -> None:
        self.labels = labels
        self.calls = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output={
                "answer_markdown": "应先检查连接池是否及时回收连接。",
                "citation_labels": list(self.labels),
            }
        )


class _Prompt:
    content = "仅按证据回答"

    def ref(self):
        return {"prompt_id": "aiops_agent.knowledge_answer_compose"}


class _Prompts:
    async def resolve(self, prompt_id):
        if prompt_id != "knowledge_answer_compose":
            raise AssertionError(prompt_id)
        return _Prompt()


class TurnCapabilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_knowledge_uses_agent_version_policy_model_and_citations(self):
        knowledge = _KnowledgeService()
        model = _Model()
        resolved = []

        async def resolve_model(model_id):
            resolved.append(model_id)
            return {
                "model_id": str(model_id),
                "technical_name": "agent-composer",
                "revision": str(model_id),
            }

        internal = _internal(uuid7())
        result = await KnowledgeQuestionAdapter(
            knowledge_service=knowledge,
            structured_model=model,
            prompt_registry=_Prompts(),
            model_resolver=resolve_model,
            caller_service="aiops-worker",
        ).execute(
            capability=AIOpsCapability.KNOWLEDGE_QA,
            plan=SimpleNamespace(),
            scope=SimpleNamespace(),
            internal_scope=internal,
        )

        self.assertTrue(result.supported)
        self.assertEqual("K1", result.evidence[0]["label"])
        self.assertIn("### 引用", result.markdown)
        self.assertEqual(2, knowledge.calls[0]["request"].max_security_level)
        self.assertEqual(4, knowledge.calls[0]["request"].max_results)
        self.assertEqual(
            tuple(internal["runtime_context"]["knowledge_policy"]["knowledge_base_ids"]),
            tuple(str(item) for item in knowledge.calls[0]["request"].knowledge_base_ids),
        )
        self.assertEqual(
            internal["runtime_context"]["models"]["composer_llm"],
            str(resolved[0]),
        )
        self.assertEqual(
            "回答应简洁",
            model.calls[0]["input_payload"]["agent_instruction"],
        )

    async def test_knowledge_rejects_model_citation_outside_evidence(self):
        async def resolve_model(model_id):
            return {
                "model_id": str(model_id),
                "technical_name": "agent-composer",
                "revision": str(model_id),
            }

        result = await KnowledgeQuestionAdapter(
            knowledge_service=_KnowledgeService(),
            structured_model=_Model(labels=("K99",)),
            prompt_registry=_Prompts(),
            model_resolver=resolve_model,
            caller_service="aiops-worker",
        ).execute(
            capability=AIOpsCapability.KNOWLEDGE_QA,
            plan=SimpleNamespace(),
            scope=SimpleNamespace(),
            internal_scope=_internal(uuid7()),
        )

        self.assertFalse(result.supported)
        self.assertEqual("KNOWLEDGE_CITATION_INVALID", result.gap_code)

    async def test_knowledge_catalog_lists_topics_without_composer(self):
        asset_id, version_id = uuid7(), uuid7()
        knowledge = SimpleNamespace(search=AsyncMock(return_value={
            "status": "READY",
            "results": [{
                "asset_id": str(asset_id),
                "asset_version_id": str(version_id),
                "asset_kind": "MANUAL",
                "title": "Oracle 运维手册",
                "version": 2,
                "topics": ["连接管理", "备份恢复"],
                "citation_pack": [],
            }],
        }))
        model = SimpleNamespace(generate_structured=AsyncMock())
        plan = SimpleNamespace(steps=(SimpleNamespace(
            capability=AIOpsCapability.KNOWLEDGE_QA,
            parameters={"query_mode": "CATALOG"},
        ),))

        result = await KnowledgeQuestionAdapter(
            knowledge_service=knowledge,
            structured_model=model,
            prompt_registry=SimpleNamespace(resolve=AsyncMock()),
            model_resolver=AsyncMock(),
            caller_service="aiops-worker",
        ).execute(
            capability=AIOpsCapability.KNOWLEDGE_QA,
            plan=plan,
            scope=SimpleNamespace(),
            internal_scope=_internal(
                uuid7(),
                question="列出已发布运维知识",
                models={},
            ),
        )

        self.assertTrue(result.supported)
        self.assertIn("连接管理、备份恢复", result.markdown)
        request = knowledge.search.await_args.kwargs["request"]
        self.assertEqual("CATALOG", request.purpose)
        self.assertEqual(("MANUAL",), request.source_kinds)
        model.generate_structured.assert_not_awaited()

    async def test_alert_query_only_reads_resolved_targets(self):
        targets = (uuid7(), uuid7())
        calls = []

        async def list_situations(**kwargs):
            calls.append(kwargs["target_id"])
            target_id = kwargs["target_id"]
            row = SimpleNamespace(
                situation_id=uuid7(),
                target_id=target_id,
                severity="HIGH",
                title="连接数偏高",
                status="OPEN",
                last_observed_at=datetime.now(UTC),
                model_dump=lambda **_: {
                    "situation_id": str(uuid7()),
                    "target_id": str(target_id),
                },
            )
            return SimpleNamespace(items=(row,))

        result = await RuntimeQueryAdapter(
            runtime_service=SimpleNamespace(list_situations=list_situations)
        ).execute(
            capability=AIOpsCapability.ALERT_QUERY,
            plan=SimpleNamespace(steps=()),
            scope=SimpleNamespace(),
            internal_scope=_internal(*targets),
        )

        self.assertTrue(result.supported)
        self.assertEqual(set(targets), set(calls))
        self.assertEqual({str(item) for item in targets}, {
            item["target_id"] for item in result.evidence
        })

    async def test_monitoring_uses_fixed_profile_view_contract(self):
        target_id = uuid7()
        source_id = uuid7()
        calls = []
        source = SimpleNamespace(source_id=source_id)
        profile = SimpleNamespace(profile_id="oracle-overview", is_default=True)
        instance = SimpleNamespace(
            instance_id=target_id, display_name="MES RAC 1"
        )
        now = datetime.now(UTC)

        async def get_target_view(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                instances=(instance,),
                profile=SimpleNamespace(display_name="Oracle 概览"),
                window=SimpleNamespace(
                    start=now - timedelta(hours=1), end=now
                ),
                panels=(
                    SimpleNamespace(
                        title="会话使用率",
                        summary={"latest": 72.0},
                        quality="GOOD",
                    ),
                ),
                model_dump=lambda **_: {"profile_id": "oracle-overview"},
            )

        service = SimpleNamespace(
            list_sources=lambda **_: _async_value((source,)),
            list_instances=lambda **_: _async_value((instance,)),
            list_target_profiles=lambda **_: _async_value((profile,)),
            get_target_view=get_target_view,
        )
        result = await MonitoringQueryAdapter(
            monitoring_service=service
        ).execute(
            capability=AIOpsCapability.MONITORING_QUERY,
            plan=SimpleNamespace(steps=()),
            scope=SimpleNamespace(),
            internal_scope=_internal(target_id),
        )

        self.assertTrue(result.supported)
        self.assertEqual(target_id, calls[0]["target_id"])
        self.assertEqual("oracle-overview", calls[0]["profile_id"])
        self.assertEqual("1h", calls[0]["window"])
        self.assertEqual("MONITORING", result.evidence[0]["source_kind"])

    async def test_report_case_search_is_repeated_with_each_allowed_target(self):
        targets = (uuid7(), uuid7())
        knowledge = _KnowledgeService()
        list_reports = AsyncMock(return_value=SimpleNamespace(items=()))

        result = await RuntimeQueryAdapter(
            runtime_service=SimpleNamespace(list_reports=list_reports),
            knowledge_service=knowledge,
            caller_service="aiops-worker",
        ).execute(
            capability=AIOpsCapability.REPORT_CASE_QUERY,
            plan=SimpleNamespace(steps=()),
            scope=SimpleNamespace(),
            internal_scope=_internal(*targets),
        )

        self.assertTrue(result.supported)
        self.assertEqual(set(targets), {
            call["target_id"] for call in knowledge.calls
        })
        self.assertTrue(all(
            call["request"].source_kinds == ("DIAGNOSIS_CASE",)
            for call in knowledge.calls
        ))
        self.assertEqual(
            {"DIAGNOSIS_CASE"},
            {item["source_kind"] for item in result.evidence},
        )
        list_reports.assert_not_awaited()

    async def test_report_case_catalog_lists_case_topics(self):
        target_id = uuid7()
        asset_id, version_id = uuid7(), uuid7()
        knowledge = SimpleNamespace(search=AsyncMock(return_value={
            "status": "READY",
            "results": [{
                "asset_id": str(asset_id),
                "asset_version_id": str(version_id),
                "asset_kind": "DIAGNOSIS_CASE",
                "title": "连接池耗尽案例",
                "authority": "VERIFIED_RESOLUTION",
                "topics": ["连接池", "会话耗尽"],
                "citation_pack": [],
            }],
        }))
        list_reports = AsyncMock()
        plan = SimpleNamespace(steps=(SimpleNamespace(
            capability=AIOpsCapability.REPORT_CASE_QUERY,
            parameters={"query_mode": "CATALOG"},
        ),))

        result = await RuntimeQueryAdapter(
            runtime_service=SimpleNamespace(list_reports=list_reports),
            knowledge_service=knowledge,
            caller_service="aiops-worker",
        ).execute(
            capability=AIOpsCapability.REPORT_CASE_QUERY,
            plan=plan,
            scope=SimpleNamespace(),
            internal_scope=_internal(target_id, question="列出历史诊断案例"),
        )

        self.assertTrue(result.supported)
        self.assertIn("连接池、会话耗尽", result.markdown)
        self.assertEqual("CATALOG", knowledge.search.await_args.kwargs["request"].purpose)
        list_reports.assert_not_awaited()


async def _async_value(value):
    return value


if __name__ == "__main__":
    unittest.main()
