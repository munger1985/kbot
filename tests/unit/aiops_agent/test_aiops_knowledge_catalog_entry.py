"""资料目录入口的两平台一致语义及内部错误隔离回归。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import ProgrammingError

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import OperationsKnowledgeAssetEntity
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
from platform_core.contracts.aiops import AIOpsCapability
from platform_core.identity import uuid7


def catalog_fixture(result=None, error=None):
    knowledge = SimpleNamespace(search=AsyncMock(return_value=result, side_effect=error))
    model = SimpleNamespace(generate_structured=AsyncMock())
    policy = {"enabled": True, "source_kinds": ["MANUAL", "DIAGNOSIS_CASE"],
        "knowledge_base_ids": [str(uuid7())], "tag_ids": [], "max_security_level": 2,
        "maximum_references": 8}
    plan = SimpleNamespace(steps=(SimpleNamespace(capability=AIOpsCapability.KNOWLEDGE_QA,
        parameters={"query_mode": "CATALOG"}),))
    arguments = dict(capability=AIOpsCapability.KNOWLEDGE_QA, plan=plan, scope=SimpleNamespace())
    constructor = dict(knowledge_service=knowledge, structured_model=model,
        prompt_registry=SimpleNamespace(resolve=AsyncMock()), model_resolver=AsyncMock())
    if OperationsKnowledgeAssetEntity.__tablename__.startswith("KBOT_"):
        from aiops_agent.application.turn_capabilities import KnowledgeQuestionAdapter
        adapter = KnowledgeQuestionAdapter(**constructor, caller_service="aiops-worker")
        arguments["internal_scope"] = {"execution_targets": [], "runtime_context": {
            "question": "列出资料和可提问主题", "models": {}, "knowledge_policy": policy,
            "domain_id": 7, "agent_id": str(uuid7()), "actor_id": "tester",
            "request_id": "catalog-test", "trace_id": "catalog-test"}}
    else:
        from aiops_agent.application.turn_planner import KnowledgeQaAdapter
        adapter = KnowledgeQaAdapter(**constructor)
        arguments.update(internal_scope={}, context=SimpleNamespace(
            question="列出资料和可提问主题", models={}, knowledge_policy=policy,
            tenant_id=uuid7(), domain_id=uuid7(), agent_id=uuid7(), auth_context=SimpleNamespace()))
    return SimpleNamespace(**locals())


def test_catalog_lists_manual_and_case_topics_without_model_or_body_evidence():
    async def run():
        rows = [{"asset_id": str(uuid7()), "asset_version_id": str(uuid7()),
            "asset_kind": kind, "title": title, "version": 2, "topics": topics, "citation_pack": []}
            for kind, title, topics in [
                ("MANUAL", "Oracle 运维手册", ["连接管理", "备份恢复"]),
                ("DIAGNOSIS_CASE", "连接池耗尽案例", ["连接池", "会话耗尽"])]]
        f = catalog_fixture({"status": "READY", "results": rows, "warnings": ["目录仅显示前两项。"]})
        result = await f.adapter.execute(**f.arguments)
        assert result.supported
        assert "运维资料，版本 2" in result.markdown and "历史诊断案例，版本 2" in result.markdown
        assert "连接管理、备份恢复" in result.markdown and "连接池、会话耗尽" in result.markdown
        assert "目录仅显示前两项。" in result.markdown and len(result.evidence) == 2
        assert f.knowledge.search.await_args.kwargs["request"].source_kinds == ("MANUAL", "DIAGNOSIS_CASE")
        f.model.generate_structured.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("status", ["KNOWLEDGE_SCOPE_CHANGED", "NO_KNOWLEDGE_BASE_BOUND", "KNOWLEDGE_DISABLED"])
def test_unavailable_catalog_scope_is_not_presented_as_empty_publication(status):
    async def run():
        f = catalog_fixture({"status": status, "results": []})
        result = await f.adapter.execute(**f.arguments)
        assert not result.supported
        assert "没有已发布" not in result.markdown
        f.model.generate_structured.assert_not_awaited()
    asyncio.run(run())


def test_directory_database_failure_is_not_a_sql_answer_or_empty_catalog():
    async def run():
        error = ProgrammingError("SELECT private_column FROM private_table", {"private": "value"},
            RuntimeError("UndefinedColumnError"))
        f = catalog_fixture(error=error)
        result = await f.adapter.execute(**f.arguments)
        assert not result.supported and result.gap_code == "OPERATIONS_KNOWLEDGE_UNAVAILABLE"
        assert "这不表示没有已发布资料" in result.markdown
        assert "private" not in result.markdown and "SELECT" not in result.markdown
    asyncio.run(run())


@pytest.mark.parametrize("source_kind", ["MANUAL", "DIAGNOSIS_CASE"])
def test_catalog_honors_single_authorized_source_kind(source_kind):
    async def run():
        f = catalog_fixture({"status": "NO_APPLICABLE_KNOWLEDGE", "results": []})
        f.policy["source_kinds"] = [source_kind]
        result = await f.adapter.execute(**f.arguments)
        assert result.supported
        assert f.knowledge.search.await_args.kwargs["request"].source_kinds == (source_kind,)
        f.model.generate_structured.assert_not_awaited()
    asyncio.run(run())


def test_catalog_does_not_claim_empty_when_scan_budget_is_incomplete():
    async def run():
        f = catalog_fixture({"status": "NO_APPLICABLE_KNOWLEDGE", "results": [],
            "coverage": {"registry_scope_complete": False}, "warnings": ["资料范围超过本次目录预算。"]})
        result = await f.adapter.execute(**f.arguments)
        assert "已扫描范围" in result.markdown and "超过本次目录预算" in result.markdown
        assert "没有已发布" not in result.markdown
    asyncio.run(run())


@pytest.mark.parametrize("business_error", [False, True])
def test_planner_terminal_failure_only_publishes_safe_error(business_error):
    async def run():
        terminal = SimpleNamespace(fail_terminal=AsyncMock())
        sink = AIOpsDomainOutboxSink(runtime_service=SimpleNamespace(), fallback=SimpleNamespace(),
            turn_planner_service=terminal)
        error = AIOpsApplicationError(code="CATALOG_NOT_AVAILABLE", message="目录暂不可用", status_code=503) if business_error else (
            ProgrammingError("SELECT private_column FROM private_table", {"private": "value"}, RuntimeError("undefined column")))
        await sink.on_terminal_failure("aiops.turn.understanding_requested", {"turn_id": str(uuid7())}, error)
        public = terminal.fail_terminal.await_args.kwargs
        assert "private" not in public["error_message"] and "SELECT" not in public["error_message"]
        assert public["error_code"] == ("CATALOG_NOT_AVAILABLE" if business_error else "AIOPS_INVESTIGATION_PLAN_INTERNAL_ERROR")
        if business_error:
            assert public["error_message"] == "目录暂不可用"
    asyncio.run(run())
