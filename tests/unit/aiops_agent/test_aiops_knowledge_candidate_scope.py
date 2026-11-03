"""知识候选预算、稳定分页、精确版本与正文检索专项；不替代真实召回质量评测。"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.knowledge_search import KnowledgeSearchBudget
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.contracts.knowledge import KnowledgeSearchRequest
from aiops_agent.repositories.knowledge import OperationsKnowledgeRepository
from platform_core.identity import uuid7
from tests.support.aiops_knowledge_scope import knowledge_agent_repository, active_knowledge_bases_fixture


def setup_search(*, size=5, max_results=1, budget=None):
    rows = [(
        SimpleNamespace(asset_id=uuid7(), asset_kind="MANUAL", display_name=f"知识-{index}"),
        SimpleNamespace(asset_version_id=uuid7(), version_no=1, profile_json={"scope": {}}),
        SimpleNamespace(index_ref_id=uuid7(), collection_id=uuid7(), bundle_revision_id=uuid7()),
    ) for index in range(size)]
    state = SimpleNamespace(rows=rows, rechecks=0, transaction_open=False)

    async def list_candidates(*, limit, after_index_id=None, asset_version_ids=(), bundle_revision_ids=(), **scope):
        assert state.transaction_open
        values = state.rows
        if after_index_id is not None:
            values = [row for row in values if row[2].index_ref_id > after_index_id]
        if asset_version_ids:
            state.rechecks += 1
            values = [row for row in values if row[1].asset_version_id in asset_version_ids]
        if bundle_revision_ids:
            values = [row for row in values if row[2].bundle_revision_id in bundle_revision_ids]
        return values if limit is None else values[:limit]

    repo = SimpleNamespace(list_published_candidates=AsyncMock(side_effect=list_candidates))
    active_knowledge_bases_fixture(repo)
    @asynccontextmanager
    async def uow_factory():
        state.transaction_open = True
        try:
            yield SimpleNamespace(operations_knowledge=repo, agents=agents)
        finally:
            state.transaction_open = False

    wanted = rows[-1]
    async def discover(**kwargs):
        assert not state.transaction_open
        if wanted[2].bundle_revision_id not in kwargs["bundle_revision_ids"]:
            return {"candidates": []}
        return {"candidates": [{
            "collection_id": str(wanted[2].collection_id), "bundle_id": str(uuid7()),
            "bundle_revision_id": str(wanted[2].bundle_revision_id), "rrf_score": 0.03,
        }]}

    async def retrieve(**kwargs):
        assert not state.transaction_open
        return {"citations": [{
            "citation_id": str(uuid7()), "collection_id": str(wanted[2].collection_id),
            "bundle_revision_id": str(wanted[2].bundle_revision_id),
            "quote": "较早相关知识的正文",
        }]}

    service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
    service._uow_factory = uow_factory
    service._knowledge_search_budget = budget or KnowledgeSearchBudget()
    service._kc = SimpleNamespace(discover=AsyncMock(side_effect=discover), retrieve_evidence=AsyncMock(side_effect=retrieve))
    request = KnowledgeSearchRequest(query="连接等待的排查方式", max_results=max_results,
        source_kinds=("MANUAL",), knowledge_base_ids=(uuid7(),))
    agents = knowledge_agent_repository(knowledge_base_ids=request.knowledge_base_ids)
    kwargs = dict(domain_id=1, agent_id=str(uuid7()), request=request, context=SimpleNamespace())
    import inspect
    if "tenant_id" in inspect.signature(service.search).parameters:
        kwargs.update(domain_id=uuid7(), tenant_id=uuid7())
    return SimpleNamespace(**locals())


def test_final_top_k_does_not_truncate_registry_before_body_search():
    async def run():
        f = setup_search()
        result = await f.service.search(**f.kwargs)
        assert result["results"][0]["asset_id"] == str(f.wanted[0].asset_id)
        assert len(result["results"]) == 1
        assert result["coverage"]["scanned_candidates"] == 5
        assert result["coverage"]["registry_scope_complete"]
        call = f.service._kc.discover.await_args.kwargs
        assert len(call["bundle_revision_ids"]) == 5
        assert call["per_collection_limit"] != f.request.max_results
        recheck = f.repo.list_published_candidates.await_args.kwargs
        assert recheck["limit"] is None and len(recheck["asset_version_ids"]) == 5
        assert f.state.rechecks == 1
    asyncio.run(run())


@pytest.mark.parametrize("failure,expected", [
    ("disabled", "KNOWLEDGE_DISABLED"), ("source_kind", "NO_AUTHORIZED_KNOWLEDGE_SOURCES"),
    ("unbound_base", "NO_KNOWLEDGE_BASE_BOUND"), ("target", "TARGET_NOT_AUTHORIZED"),
    ("inactive", "NO_AGENT_BOUND"),
    ("archived_base", "NO_KNOWLEDGE_BASE_BOUND"),
])
def test_current_agent_policy_cannot_be_bypassed_by_explicit_search_fields(failure, expected):
    async def run():
        f = setup_search()
        policy = f.agents.version_knowledge_policy.return_value
        if failure == "disabled":
            policy["enabled"] = False
        elif failure == "source_kind":
            policy["source_kinds"] = ["MANUAL"]
            f.kwargs["request"] = f.request.model_copy(update={"source_kinds": ("DIAGNOSIS_CASE",)})
        elif failure == "unbound_base":
            policy["knowledge_base_ids"] = [uuid7()]
        elif failure == "target":
            f.kwargs["target_id"] = uuid7()
            f.agents.version_has_target.return_value = False
        elif failure == "inactive":
            f.agents.get.return_value.status = "DISABLED"
        elif failure == "archived_base":
            f.repo.active_knowledge_base_ids.side_effect = None
            f.repo.active_knowledge_base_ids.return_value = []
        result = await f.service.search(**f.kwargs)
        assert result["status"] == expected and result["results"] == []
        f.service._kc.discover.assert_not_awaited()
        f.repo.list_published_candidates.assert_not_awaited()
    asyncio.run(run())


def test_policy_security_cap_and_required_tags_survive_request_overrides():
    async def run():
        f = setup_search()
        tag = uuid7()
        f.agents.version_knowledge_policy.return_value.update(max_security_level=2, tag_ids=[tag])
        result = await f.service.search(**f.kwargs)
        assert result["results"]
        for call in f.repo.list_published_candidates.await_args_list:
            assert call.kwargs["max_security_level"] == 2
            assert str(tag).upper() in call.kwargs["scope_filters"]["TAG_ID"]
        assert f.service._kc.discover.await_args.kwargs["max_security_level"] == 2
    asyncio.run(run())


def test_agent_policy_revoked_during_kc_call_discards_already_retrieved_body():
    async def run():
        f = setup_search()
        original = f.service._kc.retrieve_evidence.side_effect
        async def revoke(**kwargs):
            response = await original(**kwargs)
            f.agents.version_knowledge_policy.return_value["enabled"] = False
            return response
        f.service._kc.retrieve_evidence.side_effect = revoke
        result = await f.service.search(**f.kwargs)
        assert result["status"] == "KNOWLEDGE_SCOPE_CHANGED"
        assert result["results"] == [] and result["warnings"]
    asyncio.run(run())


def test_archived_only_knowledge_binding_does_not_unlock_domain_cases():
    async def run():
        f = setup_search()
        f.kwargs["request"] = f.request.model_copy(update={"source_kinds": ("DIAGNOSIS_CASE",)})
        f.repo.active_knowledge_base_ids.side_effect = None
        f.repo.active_knowledge_base_ids.return_value = []
        result = await f.service.search(**f.kwargs)
        assert result["status"] == "NO_KNOWLEDGE_BASE_BOUND" and not result["results"]
        f.repo.list_published_candidates.assert_not_awaited()
        f.service._kc.discover.assert_not_awaited()
    asyncio.run(run())


def test_knowledge_base_archived_during_body_retrieval_cannot_leak_results():
    async def run():
        f = setup_search()
        f.repo.active_knowledge_base_ids.side_effect = [list(f.request.knowledge_base_ids), []]
        result = await f.service.search(**f.kwargs)
        assert result["status"] == "KNOWLEDGE_SCOPE_CHANGED" and not result["results"]
    asyncio.run(run())


def test_keyset_pages_and_kc_shards_keep_exact_authorized_revisions():
    async def run():
        f = setup_search(budget=KnowledgeSearchBudget(registry_page_size=2, discovery_scope_size=2))
        result = await f.service.search(**f.kwargs)
        assert result["results"]
        assert result["coverage"]["discovery_shards"] == 3
        pages = f.repo.list_published_candidates.await_args_list[:3]
        assert pages[0].kwargs["after_index_id"] is None
        assert pages[1].kwargs["after_index_id"] == f.rows[1][2].index_ref_id
        assert pages[2].kwargs["after_index_id"] == f.rows[3][2].index_ref_id
        ranges = [call.kwargs["bundle_revision_ids"] for call in f.service._kc.discover.await_args_list]
        assert [len(values) for values in ranges] == [2, 2, 1]
        assert {str(value) for values in ranges for value in values} == {str(row[2].bundle_revision_id) for row in f.rows}
    asyncio.run(run())


def test_budget_overflow_is_explicit_not_a_false_complete_no_match():
    async def run():
        f = setup_search(budget=KnowledgeSearchBudget(max_candidates=2, registry_page_size=2))
        result = await f.service.search(**f.kwargs)
        assert result["results"] == []
        assert not result["coverage"]["registry_scope_complete"]
        assert result["coverage"]["scanned_candidates"] == 2
        assert any("超过本次检索预算" in warning for warning in result["warnings"])
    asyncio.run(run())


def test_revision_withdrawn_while_kc_runs_is_removed_before_output():
    async def run():
        f = setup_search()
        original = f.service._kc.retrieve_evidence.side_effect
        async def revoke(**kwargs):
            response = await original(**kwargs)
            f.state.rows = f.rows[:-1]
            return response
        f.service._kc.retrieve_evidence.side_effect = revoke
        result = await f.service.search(**f.kwargs)
        assert not result["results"]
        assert f.state.rechecks == 1
    asyncio.run(run())


@pytest.mark.parametrize("tamper", ["revision", "collection"])
def test_kc_candidate_outside_registry_whitelist_cannot_expand_scope(tamper):
    async def run():
        f = setup_search()
        response = await f.service._kc.discover.side_effect(
            bundle_revision_ids=[f.wanted[2].bundle_revision_id])
        response["candidates"][0]["bundle_revision_id" if tamper == "revision" else "collection_id"] = str(uuid7())
        f.service._kc.discover.side_effect = None
        f.service._kc.discover.return_value = response
        result = await f.service.search(**f.kwargs)
        assert not result["results"]
        f.service._kc.retrieve_evidence.assert_not_awaited()
    asyncio.run(run())


def test_two_citations_to_one_asset_are_one_result_not_two_requested_results():
    async def run():
        f = setup_search(max_results=3)
        original = f.service._kc.retrieve_evidence.side_effect
        async def two_quotes(**kwargs):
            response = await original(**kwargs)
            first = response["citations"][0]
            response["citations"].append({**first, "citation_id": str(uuid7()), "quote": "同一确切版本的另一个片段"})
            return response
        f.service._kc.retrieve_evidence.side_effect = two_quotes
        result = await f.service.search(**f.kwargs)
        assert len(result["results"]) == 1
        assert len(result["results"][0]["citation_pack"]) == 2
    asyncio.run(run())


@pytest.mark.parametrize("invalid", [None, "body_hash", "document_role", "document_id", "revision",
    "missing_heading", "withdrawn", "processing", "rejected"])
def test_case_context_requires_exact_kc_content_member_and_survives_in_citation_locator(monkeypatch, invalid):
    from aiops_agent.application.case_knowledge_projection import case_knowledge_markdown
    from tests.unit.aiops_agent.test_aiops_case_knowledge_projection import profile
    monkeypatch.setattr("aiops_agent.application.knowledge_search._eligible_case", AsyncMock(return_value=True))
    async def run():
        f = setup_search()
        value = profile()
        f.kwargs["request"] = f.request.model_copy(update={"source_kinds": ("DIAGNOSIS_CASE",)})
        case_knowledge_markdown(title="复盘", profile=value)
        asset, version, index = f.wanted
        asset.asset_kind = "DIAGNOSIS_CASE"
        version.profile_json = value.model_dump(mode="json")
        index.bundle_id = uuid7()
        document_id = uuid7()
        member = {"document_version_id": str(document_id), "document_role": "CONTENT",
            "content_hash_sha256": value.knowledge_projection.body_sha256, "member_status": "READY"}
        revision = {"bundle_revision_id": str(index.bundle_revision_id), "members": [member],
            "status": "READY", "approval_status": "APPROVED", "withdrawn_at": None}
        if invalid == "withdrawn":
            revision["withdrawn_at"] = "2026-11-03T00:00:00+00:00"
        if invalid == "processing":
            revision["status"] = "PROCESSING"
        if invalid == "rejected":
            revision["approval_status"] = "REJECTED"
        if invalid == "body_hash":
            member["content_hash_sha256"] = "a" * 64
        if invalid == "document_role":
            member["document_role"] = "MANIFEST"
        if invalid == "revision":
            revision["bundle_revision_id"] = str(uuid7())
        async def members(**kwargs):
            assert not f.state.transaction_open
            assert kwargs["bundle_revision_id"] == index.bundle_revision_id
            return revision
        f.service._kc.get_revision_status = AsyncMock(side_effect=members)
        original = f.service._kc.retrieve_evidence.side_effect
        async def retrieve(**kwargs):
            response = await original(**kwargs)
            citation = response["citations"][0]
            citation["document_version_id"] = str(uuid7() if invalid == "document_id" else document_id)
            citation["items"] = [{"evidence": {"heading_path": [] if invalid == "missing_heading" else ["[case-action-1] 登记措施与效果 1"],
                "content_text": "登记措施与效果"}}]
            return response
        f.service._kc.retrieve_evidence.side_effect = retrieve
        result = await f.service.search(**f.kwargs)
        if invalid:
            assert result["results"] == [] and result["warnings"]
        else:
            citation = result["results"][0]["citation_pack"][0]
            assert citation["locator"]["case_context"]["body_sha256"] == value.knowledge_projection.body_sha256
            assert citation["locator"]["case_context"]["requires_current_verification"]
            assert result["results"][0]["reuse_context"]["role"] == "HISTORICAL_KNOWLEDGE_NOT_CURRENT_EVIDENCE"
        assert f.state.rechecks == 1
    asyncio.run(run())


def test_catalog_honors_final_count_without_calling_kc():
    async def run():
        f = setup_search()
        f.kwargs["request"] = f.request.model_copy(update={"purpose": "CATALOG", "max_results": 2})
        result = await f.service.search(**f.kwargs)
        assert len(result["results"]) == 2
        assert result["warnings"]
        f.service._kc.discover.assert_not_awaited()
    asyncio.run(run())


def test_catalog_checks_current_authorization_after_registry_pagination():
    async def run():
        f = setup_search()
        f.kwargs["request"] = f.request.model_copy(update={"purpose": "CATALOG"})
        original = f.repo.list_published_candidates.side_effect
        async def revoke(**kwargs):
            values = await original(**kwargs)
            f.agents.version_knowledge_policy.return_value["enabled"] = False
            return values
        f.repo.list_published_candidates.side_effect = revoke
        result = await f.service.search(**f.kwargs)
        assert result["status"] == "KNOWLEDGE_SCOPE_CHANGED" and result["results"] == []
        f.service._kc.discover.assert_not_awaited()
    asyncio.run(run())


def test_nonadvancing_repository_cursor_fails_with_bounded_error():
    async def run():
        f = setup_search(size=2, budget=KnowledgeSearchBudget(max_candidates=10, registry_page_size=2))
        f.repo.list_published_candidates.side_effect = None
        f.repo.list_published_candidates.return_value = f.rows
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service.search(**f.kwargs)
        assert error.value.code == "AIOPS_KNOWLEDGE_CURSOR_CONFLICT"
        assert f.repo.list_published_candidates.await_count == 2
    asyncio.run(run())


def test_active_knowledge_bindings_use_one_domain_scoped_database_query():
    async def run():
        base_ids = [uuid7(), uuid7()]
        session = SimpleNamespace(scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: base_ids)))
        repo = OperationsKnowledgeRepository(session)
        from aiops_agent.entities import OperationsKnowledgeAssetEntity
        domain = 1 if OperationsKnowledgeAssetEntity.__tablename__.startswith("KBOT_") else uuid7()
        result = await repo.active_knowledge_base_ids(domain_id=domain, knowledge_base_ids=base_ids)
        assert result == base_ids and session.scalars.await_count == 1
        statement = session.scalars.await_args.args[0]
        params = statement.compile().params
        assert "ACTIVE" in params.values() and domain in params.values()
        assert any(value == base_ids for value in params.values())
        assert await repo.active_knowledge_base_ids(domain_id=domain, knowledge_base_ids=[]) == []
        assert session.scalars.await_count == 1
    asyncio.run(run())


@pytest.mark.parametrize("dialect", [postgresql.dialect(), oracle.dialect()])
def test_registry_query_uses_database_keyset_limit_and_correlated_scope_not_full_scan(dialect):
    async def run():
        result = SimpleNamespace(all=lambda: [])
        session = SimpleNamespace(execute=AsyncMock(return_value=result))
        repo = OperationsKnowledgeRepository(session)
        cursor, tag_id, version, revision = (uuid7() for _ in range(4))
        from aiops_agent.entities import OperationsKnowledgeAssetEntity
        domain = 1 if OperationsKnowledgeAssetEntity.__tablename__.startswith("KBOT_") else uuid7()
        await repo.list_published_candidates(
            domain_id=domain, target_id=uuid7(), source_kinds=("MANUAL", "DIAGNOSIS_CASE"),
            knowledge_base_ids=(uuid7(),), max_security_level=3,
            scope_filters={"TAG_ID": (str(tag_id),), "DATABASE_TYPE": ("ORACLE",)},
            limit=7, after_index_id=cursor, asset_version_ids=(version,), bundle_revision_ids=(revision,),
        )
        assert session.execute.await_count == 1
        statement = session.execute.await_args.args[0]
        assert statement._limit_clause.value == 7
        sql = str(statement.compile(dialect=dialect)).upper()
        assert "EXISTS" in sql and "INDEX_REF_ID >" in sql
        assert "PUBLISHED" in str(statement.compile().params.values())
        assert str(cursor) in str(statement.compile().params.values())
    asyncio.run(run())


@pytest.mark.parametrize("changes", [{"max_candidates": 0}, {"max_candidates": 1001},
    {"discovery_scope_size": 1001}, {"evidence_batch_size": 129}])
def test_budget_respects_both_platform_contract_limits(changes):
    with pytest.raises(ValueError):
        KnowledgeSearchBudget(**changes)
