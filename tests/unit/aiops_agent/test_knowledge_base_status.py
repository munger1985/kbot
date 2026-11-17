"""知识库启停只改变使用资格，保留模型配置、资料、索引和绑定。"""
import asyncio
import importlib
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.knowledge import KnowledgeBaseStatusCommand
from aiops_agent.entities import OperationsKnowledgeBaseEntity
from platform_clients.aiops import AIOpsManagementClient
from platform_core.identity import uuid7

ORACLE = OperationsKnowledgeBaseEntity.__tablename__.startswith("KBOT_")


def fixture(status="ACTIVE", version=7, missing=False):
    row = SimpleNamespace(status=status, row_version=version,
        configuration_json={"models": {"embedding": str(uuid7())}}, updated_by=None, updated_at=None)
    repo = SimpleNamespace(get_knowledge_base=AsyncMock(return_value=None if missing else row))
    uow = SimpleNamespace(operations_knowledge=repo, commit=AsyncMock())
    @asynccontextmanager
    async def factory():
        yield uow
    service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
    service._uow_factory = factory
    service._validate_knowledge_base_models = AsyncMock(side_effect=AssertionError("启停不能改模型"))
    service.get_knowledge_base = AsyncMock(side_effect=lambda **_: {"status": row.status})
    actor = uuid7()
    context = SimpleNamespace(principal_id=actor, asserted_user_id=str(actor), client_id="test")
    kwargs = dict(domain_id=1 if ORACLE else uuid7(), knowledge_base_id=uuid7(),
        expected_row_version=7, context=context)
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("before,after", [("ACTIVE","DISABLED"), ("DISABLED","ACTIVE")])
def test_status_changes_are_scoped_versioned_and_never_reconfigure_models(before, after):
    async def run():
        f = fixture(before)
        original = deepcopy(f.row.configuration_json)
        result = await f.service.set_knowledge_base_status(**f.kwargs, status=after)
        assert result == {"status": after}
        assert f.row.configuration_json == original
        assert f.row.updated_by == (str(f.actor) if ORACLE else f.actor)
        assert f.row.updated_at is not None
        assert f.repo.get_knowledge_base.await_args.kwargs == {
            "domain_id": f.kwargs["domain_id"], "knowledge_base_id": f.kwargs["knowledge_base_id"], "for_update": True}
        f.uow.commit.assert_awaited_once()
        f.service._validate_knowledge_base_models.assert_not_awaited()
        if ORACLE:
            assert f.row.row_version == 8
    asyncio.run(run())


@pytest.mark.parametrize("mode,code", [("stale",412), ("missing",404), ("archived",404)])
def test_stale_or_unavailable_library_cannot_change_status(mode, code):
    async def run():
        f = fixture("ARCHIVED" if mode == "archived" else "ACTIVE", 8 if mode == "stale" else 7, mode == "missing")
        with pytest.raises(AIOpsApplicationError) as captured:
            await f.service.set_knowledge_base_status(**f.kwargs, status="DISABLED")
        assert captured.value.status_code == code
        f.uow.commit.assert_not_awaited()
    asyncio.run(run())


def test_idempotent_action_does_not_rewrite_and_invalid_status_is_rejected():
    async def run():
        f = fixture()
        await f.service.set_knowledge_base_status(**f.kwargs, status="ACTIVE")
        f.uow.commit.assert_not_awaited()
        with pytest.raises(ValueError):
            await f.service.set_knowledge_base_status(**f.kwargs, status="ARCHIVED")
    asyncio.run(run())


@pytest.mark.parametrize("payload", [{}, {"expected_row_version": 0}, {"expected_row_version":7,"models":{}}])
def test_status_command_requires_current_version_and_excludes_model_edits(payload):
    with pytest.raises(ValidationError):
        KnowledgeBaseStatusCommand.model_validate(payload)


@pytest.mark.parametrize("action,status", [("enable","ACTIVE"), ("disable","DISABLED")])
def test_public_client_internal_route_preserves_scope_permission_and_version(monkeypatch, action, status):
    async def run():
        public = importlib.import_module("main_api.api.aiops_app" if ORACLE else "main_api.api.ops")
        internal = importlib.import_module("aiops_agent.api.operations_knowledge")
        domain = 1 if ORACLE else uuid7()
        context = SimpleNamespace()
        service = SimpleNamespace(set_knowledge_base_status=AsyncMock(return_value={"status": status}))
        request = SimpleNamespace(state=SimpleNamespace(auth_context=context),
            app=SimpleNamespace(state=SimpleNamespace(operations_knowledge_service=service)))
        monkeypatch.setattr(internal, "_scope", lambda *_: (domain, "actor") if ORACLE else (uuid7(), domain, uuid7()))
        client = AIOpsManagementClient.__new__(AIOpsManagementClient)
        async def dispatch(method, path, *, payload, auth_context):
            assert method == "POST" and path.endswith(":" + action)
            assert auth_context is context
            return await getattr(internal, action + "_knowledge_base")(
                identity, KnowledgeBaseStatusCommand.model_validate(payload), request, context)
        client._json = AsyncMock(side_effect=dispatch)
        monkeypatch.setattr(public, "_client", lambda _: client)
        if ORACLE:
            permission = AsyncMock()
            monkeypatch.setattr(public, "_require", permission)
        identity = uuid7()
        result = await getattr(public, action + "_operations_knowledge_base")(
            identity, public.OperationsKnowledgeBaseStatusPayload(expected_row_version=7), request)
        assert result == {"status": status}
        assert service.set_knowledge_base_status.await_args.kwargs == {
            "domain_id":domain, "knowledge_base_id":identity, "expected_row_version":7, "status":status, "context":context}
        if ORACLE:
            permission.assert_awaited_once_with(request, "aiops:knowledge_manage")
    asyncio.run(run())
