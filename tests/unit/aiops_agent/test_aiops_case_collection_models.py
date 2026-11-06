"""案例本域文档配置及权限；不访问 KC 业务表或手册绑定。"""

import asyncio
import importlib
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from aiops_agent.api import operations_knowledge as internal
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts.aiops.case_collection import CASE_MODEL_ROLES, CaseCollectionModels
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_output_job_api import api, human_context
from tests.unit.aiops_agent.test_aiops_output_jobs import Store


def fixture():
    from contextlib import asynccontextmanager
    from aiops_agent.application.operations_documents import OperationsDocuments
    from aiops_agent.domain.knowledge_retrieval import RetrievalAccess
    store = Store()
    context = human_context(store)
    models = {role: uuid7() for role in CASE_MODEL_ROLES}
    access = RetrievalAccess(store.domain, store.tenant, str(getattr(context, "principal_id", None) or context.asserted_user_id))
    state = SimpleNamespace(configuration=None)
    async def configure_cases(access, *, models, profile_ref):
        state.configuration = SimpleNamespace(models_json=models, profile_ref=profile_ref, row_version=1)
        return state.configuration
    repository = SimpleNamespace(case_configuration=AsyncMock(side_effect=lambda access: state.configuration),
        configure_cases=AsyncMock(side_effect=configure_cases))
    uow = SimpleNamespace(knowledge_documents=repository, commit=AsyncMock())
    @asynccontextmanager
    async def factory():
        yield uow
    documents = SimpleNamespace(access=lambda _: access,
        configuration=lambda models: {"models": {k: str(v) for k,v in models.items()}, "profile_ref":"test-profile"},
        retrieval=SimpleNamespace(authorization=SimpleNamespace(service=None)))
    categories = {str(model): (2 if role == "embedding" else 1) for role, model in models.items()}
    if store.tenant is not None:
        catalog = SimpleNamespace(list_tenant_models=AsyncMock(return_value=[
            {"model_id": key, "category": category, "status": 1,
                "entitled": True, "tenant_status": "ACTIVE", "tenant_configured": True}
            for key, category in categories.items()]))
        service = OperationsKnowledgeService(uow_factory=factory, documents=documents, model_catalog=catalog)
    else:
        catalog = {category: SimpleNamespace(get_model=AsyncMock(
            side_effect=lambda model_id, **kwargs: {"model_id": str(model_id),
                "category": categories[str(model_id)], "status": "ACTIVE"})) for category in (1, 2)}
        service = OperationsKnowledgeService(uow_factory=factory, documents=documents, model_catalogs=catalog)
    scope = {"domain_id": store.domain, "context": context}
    if store.tenant is not None:
        scope["tenant_id"] = store.tenant
    return SimpleNamespace(**locals())


def test_case_configuration_is_app_local_text_only_and_publication_gated():
    async def run():
        f = fixture()
        before = await f.service.get_case_collection_models(**f.scope)
        assert before["configuration_status"] == "MODELS_REQUIRED"
        assert before["missing_model_roles"] == sorted(CASE_MODEL_ROLES)
        configured = await f.service.configure_case_collection_models(**f.scope, models=f.models)
        assert configured["configuration_status"] == "CONFIGURED"
        assert configured["retrieval_requires_publication"]
        assert configured["profile_ref"] == "test-profile"
        assert configured["models"] == {k:str(v) for k,v in f.models.items()}
        assert "parser_ocr" not in configured["models"]
        f.uow.commit.assert_awaited_once()
        params = {"models": f.models}
        params.update({"tenant_id": f.store.tenant} if f.store.tenant else {"context": f.context})
        with pytest.raises(AIOpsApplicationError):
            await f.service._validate_knowledge_base_models(**params)
    asyncio.run(run())


@pytest.mark.parametrize("dimension", ["domain", "tenant"])
def test_case_configuration_rejects_scope_mismatch_before_repository_read(dimension):
    async def run():
        f = fixture()
        if dimension == "tenant" and not f.store.tenant:
            return
        scope = {**f.scope, dimension + "_id": uuid7()}
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service.get_case_collection_models(**scope)
        assert error.value.status_code == 403
        f.repository.case_configuration.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["missing_role", "extra_role", "unknown_field"])
def test_case_model_command_rejects_implicit_or_unrelated_roles(mode):
    f = fixture()
    payload = {"models": f.models}
    if mode == "missing_role":
        payload["models"].pop("embedding")
    elif mode == "extra_role":
        payload["models"]["parser_ocr"] = uuid7()
    else:
        payload["knowledge_base_id"] = uuid7()
    with pytest.raises(ValidationError):
        CaseCollectionModels.model_validate(payload)


def test_unavailable_model_is_rejected_before_app_configuration_write():
    async def run():
        f = fixture()
        if f.store.tenant:
            f.catalog.list_tenant_models.return_value = []
        else:
            f.catalog[2].get_model.side_effect = None
            f.catalog[2].get_model.return_value = {"category":1, "status":"ACTIVE"}
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service.configure_case_collection_models(**f.scope, models=f.models)
        assert error.value.code == "AIOPS_KNOWLEDGE_MODEL_UNAVAILABLE"
        f.repository.configure_cases.assert_not_awaited()
    asyncio.run(run())


def test_case_configuration_internal_http_requires_manage_scope_and_validates_roles():
    async def run():
        f = fixture()
        path = "/internal/v1/aiops/operations-knowledge/case-collection/models"
        for scopes in [("aiops.run",), ("aiops.manage",)]:
            app = api(f.store, scopes=scopes)
            app.include_router(internal.router)
            app.state.operations_knowledge_service = f.service
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
                read = await client.get(path)
                assert read.status_code == (200 if "aiops.manage" in scopes else 403), read.text
                if "aiops.manage" not in scopes:
                    continue
                missing = await client.put(path, json={"models": {"embedding": str(f.models["embedding"])}})
                assert missing.status_code == 422
                f.repository.configure_cases.assert_not_awaited()
                write = await client.put(path, json=CaseCollectionModels(models=f.models).model_dump(mode="json"))
                assert write.status_code == 200, write.text
                assert write.json()["owner_app_id"] == "aiops"
    asyncio.run(run())


def test_case_configuration_bff_and_narrow_client_preserve_identity_and_role_mapping():
    async def run():
        f = fixture()
        module = importlib.import_module("main_api.api." + ("ops" if f.store.tenant else "aiops_app"))
        client = SimpleNamespace(get_case_collection_models=AsyncMock(return_value={"models": {}}),
            configure_case_collection_models=AsyncMock(return_value={"configuration_status": "CONFIGURED"}))
        request = SimpleNamespace(state=SimpleNamespace(auth_context=f.context),
            app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)))
        if f.store.tenant is None:
            with patch.object(module, "_require", new=AsyncMock()) as permission:
                await module.get_case_collection_models(request)
                await module.configure_case_collection_models(CaseCollectionModels(models=f.models), request)
                assert permission.await_count == 2
                assert all(call.args[1] == "aiops:knowledge_manage" for call in permission.await_args_list)
        else:
            await module.get_case_collection_models(request)
            await module.configure_case_collection_models(CaseCollectionModels(models=f.models), request)
            from platform_core.authorization.registered_routes import AIOPS_ROUTE_PERMISSIONS
            route = "/api/v1/apps/aiops/domains/{domain_code}/ops/operations-knowledge/case-collection/models"
            assert ("GET", route, "aiops:knowledge_manage") in AIOPS_ROUTE_PERMISSIONS
            assert ("PUT", route, "aiops:knowledge_manage") in AIOPS_ROUTE_PERMISSIONS
        assert client.configure_case_collection_models.await_args.kwargs == {
            "models": f.models, "auth_context": f.context}
        narrow = object.__new__(AIOpsManagementClient)
        narrow._json = AsyncMock(return_value={})
        await narrow.get_case_collection_models(auth_context=f.context)
        assert narrow._json.await_args.args == ("GET",
            "/internal/v1/aiops/operations-knowledge/case-collection/models")
        await narrow.configure_case_collection_models(models=f.models, auth_context=f.context)
        call = narrow._json.await_args
        assert call.args[0] == "PUT" and call.kwargs["auth_context"] is f.context
        assert call.kwargs["payload"] == {"models": {key: str(value) for key, value in f.models.items()}}
    asyncio.run(run())
