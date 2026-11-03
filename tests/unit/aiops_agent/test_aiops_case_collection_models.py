"""案例固定集合的显式配置及权限；复用 KC，不读取手册绑定。"""

import asyncio
import importlib
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from aiops_agent.api import operations_knowledge as internal
from aiops_agent.application.case_collection import case_collection_configuration
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from platform_clients.aiops import AIOpsManagementClient
from platform_clients.knowledge_core import KnowledgeCoreClientError
from platform_core.contracts.aiops.case_collection import CASE_MODEL_ROLES, CaseCollectionModels
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_output_job_api import api, human_context
from tests.unit.aiops_agent.test_aiops_output_jobs import Store


def fixture():
    store = Store()
    context = human_context(store)
    models = {role: uuid7() for role in CASE_MODEL_ROLES}
    collection_id, manual_id = uuid7(), uuid7()
    collection = {
        "collection_id": str(collection_id), "display_name": "diagnosis-cases",
        "status": "ACTIVE", "row_version": 8, "models": {},
        "metadata": {"owner_app_id": "aiops", "fixed_resource": True,
            "domain_id": str(store.domain)},
        "source_system": "aiops", "external_key": f"diagnosis-cases:{store.domain}",
        "parse_policy": {"parse_strategy": "AUTO"},
    }
    manual = {**deepcopy(collection), "collection_id": str(manual_id),
        "display_name": "operations-manuals", "external_key": f"operations-manuals:{store.domain}",
        "models": {"parser_ocr": str(uuid7()), "embedding": str(uuid7())}}
    kc = SimpleNamespace(list_collections=AsyncMock(return_value={"collections": [manual, collection]}),
        get_collection=AsyncMock(side_effect=lambda **kwargs: deepcopy(collection)),
        update_collection_models=AsyncMock(side_effect=lambda **kwargs:
            {**deepcopy(collection), "models": kwargs["payload"]["models"]}))
    categories = {str(model): (2 if role == "embedding" else 1) for role, model in models.items()}
    if store.tenant is not None:
        catalog = SimpleNamespace(list_tenant_models=AsyncMock(return_value=[
            {"model_id": key, "category": category, "status": 1,
                "entitled": True, "tenant_status": "ACTIVE", "tenant_configured": True}
            for key, category in categories.items()]))
        service = OperationsKnowledgeService(uow_factory=None, knowledge_core=kc, model_catalog=catalog)
    else:
        catalog = {category: SimpleNamespace(get_model=AsyncMock(
            side_effect=lambda model_id, **kwargs: {"model_id": str(model_id),
                "category": categories[str(model_id)], "status": "ACTIVE"}))
            for category in (1, 2)}
        service = OperationsKnowledgeService(uow_factory=None, knowledge_core=kc, model_catalogs=catalog)
    scope = {"domain_id": store.domain, "context": context}
    if store.tenant is not None:
        scope["tenant_id"] = store.tenant
    return SimpleNamespace(**locals())


def test_case_configuration_has_own_roles_and_preserves_manual_ocr_requirement():
    async def run():
        f = fixture()
        before = await f.service.get_case_collection_models(**f.scope)
        assert before["configuration_status"] == "MODELS_REQUIRED"
        assert before["missing_model_roles"] == sorted(CASE_MODEL_ROLES)
        configured = await f.service.configure_case_collection_models(**f.scope, models=f.models)
        assert configured["configuration_status"] == "CONFIGURED"
        assert configured["retrieval_requires_publication"] is True
        call = f.kc.update_collection_models.await_args.kwargs
        assert call["collection_id"] == f.collection_id and call["collection_id"] != f.manual_id
        assert call["payload"]["models"] == {key: str(value) for key, value in f.models.items()}
        assert "parser_ocr" not in call["payload"]["models"]
        if f.store.tenant is None:
            assert call["payload"]["expected_row_version"] == 8
        assert f.manual["models"].keys() == {"parser_ocr", "embedding"}
        params = {"models": f.models}
        params.update({"tenant_id": f.store.tenant} if f.store.tenant else {"context": f.context})
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service._validate_knowledge_base_models(**params)
        assert error.value.code == "AIOPS_KNOWLEDGE_COLLECTION_MODELS_REQUIRED"
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["missing", "duplicate", "owner", "inactive", "unavailable"])
def test_invalid_case_resource_does_not_fall_back_to_manual_or_modify_any_models(mode):
    async def run():
        f = fixture()
        if mode == "missing":
            f.kc.list_collections.return_value["collections"].remove(f.collection)
        elif mode == "duplicate":
            f.kc.list_collections.return_value["collections"].append(deepcopy(f.collection))
        elif mode == "owner":
            f.collection["metadata"]["owner_app_id"] = "another-app"
        elif mode == "inactive":
            f.collection["status"] = "DISABLED"
        else:
            f.kc.list_collections.side_effect = KnowledgeCoreClientError(
                status_code=503, code="KC_UNAVAILABLE", message="不可用")
        with pytest.raises(AIOpsApplicationError):
            await f.service.configure_case_collection_models(**f.scope, models=f.models)
        f.kc.update_collection_models.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("dimension", ["domain", "tenant"])
def test_case_configuration_rejects_scope_mismatch_before_external_calls(dimension):
    async def run():
        f = fixture()
        if dimension == "tenant" and not f.store.tenant:
            return
        scope = dict(f.scope)
        scope[dimension + "_id"] = uuid7()
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service.get_case_collection_models(**scope)
        assert error.value.status_code == 403
        f.kc.list_collections.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("strategy", ["VISUAL", "UNKNOWN"])
def test_case_configuration_diagnoses_incompatible_policy_without_overriding_it(strategy):
    async def run():
        f = fixture()
        f.collection["parse_policy"]["parse_strategy"] = strategy
        result = await f.service.get_case_collection_models(**f.scope)
        assert result["configuration_status"] == "TEXT_POLICY_REQUIRED"
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service.configure_case_collection_models(**f.scope, models=f.models)
        assert error.value.code == "AIOPS_CASE_TEXT_POLICY_REQUIRED"
        f.kc.update_collection_models.assert_not_awaited()
        assert f.collection["parse_policy"]["parse_strategy"] == strategy
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["missing_role", "extra_role", "unknown_field"])
def test_case_model_command_does_not_accept_implicit_or_visual_models(mode):
    f = fixture()
    payload = {"models": f.models}
    if mode == "missing_role":
        payload["models"].pop("embedding")
    elif mode == "extra_role":
        payload["models"]["parser_ocr"] = uuid7()
    else:
        payload["knowledge_base_id"] = f.manual_id
    with pytest.raises(ValidationError):
        CaseCollectionModels.model_validate(payload)


def test_case_model_catalog_unavailability_is_checked_before_kc_write():
    async def run():
        f = fixture()
        if f.store.tenant:
            f.catalog.list_tenant_models.return_value = []
        else:
            f.catalog[2].get_model.return_value = {"category": 1, "status": "ACTIVE"}
            f.catalog[2].get_model.side_effect = None
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service.configure_case_collection_models(**f.scope, models=f.models)
        assert error.value.code == "AIOPS_KNOWLEDGE_MODEL_UNAVAILABLE"
        f.kc.update_collection_models.assert_not_awaited()
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
                f.kc.update_collection_models.assert_not_awaited()
                write = await client.put(path, json=CaseCollectionModels(models=f.models).model_dump(mode="json"))
                assert write.status_code == 200, write.text
                assert write.json()["collection_id"] == str(f.collection_id)
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


def test_configuration_read_does_not_claim_model_health_or_publication():
    value = case_collection_configuration({"collection_id": str(uuid7()),
        "models": {role: str(uuid7()) for role in CASE_MODEL_ROLES}})
    assert value["configuration_status"] == "CONFIGURED"
    assert "ready" not in value and "published" not in value


@pytest.mark.parametrize("status", [409, 422, 503])
def test_case_model_configuration_keeps_kc_immutability_and_version_guards(status):
    async def run():
        f = fixture()
        f.kc.update_collection_models.side_effect = KnowledgeCoreClientError(
            status_code=status, code="KC_MODEL_CHANGE_REJECTED",
            message="内部模型目录细节不进入案例配置错误")
        with pytest.raises(AIOpsApplicationError) as error:
            await f.service.configure_case_collection_models(**f.scope, models=f.models)
        assert error.value.code == "AIOPS_CASE_MODEL_CONFIG_FAILED"
        assert error.value.status_code == status
        assert error.value.retryable == (status >= 500)
        assert "内部模型目录" not in error.value.message
        assert f.collection["models"] == {} and f.manual["models"]
        f.kc.update_collection_models.assert_awaited_once()
    asyncio.run(run())
