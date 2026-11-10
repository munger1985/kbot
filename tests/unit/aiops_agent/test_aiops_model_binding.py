"""AIOps 模型由用户选择，环境不限制提供方，绑定与资源边界仍须有效。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from uuid import UUID

from platform_core.identity import uuid7
from platform_core.dictionary import LLMProvider
from platform_core.contracts.aiops import AIOpsCapability, AgentKnowledgePolicy

from aiops_agent.adapters.agent_catalog import AIOpsAgentValidator
from aiops_agent.application.agents import (
    AIOpsAgentError,
    AIOpsAgentService,
    CreateAIOpsAgentCommand,
    UpdateAIOpsAgentCommand,
)
from aiops_agent.application.errors import AIOpsApplicationError

from tests.unit.aiops_agent.test_aiops_agent_creation import _UnitOfWork, _database_row

ACTOR_ID = "kbotui_dev"
SCOPE = {"domain_id": 100}
RESOLUTION_SCOPE = SCOPE
MODEL_NAME_FIELD = "served_model_name"


def _unit_of_work():
    return _UnitOfWork(rows=(_database_row(),))


def _command(uow, *, models, status="ACTIVE"):
    return CreateAIOpsAgentCommand(
        **SCOPE, display_name="用户自主模型助手",
        allowed_business_system_ids=(uow.system.business_system_id,),
        capabilities=(AIOpsCapability.ACTIVE_DIAGNOSIS,),
        knowledge_policy=AgentKnowledgePolicy(enabled=False),
        models=models, status=status, actor_id=ACTOR_ID,
    )


ROLES = ("planner_llm", "diagnosis_llm", "composer_llm")
ENVIRONMENTS = ("development", "prod", "production", "live", "PROD")


def _models():
    return {role: uuid7() for role in ROLES}


def _model_client(providers=None):
    client = AsyncMock()
    client.get_model.side_effect = lambda model_id: {
        "provider": (providers or {}).get(model_id, "chatgpt"),
        MODEL_NAME_FIELD: f"selected-{model_id}",
        "model_id": str(model_id),
    }
    return client


class ModelBindingServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_each_environment_accepts_each_configured_provider(self):
        for environment in ENVIRONMENTS:
            for provider in LLMProvider:
                with self.subTest(environment=environment, provider=provider.value):
                    uow = _unit_of_work()
                    models = _models()
                    client = _model_client({model_id: provider.value for model_id in models.values()})
                    service = AIOpsAgentService(
                        uow_factory=lambda: uow, environment=environment, model_client=client,
                    )
                    result = await service.create(_command(uow, models=models))
                    self.assertEqual("ACTIVE", result["status"])
                    for role, model_id in models.items():
                        self.assertEqual(str(model_id), str(result["models"][role]))
                    if environment != "development":
                        self.assertEqual(list(models.values()), [call.args[0] for call in client.get_model.await_args_list])

    async def test_production_mixed_providers_survive_create_enable_and_update(self):
        for initial_status in ("DRAFT", "ACTIVE"):
            with self.subTest(initial_status=initial_status):
                uow = _unit_of_work()
                models = _models()
                providers = dict(zip(models.values(), ("chatgpt", "local_deepseek", "api_qwen")))
                client = _model_client(providers)
                service = AIOpsAgentService(
                    uow_factory=lambda: uow, environment="production", model_client=client,
                )
                created = await service.create(_command(uow, models=models, status=initial_status))
                updated = await service.update(UpdateAIOpsAgentCommand(
                    **SCOPE, agent_id=created["agent_id"],
                    expected_row_version=created["row_version"], actor_id=ACTOR_ID,
                    status="ACTIVE", models=models,
                ))
                self.assertEqual("ACTIVE", updated["status"])
                self.assertEqual(2, updated["version_no"])
                for role, model_id in models.items():
                    self.assertEqual(str(model_id), str(updated["models"][role]))
                resolver = AIOpsAgentValidator(service, model_client=client)
                for role in ("planner_llm", "diagnosis_llm"):
                    result = await getattr(resolver, f"resolve_{role.removesuffix('_llm')}_model")(
                        **RESOLUTION_SCOPE, agent_id=UUID(str(created["agent_id"])), trace_id="mixed-provider",
                    )
                    self.assertEqual(f"selected-{models[role]}", result["technical_name"])
                    self.assertEqual(str(models[role]), result["revision"])

    async def test_draft_and_development_do_not_require_directory(self):
        for environment, status in (("production", "DRAFT"), ("development", "ACTIVE")):
            with self.subTest(environment=environment, status=status):
                uow = _unit_of_work()
                service = AIOpsAgentService(uow_factory=lambda: uow, environment=environment)
                result = await service.create(_command(uow, models=_models(), status=status))
                self.assertEqual(status, result["status"])

    async def test_production_without_directory_remains_unavailable(self):
        uow = _unit_of_work()
        service = AIOpsAgentService(uow_factory=lambda: uow, environment="production")
        with self.assertRaises(AIOpsAgentError) as caught:
            await service.create(_command(uow, models=_models()))
        self.assertEqual("AIOPS_AGENT_MODEL_DIRECTORY_UNAVAILABLE", caught.exception.code)
        self.assertEqual(503, caught.exception.status_code)
        self.assertEqual(0, uow.commit_count)

    async def test_production_directory_failure_does_not_fallback(self):
        for error in (LookupError("missing"), RuntimeError("unavailable")):
            with self.subTest(error=type(error).__name__):
                uow = _unit_of_work()
                client = _model_client()
                client.get_model.side_effect = error
                service = AIOpsAgentService(
                    uow_factory=lambda: uow, environment="production", model_client=client,
                )
                with self.assertRaises(AIOpsAgentError) as caught:
                    await service.create(_command(uow, models=_models()))
                self.assertEqual("AIOPS_AGENT_MODEL_DIRECTORY_UNAVAILABLE", caught.exception.code)
                self.assertEqual(503, caught.exception.status_code)
                self.assertEqual(0, uow.commit_count)

    async def test_production_still_requires_every_runtime_role(self):
        for role in ROLES:
            with self.subTest(role=role):
                uow = _unit_of_work()
                models = _models()
                models.pop(role)
                service = AIOpsAgentService(
                    uow_factory=lambda: uow, environment="production", model_client=_model_client(),
                )
                with self.assertRaises(AIOpsAgentError) as caught:
                    await service.create(_command(uow, models=models))
                self.assertEqual(422, caught.exception.status_code)
                self.assertIn(role.removesuffix("_llm").upper(), caught.exception.code)
                self.assertEqual(0, uow.commit_count)

    async def test_production_rejects_unresolvable_technical_name_for_each_role(self):
        for role in ROLES:
            with self.subTest(role=role):
                uow = _unit_of_work()
                models = _models()
                client = _model_client()
                client.get_model.side_effect = lambda model_id: {
                    "provider": "chatgpt",
                    MODEL_NAME_FIELD: " " if model_id == models[role] else "available-model",
                }
                service = AIOpsAgentService(
                    uow_factory=lambda: uow, environment="production", model_client=client,
                )
                with self.assertRaises(AIOpsAgentError) as caught:
                    await service.create(_command(uow, models=models))
                self.assertEqual("AIOPS_AGENT_MODEL_BINDING_INVALID", caught.exception.code)
                self.assertEqual(422, caught.exception.status_code)
                self.assertEqual(0, uow.commit_count)


class ModelResolveTest(unittest.IsolatedAsyncioTestCase):
    def _resolver(self, *, provider="chatgpt", status="ACTIVE", models=None):
        self.agent_id = uuid7()
        self.models = _models() if models is None else models
        self.service = AsyncMock()
        self.service.get.return_value = {
            "agent_id": str(self.agent_id), "status": status,
            "models": {role: str(model_id) for role, model_id in self.models.items()},
        }
        self.client = _model_client({model_id: provider for model_id in self.models.values()})
        return AIOpsAgentValidator(self.service, model_client=self.client)

    async def _resolve(self, resolver, role):
        return await getattr(resolver, f"resolve_{role}_model")(
            **RESOLUTION_SCOPE, agent_id=self.agent_id, trace_id="model-choice",
        )

    async def test_runtime_uses_selected_model_for_every_provider(self):
        for provider in LLMProvider:
            for role in ("planner", "diagnosis"):
                with self.subTest(provider=provider.value, role=role):
                    resolver = self._resolver(provider=provider.value)
                    result = await self._resolve(resolver, role)
                    model_id = self.models[f"{role}_llm"]
                    self.assertEqual(f"selected-{model_id}", result["technical_name"])
                    self.assertEqual(str(model_id), result["revision"])
                    self.client.get_model.assert_awaited_once_with(model_id)
                    if "tenant_id" in RESOLUTION_SCOPE:
                        self.assertEqual(str(RESOLUTION_SCOPE["tenant_id"]), result["tenant_id"])
                        self.assertEqual(str(model_id), result["model_id"])

    async def test_runtime_still_rejects_missing_role(self):
        for role in ("planner", "diagnosis"):
            with self.subTest(role=role):
                resolver = self._resolver(models={})
                with self.assertRaises(AIOpsApplicationError) as caught:
                    await self._resolve(resolver, role)
                self.assertEqual(422, caught.exception.status_code)
                self.client.get_model.assert_not_awaited()

    async def test_runtime_still_requires_active_agent(self):
        for status in ("DRAFT", "DISABLED", "ARCHIVED"):
            with self.subTest(status=status):
                resolver = self._resolver(status=status)
                with self.assertRaises(AIOpsApplicationError) as caught:
                    await self._resolve(resolver, "diagnosis")
                self.assertEqual(422, caught.exception.status_code)
                self.client.get_model.assert_not_awaited()

    async def test_runtime_still_respects_domain_scope(self):
        resolver = self._resolver()
        self.service.get.side_effect = AIOpsAgentError("AIOPS_AGENT_NOT_FOUND", "不在当前空间", status_code=404)
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self._resolve(resolver, "diagnosis")
        self.assertEqual(422, caught.exception.status_code)
        self.service.get.assert_awaited_once_with(**SCOPE, agent_id=self.agent_id)
        self.client.get_model.assert_not_awaited()

    async def test_runtime_directory_errors_never_switch_model(self):
        for error in (LookupError("missing"), RuntimeError("unavailable")):
            with self.subTest(error=type(error).__name__):
                resolver = self._resolver()
                self.client.get_model.side_effect = error
                with self.assertRaises(AIOpsApplicationError) as caught:
                    await self._resolve(resolver, "planner")
                self.assertEqual(503, caught.exception.status_code)
                self.client.get_model.assert_awaited_once_with(self.models["planner_llm"])

    async def test_runtime_still_requires_directory(self):
        self._resolver()
        resolver = AIOpsAgentValidator(self.service)
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self._resolve(resolver, "diagnosis")
        self.assertEqual(503, caught.exception.status_code)
        self.service.get.assert_not_awaited()

    async def test_runtime_still_requires_technical_name(self):
        resolver = self._resolver()
        self.client.get_model.side_effect = None
        self.client.get_model.return_value = {"provider": "chatgpt", MODEL_NAME_FIELD: " "}
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self._resolve(resolver, "diagnosis")
        self.assertEqual(422, caught.exception.status_code)
        self.assertIn(MODEL_NAME_FIELD, caught.exception.message)


if __name__ == "__main__":
    unittest.main()
