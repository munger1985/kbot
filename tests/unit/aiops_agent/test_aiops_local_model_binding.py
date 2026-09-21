"""AIOps 生产环境必须绑定本地 DeepSeek。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from platform_core.identity import uuid7

from aiops_agent.adapters.agent_catalog import AIOpsAgentValidator
from aiops_agent.application.agents import (
    AIOpsAgentError,
    AIOpsAgentService,
    CreateAIOpsAgentCommand,
)
from aiops_agent.application.diagnosis.local_models import (
    LocalModelBindingError,
    is_production_environment,
    require_production_local_llm,
)
from aiops_agent.application.errors import AIOpsApplicationError
from tests.unit.aiops_agent.test_aiops_agent_creation import (
    _AgentRepository,
    _UnitOfWork,
)


class LocalModelHelperTest(unittest.TestCase):
    def test_production_environment_names_match_settings(self):
        for name in ("prod", "production", "live", "PROD"):
            self.assertTrue(is_production_environment(name))
        for name in ("development", "dev", "test", None, ""):
            self.assertFalse(is_production_environment(name))

    def test_development_allows_cloud_provider(self):
        require_production_local_llm(
            environment="development",
            provider="chatgpt",
            role_label="诊断",
        )

    def test_production_accepts_local_deepseek(self):
        require_production_local_llm(
            environment="production",
            provider="local_deepseek",
            role_label="诊断",
        )

    def test_production_rejects_cloud_provider(self):
        with self.assertRaises(LocalModelBindingError) as raised:
            require_production_local_llm(
                environment="production",
                provider="chatgpt",
                role_label="规划",
            )
        self.assertEqual(
            "AIOPS_AGENT_PRODUCTION_LOCAL_MODEL_REQUIRED",
            raised.exception.code,
        )
        self.assertIn("本地部署的 DeepSeek", raised.exception.message)


class LocalModelBindingServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_development_active_agent_allows_any_bound_model(self):
        source_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        service = AIOpsAgentService(uow_factory=lambda: unit_of_work)

        result = await service.create(
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="开发诊断助手",
                diagnostic_source_ids=(source_id,),
                target_ids=(unit_of_work.target.target_id,),
                models={
                    "planner_llm": uuid7(),
                    "diagnosis_llm": uuid7(),
                },
                status="ACTIVE",
                actor_id="kbotui_dev",
            )
        )
        self.assertEqual("ACTIVE", result["status"])

    async def test_production_draft_does_not_require_local_model(self):
        source_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        service = AIOpsAgentService(
            uow_factory=lambda: unit_of_work,
            environment="production",
        )

        result = await service.create(
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="生产草稿",
                diagnostic_source_ids=(source_id,),
                target_ids=(unit_of_work.target.target_id,),
                models={
                    "planner_llm": uuid7(),
                    "diagnosis_llm": uuid7(),
                },
                status="DRAFT",
                actor_id="kbotui_dev",
            )
        )
        self.assertEqual("DRAFT", result["status"])

    async def test_production_active_without_directory_is_unavailable(self):
        source_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        service = AIOpsAgentService(
            uow_factory=lambda: unit_of_work,
            environment="production",
        )

        with self.assertRaises(AIOpsAgentError) as raised:
            await service.create(
                CreateAIOpsAgentCommand(
                    domain_id=100,
                    display_name="生产诊断助手",
                    diagnostic_source_ids=(source_id,),
                    target_ids=(unit_of_work.target.target_id,),
                    models={
                        "planner_llm": uuid7(),
                        "diagnosis_llm": uuid7(),
                    },
                    status="ACTIVE",
                    actor_id="kbotui_dev",
                )
            )
        self.assertEqual(
            "AIOPS_AGENT_MODEL_DIRECTORY_UNAVAILABLE",
            raised.exception.code,
        )
        self.assertEqual(503, raised.exception.status_code)

    async def test_production_active_rejects_cloud_provider(self):
        source_id = uuid7()
        planner_id = uuid7()
        diagnosis_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        model_client = AsyncMock()
        model_client.get_model.side_effect = lambda model_id: {
            "provider": "chatgpt",
            "served_model_name": "gpt-test",
            "model_id": str(model_id),
        }
        service = AIOpsAgentService(
            uow_factory=lambda: unit_of_work,
            environment="production",
            model_client=model_client,
        )

        with self.assertRaises(AIOpsAgentError) as raised:
            await service.create(
                CreateAIOpsAgentCommand(
                    domain_id=100,
                    display_name="生产诊断助手",
                    diagnostic_source_ids=(source_id,),
                    target_ids=(unit_of_work.target.target_id,),
                    models={
                        "planner_llm": planner_id,
                        "diagnosis_llm": diagnosis_id,
                    },
                    status="ACTIVE",
                    actor_id="kbotui_dev",
                )
            )
        self.assertEqual(
            "AIOPS_AGENT_PRODUCTION_LOCAL_MODEL_REQUIRED",
            raised.exception.code,
        )
        self.assertEqual(422, raised.exception.status_code)

    async def test_production_active_accepts_local_deepseek(self):
        source_id = uuid7()
        planner_id = uuid7()
        diagnosis_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        model_client = AsyncMock()
        model_client.get_model.side_effect = lambda model_id: {
            "provider": "local_deepseek",
            "served_model_name": "deepseek-v4",
            "model_id": str(model_id),
        }
        service = AIOpsAgentService(
            uow_factory=lambda: unit_of_work,
            environment="production",
            model_client=model_client,
        )

        result = await service.create(
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="生产诊断助手",
                diagnostic_source_ids=(source_id,),
                target_ids=(unit_of_work.target.target_id,),
                models={
                    "planner_llm": planner_id,
                    "diagnosis_llm": diagnosis_id,
                },
                status="ACTIVE",
                actor_id="kbotui_dev",
            )
        )
        self.assertEqual("ACTIVE", result["status"])


class LocalModelResolveTest(unittest.IsolatedAsyncioTestCase):
    async def test_development_resolve_allows_cloud_provider(self):
        agent_id = uuid7()
        model_id = uuid7()
        agent_service = AsyncMock()
        agent_service.get.return_value = {
            "agent_id": str(agent_id),
            "domain_id": 100,
            "status": "ACTIVE",
            "models": {"diagnosis_llm": str(model_id)},
        }
        model_client = AsyncMock()
        model_client.get_model.return_value = {
            "provider": "chatgpt",
            "served_model_name": "gpt-diagnosis",
        }
        resolver = AIOpsAgentValidator(
            agent_service,
            model_client=model_client,
            environment="development",
        )

        result = await resolver.resolve_diagnosis_model(
            agent_id=agent_id,
            domain_id=100,
            trace_id="trace-dev",
        )
        self.assertEqual("gpt-diagnosis", result["technical_name"])

    async def test_production_resolve_rejects_cloud_provider(self):
        agent_id = uuid7()
        model_id = uuid7()
        agent_service = AsyncMock()
        agent_service.get.return_value = {
            "agent_id": str(agent_id),
            "domain_id": 100,
            "status": "ACTIVE",
            "models": {"diagnosis_llm": str(model_id)},
        }
        model_client = AsyncMock()
        model_client.get_model.return_value = {
            "provider": "chatgpt",
            "served_model_name": "gpt-diagnosis",
        }
        resolver = AIOpsAgentValidator(
            agent_service,
            model_client=model_client,
            environment="production",
        )

        with self.assertRaises(AIOpsApplicationError) as caught:
            await resolver.resolve_diagnosis_model(
                agent_id=agent_id,
                domain_id=100,
                trace_id="trace-prod",
            )
        self.assertEqual(422, caught.exception.status_code)
        self.assertIn("本地部署的 DeepSeek", caught.exception.message)

    async def test_production_resolve_accepts_local_deepseek(self):
        agent_id = uuid7()
        model_id = uuid7()
        agent_service = AsyncMock()
        agent_service.get.return_value = {
            "agent_id": str(agent_id),
            "domain_id": 100,
            "status": "ACTIVE",
            "models": {"planner_llm": str(model_id)},
        }
        model_client = AsyncMock()
        model_client.get_model.return_value = {
            "provider": "local_deepseek",
            "served_model_name": "deepseek-v4",
        }
        resolver = AIOpsAgentValidator(
            agent_service,
            model_client=model_client,
            environment="live",
        )

        result = await resolver.resolve_planner_model(
            agent_id=agent_id,
            domain_id=100,
            trace_id="trace-local",
        )
        self.assertEqual("deepseek-v4", result["technical_name"])
        self.assertEqual(str(model_id), result["revision"])


if __name__ == "__main__":
    unittest.main()
