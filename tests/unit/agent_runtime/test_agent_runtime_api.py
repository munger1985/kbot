"""Agent Runtime 内部 API 的身份派生与错误映射测试。"""

import unittest
from unittest.mock import AsyncMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from agent_runtime.api import internal_router, task_router
from agent_runtime.application import AgentRuntimeConflict
from platform_core.contracts import AgentRunReceipt, AuthContext, PrincipalKind
from platform_core.identity import uuid7


class AgentRuntimeApiTest(unittest.TestCase):
    def setUp(self):
        self.service = AsyncMock()
        self.run_id = uuid7()
        self.service.create_run.return_value = AgentRunReceipt(
            run_id=self.run_id,
            status="CREATED",
            event_cursor=1,
            events_url=f"/api/v1/runs/{self.run_id}/events",
        )
        app = FastAPI()
        app.state.agent_runtime_service = self.service
        app.state.platform_app_id = 7
        app.state.agent_runtime_budget = {"max_tasks": 16}

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.auth_context = AuthContext(
                principal_kind=PrincipalKind.PORTAL,
                client_id="portal",
                api_key_id="portal-key",
                domain_id="20",
                asserted_user_id="user-1",
                request_id="request-1",
                trace_id="trace-1",
            )
            return await call_next(request)

        app.include_router(internal_router)
        app.include_router(task_router)
        self.client = TestClient(app)

    def test_create_derives_scope_and_actor_from_auth_context(self):
        response = self.client.post(
            "/internal/v1/runs",
            headers={"Idempotency-Key": "create-1"},
            json={
                "agent_id": str(uuid7()),
                "input": "查询文档",
                "collection_ids": [],
                "security_level": 2,
                "client_metadata": {"channel": "portal"},
            },
        )

        self.assertEqual(response.status_code, 202)
        command = self.service.create_run.await_args.args[0]
        self.assertEqual(command.app_id, 7)
        self.assertEqual(command.domain_id, 20)
        self.assertEqual(command.actor_id, "user-1")
        self.assertEqual(command.request_id, "request-1")

    def test_idempotency_conflict_uses_stable_error_code(self):
        self.service.create_run.side_effect = AgentRuntimeConflict(
            "IDEMPOTENCY_CONFLICT", "请求内容不同"
        )

        response = self.client.post(
            "/internal/v1/runs",
            headers={"Idempotency-Key": "create-1"},
            json={
                "agent_id": str(uuid7()),
                "input": "查询文档",
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"]["code"], "IDEMPOTENCY_CONFLICT"
        )


if __name__ == "__main__":
    unittest.main()
