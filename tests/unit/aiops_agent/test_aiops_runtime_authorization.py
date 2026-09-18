"""AIOps Runtime 资源级机器白名单测试。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from uuid import UUID

from fastapi import HTTPException
from starlette.requests import Request

from aiops_agent.api.runtime.routes import get_pending_input
from platform_core.contracts import AuthContext, PrincipalKind, ServiceIdentity


ALLOWED_AGENT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
OTHER_AGENT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a002")
RUN_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a003")


class _RuntimeService:
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.pending_input_called = False

    async def get_run(self, *, ops_run_id, domain_id):
        return SimpleNamespace(
            ops_run_id=ops_run_id,
            domain_id=domain_id,
            agent_id=self.agent_id,
        )

    async def get_pending_input(self, **kwargs):
        self.pending_input_called = True
        return kwargs


def _request() -> Request:
    request = Request({
        "type": "http",
        "method": "GET",
        "path": f"/internal/v1/aiops/runs/{RUN_ID}/pending-input",
        "headers": [],
    })
    now = datetime.now(timezone.utc)
    request.state.service_identity = ServiceIdentity(
        issuer="kbot",
        subject="kbot-main-api",
        audience="aiops-agent",
        scopes=("aiops.hitl",),
        issued_at=now,
        expires_at=now + timedelta(minutes=5),
        token_id=RUN_ID,
    )
    return request


def _context() -> AuthContext:
    return AuthContext(
        principal_kind=PrincipalKind.APP_API_CLIENT,
        client_id="api-client",
        api_key_id="credential",
        app_id="aiops",
        domain_id="100",
        asserted_user_id="aiops-service",
        authorized_agent_ids=(ALLOWED_AGENT_ID,),
        request_id="request",
        trace_id="trace",
    )


class AIOpsRuntimeAuthorizationTest(unittest.IsolatedAsyncioTestCase):
    async def test_pending_input_rejects_agent_outside_client_allowlist(self):
        service = _RuntimeService(OTHER_AGENT_ID)

        with self.assertRaises(HTTPException) as rejected:
            await get_pending_input(
                run_id=RUN_ID,
                request=_request(),
                service=service,
                context=_context(),
            )

        self.assertEqual(404, rejected.exception.status_code)
        self.assertFalse(service.pending_input_called)

    async def test_pending_input_allows_bound_agent(self):
        service = _RuntimeService(ALLOWED_AGENT_ID)

        await get_pending_input(
            run_id=RUN_ID,
            request=_request(),
            service=service,
            context=_context(),
        )

        self.assertTrue(service.pending_input_called)


if __name__ == "__main__":
    unittest.main()
