"""显式轮次通过公开入口、窄客户端与内部入口保真传递，不退回默认范围。"""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from aiops_agent.api import operations_knowledge as internal
from aiops_agent.entities import ReportSourceEntity
from platform_clients.aiops import AIOpsManagementClient
from platform_core.identity import uuid7


@pytest.mark.parametrize("selection", ["default", "explicit", "duplicate", "invalid", "too_many"])
def test_turn_selection_survives_http_and_client_boundaries(selection, monkeypatch):
    async def run():
        oracle = ReportSourceEntity.__tablename__.startswith("KBOT_")
        public = importlib.import_module("main_api.api.aiops_app" if oracle else "main_api.api.ops")
        ids = [uuid7(), uuid7()]
        values = {"default": [], "explicit": list(map(str, ids)), "duplicate": [str(ids[0])] * 2,
            "invalid": ["not-a-uuid"], "too_many": [str(ids[0])] * 201}[selection]
        context, conversation_id, domain = SimpleNamespace(), uuid7(), uuid7()
        service = SimpleNamespace(conversation_case_preview=AsyncMock(return_value={"eligible": True}))
        request = SimpleNamespace(state=SimpleNamespace(auth_context=context),
            app=SimpleNamespace(state=SimpleNamespace(operations_knowledge_service=service)))
        monkeypatch.setattr(internal, "_use_scope", lambda *_: (domain, None) if oracle else (uuid7(), domain, None))
        client = AIOpsManagementClient.__new__(AIOpsManagementClient)
        async def dispatch(method, path, *, auth_context):
            assert method == "GET" and auth_context is context
            query = parse_qs(urlsplit(path).query)
            selected = [type(ids[0])(value) for value in query["turn_ids"]] if "turn_ids" in query else None
            return await internal.conversation_case_candidate(conversation_id=conversation_id,
                request=request, turn_ids=selected, context=auth_context)
        client._json = AsyncMock(side_effect=dispatch)
        monkeypatch.setattr(public, "_client", lambda _: client)
        access = AsyncMock()
        monkeypatch.setattr(public, "_conversation_with_access", access)
        app = FastAPI()
        app.add_api_route("/candidate/{conversation_id}", public.get_conversation_case_candidate)
        @app.middleware("http")
        async def set_context(req, call_next):
            req.state.auth_context = context
            return await call_next(req)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            response = await http.get(f"/candidate/{conversation_id}", params=[("turn_ids", value) for value in values])
        if selection in {"invalid", "too_many"}:
            assert response.status_code == 422
            client._json.assert_not_awaited()
            service.conversation_case_preview.assert_not_awaited()
        else:
            assert response.status_code == 200
            access.assert_awaited_once()
            expected = tuple(type(ids[0])(value) for value in values) if values else None
            service.conversation_case_preview.assert_awaited_once_with(domain_id=domain,
                conversation_id=conversation_id, context=context, included_turn_ids=expected)
    asyncio.run(run())
