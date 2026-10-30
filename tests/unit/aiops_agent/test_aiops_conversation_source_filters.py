"""来源会话筛选贯穿 API、客户端、应用与隔离仓储的合同。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.api import conversations as internal_api
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.repositories.conversation import ConversationRepository
from main_api.api import aiops_app as public_api
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts import AuthContext, PrincipalKind


def _context():
    return AuthContext(
        principal_kind=PrincipalKind.PORTAL,
        client_id="aiops-source-test", api_key_id="isolated-test",
        request_id="source-filter-test", trace_id="source-filter-test",
        app_id="aiops", domain_id="42", asserted_user_id="dba",
    )


class ConversationSourceFiltersTest(unittest.IsolatedAsyncioTestCase):
    async def test_repository_keeps_domain_actor_archive_and_source_boundaries(self):
        agent_id, situation_id, run_id = uuid4(), uuid4(), uuid4()
        session = SimpleNamespace(scalars=AsyncMock(return_value=[]))
        repository = ConversationRepository(session)
        self.assertEqual([], await repository.list_conversations(
            domain_id=42, created_by="dba", agent_id=agent_id,
            situation_id=situation_id, run_id=run_id, limit=7,
        ))
        statement = session.scalars.await_args.args[0]
        for dialect in (oracle.dialect(), postgresql.dialect()):
            with self.subTest(dialect=dialect.name):
                compiled = statement.compile(dialect=dialect)
                sql = str(compiled).lower()
                self.assertIn("left outer join", sql)
                self.assertIn("source_situation_id =", sql)
                self.assertIn("or", sql)
                self.assertIn("situation_id =", sql)
                self.assertIn("source_run_id =", sql)
                self.assertIn("created_by =", sql)
                self.assertIn("status !=", sql)
                self.assertIn("created_at desc", sql)
                self.assertIn("conversation_id desc", sql)
                self.assertNotIn("updated_at desc", sql)
                self.assertGreaterEqual(sql.count("domain_id ="), 2)
                for value in (42, "dba", "ARCHIVED", agent_id, situation_id, run_id, 7):
                    self.assertIn(value, compiled.params.values())

    async def test_repository_without_situation_does_not_join_run(self):
        session = SimpleNamespace(scalars=AsyncMock(return_value=[]))
        await ConversationRepository(session).list_conversations(
            domain_id=42, created_by="dba", run_id=uuid4(),
        )
        sql = str(session.scalars.await_args.args[0]).lower()
        self.assertNotIn("join", sql)
        self.assertIn("source_run_id =", sql)

    async def test_application_passes_both_source_filters_with_actor(self):
        repository = SimpleNamespace(list_conversations=AsyncMock(return_value=[]))
        uow = AsyncMock()
        uow.__aenter__.return_value = SimpleNamespace(conversations=repository)
        service = ConversationTurnService(uow_factory=lambda: uow)
        situation_id, run_id = uuid4(), uuid4()
        self.assertEqual([], await service.list_conversations(
            domain_id=42, actor_id="dba", situation_id=situation_id,
            run_id=run_id, limit=7,
        ))
        repository.list_conversations.assert_awaited_once_with(
            domain_id=42, created_by="dba", agent_id=None,
            situation_id=situation_id, run_id=run_id, limit=7,
        )

    async def test_internal_api_uses_trusted_scope_with_source_filters(self):
        context = _context()
        service = SimpleNamespace(list_conversations=AsyncMock(return_value=[]))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            conversation_turn_service=service,
        )))
        situation_id, run_id = uuid4(), uuid4()
        with patch.object(internal_api, "_scope", return_value=(42, "dba")) as scope:
            self.assertEqual([], await internal_api.list_conversations(
                request, situation_id=situation_id, run_id=run_id,
                limit=7, context=context,
            ))
        scope.assert_called_once_with(request, context)
        service.list_conversations.assert_awaited_once_with(
            domain_id=42, actor_id="dba", agent_id=None,
            situation_id=situation_id, run_id=run_id, limit=7,
        )

    async def test_public_api_forwards_filters_after_authorization(self):
        context = _context()
        client = SimpleNamespace(list_conversations=AsyncMock(return_value=[]))
        request = SimpleNamespace(
            state=SimpleNamespace(auth_context=context),
            app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)),
        )
        situation_id, run_id = uuid4(), uuid4()
        with patch.object(public_api, "_require", new_callable=AsyncMock) as require:
            self.assertEqual([], await public_api.list_conversations(
                request, situation_id=situation_id, run_id=run_id, limit=7,
            ))
        require.assert_awaited_once_with(request, "aiops:use")
        client.list_conversations.assert_awaited_once_with(
            agent_id=None, situation_id=situation_id, run_id=run_id,
            limit=7, auth_context=context,
        )

    async def test_client_encodes_source_filters_without_losing_agent_or_auth(self):
        client = object.__new__(AIOpsManagementClient)
        client._json = AsyncMock(return_value=[])
        context = _context()
        agent_id, situation_id, run_id = uuid4(), uuid4(), uuid4()
        self.assertEqual([], await client.list_conversations(
            agent_id=agent_id, situation_id=situation_id, run_id=run_id,
            limit=7, auth_context=context,
        ))
        args, kwargs = client._json.await_args
        self.assertEqual("GET", args[0])
        self.assertEqual("/internal/v1/aiops/conversations", urlsplit(args[1]).path)
        self.assertEqual({
            "limit": ["7"], "agent_id": [str(agent_id)],
            "situation_id": [str(situation_id)], "run_id": [str(run_id)],
        }, parse_qs(urlsplit(args[1]).query))
        self.assertEqual({"auth_context": context}, kwargs)

    async def test_public_source_filter_does_not_expand_app_client_agent_scope(self):
        allowed_agent, other_agent = uuid4(), uuid4()
        context = _context().model_copy(update={
            "principal_kind": PrincipalKind.APP_API_CLIENT,
            "authorized_agent_ids": (allowed_agent,),
        })
        expected = {"conversation_id": "permitted", "agent_id": str(allowed_agent)}
        client = SimpleNamespace(list_conversations=AsyncMock(return_value=[
            expected, {"conversation_id": "outside", "agent_id": str(other_agent)},
        ]))
        request = SimpleNamespace(
            state=SimpleNamespace(auth_context=context),
            app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)),
        )
        with patch.object(public_api, "_require", new_callable=AsyncMock):
            result = await public_api.list_conversations(
                request, situation_id=uuid4(), run_id=uuid4(), limit=7,
            )
        self.assertEqual([expected], result)

    async def test_public_authorization_failure_never_queries_source_conversations(self):
        client = SimpleNamespace(list_conversations=AsyncMock())
        request = SimpleNamespace(
            state=SimpleNamespace(auth_context=_context()),
            app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)),
        )
        with patch.object(public_api, "_require", AsyncMock(side_effect=PermissionError("无权使用"))):
            with self.assertRaises(PermissionError):
                await public_api.list_conversations(request, situation_id=uuid4(), limit=7)
        client.list_conversations.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
