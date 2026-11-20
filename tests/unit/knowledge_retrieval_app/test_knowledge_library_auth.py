"""资料库入口区分服务调用 scope 与用户业务身份。"""

import unittest
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from knowledge_retrieval_app.api.knowledge_library import create_library_router
from knowledge_retrieval_app.domain.knowledge_retrieval import RetrievalAccess
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.security import (
    build_scoped_internal_auth_headers,
    create_scoped_internal_auth_middleware,
)
from platform_core.security.auth_context import AuthContextJWTCodec
from platform_core.security.service_identity import ServiceIdentityJWTCodec


class KnowledgeLibraryAuthTest(unittest.TestCase):
    audience = "kbot-knowledge-retrieval-app"
    caller = "kbot-main-api"
    path = "/internal/v1/knowledge-retrieval/knowledge/libraries"
    library_scope = "knowledge_retrieval.manage"

    def setUp(self) -> None:
        self.auth_codec = AuthContextJWTCodec(secret="a" * 32, issuer="test")
        self.identity_codec = ServiceIdentityJWTCodec(secret="b" * 32, issuer="test")
        self.context = AuthContext(
            principal_kind=PrincipalKind.PORTAL,
            client_id="user-session",
            api_key_id="user-jwt",
            app_id="knowledge_retrieval",
            domain_id="17",
            asserted_user_id="test-user",
            request_id="test-request",
            trace_id="test-trace",
        )
        self.list_libraries = AsyncMock(return_value=[])
        app = FastAPI()
        app.state.knowledge_library_service = Mock(list=self.list_libraries)
        app.middleware("http")(create_scoped_internal_auth_middleware(
            audience=self.audience,
            allowed_callers={self.caller: frozenset({self.library_scope, "knowledge.retrieve"})},
            auth_context_codec=self.auth_codec,
            service_identity_codec=self.identity_codec,
        ))
        app.include_router(create_library_router())
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def headers(self, context: AuthContext, *, scope: str) -> dict[str, str]:
        return build_scoped_internal_auth_headers(
            audience=self.audience,
            caller_service=self.caller,
            scopes=(scope,),
            context=context,
            auth_context_codec=self.auth_codec,
            service_identity_codec=self.identity_codec,
        )

    def test_user_session_without_internal_scope_can_list_libraries(self) -> None:
        response = self.client.get(self.path, headers=self.headers(
            self.context, scope=self.library_scope,
        ))

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), [])
        self.list_libraries.assert_awaited_once_with(RetrievalAccess(17, None, "test-user"))

    def test_user_scope_cannot_replace_the_service_scope(self) -> None:
        context = self.context.model_copy(update={"scopes": (self.library_scope,)})
        response = self.client.get(self.path, headers=self.headers(
            context, scope="knowledge.retrieve",
        ))

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(response.json()["detail"]["code"], "KNOWLEDGE_LIBRARY_CONTEXT_REQUIRED")
        self.list_libraries.assert_not_awaited()

    def test_wrong_app_or_invalid_domain_does_not_reach_library_service(self) -> None:
        for update, error in (
            ({"app_id": "aiops"}, "KNOWLEDGE_LIBRARY_CONTEXT_REQUIRED"),
            ({"domain_id": "invalid"}, "KNOWLEDGE_LIBRARY_CONTEXT_INVALID"),
        ):
            with self.subTest(update=update):
                context = self.context.model_copy(update=update)
                response = self.client.get(self.path, headers=self.headers(
                    context, scope=self.library_scope,
                ))

                self.assertEqual(response.status_code, 403, response.text)
                self.assertEqual(response.json()["detail"]["code"], error)
                self.list_libraries.assert_not_awaited()
