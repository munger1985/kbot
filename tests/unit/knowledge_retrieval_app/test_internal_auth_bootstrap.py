"""知识检索应用内部身份验证器启动测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from knowledge_retrieval_app.entrypoints import api


class InternalAuthBootstrapTest(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_initializes_internal_auth_codecs(self) -> None:
        database_runtime = SimpleNamespace(
            session_factory=Mock(),
            close=AsyncMock(),
        )
        auth_context_codec = object()
        service_identity_codec = object()

        with (
            patch.object(
                api,
                "create_database_runtime",
                return_value=database_runtime,
            ),
            patch.object(api, "create_knowledge_retrieval_app_uow"),
            patch.object(api, "LogManager"),
            patch.object(
                api,
                "create_auth_context_codec",
                return_value=auth_context_codec,
            ),
            patch.object(
                api,
                "create_service_identity_codec",
                return_value=service_identity_codec,
            ),
        ):
            async with api.lifespan(api.app):
                self.assertIs(
                    auth_context_codec,
                    api.app.state.auth_context_codec,
                )
                self.assertIs(
                    service_identity_codec,
                    api.app.state.service_identity_codec,
                )

        database_runtime.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
