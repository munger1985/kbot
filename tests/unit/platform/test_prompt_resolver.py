"""数据库 Prompt Resolver 的失败和版本约束测试。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from platform_core.prompts import PromptNotFoundError, PromptResolver


class _SessionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *_args):
        return None


class PromptResolverTest(unittest.IsolatedAsyncioTestCase):
    async def test_database_failure_is_not_hidden_by_file_fallback(self) -> None:
        repository = AsyncMock()
        repository.get_active.side_effect = RuntimeError("database unavailable")
        resolver = PromptResolver(session_factory=lambda: _SessionContext())

        with patch(
            "platform_core.prompts.resolver.PlatformPromptRepository",
            return_value=repository,
        ):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await resolver.resolve("agent_runtime.intent_route")

    async def test_missing_database_prompt_fails_explicitly(self) -> None:
        repository = AsyncMock()
        repository.get_active.return_value = None
        resolver = PromptResolver(session_factory=lambda: _SessionContext())

        with patch(
            "platform_core.prompts.resolver.PlatformPromptRepository",
            return_value=repository,
        ):
            with self.assertRaisesRegex(
                PromptNotFoundError,
                "数据库 Prompt 不存在",
            ):
                await resolver.resolve("agent_runtime.intent_route")


if __name__ == "__main__":
    unittest.main()
