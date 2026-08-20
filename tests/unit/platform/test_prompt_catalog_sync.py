"""Prompt Catalog 数据库同步测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from platform_core.prompts.sync import sync_prompt_catalog


class _Result:
    def __init__(self, row=None):
        self._row = row

    def one_or_none(self):
        return self._row


class _Connection:
    def __init__(self, *, prompt_row=None, version_row=None):
        self.prompt_row = prompt_row
        self.version_row = version_row
        self.statements: list[tuple[str, dict | None]] = []
        self.committed = False

    async def execute(self, statement, parameters=None):
        sql = str(statement)
        self.statements.append((sql, parameters))
        if "SELECT prompt_id, active_version_id" in sql:
            return _Result(self.prompt_row)
        if "SELECT prompt_version_id, content_sha256, source" in sql:
            return _Result(self.version_row)
        return _Result()

    async def commit(self):
        self.committed = True


def _catalog(*, sha256: str = "a" * 64, version: str = "1.0.0"):
    entry = SimpleNamespace(
        prompt_key="agent_runtime.data_query_plan",
        owner_service="agent_runtime",
        version=version,
        active=True,
        purpose="生成问数计划",
        input_variables=(),
        output_schema="DataQueryPlan.v1",
        content="只输出 JSON。\n",
        sha256=sha256,
    )
    return SimpleNamespace(for_services=lambda services: (entry,))


class PromptCatalogSyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_new_version_is_inserted_and_activated(self):
        connection = _Connection()

        with patch(
            "platform_core.prompts.sync.load_prompt_catalog",
            return_value=_catalog(),
        ):
            count = await sync_prompt_catalog(
                connection,
                selected_services={"agent_runtime"},
                environment="development",
            )

        sql = "\n".join(statement for statement, _ in connection.statements)
        self.assertEqual(1, count)
        self.assertTrue(connection.committed)
        self.assertIn("INSERT INTO KBOT_PLATFORM_PROMPT (", sql)
        self.assertIn("INSERT INTO KBOT_PLATFORM_PROMPT_VERSION (", sql)
        self.assertIn("SET active_version_id = :prompt_version_id", sql)

    async def test_same_version_with_different_hash_is_rejected(self):
        connection = _Connection(
            prompt_row=(b"p" * 16, None),
            version_row=(b"v" * 16, "b" * 64, "FILE_SEED"),
        )

        with patch(
            "platform_core.prompts.sync.load_prompt_catalog",
            return_value=_catalog(sha256="a" * 64),
        ):
            with self.assertRaisesRegex(RuntimeError, "Hash 冲突"):
                await sync_prompt_catalog(
                    connection,
                    selected_services={"agent_runtime"},
                    environment="production",
                )

        self.assertFalse(connection.committed)

    async def test_development_updates_file_seed_in_place(self):
        connection = _Connection(
            prompt_row=(b"p" * 16, None),
            version_row=(b"v" * 16, "b" * 64, "FILE_SEED"),
        )

        with patch(
            "platform_core.prompts.sync.load_prompt_catalog",
            return_value=_catalog(sha256="a" * 64),
        ):
            count = await sync_prompt_catalog(
                connection,
                selected_services={"agent_runtime"},
                environment="development",
            )

        sql = "\n".join(statement for statement, _ in connection.statements)
        self.assertEqual(1, count)
        self.assertTrue(connection.committed)
        self.assertIn("SET content = :content", sql)
        self.assertNotIn("INSERT INTO KBOT_PLATFORM_PROMPT_VERSION (", sql)

    async def test_development_rejects_file_version_upgrade(self):
        connection = _Connection()

        with patch(
            "platform_core.prompts.sync.load_prompt_catalog",
            return_value=_catalog(version="1.0.1"),
        ):
            with self.assertRaisesRegex(RuntimeError, "必须固定为 1.0.0"):
                await sync_prompt_catalog(
                    connection,
                    selected_services={"agent_runtime"},
                    environment="development",
                )

        self.assertEqual([], connection.statements)
        self.assertFalse(connection.committed)

    async def test_development_does_not_overwrite_database_version(self):
        connection = _Connection(
            prompt_row=(b"p" * 16, None),
            version_row=(b"v" * 16, "b" * 64, "DATABASE"),
        )

        with patch(
            "platform_core.prompts.sync.load_prompt_catalog",
            return_value=_catalog(sha256="a" * 64),
        ):
            with self.assertRaisesRegex(RuntimeError, "不是 FILE_SEED"):
                await sync_prompt_catalog(
                    connection,
                    selected_services={"agent_runtime"},
                    environment="development",
                )

        self.assertFalse(connection.committed)


if __name__ == "__main__":
    unittest.main()
