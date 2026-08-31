"""AIOps 数据库 Prompt 注册表测试。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from aiops_agent.orchestration.diagnosis import AIOpsPromptRegistry
from platform_core.identity import uuid7
from platform_core.prompts import PromptIntegrityError, ResolvedPrompt


def _resolved_prompt(
    *,
    prompt_version_id=None,
    version: str = "1.1.0",
    sha256: str = "a" * 64,
    source: str = "DATABASE",
) -> ResolvedPrompt:
    return ResolvedPrompt(
        prompt_key="aiops_agent.investigation_planner",
        version=version,
        sha256=sha256,
        content="只输出调查计划。",
        input_variables=(),
        output_schema="InvestigationPlanningOutput",
        source=source,
        prompt_version_id=prompt_version_id,
    )


class AIOpsPromptRegistryTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_active_database_version_and_builds_snapshot(
        self,
    ) -> None:
        version_id = uuid7()
        resolver = AsyncMock()
        resolver.resolve.return_value = _resolved_prompt(
            prompt_version_id=version_id
        )
        registry = AIOpsPromptRegistry(resolver)

        snapshot = await registry.snapshot(("investigation_planner",))

        resolver.resolve.assert_awaited_once_with(
            "aiops_agent.investigation_planner",
            prompt_version_id=None,
        )
        self.assertEqual(
            {
                "prompt_id": "aiops_agent.investigation_planner",
                "prompt_version": "1.1.0",
                "prompt_sha256": "a" * 64,
                "prompt_version_id": str(version_id),
                "prompt_source": "DATABASE",
            },
            snapshot["investigation_planner"],
        )

    async def test_frozen_snapshot_replays_exact_version_id(self) -> None:
        version_id = uuid7()
        resolver = AsyncMock()
        resolver.resolve.return_value = _resolved_prompt(
            prompt_version_id=version_id
        )
        registry = AIOpsPromptRegistry(resolver)
        frozen = {
            "investigation_planner": {
                "prompt_id": "aiops_agent.investigation_planner",
                "prompt_version": "1.1.0",
                "prompt_sha256": "a" * 64,
                "prompt_version_id": str(version_id),
                "prompt_source": "DATABASE",
            }
        }

        await registry.resolve(
            "investigation_planner", frozen_prompts=frozen
        )

        resolver.resolve.assert_awaited_once_with(
            "aiops_agent.investigation_planner",
            prompt_version_id=version_id,
        )

    async def test_frozen_snapshot_rejects_hash_drift(self) -> None:
        version_id = uuid7()
        resolver = AsyncMock()
        resolver.resolve.return_value = _resolved_prompt(
            prompt_version_id=version_id,
            sha256="b" * 64,
        )
        registry = AIOpsPromptRegistry(resolver)
        frozen = {
            "investigation_planner": {
                "prompt_id": "aiops_agent.investigation_planner",
                "prompt_version": "1.1.0",
                "prompt_sha256": "a" * 64,
                "prompt_version_id": str(version_id),
                "prompt_source": "DATABASE",
            }
        }

        with self.assertRaisesRegex(PromptIntegrityError, "已漂移"):
            await registry.resolve(
                "investigation_planner", frozen_prompts=frozen
            )

    async def test_rejects_non_database_prompt_source(self) -> None:
        resolver = AsyncMock()
        resolver.resolve.return_value = _resolved_prompt(
            prompt_version_id=uuid7(), source="FILE_FALLBACK"
        )
        registry = AIOpsPromptRegistry(resolver)

        with self.assertRaisesRegex(PromptIntegrityError, "必须来自数据库"):
            await registry.resolve("investigation_planner")


if __name__ == "__main__":
    unittest.main()
