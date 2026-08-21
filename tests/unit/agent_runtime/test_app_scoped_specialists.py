"""App 专属 Agent 实现分派的架构边界测试。"""

import unittest
from types import SimpleNamespace

from agent_runtime.specialists.app_scoped import AppScopedSkill, owner_app_id


class _Skill:
    def __init__(self, name: str) -> None:
        self.name = name

    async def execute(self, context):
        del context
        return self.name


class AppScopedSkillTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_owner_uses_dedicated_implementation(self):
        context = SimpleNamespace(config_snapshot={
            "agent": {"owner_app_id": "km_asset"},
        })
        skill = AppScopedSkill(
            default=_Skill("default"),
            implementations={"km_asset": _Skill("km")},
        )

        self.assertEqual("km_asset", owner_app_id(context))
        self.assertEqual("km", await skill.execute(context))

    async def test_unknown_owner_uses_generic_implementation(self):
        context = SimpleNamespace(config_snapshot={
            "agent": {"owner_app_id": "future_app"},
        })
        skill = AppScopedSkill(
            default=_Skill("default"),
            implementations={"km_asset": _Skill("km")},
        )

        self.assertEqual("default", await skill.execute(context))

    async def test_owner_cannot_be_overridden_outside_agent_snapshot(self):
        context = SimpleNamespace(config_snapshot={
            "owner_app_id": "km_asset",
            "agent": {"owner_app_id": "knowledge_retrieval"},
        })
        skill = AppScopedSkill(
            default=_Skill("default"),
            implementations={"km_asset": _Skill("km")},
        )

        self.assertEqual("knowledge_retrieval", owner_app_id(context))
        self.assertEqual("default", await skill.execute(context))


if __name__ == "__main__":
    unittest.main()
