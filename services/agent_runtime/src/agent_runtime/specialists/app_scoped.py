"""按 Agent 所属 App 显式分派专属 Skill 实现。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from agent_runtime.runtime import ExecutionContext, SkillResult


def owner_app_id(context: ExecutionContext) -> str:
    """只从冻结 Agent Snapshot 读取所属 App。"""
    agent = context.config_snapshot.get("agent") or {}
    if not isinstance(agent, Mapping):
        return ""
    return str(agent.get("owner_app_id") or "").strip()


class AppScopedSkill:
    """让共享 Skill ID 在运行时选择显式注册的 App 专属实现。"""

    def __init__(
        self,
        *,
        default: Any,
        implementations: Mapping[str, Any] | None = None,
    ) -> None:
        self._default = default
        self._implementations = dict(implementations or {})

    def implementation_for(self, context: ExecutionContext) -> Any:
        """未注册专属实现时稳定回落到通用实现。"""
        return self._implementations.get(owner_app_id(context), self._default)

    async def execute(self, context: ExecutionContext) -> SkillResult:
        implementation = self.implementation_for(context)
        return await implementation.execute(context)

    async def execute_stream(
        self, context: ExecutionContext
    ) -> AsyncIterator[Any]:
        implementation = self.implementation_for(context)
        if hasattr(implementation, "execute_stream"):
            async for item in implementation.execute_stream(context):
                yield item
            return
        yield await implementation.execute(context)
