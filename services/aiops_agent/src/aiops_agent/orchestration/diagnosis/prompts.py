"""AIOps 数据库 Prompt 解析与运行版本冻结。"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from platform_core.prompts import PromptIntegrityError, ResolvedPrompt


PROMPT_KEYS = {
    "compact_planner": "aiops_agent.compact_planner",
    "investigation_planner": "aiops_agent.investigation_planner",
    "investigation_policy_repair": (
        "aiops_agent.investigation_policy_repair"
    ),
    "investigation_replanner": "aiops_agent.investigation_replanner",
    "investigation_assessor": "aiops_agent.investigation_assessor",
    "round_draft": "aiops_agent.round_draft",
    "round_assess": "aiops_agent.round_assess",
    "grounding_verify": "aiops_agent.grounding_verify",
    "answer_compose": "aiops_agent.answer_compose",
    "answer_stream": "aiops_agent.answer_stream",
}

DIAGNOSIS_PROMPT_IDS = (
    "round_draft",
    "round_assess",
    "grounding_verify",
)
TURN_PROMPT_IDS = (
    "compact_planner",
    "investigation_planner",
    "investigation_policy_repair",
    "investigation_replanner",
    "investigation_assessor",
    "answer_compose",
    "answer_stream",
)


@dataclass(frozen=True)
class PromptAsset:
    prompt_id: str
    version: str
    sha256: str
    content: str
    prompt_version_id: UUID
    source: str

    def ref(self) -> dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.version,
            "prompt_sha256": self.sha256,
            "prompt_version_id": str(self.prompt_version_id),
            "prompt_source": self.source,
        }

    def snapshot(self) -> dict[str, str]:
        return self.ref()


class AIOpsPromptRegistry:
    """只从平台数据库解析 Prompt，并校验冻结版本没有漂移。"""

    def __init__(self, resolver) -> None:
        self._resolver = resolver

    async def resolve(
        self,
        prompt_id: str,
        *,
        frozen_prompts: dict[str, dict[str, str]] | None = None,
    ) -> PromptAsset:
        try:
            prompt_key = PROMPT_KEYS[prompt_id]
        except KeyError as exc:
            raise LookupError(f"AIOps Prompt ID 未注册：{prompt_id}") from exc
        frozen = (frozen_prompts or {}).get(prompt_id)
        resolved = await self._resolver.resolve(
            prompt_key,
            prompt_version_id=(
                UUID(str(frozen["prompt_version_id"])) if frozen else None
            ),
        )
        asset = self._asset(resolved)
        if frozen is not None:
            expected = {
                "prompt_id": str(frozen.get("prompt_id") or ""),
                "prompt_version": str(frozen.get("prompt_version") or ""),
                "prompt_sha256": str(frozen.get("prompt_sha256") or ""),
                "prompt_version_id": str(
                    frozen.get("prompt_version_id") or ""
                ),
                "prompt_source": str(frozen.get("prompt_source") or ""),
            }
            if asset.snapshot() != expected:
                raise PromptIntegrityError(
                    f"AIOps 冻结 Prompt 已漂移：{prompt_id}"
                )
        return asset

    async def snapshot(
        self,
        prompt_ids: tuple[str, ...],
        *,
        frozen_prompts: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, dict[str, str]]:
        result = {}
        for prompt_id in prompt_ids:
            result[prompt_id] = (
                await self.resolve(
                    prompt_id,
                    frozen_prompts=frozen_prompts,
                )
            ).snapshot()
        return result

    @staticmethod
    def _asset(resolved: ResolvedPrompt) -> PromptAsset:
        if resolved.source != "DATABASE" or resolved.prompt_version_id is None:
            raise PromptIntegrityError("AIOps Prompt 必须来自数据库版本")
        return PromptAsset(
            prompt_id=resolved.prompt_key,
            version=resolved.version,
            sha256=resolved.sha256,
            content=resolved.content,
            prompt_version_id=resolved.prompt_version_id,
            source=resolved.source,
        )
