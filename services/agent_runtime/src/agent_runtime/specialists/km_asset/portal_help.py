"""KM Portal 固定产品介绍和使用帮助。"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from agent_runtime.language import response_language
from agent_runtime.runtime import SkillArtifact, SkillProgress, SkillResult
from agent_runtime.specialists.conversation_response import (
    ConversationResponseSkill,
)


_RESOURCE_NAMES = {
    "zh-CN": "portal_help.zh-CN.md",
    "en-US": "portal_help.en-US.md",
    "ja-JP": "portal_help.ja-JP.md",
    "ko-KR": "portal_help.ko-KR.md",
}


@lru_cache(maxsize=1)
def _portal_help_contents() -> dict[str, str]:
    """一次性加载并校验四种受支持语言的固定回复。"""
    resource_root = files(
        "agent_runtime.specialists.km_asset"
    ).joinpath("resources")
    contents: dict[str, str] = {}
    for language, resource_name in _RESOURCE_NAMES.items():
        content = resource_root.joinpath(resource_name).read_text(
            encoding="utf-8"
        ).strip()
        if not content:
            raise ValueError(f"KM Portal 帮助资源为空：{resource_name}")
        contents[language] = content
    return contents


class KmAssetConversationResponseSkill(ConversationResponseSkill):
    """为 KM Portal 帮助路由返回受控 Markdown，其他对话沿用通用实现。"""

    def __init__(self, *, model_client, prompt_resolver) -> None:
        super().__init__(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        )
        _portal_help_contents()

    async def execute_stream(self, context):
        route = dict(context.config_snapshot.get("route") or {})
        if route.get("answer_basis") != "PORTAL_HELP":
            async for item in super().execute_stream(context):
                yield item
            return

        language = response_language(
            context.config_snapshot, context.original_input
        )
        effective_language = (
            language if language in _RESOURCE_NAMES else "en-US"
        )
        answer = _portal_help_contents()[effective_language]
        for index, chunk in enumerate(
            answer.splitlines(keepends=True), start=1
        ):
            yield SkillProgress(
                event_type="answer.delta",
                payload={"chunk_index": index, "delta": chunk},
            )
        yield SkillResult(
            artifact=SkillArtifact(
                artifact_type="GROUNDED_ANSWER",
                schema_version="GroundedAnswer.v1",
                payload={
                    "answer": answer,
                    "status": "READY",
                    "used_citation_labels": [],
                    "references": [],
                    "warnings": [],
                },
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                    "answer_mode": "KM_PORTAL_HELP",
                    "response_language": effective_language,
                    "resource": _RESOURCE_NAMES[effective_language],
                },
            )
        )
