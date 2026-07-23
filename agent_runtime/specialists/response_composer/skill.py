"""仅消费已验证 Artifact 的最终回答组合器。"""

import json
import re
from typing import Any

from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from agent_runtime.specialists.document.contracts import (
    DocumentRetrievalResult,
)

from .contracts import GroundedAnswer, ReferenceCard


_CITATION_PATTERN = re.compile(r"\[([A-Z]\d+)\]")


class ResponseComposerSkill:
    def __init__(self, *, model_client):
        self._model_client = model_client

    async def execute(self, context: ExecutionContext) -> SkillResult:
        retrieval = self._document_result(context)
        if retrieval is None or not retrieval.citation_pack.citations:
            answer = GroundedAnswer(
                answer="当前授权知识范围内没有找到足够的可引用证据。",
                status="INSUFFICIENT_EVIDENCE",
                warnings=("回答未调用模型补写无来源内容",),
            )
            return self._result(context, answer)

        model_name = str(
            context.config_snapshot.get("agent", {}).get(
                "composer_model_name", ""
            )
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 composer_model_name")

        allowed = {
            item.citation_label: item
            for item in retrieval.citation_pack.citations
        }
        prompt = self._prompt(context, retrieval)
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=prompt,
            max_tokens=4096,
        )
        answer_text, used_labels = self._validate_model_answer(
            response, allowed
        )
        references = tuple(
            ReferenceCard(
                citation_label=label,
                collection_id=allowed[label].collection_id,
                bundle_id=allowed[label].bundle_id,
                document_id=allowed[label].document_id,
                document_version_id=allowed[label].document_version_id,
                title=allowed[label].title,
                locator=allowed[label].locator,
            )
            for label in used_labels
        )
        grounded = GroundedAnswer(
            answer=answer_text,
            status="READY",
            used_citation_labels=used_labels,
            references=references,
            warnings=retrieval.warnings,
        )
        return self._result(context, grounded)

    @staticmethod
    def _document_result(
        context: ExecutionContext,
    ) -> DocumentRetrievalResult | None:
        artifacts = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "CITATION_PACK"
        ]
        if not artifacts:
            return None
        latest = artifacts[-1]
        return DocumentRetrievalResult.model_validate(latest.payload)

    @staticmethod
    def _prompt(
        context: ExecutionContext,
        retrieval: DocumentRetrievalResult,
    ) -> list[dict[str, str]]:
        instruction = (
            context.config_snapshot.get("agent", {}).get("instruction")
            or "请基于证据准确、简洁地回答。"
        )
        evidence = [
            {
                "citation_label": item.citation_label,
                "title": item.title,
                "excerpt": item.excerpt,
                "locator": item.locator,
                "heading_path": list(item.heading_path),
            }
            for item in retrieval.citation_pack.citations
        ]
        return [
            {
                "role": "system",
                "content": (
                    f"{instruction}\n"
                    "只能使用给定证据陈述文档事实。每个事实后必须使用"
                    "[C1] 形式标注真实使用的证据。不得创建不存在的标签。"
                    "输出 JSON："
                    '{"answer":"...","used_citation_labels":["C1"]}。'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": context.original_input,
                        "evidence": evidence,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _validate_model_answer(
        response: dict[str, Any],
        allowed: dict[str, Any],
    ) -> tuple[str, tuple[str, ...]]:
        answer = str(response.get("answer") or "").strip()
        if not answer:
            raise ValueError("模型返回的 answer 为空")
        mentioned = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
        unknown = set(mentioned) - allowed.keys()
        if unknown:
            raise ValueError(f"模型使用了未知引用标签：{sorted(unknown)}")
        reported = tuple(
            str(value)
            for value in response.get("used_citation_labels", [])
        )
        if set(reported) != set(mentioned):
            raise ValueError("回答中的引用标签与声明的使用列表不一致")
        if not mentioned:
            raise ValueError("有文档事实的回答必须实际包含引用标签")
        return answer, mentioned

    @staticmethod
    def _result(
        context: ExecutionContext,
        answer: GroundedAnswer,
    ) -> SkillResult:
        source_ids = [
            str(item.artifact_id)
            for item in context.input_artifacts
        ]
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="GROUNDED_ANSWER",
                schema_version="GroundedAnswer.v1",
                payload=answer.model_dump(mode="json"),
                provenance={
                    "input_artifact_ids": source_ids,
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                },
                security_level=max(
                    (
                        item.security_level
                        for item in context.input_artifacts
                    ),
                    default=0,
                ),
            ),
            warnings=answer.warnings,
        )
