"""Answer-model boundary for RootAgentV2.

The model receives a bounded, labelled Citation Pack and must return a small
structured draft.  It never receives database identifiers as a citation
authority: labels are request-local and are checked again by grounding.
"""
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, Sequence

from knowledge_core.application.grounding import AnswerClaim, AnswerDraft
from knowledge_core.application.task_dto import KnowledgeTask
from platform_clients import AIModelClient


class AnswerGenerator(Protocol):
    async def generate(self, *, task: KnowledgeTask, citation_pack: dict[str, Any]) -> AnswerDraft: ...


class EmptyAnswerGenerator:
    """Safe default used when a route has no answer-model adapter configured."""

    async def generate(self, *, task: KnowledgeTask, citation_pack: dict[str, Any]) -> AnswerDraft:
        return AnswerDraft(answer_markdown="")


class LLMAnswerGenerator:
    """Generate a citation-aware answer through the existing LLM service."""

    def __init__(
        self,
        *,
        model_client: AIModelClient | None = None,
        model_resolver: Callable[[KnowledgeTask], Awaitable[str]] | None = None,
        max_evidence_chars: int = 18000,
    ):
        self._model_client = model_client or AIModelClient()
        self._model_resolver = model_resolver
        self._max_evidence_chars = max_evidence_chars

    async def generate(self, *, task: KnowledgeTask, citation_pack: dict[str, Any]) -> AnswerDraft:
        model_name = task.answer_model
        if not model_name and self._model_resolver:
            model_name = await self._model_resolver(task)
        if not model_name:
            return AnswerDraft(answer_markdown="")
        prompt = _build_prompt(task.original_query or task.standalone_query, citation_pack, self._max_evidence_chars)
        response = await self._model_client.get_llm_answer(
            model_name=model_name,
            prompt=prompt,
            temperature=0.2,
            max_tokens=4096,
        )
        return _parse_draft(response)


def _build_prompt(question: str, citation_pack: dict[str, Any], max_chars: int) -> list[dict[str, str]]:
    citations = citation_pack.get("citations") or []
    blocks: list[str] = []
    used = 0
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        label = citation.get("citation_label") or citation.get("label")
        if not label:
            continue
        # Bundle/Document database identifiers are intentionally not shown to
        # the model.  The request-local label is the only citation authority.
        lines = [f"[{label}]" ]
        for item in citation.get("items") or []:
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence") or {}
            if item.get("final_role") not in ("PRIMARY", "STRUCTURAL_CONTEXT"):
                continue
            text = str(evidence.get("content_text") or evidence.get("retrieval_text") or "").strip()
            if text:
                lines.append(f"{item.get('final_role')}: {text}")
        block = "\n".join(lines)
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    context = "\n\n".join(blocks) or "(no usable evidence)"
    system = (
        "你是知识库问答模型。只能根据给出的证据回答，不得补造事实。\n"
        "必须输出 JSON，不要 Markdown 代码围栏："
        '{"answer_markdown":"...","claims":[{"claim_id":"claim-1","text":"...",'
        '"citation_labels":["C1"]}],"used_citation_labels":["C1"],'
        '"selected_bundle_ids":[]}。selected_bundle_ids 可留空；每个有事实内容的 claim '
        '至少引用一个证据标签。不要输出数据库 ID 作为引用。'
    )
    user = f"问题：{question}\n\n证据：\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_draft(response: str) -> AnswerDraft:
    response = (response or "").strip()
    if not response:
        return AnswerDraft(answer_markdown="")
    raw = _extract_json(response)
    if isinstance(raw, dict) and isinstance(raw.get("answer_markdown"), str):
        claims: list[AnswerClaim] = []
        for index, claim in enumerate(raw.get("claims") or [], 1):
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text") or "")
            claims.append(AnswerClaim(
                claim_id=str(claim.get("claim_id") or f"claim-{index}"),
                text=text,
                citation_labels=tuple(str(label) for label in claim.get("citation_labels") or []),
            ))
        return AnswerDraft(
            answer_markdown=raw["answer_markdown"],
            claims=tuple(claims),
            used_citation_labels=tuple(str(label) for label in raw.get("used_citation_labels") or []),
            selected_bundle_ids=tuple(int(value) for value in raw.get("selected_bundle_ids") or []),
        )
    # A non-JSON response is still deliverable, but has no trusted citation
    # claims.  Grounding will mark it INSUFFICIENT rather than inventing refs.
    return AnswerDraft(
        answer_markdown=response,
        claims=(AnswerClaim("claim-1", response, tuple(sorted(set(re.findall(r"\bC\d+\b", response))))),),
        used_citation_labels=tuple(sorted(set(re.findall(r"\bC\d+\b", response)))),
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string and char == "{":
            depth += 1
        elif not in_string and char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
