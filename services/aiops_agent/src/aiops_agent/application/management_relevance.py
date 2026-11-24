"""资料检索正文准入；召回名次不能代替查询与原文之间的证据。"""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime, timedelta
import re
import unicodedata
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.application.errors import AIOpsApplicationError
from knowledge_algorithms.contracts.identity import fingerprint
from platform_core.prompts import PromptIntegrityError


PASSAGE_CHARS = 4000
_SEMANTIC_CANDIDATES = 24
_BATCH_SIZE = 6
_TIMEOUT_SECONDS = 20
_QUERY_PARTS = re.compile(r"[a-z0-9]+(?:[._:+/\-][a-z0-9]+)*|[\u3400-\u9fff]+|[^\W_]", re.IGNORECASE)
_SENTENCES = re.compile(r"[。！？!?；;\r\n]+|\.\s+")


class PassageJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passage_id: int = Field(strict=True, ge=1)
    support: str | None = Field(max_length=400)


class PassageRelevanceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["OPS_PASSAGE_RELEVANCE.v1"] = "OPS_PASSAGE_RELEVANCE.v1"
    judgments: list[PassageJudgment] = Field(max_length=_BATCH_SIZE)


def _literal_matcher(query):
    """全查询成分必须按顺序出现在同一句；技术标识不接受单词内部的子串。"""
    parts = _QUERY_PARTS.findall(unicodedata.normalize("NFKC", query).casefold())
    if not parts:
        return None
    patterns = [
        r"(?<![a-z0-9])" + re.escape(part) + r"(?![a-z0-9])"
        if part[0].isascii() else re.escape(part)
        for part in parts
    ]
    if len(patterns) > 16:
        return None
    return tuple((re.compile(pattern), len(part)) for pattern, part in zip(patterns, parts, strict=True))


def _literal_support(matcher, quote):
    if matcher is None:
        return False
    for sentence in _SENTENCES.split(unicodedata.normalize("NFKC", quote).casefold()):
        for start in matcher[0][0].finditer(sentence):
            cursor = start.end()
            for pattern, length in matcher[1:]:
                match = pattern.search(sentence, cursor, cursor + 40 + length + 1)
                if match is None or match.start() - cursor > 40:
                    break
                cursor = match.end()
            else:
                return True
    return False


async def _bindings(service, access, source_kind, base_ids):
    """只从当前工作域读取各资料源的显式绑定；手册与案例不互借模型。"""
    async with service._uow_factory() as uow:
        if source_kind == "DIAGNOSIS_CASE":
            config = await uow.knowledge_documents.case_configuration(access)
            return {None: (config.models_json or {}).get("retrieval_llm")} if config else {}
        rows = await uow.operations_knowledge.list_knowledge_bases(domain_id=access.domain_id)
        return {row.knowledge_base_id: ((row.configuration_json or {}).get("models") or {}).get("parser_llm")
                for row in rows if row.status == "ACTIVE" and row.knowledge_base_id in base_ids}


async def qualify_passages(service, *, access, context, query, source_kind, evidence_items, base_by_document):
    """返回可交付片段、直接支持的位置及未能完整复核的原因。"""
    matcher = _literal_matcher(query)
    direct, uncertain = set(), []
    for position, (citation, _) in enumerate(evidence_items, 1):
        excerpt = citation["quote"][:PASSAGE_CHARS]
        if _literal_support(matcher, excerpt):
            direct.add(position)
        else:
            uncertain.append((position, excerpt, base_by_document.get(citation["document_version_id"])))
    if not uncertain:
        return direct, direct, False

    verified = set(direct)
    model = getattr(service, "_structured_model", None)
    prompts = getattr(service, "_prompt_registry", None)
    if model is None or prompts is None:
        return verified, direct, True

    incomplete = len(uncertain) > _SEMANTIC_CANDIDATES
    uncertain = uncertain[:_SEMANTIC_CANDIDATES]
    bindings = await _bindings(service, access, source_kind, {base for _, _, base in uncertain})
    batches = defaultdict(list)
    for position, excerpt, base_id in uncertain:
        binding = bindings.get(base_id if source_kind == "MANUAL" else None)
        if not binding:
            incomplete = True
            continue
        batches[str(binding)].append({"passage_id": position, "text": excerpt})
    if not batches:
        return verified, direct, True

    deadline = datetime.now(UTC) + timedelta(seconds=_TIMEOUT_SECONDS)
    slots = asyncio.Semaphore(3)

    async def bounded_assess(model_id, passages, prompt):
        async with slots:
            return await assess(model_id, passages, prompt)

    async def assess(model_id, passages, prompt):
        try:
            snapshot = await service._management_relevance_model_snapshot(
                UUID(model_id), access=access, context=context)
            accepted = set()
            complete = True
            for offset in range(0, len(passages), _BATCH_SIZE):
                batch = passages[offset:offset + _BATCH_SIZE]
                payload = {"query": query, "passages": batch}
                generated = await model.generate_structured(
                    purpose="operations_knowledge.passage_relevance", output_model=PassageRelevanceAssessment,
                    model_snapshot=snapshot, prompt_ref={**prompt.ref(), "content": prompt.content},
                    input_payload=payload, deadline=deadline,
                    idempotency_key="passage-relevance:" + fingerprint({
                        "input": payload, "model": snapshot, "prompt": prompt.ref(),
                    }),
                )
                assessment = PassageRelevanceAssessment.model_validate(generated.output)
                expected = {item["passage_id"]: item["text"] for item in batch}
                judgments = {item.passage_id: item.support for item in assessment.judgments}
                if (set(judgments) != set(expected) or len(judgments) != len(assessment.judgments)):
                    complete = False
                    continue
                for position, support in judgments.items():
                    if support is None:
                        continue
                    if support.strip() and support in expected[position]:
                        accepted.add(position)
                    else:
                        complete = False
            return accepted, complete
        except (AIOpsApplicationError, AIOpsModelError, PromptIntegrityError, TimeoutError,
                KeyError, TypeError, ValueError, LookupError):
            return set(), False

    try:
        # 多库并行，各个模型单独审核各自资料；超时不得将未核验候选回填为命中。
        async with asyncio.timeout(_TIMEOUT_SECONDS):
            prompt = await prompts.resolve("passage_relevance")
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(bounded_assess(model_id, passages, prompt))
                         for model_id, passages in batches.items()]
    except (PromptIntegrityError, TimeoutError, LookupError):
        return verified, direct, True
    for task in tasks:
        matches, complete = task.result()
        verified.update(matches)
        incomplete |= not complete
    return verified, direct, incomplete
