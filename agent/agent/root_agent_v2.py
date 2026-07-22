"""Explicit V2 knowledge route; the legacy RootAgent remains untouched."""
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any

from agent.agent.document_agent_v2 import DocumentAgentV2
from knowledge_core.application.answer_generation import AnswerGenerator, EmptyAnswerGenerator
from knowledge_core.application.grounding import AnswerGroundingVerifier, AnswerDraft, citation_groups_from_payload
from knowledge_core.application.sse_v2 import build_grounded_sse_payload
from knowledge_core.application.task_dto import KnowledgeTask, KnowledgeTaskResult


class RootAgentV2:
    def __init__(
        self,
        *,
        document_agent: DocumentAgentV2,
        answer_generator: AnswerGenerator | None = None,
        grounding_verifier: AnswerGroundingVerifier | None = None,
    ):
        self._document_agent = document_agent
        self._answer_generator = answer_generator or EmptyAnswerGenerator()
        self._grounding_verifier = grounding_verifier or AnswerGroundingVerifier()

    async def retrieve(self, task: KnowledgeTask) -> KnowledgeTaskResult:
        return await self._document_agent.retrieve(task)

    async def complete(self, task: KnowledgeTask) -> dict[str, Any]:
        """Non-streaming equivalent of the grounded V2 SSE terminal payload."""
        result = await self.retrieve(task)
        grounded = await self._ground(task, result)
        return {
            "knowledge_task_result": asdict(result),
            **build_grounded_sse_payload(answer_markdown=grounded.answer_markdown, result=grounded),
            "grounding_status": grounded.grounding_status,
        }

    async def stream(self, task: KnowledgeTask) -> AsyncGenerator[bytes, None]:
        yield self._event("metadata", {"task_id": task.task_id, "parent_run_id": task.parent_run_id})
        yield self._event("thought", {"message": "Knowledge Core V2 retrieval started"})
        result = await self.retrieve(task)
        yield self._event("knowledge_task_result", asdict(result))
        if result.citation_pack:
            # Candidate context is deliberately separate from the final
            # citations.  Only the labels the answer model actually used are
            # allowed into the terminal doc_results_v2 projection.
            yield self._event("citation_candidates_v2", result.citation_pack)
        grounded = await self._ground(task, result)
        payload = build_grounded_sse_payload(answer_markdown=grounded.answer_markdown, result=grounded)
        yield self._event("answer", {"answer": grounded.answer_markdown})
        yield self._event("grounded_answer", payload)
        yield self._event("doc_results_v2", payload["doc_results_v2"])
        yield self._event("grounding_status", {
            "status": grounded.grounding_status,
            "dropped_citation_labels": grounded.dropped_citation_labels,
            "unsupported_claim_ids": grounded.unsupported_claim_ids,
        })
        yield self._event("done", {"status": result.status, "grounding_status": grounded.grounding_status})

    async def _ground(self, task: KnowledgeTask, result: KnowledgeTaskResult):
        draft = AnswerDraft(answer_markdown="")
        if result.citation_pack:
            try:
                draft = await self._answer_generator.generate(
                    task=task, citation_pack=result.citation_pack,
                )
            except Exception:
                # Retrieval remains useful even when the answer provider is
                # unavailable; the terminal payload explicitly reports that
                # no grounded answer was produced.
                draft = AnswerDraft(answer_markdown="")
        citation_groups = citation_groups_from_payload(result.citation_pack or {})
        return self._grounding_verifier.verify(draft=draft, citation_pack=citation_groups)

    @staticmethod
    def _event(name: str, content: Any) -> bytes:
        return f"event: {name}\ndata: {json.dumps(content, ensure_ascii=False)}\n\n".encode("utf-8")
