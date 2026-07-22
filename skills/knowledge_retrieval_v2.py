"""Stateless KnowledgeRetrievalSkillV2 implementation."""
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from agent.common import ContextMemory
from platform_core.dictionary import PacketType
from knowledge_core.application.task_dto import KnowledgeTask, KnowledgeTaskResult
from knowledge_core.client import KnowledgeCoreClient
from skills import BaseSkill, SkillDomain, SkillMeta, SkillRunMode


class KnowledgeRetrievalSkillV2(BaseSkill):
    meta = SkillMeta(
        name="knowledge-retrieval-skill-v2",
        description="通过 Knowledge Core V2 Discovery/Evidence API 检索可引用知识",
        domain=SkillDomain.BUSINESS,
        run_mode=SkillRunMode.READ_ONLY,
    )

    def __init__(self, *, kc_client: KnowledgeCoreClient | None = None, kc_url: str = "http://127.0.0.1:18090"):
        self._kc_client = kc_client or KnowledgeCoreClient(base_url=kc_url)

    async def execute(self, task: KnowledgeTask) -> KnowledgeTaskResult:
        if not task.standalone_query.strip() or not task.collection_ids:
            return KnowledgeTaskResult(task_id=task.task_id, status="INSUFFICIENT_EVIDENCE", coverage_gaps=["缺少可检索问题或已授权 Collection"])
        discovery = await self._kc_client.discover(
            query=task.standalone_query, collection_ids=task.collection_ids,
            domain_id=task.domain_id, agent_id=task.agent_id,
            max_security_level=task.security_level,
        )
        candidates = discovery.get("candidates") or []
        if not candidates:
            return KnowledgeTaskResult(task_id=task.task_id, status="INSUFFICIENT_EVIDENCE", coverage_gaps=["没有找到相关知识对象"])
        evidence = await self._kc_client.retrieve_evidence(
            query=task.standalone_query,
            candidates=[{"collection_id": item["collection_id"], "bundle_id": item["bundle_id"], "bundle_revision_id": item["bundle_revision_id"]} for item in candidates],
            domain_id=task.domain_id, agent_id=task.agent_id,
            max_security_level=task.security_level,
        )
        citations = evidence.get("citations") or []
        if not citations:
            return KnowledgeTaskResult(task_id=task.task_id, status="INSUFFICIENT_EVIDENCE", coverage_gaps=["找到相关对象，但没有可引用 Evidence"])
        return KnowledgeTaskResult(
            task_id=task.task_id, status="READY",
            citation_pack={"question": task.standalone_query, "discovery_summary": {"candidate_count": len(candidates)}, "citations": citations, "coverage": {"candidate_count": len(candidates), "evidence_count": len(citations)}},
        )

    async def run_stream(self, context: ContextMemory, **kwargs) -> AsyncGenerator[dict[str, Any], None]:
        current = context.get("current_execution") or {}
        query = current.get("resolved_input") or context.get("standalone_query") or context.get("question") or ""
        collection_ids = tuple(context.get("kc_collection_ids") or context.get("collection_ids") or ())
        task = KnowledgeTask(
            task_id=str(uuid.uuid4()), parent_run_id=str(context.get("run_id") or ""),
            domain_id=int(context.get("domain_id") or 0), agent_id=int(context.get("agent_id") or 0),
            original_query=str(context.get("question") or query), standalone_query=str(query),
            collection_ids=collection_ids, security_level=int(context.get("security_level") or 3),
        )
        yield {"type": PacketType.THOUGHT, "content": "正在通过 Knowledge Core V2 检索可引用证据……"}
        try:
            result = await self.execute(task)
            context["knowledge_task_result"] = result.__dict__
            yield {"type": "knowledge_task_result", "content": result.__dict__}
            if result.citation_pack:
                yield {"type": PacketType.DOC_RESULTS, "content": result.citation_pack}
        except Exception as exc:
            yield {"type": PacketType.ERROR, "content": f"Knowledge Core V2 检索失败：{exc}"}
