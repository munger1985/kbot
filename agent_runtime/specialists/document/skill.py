"""基于 Knowledge Core 两阶段检索的 Document Skill。"""

from typing import Any
from uuid import UUID

from platform_core.contracts import AuthContext, PrincipalKind

from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult

from .contracts import (
    Citation,
    CitationPack,
    DocumentRetrievalResult,
    RetrievalCoverage,
)


class KnowledgeRetrievalSkill:
    """只调用 KC API，不访问 KC Entity、Repository 或向量表。"""

    def __init__(self, *, knowledge_core_client, service_name: str):
        self._client = knowledge_core_client
        self._service_name = service_name

    async def execute(self, context: ExecutionContext) -> SkillResult:
        collection_ids = await self._resolve_collection_ids(context)
        if not collection_ids:
            return self._empty_result(
                context,
                status="INSUFFICIENT_EVIDENCE",
                warning="当前 Agent 没有可用的 Collection 绑定",
            )

        retrieval_config = self._retrieval_config(context)
        discovery = await self._client.discover(
            query=context.original_input,
            collection_ids=collection_ids,
            domain_id=context.domain_id,
            agent_id=str(context.agent_id),
            auth_context=self._auth_context(context),
            max_security_level=self._security_level(context),
            per_collection_limit=retrieval_config["max_bundles"],
        )
        candidates = list(discovery.get("candidates") or [])
        candidates = candidates[: retrieval_config["max_bundles"]]
        if not candidates:
            return self._empty_result(
                context,
                status="INSUFFICIENT_EVIDENCE",
                warning="Knowledge Core 未发现相关 Bundle",
            )

        evidence = await self._client.retrieve_evidence(
            query=context.original_input,
            candidates=[
                {
                    "collection_id": item["collection_id"],
                    "bundle_id": item["bundle_id"],
                    "bundle_revision_id": item["bundle_revision_id"],
                    "document_version_ids": [],
                }
                for item in candidates
            ],
            domain_id=context.domain_id,
            agent_id=str(context.agent_id),
            auth_context=self._auth_context(context),
            max_security_level=self._security_level(context),
            max_evidence=retrieval_config["max_citations"],
            context_limit=retrieval_config["context_limit"],
        )
        raw_citations = list(evidence.get("citations") or [])
        citations = self._map_citations(
            raw_citations,
            candidates=candidates,
        )
        status = "READY" if citations else "INSUFFICIENT_EVIDENCE"
        gaps = () if citations else ("未找到可引用的正文证据",)
        result = DocumentRetrievalResult(
            status=status,
            citation_pack=CitationPack(
                question=context.original_input,
                query_plan={
                    "strategy": "KC_TWO_STAGE",
                    "target_level": "AUTO",
                    "collection_ids": [
                        str(value) for value in collection_ids
                    ],
                    "max_bundles": retrieval_config["max_bundles"],
                    "max_citations": retrieval_config["max_citations"],
                },
                bundle_candidates=tuple(candidates),
                citations=tuple(citations),
                coverage=RetrievalCoverage(
                    candidate_bundle_count=len(candidates),
                    selected_document_count=len(
                        {item.document_id for item in citations}
                    ),
                    selected_evidence_count=sum(
                        len(item.evidence_ids) for item in citations
                    ),
                    uncovered_aspects=gaps,
                ),
            ),
            retrieval_report={
                "strategy_version": "kc-two-stage-v1",
                "discovery_candidate_count": len(candidates),
                "citation_count": len(citations),
                "selector": "deterministic-group-selection-v1",
            },
            coverage_gaps=gaps,
        )
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="CITATION_PACK",
                schema_version="DocumentRetrievalResult.v1",
                payload=result.model_dump(mode="json"),
                provenance={
                    "knowledge_core_api": "internal/v1",
                    "strategy": "KC_TWO_STAGE",
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                },
                security_level=self._security_level(context),
            )
        )

    async def _resolve_collection_ids(
        self, context: ExecutionContext
    ) -> tuple[UUID, ...]:
        configured = (
            context.config_snapshot.get("retrieval", {})
            .get("collection_ids", [])
        )
        if configured:
            return tuple(UUID(str(value)) for value in configured)
        response = await self._client.list_agent_bindings(
            domain_id=context.domain_id,
            agent_id=context.agent_id,
            auth_context=self._auth_context(context),
        )
        return tuple(
            UUID(str(item["collection_id"]))
            for item in response.get("bindings", [])
            if item.get("status") == "ACTIVE"
        )

    def _auth_context(self, context: ExecutionContext) -> AuthContext:
        return AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id=self._service_name,
            calling_service=self._service_name,
            request_id=context.request_id,
            trace_id=context.trace_id,
            domain_id=str(context.domain_id),
            asserted_user_id=context.actor_id,
        )

    @staticmethod
    def _security_level(context: ExecutionContext) -> int:
        value = int(
            context.config_snapshot.get("retrieval", {})
            .get("security_level", 0)
        )
        return max(0, min(value, 3))

    @staticmethod
    def _retrieval_config(context: ExecutionContext) -> dict[str, int]:
        agent_config = (
            context.config_snapshot.get("agent", {}).get("config", {})
        )
        retrieval = agent_config.get("retrieval", {})
        return {
            "max_bundles": max(
                1, min(int(retrieval.get("max_bundles", 10)), 50)
            ),
            "max_citations": max(
                1, min(int(retrieval.get("max_citations", 12)), 50)
            ),
            "context_limit": max(
                0, min(int(retrieval.get("context_limit", 4)), 20)
            ),
        }

    @staticmethod
    def _map_citations(
        raw_citations: list[dict[str, Any]],
        *,
        candidates: list[dict[str, Any]],
    ) -> list[Citation]:
        titles = {
            str(item["bundle_id"]): str(item.get("display_title") or "")
            for item in candidates
        }
        result: list[Citation] = []
        for group in raw_citations:
            items = list(group.get("items") or [])
            primary = [
                item
                for item in items
                if item.get("final_role") == "PRIMARY"
            ]
            selected = primary or items
            if not selected:
                continue
            first = selected[0].get("evidence") or {}
            if not first.get("document_id"):
                continue
            excerpts = [
                str((item.get("evidence") or {}).get("content_text") or "")
                for item in selected
            ]
            excerpt = "\n".join(
                value.strip() for value in excerpts if value.strip()
            )[:4000]
            evidence_ids = tuple(
                UUID(str(value))
                for value in group.get("primary_evidence_ids", [])
            )
            if not evidence_ids:
                evidence_ids = tuple(
                    UUID(str((item.get("evidence") or {})["evidence_id"]))
                    for item in selected
                    if (item.get("evidence") or {}).get("evidence_id")
                )
            provenance = first.get("provenance") or {}
            bundle_id = UUID(str(group["bundle_id"]))
            bundle_title = (
                str(first.get("bundle_title") or "").strip()
                or titles.get(str(bundle_id))
                or "未命名 Bundle"
            )
            document_name = str(
                first.get("document_name")
                or first.get("external_document_id")
                or bundle_title
            )
            result.append(
                Citation(
                    citation_label=str(group["citation_label"]),
                    collection_id=UUID(str(group["collection_id"])),
                    bundle_id=bundle_id,
                    bundle_revision_id=UUID(
                        str(group["bundle_revision_id"])
                    ),
                    document_id=UUID(str(first["document_id"])),
                    document_version_id=UUID(
                        str(group["document_version_id"])
                    ),
                    evidence_ids=evidence_ids,
                    title=document_name,
                    bundle_title=bundle_title,
                    external_document_id=first.get(
                        "external_document_id"
                    ),
                    document_role=first.get("document_role"),
                    excerpt=excerpt,
                    locator=dict(first.get("locator") or {}),
                    heading_path=tuple(first.get("heading_path") or ()),
                    relevance_reason="由 KC 两阶段检索选中的正文证据组",
                    source_hash=(
                        first.get("content_hash")
                        or provenance.get("source_hash")
                        or provenance.get("content_hash")
                    ),
                )
            )
        return result

    @staticmethod
    def _empty_result(
        context: ExecutionContext,
        *,
        status: str,
        warning: str,
    ) -> SkillResult:
        result = DocumentRetrievalResult(
            status=status,
            citation_pack=CitationPack(
                question=context.original_input,
                query_plan={"strategy": "KC_TWO_STAGE"},
                bundle_candidates=(),
                citations=(),
                coverage=RetrievalCoverage(
                    candidate_bundle_count=0,
                    selected_document_count=0,
                    selected_evidence_count=0,
                    uncovered_aspects=(warning,),
                ),
            ),
            retrieval_report={
                "strategy_version": "kc-two-stage-v1",
                "discovery_candidate_count": 0,
                "citation_count": 0,
            },
            coverage_gaps=(warning,),
            warnings=(warning,),
        )
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="CITATION_PACK",
                schema_version="DocumentRetrievalResult.v1",
                payload=result.model_dump(mode="json"),
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                },
                security_level=(
                    KnowledgeRetrievalSkill._security_level(context)
                ),
            ),
            warnings=(warning,),
        )
