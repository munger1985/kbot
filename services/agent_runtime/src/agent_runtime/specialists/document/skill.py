"""基于 Knowledge Core 两阶段检索的 Document Skill。"""

import base64
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

from PIL import Image
from platform_core.contracts import AuthContext, PrincipalKind

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult

from .contracts import (
    Citation,
    CitationPack,
    DocumentRetrievalResult,
    RetrievalCoverage,
)


class KnowledgeRetrievalSkill:
    """只调用 KC API，不访问 KC Entity、Repository 或向量表。"""

    def __init__(
        self,
        *,
        knowledge_core_client,
        service_name: str,
        model_client=None,
        prompt_resolver=None,
    ):
        self._client = knowledge_core_client
        self._service_name = service_name
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        query = self._standalone_query(context)
        collection_ids = await self._resolve_collection_ids(context)
        if not collection_ids:
            return self._empty_result(
                context,
                status="INSUFFICIENT_EVIDENCE",
                warning="当前 Agent 没有可用的 Collection 绑定",
            )

        retrieval_config = self._retrieval_config(context)
        image_payloads = await self._load_query_images(context)
        image_processing: dict[str, Any] = {
            "image_count": len(image_payloads),
            "visual_search": "NOT_REQUESTED",
            "vlm_text_search": "NOT_REQUESTED",
        }
        warnings: list[str] = []
        vlm_descriptions, vlm_prompt_ref = await self._describe_images(
            context, image_payloads, image_processing, warnings
        )
        retrieval_query = self._multimodal_query(
            query, vlm_descriptions
        )
        visual_outcome = await self._visual_hits(
            context,
            collection_ids,
            retrieval_config,
            image_payloads,
            image_processing,
            warnings,
        )
        visual_hits = list(visual_outcome.get("results") or [])
        self._finalize_image_warnings(
            image_processing=image_processing,
            warnings=warnings,
        )
        discovery = await self._client.discover(
            query=retrieval_query,
            collection_ids=collection_ids,
            domain_id=context.domain_id,
            agent_id=str(context.agent_id),
            auth_context=self._auth_context(context),
            max_security_level=self._security_level(context),
            per_collection_limit=retrieval_config["max_bundles"],
            do_rerank=retrieval_config["do_rerank"],
            run_id=context.run_id,
            task_id=context.task_id,
        )
        warnings.extend(discovery.get("warnings") or [])
        discovery_rerank = dict(discovery.get("rerank") or {})
        discovery_diagnostics = dict(
            discovery.get("diagnostics") or {}
        )
        candidates = list(discovery.get("candidates") or [])
        candidates = self._merge_candidates(
            visual_hits,
            candidates,
            limit=retrieval_config["max_bundles"],
        )
        if not candidates:
            return self._empty_result(
                context,
                status="INSUFFICIENT_EVIDENCE",
                warning="Knowledge Core 未发现相关 Bundle",
                warnings=tuple(warnings),
                query_plan={
                    "image_processing": image_processing,
                    "do_rerank": retrieval_config["do_rerank"],
                    "rerank": {"discovery": discovery_rerank},
                    "diagnostics": {
                        "discovery": discovery_diagnostics,
                    },
                },
            )

        evidence = await self._client.retrieve_evidence(
            query=retrieval_query,
            candidates=[
                {
                    "collection_id": item["collection_id"],
                    "bundle_id": item["bundle_id"],
                    "bundle_revision_id": item["bundle_revision_id"],
                    "document_version_ids": (
                        [item["document_version_id"]]
                        if item.get("document_version_id")
                        else []
                    ),
                }
                for item in candidates
            ],
            domain_id=context.domain_id,
            agent_id=str(context.agent_id),
            auth_context=self._auth_context(context),
            max_security_level=self._security_level(context),
            max_evidence=retrieval_config["max_citations"],
            context_limit=retrieval_config["context_limit"],
            do_rerank=retrieval_config["do_rerank"],
            run_id=context.run_id,
            task_id=context.task_id,
        )
        warnings.extend(evidence.get("warnings") or [])
        evidence_rerank = dict(evidence.get("rerank") or {})
        evidence_diagnostics = dict(
            evidence.get("diagnostics") or {}
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
                question=query,
                query_plan={
                    "strategy": "KC_TWO_STAGE",
                    "visual_query_count": len(
                        self._query_image_descriptors(context)
                    ),
                    "image_processing": image_processing,
                    "vlm_prompt": vlm_prompt_ref,
                    "target_level": "AUTO",
                    "collection_ids": [
                        str(value) for value in collection_ids
                    ],
                    "max_bundles": retrieval_config["max_bundles"],
                    "max_citations": retrieval_config["max_citations"],
                    "do_rerank": retrieval_config["do_rerank"],
                    "rerank": {
                        "discovery": discovery_rerank,
                        "evidence": evidence_rerank,
                    },
                    "diagnostics": {
                        "discovery": discovery_diagnostics,
                        "evidence": evidence_diagnostics,
                    },
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
                "selector": (
                    "llm-object-and-evidence-group-v1"
                    if retrieval_config["do_rerank"]
                    else "deterministic-group-selection-v1"
                ),
                "rerank": {
                    "enabled": retrieval_config["do_rerank"],
                    "discovery": discovery_rerank,
                    "evidence": evidence_rerank,
                },
                "diagnostics": {
                    "discovery": discovery_diagnostics,
                    "evidence": evidence_diagnostics,
                },
                "visual_hit_count": len(visual_hits),
                "vlm_description_count": len(vlm_descriptions),
            },
            coverage_gaps=gaps,
            warnings=tuple(warnings),
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

    async def _visual_hits(
        self,
        context: ExecutionContext,
        collection_ids: tuple[UUID, ...],
        retrieval_config: dict[str, int | bool],
        image_payloads: list[bytes],
        image_processing: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        if not image_payloads:
            return {"results": []}
        try:
            response = await self._client.search_visual(
                images_base64=[
                    base64.b64encode(payload).decode("ascii")
                    for payload in image_payloads
                ],
                collection_ids=collection_ids,
                domain_id=context.domain_id,
                agent_id=context.agent_id,
                auth_context=self._auth_context(context),
                per_image_limit=retrieval_config["max_bundles"],
                result_limit=retrieval_config["max_bundles"],
            )
        except Exception:
            image_processing["visual_search"] = "FAILED"
            warnings.append("图片相似检索暂时不可用，已继续其他检索路径")
            return {"results": []}
        searched = list(response.get("searched_collection_ids") or [])
        skipped = list(response.get("skipped_collection_ids") or [])
        if searched and skipped:
            image_processing["visual_search"] = "PARTIAL"
            warnings.append(
                "部分 Collection 未配置 Visual Embedding，已跳过其图搜图路径"
            )
        elif searched:
            image_processing["visual_search"] = "EXECUTED"
        else:
            image_processing["visual_search"] = "SKIPPED_NOT_CONFIGURED"
        image_processing["visual_searched_collection_ids"] = searched
        image_processing["visual_skipped_collection_ids"] = skipped
        return response

    async def _load_query_images(
        self, context: ExecutionContext
    ) -> list[bytes]:
        payloads: list[bytes] = []
        for descriptor in self._query_image_descriptors(context):
            uri = str(descriptor.get("storage_uri") or "")
            if uri:
                payloads.append(
                    await asyncio.to_thread(Path(uri).read_bytes)
                )
        return payloads

    async def _describe_images(
        self,
        context: ExecutionContext,
        image_payloads: list[bytes],
        image_processing: dict[str, Any],
        warnings: list[str],
    ) -> tuple[list[str], dict[str, Any] | None]:
        if not image_payloads:
            return [], None
        model_name = str(
            agent_model_name(
                context.config_snapshot.get("agent", {}), "query_vlm"
            )
            or ""
        ).strip()
        if not model_name:
            image_processing["vlm_text_search"] = (
                "SKIPPED_NOT_CONFIGURED"
            )
            return [], None
        if self._model_client is None or self._prompt_resolver is None:
            image_processing["vlm_text_search"] = "FAILED"
            warnings.append("图片理解服务未初始化，已跳过图片转文字检索")
            return [], None
        try:
            prompt = await self._prompt_resolver.resolve(
                "agent_runtime.query_image_description"
            )
            descriptions = await asyncio.gather(
                *(
                    self._model_client.get_vlm_answer(
                        model_name,
                        Image.open(BytesIO(payload)).convert("RGB"),
                        prompt=prompt.content,
                        temperature=0.1,
                        max_tokens=1024,
                    )
                    for payload in image_payloads
                )
            )
        except Exception:
            image_processing["vlm_text_search"] = "FAILED"
            warnings.append("图片理解暂时不可用，已继续其他检索路径")
            return [], (
                prompt.ref() if "prompt" in locals() else None
            )
        normalized = [
            str(value).strip()[:3000]
            for value in descriptions
            if str(value).strip()
        ]
        image_processing["vlm_text_search"] = (
            "EXECUTED" if normalized else "FAILED"
        )
        if not normalized:
            warnings.append("图片理解未产生可检索文本")
        return normalized, prompt.ref()

    @staticmethod
    def _multimodal_query(
        query: str, descriptions: list[str]
    ) -> str:
        if not descriptions:
            return query
        supplement = "\n".join(
            f"查询图片{i + 1}：{value}"
            for i, value in enumerate(descriptions)
        )
        return f"{query}\n\n{supplement}"[:8000]

    @staticmethod
    def _finalize_image_warnings(
        *,
        image_processing: dict[str, Any],
        warnings: list[str],
    ) -> None:
        if not image_processing.get("image_count"):
            return
        visual = image_processing["visual_search"]
        vlm = image_processing["vlm_text_search"]
        if (
            visual == "SKIPPED_NOT_CONFIGURED"
            and vlm == "SKIPPED_NOT_CONFIGURED"
        ):
            warnings.append(
                "未配置 Visual Embedding 或查询 VLM，已忽略上传图片并仅处理文字"
            )
        elif visual == "SKIPPED_NOT_CONFIGURED":
            warnings.append(
                "未配置 Visual Embedding，已仅使用 VLM 图片转文字检索"
            )
        elif vlm == "SKIPPED_NOT_CONFIGURED":
            warnings.append(
                "未配置查询 VLM，已仅执行图片相似检索"
            )

    @staticmethod
    def _query_image_descriptors(
        context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        return list(
            context.config_snapshot.get("client_metadata", {}).get(
                "query_images", []
            )
            or []
        )

    @staticmethod
    def _merge_candidates(
        visual_hits: list[dict[str, Any]],
        text_candidates: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """视觉候选优先保留，再补充文本 Discovery 候选。"""
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*visual_hits, *text_candidates]:
            key = str(item.get("bundle_revision_id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(item)
            if len(output) >= limit:
                break
        return output

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

    @staticmethod
    def _standalone_query(context: ExecutionContext) -> str:
        document_scopes = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "DOCUMENT_SCOPE"
        ]
        if document_scopes:
            scoped_query = str(
                (document_scopes[-1].payload or {}).get("query") or ""
            ).strip()
            if scoped_query:
                return scoped_query
        artifacts = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "CONTEXT_REWRITE"
        ]
        if not artifacts:
            return context.original_input
        query = str(
            (artifacts[-1].payload or {}).get("standalone_query") or ""
        ).strip()
        return query or context.original_input

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
    def _retrieval_config(
        context: ExecutionContext,
    ) -> dict[str, int | bool]:
        agent_snapshot = context.config_snapshot.get("agent", {})
        agent_config = agent_snapshot.get("config", {})
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
            "do_rerank": bool(agent_snapshot.get("do_rerank", False)),
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
                    locator_schema_version=str(
                        first["locator_schema_version"]
                    ),
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
        warnings: tuple[str, ...] = (),
        query_plan: dict[str, Any] | None = None,
    ) -> SkillResult:
        result = DocumentRetrievalResult(
            status=status,
            citation_pack=CitationPack(
                question=context.original_input,
                query_plan={
                    "strategy": "KC_TWO_STAGE",
                    **(query_plan or {}),
                },
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
            warnings=tuple(dict.fromkeys((*warnings, warning))),
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
            warnings=tuple(dict.fromkeys((*warnings, warning))),
        )
