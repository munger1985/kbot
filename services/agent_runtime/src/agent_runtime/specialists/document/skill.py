"""基于 Knowledge Core 两阶段检索的 Document Skill。"""

import base64
import asyncio
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

from loguru import logger
from PIL import Image
from platform_core.contracts import (
    AuthContext,
    KNOWLEDGE_DISCOVERY_REVISION_SCOPE_LIMIT,
    KnowledgeScopeSnapshotV1,
    PrincipalKind,
)

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
        knowledge_scope = self._knowledge_scope(context)
        collection_ids = knowledge_scope.collection_ids
        if not collection_ids:
            return self._empty_result(
                context,
                status="INSUFFICIENT_EVIDENCE",
                warning="当前 Agent 没有可用的 Collection 绑定",
            )

        retrieval_config = self._retrieval_config(context)
        coverage_mode = str(
            (context.config_snapshot.get("route") or {}).get(
                "coverage_mode", "BALANCED"
            )
        ).upper()
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
            knowledge_scope.bundle_revision_ids,
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
        scoped_candidates = await self._scoped_candidates(
            context=context,
            allowed_collection_ids=collection_ids,
            warnings=warnings,
        )
        if scoped_candidates is None:
            discovery = await self._discover_exact_scope(
                context=context,
                query=retrieval_query,
                collection_ids=collection_ids,
                bundle_revision_ids=knowledge_scope.bundle_revision_ids,
                per_collection_limit=retrieval_config["max_bundles"],
                coverage_mode=coverage_mode,
            )
            warnings.extend(discovery.get("warnings") or [])
            discovery_diagnostics = dict(
                discovery.get("diagnostics") or {}
            )
            candidates = self._merge_candidates(
                visual_hits,
                list(discovery.get("candidates") or []),
                limit=retrieval_config["max_bundles"],
            )
        else:
            candidates = scoped_candidates
            discovery_diagnostics = {
                "strategy": "QUERY_RESULT_BUNDLE_SCOPE",
                "target_count": len(candidates),
            }
        self._assert_scope_closure(
            candidates, knowledge_scope.bundle_revision_ids
        )
        if not candidates:
            return self._empty_result(
                context,
                status="INSUFFICIENT_EVIDENCE",
                warning="Knowledge Core 未发现相关 Bundle",
                warnings=tuple(warnings),
                query_plan={
                    "image_processing": image_processing,
                    "diagnostics": {
                        "discovery": discovery_diagnostics,
                    },
                },
            )

        evidence, candidates = await self._retrieve_evidence(
            context=context,
            query=retrieval_query,
            candidates=candidates,
            scoped=scoped_candidates is not None,
            retrieval_config=retrieval_config,
            coverage_mode=coverage_mode,
        )
        warnings.extend(evidence.get("warnings") or [])
        evidence_diagnostics = dict(
            evidence.get("diagnostics") or {}
        )
        raw_citations = list(evidence.get("citations") or [])
        citations = self._map_citations(
            raw_citations,
            candidates=candidates,
            prefer_evidence_order=self._prefer_evidence_order(context),
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
                    "scope_hash": knowledge_scope.scope_hash,
                    "revision_scope_count": len(
                        knowledge_scope.bundle_revision_ids
                    ),
                    "max_bundles": retrieval_config["max_bundles"],
                    "max_citations": retrieval_config["max_citations"],
                    "coverage_mode": coverage_mode,
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
                "selector": "bundle-evidence-aggregation-v1",
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

    async def _discover_exact_scope(
        self,
        *,
        context: ExecutionContext,
        query: str,
        collection_ids: tuple[UUID, ...],
        bundle_revision_ids: tuple[UUID, ...],
        per_collection_limit: int,
        coverage_mode: str,
    ) -> dict[str, Any]:
        """按稳定 Revision 分片检索，并在 Runtime 统一合并候选。"""
        ordered = tuple(sorted(bundle_revision_ids, key=lambda item: item.bytes))
        chunks = tuple(
            ordered[offset:offset + KNOWLEDGE_DISCOVERY_REVISION_SCOPE_LIMIT]
            for offset in range(
                0, len(ordered), KNOWLEDGE_DISCOVERY_REVISION_SCOPE_LIMIT
            )
        )
        responses = await asyncio.gather(*(
            self._client.discover(
                query=query,
                collection_ids=collection_ids,
                bundle_revision_ids=chunk,
                domain_id=context.domain_id,
                agent_id=str(context.agent_id),
                auth_context=self._auth_context(context),
                max_security_level=self._security_level(context),
                per_collection_limit=per_collection_limit,
                coverage_mode=coverage_mode,
                run_id=context.run_id,
                task_id=context.task_id,
            )
            for chunk in chunks
        ))
        by_revision: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        for chunk_index, response in enumerate(responses, 1):
            chunk_scope = {
                str(value) for value in chunks[chunk_index - 1]
            }
            warnings.extend(str(item) for item in response.get("warnings") or [])
            diagnostics.append({
                "chunk_index": chunk_index,
                "revision_count": len(chunks[chunk_index - 1]),
                **dict(response.get("diagnostics") or {}),
            })
            for candidate in response.get("candidates") or []:
                revision_id = str(candidate.get("bundle_revision_id") or "")
                if revision_id not in chunk_scope:
                    raise RuntimeError("KNOWLEDGE_SCOPE_CLOSURE_BROKEN")
                current = by_revision.get(revision_id)
                if current is None or self._candidate_sort_key(candidate) < self._candidate_sort_key(current):
                    by_revision[revision_id] = candidate
        candidates = sorted(by_revision.values(), key=self._candidate_sort_key)
        return {
            "candidates": candidates[:per_collection_limit],
            "warnings": list(dict.fromkeys(warnings)),
            "diagnostics": {
                "strategy": "EXACT_REVISION_SCOPE",
                "scope_revision_count": len(ordered),
                "scope_chunk_count": len(chunks),
                "chunks": diagnostics,
            },
        }

    @staticmethod
    def _candidate_sort_key(item: dict[str, Any]) -> tuple[float, str, str]:
        return (
            -float(item.get("rrf_score") or 0),
            str(item.get("collection_id") or ""),
            str(item.get("bundle_revision_id") or ""),
        )

    @staticmethod
    def _assert_scope_closure(
        candidates: list[dict[str, Any]],
        bundle_revision_ids: tuple[UUID, ...],
    ) -> None:
        allowed = {str(value) for value in bundle_revision_ids}
        if any(
            str(item.get("bundle_revision_id") or "") not in allowed
            for item in candidates
        ):
            raise RuntimeError("KNOWLEDGE_SCOPE_CLOSURE_BROKEN")

    async def _retrieve_evidence(
        self,
        *,
        context: ExecutionContext,
        query: str,
        candidates: list[dict[str, Any]],
        scoped: bool,
        retrieval_config: dict[str, int],
        coverage_mode: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """通用 Document Skill 只执行标准 KC 证据检索。"""
        del scoped
        evidence = await self._client.retrieve_evidence(
            query=query,
            candidates=self._evidence_candidates(candidates),
            domain_id=context.domain_id,
            agent_id=str(context.agent_id),
            auth_context=self._auth_context(context),
            max_security_level=self._security_level(context),
            max_evidence=retrieval_config["max_citations"],
            context_limit=retrieval_config["context_limit"],
            coverage_mode=coverage_mode,
            run_id=context.run_id,
            task_id=context.task_id,
        )
        return evidence, candidates


    @staticmethod
    def _evidence_candidates(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """只向 KC 发送 Evidence API 声明的候选字段。"""
        return [{
            "collection_id": item["collection_id"],
            "bundle_id": item["bundle_id"],
            "bundle_revision_id": item["bundle_revision_id"],
            "document_version_ids": (
                [item["document_version_id"]]
                if item.get("document_version_id") else []
            ),
        } for item in candidates]


    async def _scoped_candidates(
        self,
        *,
        context: ExecutionContext,
        allowed_collection_ids: tuple[UUID, ...],
        warnings: list[str],
    ) -> list[dict[str, Any]] | None:
        """把问数选定的 Bundle 转成 KC Evidence 的确定性候选。"""
        scope = next(
            (
                item.payload or {}
                for item in reversed(context.input_artifacts)
                if item.artifact_type == "DOCUMENT_SCOPE"
            ),
            None,
        )
        if not isinstance(scope, dict):
            return None
        raw_targets = [
            item
            for item in list(scope.get("bundle_targets") or [])
            if isinstance(item, dict)
        ]
        if not raw_targets:
            return None
        allowed = {str(value) for value in allowed_collection_ids}
        candidates: list[dict[str, Any]] = []
        retrieval_config = self._retrieval_config(context)
        scoped_targets = await self._prepare_scoped_targets(
            context=context,
            allowed_collection_ids=allowed_collection_ids,
            raw_targets=raw_targets,
            retrieval_config=retrieval_config,
            warnings=warnings,
        )
        semaphore = asyncio.Semaphore(16)

        async def resolve_target(target):
            if not isinstance(target, dict):
                return None, None
            try:
                bundle_id = UUID(str(target["bundle_id"]))
                revision_id = UUID(str(target["bundle_revision_id"]))
                async with semaphore:
                    status = await self._client.get_bundle_status(
                        domain_id=context.domain_id,
                        bundle_id=bundle_id,
                        auth_context=self._auth_context(context),
                    )
                availability = str(
                    status.get("availability_status") or ""
                ).upper()
                current_revision_id = str(
                    status.get("current_revision_id") or ""
                )
                if availability not in {"READY", "PARTIAL"}:
                    return None, "问数命中的 Asset 当前 Bundle 不可检索"
                if current_revision_id != str(revision_id):
                    return None, "问数命中的 Asset 已切换到其他 Bundle Revision"
                collection_id = str(status.get("collection_id") or "")
                if collection_id not in allowed:
                    return None, "问数命中的 Asset 不属于当前 Agent Collection"
                return {
                    "collection_id": collection_id,
                    "bundle_id": str(bundle_id),
                    "bundle_revision_id": str(revision_id),
                    "document_version_ids": [],
                    "display_title": str(target.get("title") or ""),
                }, None
            except (KeyError, TypeError, ValueError):
                return None, "问数命中的 Asset 缺少有效 Bundle 定位信息"
            except Exception:
                return None, "部分问数命中的 Asset 无法解析对应 Bundle"

        resolved = await asyncio.gather(*(
            resolve_target(target) for target in scoped_targets
        ))
        for candidate, warning in resolved:
            if candidate is not None:
                candidates.append(candidate)
            if warning:
                warnings.append(warning)
        return candidates

    async def _prepare_scoped_targets(
        self,
        *,
        context: ExecutionContext,
        allowed_collection_ids: tuple[UUID, ...],
        raw_targets: list[dict[str, Any]],
        retrieval_config: dict[str, int],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """通用路径只应用受控范围上限，专属 Agent 可覆盖候选召回。"""
        del context, allowed_collection_ids, warnings
        return raw_targets[:retrieval_config["candidate_scope_limit"]]

    async def _visual_hits(
        self,
        context: ExecutionContext,
        collection_ids: tuple[UUID, ...],
        bundle_revision_ids: tuple[UUID, ...],
        retrieval_config: dict[str, int | bool],
        image_payloads: list[bytes],
        image_processing: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        if not image_payloads:
            return {"results": []}
        ordered = tuple(
            sorted(bundle_revision_ids, key=lambda item: item.bytes)
        )
        chunks = tuple(
            ordered[
                offset:offset + KNOWLEDGE_DISCOVERY_REVISION_SCOPE_LIMIT
            ]
            for offset in range(
                0,
                len(ordered),
                KNOWLEDGE_DISCOVERY_REVISION_SCOPE_LIMIT,
            )
        )
        encoded_images = [
            base64.b64encode(payload).decode("ascii")
            for payload in image_payloads
        ]
        try:
            responses = await asyncio.gather(*(
                self._client.search_visual(
                    images_base64=encoded_images,
                    collection_ids=collection_ids,
                    bundle_revision_ids=chunk,
                    domain_id=context.domain_id,
                    agent_id=context.agent_id,
                    auth_context=self._auth_context(context),
                    per_image_limit=retrieval_config["max_bundles"],
                    result_limit=retrieval_config["max_bundles"],
                )
                for chunk in chunks
            ))
        except Exception:
            image_processing["visual_search"] = "FAILED"
            warnings.append("图片相似检索暂时不可用，已继续其他检索路径")
            return {"results": []}
        results: list[dict[str, Any]] = []
        searched_set: set[str] = set()
        skipped_set: set[str] = set()
        for chunk, response in zip(chunks, responses, strict=True):
            chunk_results = list(response.get("results") or [])
            self._assert_scope_closure(chunk_results, chunk)
            results.extend(chunk_results)
            searched_set.update(
                str(value)
                for value in response.get("searched_collection_ids") or []
            )
            skipped_set.update(
                str(value)
                for value in response.get("skipped_collection_ids") or []
            )
        results.sort(
            key=lambda item: (
                -float(item.get("score") or 0),
                str(item.get("bundle_revision_id") or ""),
                str(item.get("visual_asset_id") or ""),
            )
        )
        searched = sorted(searched_set)
        skipped = sorted(skipped_set - searched_set)
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
        return {
            "results": results[:retrieval_config["max_bundles"]],
            "searched_collection_ids": searched,
            "skipped_collection_ids": skipped,
        }

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

    @staticmethod
    def _knowledge_scope(context: ExecutionContext) -> KnowledgeScopeSnapshotV1:
        """只读取 Run 已冻结的 Scope，禁止从绑定或页面动态扩展范围。"""
        raw = context.config_snapshot.get("retrieval", {}).get(
            "knowledge_scope"
        )
        try:
            scope = KnowledgeScopeSnapshotV1.model_validate(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("KNOWLEDGE_SCOPE_INVALID") from exc
        if scope.domain_id != context.domain_id:
            raise RuntimeError("KNOWLEDGE_SCOPE_DOMAIN_MISMATCH")
        if scope.consumer_agent_id != context.agent_id:
            raise RuntimeError("KNOWLEDGE_SCOPE_AGENT_MISMATCH")
        return scope

    @staticmethod
    def _scope_keywords(payload: dict[str, Any]) -> tuple[str, ...]:
        raw = payload.get("keywords") or []
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple)):
            values = list(raw)
        else:
            return ()
        keywords: list[str] = []
        seen: set[str] = set()
        for item in values:
            text = str(item or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            keywords.append(text)
        return tuple(keywords)

    @staticmethod
    def _join_scope_query(query: str, keywords: tuple[str, ...]) -> str:
        parts: list[str] = []
        existing = query.casefold()
        if query:
            parts.append(query)
        for word in keywords:
            if word.casefold() in existing:
                continue
            parts.append(word)
            existing = f"{existing} {word.casefold()}"
        return " ".join(parts).strip()

    @staticmethod
    def _standalone_query(context: ExecutionContext) -> str:
        document_scopes = [
            item
            for item in context.input_artifacts
            if item.artifact_type == "DOCUMENT_SCOPE"
        ]
        if document_scopes:
            payload = document_scopes[-1].payload or {}
            scoped_query = str(payload.get("query") or "").strip()
            keywords = KnowledgeRetrievalSkill._scope_keywords(payload)
            combined = KnowledgeRetrievalSkill._join_scope_query(
                scoped_query, keywords
            )
            if combined:
                return combined
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
        return KnowledgeRetrievalSkill._knowledge_scope(
            context
        ).security_level

    @staticmethod
    def _retrieval_config(
        context: ExecutionContext,
    ) -> dict[str, int]:
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
            "candidate_scope_limit": max(
                1,
                min(int(retrieval.get("candidate_scope_limit", 1000)), 1000),
            ),
        }

    @staticmethod
    def _map_citations(
        raw_citations: list[dict[str, Any]],
        *,
        candidates: list[dict[str, Any]],
        prefer_evidence_order: bool = False,
    ) -> list[Citation]:
        titles = {
            str(item["bundle_id"]): str(item.get("display_title") or "")
            for item in candidates
        }
        groups_by_bundle: dict[str, list[dict[str, Any]]] = {}
        for group in raw_citations:
            groups_by_bundle.setdefault(str(group["bundle_id"]), []).append(
                group
            )
        candidate_order = [str(item["bundle_id"]) for item in candidates]
        ordered_bundle_ids = list(dict.fromkeys(
            (*groups_by_bundle.keys(), *candidate_order)
            if prefer_evidence_order
            else (*candidate_order, *groups_by_bundle.keys())
        ))
        result: list[Citation] = []
        for bundle_key in ordered_bundle_ids:
            groups = groups_by_bundle.get(bundle_key, [])
            if not groups:
                continue
            selected: list[dict[str, Any]] = []
            for group in groups:
                items = list(group.get("items") or [])
                # 引用正文必须带上锚点邻接上下文，否则会议“目的”命中后讨论表进不了模型。
                usable = [
                    item
                    for item in items
                    if str(item.get("final_role") or "PRIMARY") in {
                        "PRIMARY",
                        "STRUCTURAL_CONTEXT",
                        "NEIGHBOR",
                    }
                ]
                selected.extend(usable or items)
            if not selected:
                continue
            first_group = groups[0]
            first = selected[0].get("evidence") or {}
            if not first.get("document_id"):
                continue
            excerpt_parts: list[str] = []
            evidence_ids: list[UUID] = []
            for item in selected:
                evidence = item.get("evidence") or {}
                content = str(evidence.get("content_text") or "").strip()
                if content:
                    document_name = str(
                        evidence.get("document_name")
                        or evidence.get("external_document_id")
                        or "Bundle 正文"
                    )
                    heading = " > ".join(evidence.get("heading_path") or [])
                    location = f" · {heading}" if heading else ""
                    excerpt_parts.append(
                        f"文档：{document_name}{location}\n{content}"
                    )
                evidence_id = evidence.get("evidence_id")
                if evidence_id:
                    evidence_ids.append(UUID(str(evidence_id)))
            excerpt = "\n\n".join(dict.fromkeys(excerpt_parts))[:4000]
            provenance = first.get("provenance") or {}
            bundle_id = UUID(bundle_key)
            bundle_title = (
                str(first.get("bundle_title") or "").strip()
                or titles.get(str(bundle_id))
                or "未命名 Bundle"
            )
            result.append(
                Citation(
                    citation_id=UUID(str(first_group["citation_id"])),
                    citation_label=f"C{len(result) + 1}",
                    collection_id=UUID(str(first_group["collection_id"])),
                    bundle_id=bundle_id,
                    bundle_revision_id=UUID(
                        str(first_group["bundle_revision_id"])
                    ),
                    document_id=UUID(str(first["document_id"])),
                    document_version_id=UUID(
                        str(first_group["document_version_id"])
                    ),
                    evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                    title=bundle_title,
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
                    relevance_reason="混合检索候选 Bundle 的正文证据",
                    source_hash=(
                        first.get("content_hash")
                        or provenance.get("source_hash")
                        or provenance.get("content_hash")
                    ),
                )
            )
        return result

    @staticmethod
    def _prefer_evidence_order(context: ExecutionContext) -> bool:
        del context
        return False

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
