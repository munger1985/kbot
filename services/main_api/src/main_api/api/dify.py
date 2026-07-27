"""Dify External Knowledge Base Retrieval 协议适配器。"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from platform_clients import KnowledgeCoreClient
from platform_core.contracts import PUBLIC_API_V1


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DifyRetrievalSetting(_Contract):
    top_k: int = Field(default=10, ge=1, le=50)
    score_threshold: float = Field(default=0, ge=0, le=1)


class DifyRetrievalRequest(_Contract):
    knowledge_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=8000)
    retrieval_setting: DifyRetrievalSetting
    metadata_condition: dict[str, Any] | None = None


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/integrations/dify",
    tags=["Dify Integration"],
)


@router.post("/retrieval")
async def retrieve_for_dify(
    payload: DifyRetrievalRequest, request: Request
) -> dict[str, list[dict[str, Any]]]:
    try:
        agent_id = UUID(payload.knowledge_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_KNOWLEDGE_ID",
                "message": "knowledge_id 必须是 KBot Agent UUID",
            },
        ) from exc
    if payload.metadata_condition:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "DIFY_METADATA_FILTER_UNSUPPORTED",
                "message": "当前 Dify Adapter 不支持额外元数据过滤",
            },
        )
    context = request.state.auth_context
    try:
        domain_id = int(context.domain_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_DOMAIN", "message": "Domain 无效"},
        ) from exc
    client = cast(
        KnowledgeCoreClient, request.app.state.knowledge_core_client
    )
    bindings = await client.list_agent_bindings(
        domain_id=domain_id,
        agent_id=agent_id,
        auth_context=context,
    )
    collection_ids = [
        UUID(str(item["collection_id"]))
        for item in bindings.get("bindings", [])
        if item.get("status") == "ACTIVE"
    ]
    if not collection_ids:
        return {"records": []}
    limit = payload.retrieval_setting.top_k
    discovery = await client.discover(
        query=payload.query,
        collection_ids=collection_ids,
        domain_id=domain_id,
        agent_id=str(agent_id),
        auth_context=context,
        per_collection_limit=limit,
    )
    candidates = list(discovery.get("candidates") or [])[:limit]
    if not candidates:
        return {"records": []}
    evidence = await client.retrieve_evidence(
        query=payload.query,
        candidates=[
            {
                "collection_id": item["collection_id"],
                "bundle_id": item["bundle_id"],
                "bundle_revision_id": item["bundle_revision_id"],
                "document_version_ids": [],
            }
            for item in candidates
        ],
        domain_id=domain_id,
        agent_id=str(agent_id),
        auth_context=context,
        max_evidence=limit,
        context_limit=2,
    )
    return {
        "records": _records(
            candidates=candidates,
            citations=list(evidence.get("citations") or []),
            limit=limit,
            threshold=payload.retrieval_setting.score_threshold,
        )
    }


def _records(
    *,
    candidates: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    limit: int,
    threshold: float,
) -> list[dict[str, Any]]:
    raw_scores = [
        max(0.0, float(item.get("rrf_score") or 0))
        for item in candidates
    ]
    maximum = max(raw_scores, default=0)
    by_bundle = {
        str(item["bundle_id"]): {
            **item,
            "normalized_score": (
                float(item.get("rrf_score") or 0) / maximum
                if maximum > 0
                else 0
            ),
        }
        for item in candidates
    }
    output: list[dict[str, Any]] = []
    for citation in citations:
        candidate = by_bundle.get(str(citation.get("bundle_id")))
        if candidate is None:
            continue
        score = float(candidate["normalized_score"])
        if score < threshold:
            continue
        items = [
            item
            for item in citation.get("items", [])
            if item.get("final_role") == "PRIMARY"
        ]
        if not items:
            continue
        evidences = [
            item.get("evidence") or {}
            for item in items
        ]
        content = "\n".join(
            str(item.get("content_text") or "").strip()
            for item in evidences
            if str(item.get("content_text") or "").strip()
        )
        if not content:
            continue
        first = evidences[0]
        output.append(
            {
                "metadata": {
                    "source": first.get("document_name")
                    or candidate.get("display_title"),
                    "collection_id": candidate.get("collection_id"),
                    "bundle_id": candidate.get("bundle_id"),
                    "document_id": first.get("document_id"),
                    "document_version_id": first.get(
                        "document_version_id"
                    ),
                    "evidence_ids": [
                        item.get("evidence_id") for item in evidences
                    ],
                    "locator": first.get("locator") or {},
                },
                "score": round(score, 6),
                "title": first.get("document_name")
                or candidate.get("display_title")
                or "Untitled",
                "content": content[:12000],
            }
        )
        if len(output) >= limit:
            break
    return output
