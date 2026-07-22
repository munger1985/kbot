"""Post-generation grounding and frontend document projection."""
from dataclasses import dataclass, field
import re
from typing import Any, Sequence

from knowledge_core.application.evidence_retrieval import CitationGroup, EvidenceGroupItem, EvidenceHit


@dataclass(frozen=True)
class AnswerClaim:
    claim_id: str
    text: str
    citation_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerDraft:
    answer_markdown: str
    claims: tuple[AnswerClaim, ...] = ()
    used_citation_labels: tuple[str, ...] = ()
    selected_bundle_ids: tuple[int, ...] = ()


@dataclass
class DocumentResultV2:
    bundle_id: int
    bundle_revision_id: int
    title: str
    collection_id: int
    citation_labels: list[str]
    used_evidence_ids: list[int]
    document_version_ids: list[int]
    locators: list[dict[str, Any]]


@dataclass
class GroundingResult:
    answer_markdown: str
    claims: list[AnswerClaim]
    used_citation_labels: list[str]
    citations: list[CitationGroup]
    doc_results_v2: list[DocumentResultV2]
    grounding_status: str
    dropped_citation_labels: list[str] = field(default_factory=list)
    unsupported_claim_ids: list[str] = field(default_factory=list)


class AnswerGroundingVerifier:
    """Validate model self-reported citations against a verified Citation Pack."""

    def verify(self, *, draft: AnswerDraft, citation_pack: Sequence[CitationGroup]) -> GroundingResult:
        pack = {citation.citation_label: citation for citation in citation_pack}
        requested = list(dict.fromkeys(draft.used_citation_labels))
        valid_labels = [
            label for label in requested
            if label in pack and pack[label].primary_evidence_ids
        ]
        dropped = [label for label in requested if label not in valid_labels]
        used = set(valid_labels)
        unsupported: list[str] = []
        normalized_claims: list[AnswerClaim] = []
        for claim in draft.claims:
            normalized_labels = tuple(label for label in claim.citation_labels if label in used)
            normalized_claims.append(AnswerClaim(claim.claim_id, claim.text, normalized_labels))
            if claim.citation_labels and not set(claim.citation_labels).issubset(used):
                unsupported.append(claim.claim_id)
            elif claim.citation_labels and not set(claim.citation_labels) & used:
                unsupported.append(claim.claim_id)
            elif not claim.citation_labels and claim.text.strip():
                unsupported.append(claim.claim_id)
        if unsupported:
            status = "PARTIAL" if valid_labels else "INSUFFICIENT"
        elif draft.claims or valid_labels:
            status = "VERIFIED"
        else:
            status = "INSUFFICIENT"
        citations = [pack[label] for label in valid_labels]
        selected = set(draft.selected_bundle_ids)
        if selected:
            cited_bundles = {citation.bundle_id for citation in citations}
            # A model may not surface an Asset that has no used Citation Group.
            selected = selected & cited_bundles
        else:
            selected = {citation.bundle_id for citation in citations}
        docs = _project_doc_results(citations, selected)
        return GroundingResult(
            answer_markdown=_strip_invalid_labels(draft.answer_markdown, set(valid_labels)),
            claims=normalized_claims, used_citation_labels=valid_labels,
            citations=citations, doc_results_v2=docs,
            grounding_status=status, dropped_citation_labels=dropped,
            unsupported_claim_ids=unsupported,
        )


def citation_groups_from_payload(payload: Any) -> list[CitationGroup]:
    """Hydrate the JSON Citation Pack returned by KC into verifier DTOs.

    The HTTP boundary intentionally carries dictionaries, while grounding is
    kept on typed domain objects.  This adapter is tolerant of omitted
    optional locator/provenance fields so a partially populated test fixture or
    an older KC node cannot make the answer path fail before verification.
    """
    if isinstance(payload, dict):
        payload = payload.get("citations") or []
    if not isinstance(payload, (list, tuple)):
        return []
    result: list[CitationGroup] = []
    for raw in payload:
        if isinstance(raw, CitationGroup):
            result.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        items: list[EvidenceGroupItem] = []
        for item in raw.get("items") or []:
            if not isinstance(item, dict):
                continue
            evidence_raw = item.get("evidence") or {}
            if not isinstance(evidence_raw, dict):
                continue
            evidence = _evidence_hit_from_payload(evidence_raw, raw)
            items.append(EvidenceGroupItem(
                item_label=str(item.get("item_label") or ""), evidence=evidence,
                input_role=str(item.get("input_role") or ""),
                final_role=str(item.get("final_role") or ""),
                promoted_from_context=bool(item.get("promoted_from_context", False)),
            ))
        result.append(CitationGroup(
            citation_label=str(raw.get("citation_label") or raw.get("label") or ""),
            collection_id=int(raw.get("collection_id") or 0),
            bundle_id=int(raw.get("bundle_id") or 0),
            bundle_revision_id=int(raw.get("bundle_revision_id") or 0),
            document_version_id=int(raw.get("document_version_id") or 0),
            parse_view_id=int(raw.get("parse_view_id") or 0),
            primary_evidence_ids=[int(value) for value in raw.get("primary_evidence_ids") or []],
            structural_context_ids=[int(value) for value in raw.get("structural_context_ids") or []],
            neighbor_evidence_ids=[int(value) for value in raw.get("neighbor_evidence_ids") or []],
            items=items,
        ))
    return [item for item in result if item.citation_label]


def _evidence_hit_from_payload(raw: dict[str, Any], citation: dict[str, Any]) -> EvidenceHit:
    return EvidenceHit(
        evidence_id=int(raw.get("evidence_id") or 0),
        collection_id=int(raw.get("collection_id") or citation.get("collection_id") or 0),
        bundle_id=int(raw.get("bundle_id") or citation.get("bundle_id") or 0),
        bundle_revision_id=int(raw.get("bundle_revision_id") or citation.get("bundle_revision_id") or 0),
        bundle_revision_document_id=_optional_int(raw.get("bundle_revision_document_id")),
        document_id=int(raw.get("document_id") or 0),
        document_version_id=int(raw.get("document_version_id") or citation.get("document_version_id") or 0),
        parse_view_id=int(raw.get("parse_view_id") or citation.get("parse_view_id") or 0),
        evidence_key=str(raw.get("evidence_key") or ""),
        evidence_type=str(raw.get("evidence_type") or "TEXT"),
        content_text=str(raw.get("content_text") or ""),
        retrieval_text=str(raw.get("retrieval_text") or raw.get("content_text") or ""),
        heading_path=tuple(str(value) for value in raw.get("heading_path") or ()),
        locator=dict(raw.get("locator") or {}),
        source_spans=tuple(raw.get("source_spans") or ()),
        provenance=dict(raw.get("provenance") or {}),
        section_key=raw.get("section_key"),
        parent_evidence_key=raw.get("parent_evidence_key"),
        ordinal=int(raw.get("ordinal") or 0),
        quality_score=raw.get("quality_score"),
        local_rank=int(raw.get("local_rank") or 0),
        channel=str(raw.get("channel") or "unknown"),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _project_doc_results(citations: Sequence[CitationGroup], selected_bundles: set[int]) -> list[DocumentResultV2]:
    grouped: dict[tuple[int, int], DocumentResultV2] = {}
    for citation in citations:
        if citation.bundle_id not in selected_bundles:
            continue
        key = (citation.collection_id, citation.bundle_id)
        result = grouped.setdefault(key, DocumentResultV2(
            bundle_id=citation.bundle_id, bundle_revision_id=citation.bundle_revision_id,
            title=f"Bundle {citation.bundle_id}", collection_id=citation.collection_id,
            citation_labels=[], used_evidence_ids=[], document_version_ids=[], locators=[],
        ))
        if citation.citation_label not in result.citation_labels:
            result.citation_labels.append(citation.citation_label)
        for item in citation.items:
            if item.evidence.evidence_id not in citation.primary_evidence_ids:
                continue
            if item.evidence.evidence_id not in result.used_evidence_ids:
                result.used_evidence_ids.append(item.evidence.evidence_id)
            if item.evidence.document_version_id not in result.document_version_ids:
                result.document_version_ids.append(item.evidence.document_version_id)
            result.locators.append(item.evidence.locator)
    return list(grouped.values())


def _strip_invalid_labels(markdown: str, valid_labels: set[str]) -> str:
    """Remove bracketed fabricated citation labels before SSE delivery."""
    return re.sub(
        r"\[C\d+\]",
        lambda match: match.group(0) if match.group(0)[1:-1] in valid_labels else "",
        markdown,
    )
