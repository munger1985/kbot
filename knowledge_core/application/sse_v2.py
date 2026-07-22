"""Serialization boundary for V2 grounded answer events."""
from dataclasses import asdict

from knowledge_core.application.grounding import GroundingResult


def build_grounded_sse_payload(*, answer_markdown: str, result: GroundingResult) -> dict:
    """Return the final V2 event without mixing V1 ``doc_results``."""
    return {
        "answer": answer_markdown,
        "citations_v2": [
            {"label": citation.citation_label, "citation_group": asdict(citation)}
            for citation in result.citations
        ],
        "doc_results_v2": [asdict(item) for item in result.doc_results_v2],
        "grounding_status": result.grounding_status,
        "unsupported_claim_ids": result.unsupported_claim_ids,
    }
