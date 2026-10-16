"""Build additive Evidence rows from asynchronous full-page VLM results."""

import re
from uuid import UUID

from knowledge_core.parsing import build_evidence_key


_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


def build_visual_page_evidence(
    *, parse_view_id: UUID, enrichment, max_tokens: int = 900,
) -> list[dict]:
    output: list[dict] = []
    for page in sorted(enrichment.page_results, key=lambda item: item.page_no):
        content = page.markdown.strip()
        matches = list(_TOKEN.finditer(content))
        if not matches:
            continue
        atom_id = f"visual-page:{page.page_no}"
        model = str(
            getattr(page, "served_model_name", None)
            or getattr(page, "model_id", "")
        )
        for fragment_index, offset in enumerate(
            range(0, len(matches), max_tokens)
        ):
            group = matches[offset : offset + max_tokens]
            start, end = group[0].start(), group[-1].end()
            fragment = content[start:end].strip()
            if not fragment:
                continue
            source_spans = [{
                "atom_id": atom_id,
                "char_start": start,
                "char_end": end,
            }]
            output.append({
                "evidence_key": build_evidence_key(
                    parse_view_id=parse_view_id,
                    source_spans=source_spans,
                    fragment_index=fragment_index,
                    evidence_type="IMAGE",
                ),
                "evidence_type": "IMAGE",
                "ordinal": 1_000_000 + page.page_no * 1_000 + fragment_index,
                "fragment_index": fragment_index,
                "content_text": fragment,
                "source_spans": source_spans,
                "locator_schema_version": "document/v1",
                "locator": {"pages": [{
                    "page_no": page.page_no,
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "coordinate_space": "page_normalized_top_left",
                }]},
                "provenance": {
                    "planner": "visual-page-evidence/v1",
                    "extractors": ["VLM"],
                    "model": model,
                    "page_no": page.page_no,
                },
                "parent_evidence_key": None,
                "source_item_ref": f"#/visual-pages/{page.page_no}",
                "heading_path": [],
                "section_key": None,
                "hierarchy_depth": 0,
                "heading_level": None,
                "page_start": page.page_no,
                "page_end": page.page_no,
                "language_code": None,
                "token_count": len(group),
                "quality_score": page.confidence,
            })
    return output
