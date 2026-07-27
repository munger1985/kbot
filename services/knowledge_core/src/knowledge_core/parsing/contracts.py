"""Validation and hashing rules shared by Parser Worker and KC."""

from uuid import UUID
import hashlib
import json
import re
from typing import Any


EVIDENCE_TYPES = frozenset({
    "DOCUMENT", "SECTION", "PARAGRAPH", "TABLE", "TABLE_ROW",
    "IMAGE", "SHEET", "CELL_RANGE",
})
REQUIRED_ARTIFACTS = frozenset({
    "raw_docling", "atom_ir", "structure_ir", "evidence_manifest",
})
OPTIONAL_ARTIFACTS = frozenset({
    "deepseek_ocr_analysis",
    "spreadsheet_artifact",
    "visual_analysis",
})


def _canonical_json_default(item: Any) -> str:
    if isinstance(item, UUID):
        return str(item)
    raise TypeError(f"不支持规范 JSON 序列化的类型：{type(item).__name__}")


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        default=_canonical_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_source_spans(source_spans: list[dict[str, Any]]) -> None:
    if not source_spans:
        raise ValueError("source_spans must contain at least one Atom reference")
    for index, span in enumerate(source_spans):
        atom_id = span.get("atom_id")
        if not isinstance(atom_id, str) or not atom_id.strip():
            raise ValueError(f"source_spans[{index}].atom_id is required")
        start, end = span.get("char_start"), span.get("char_end")
        if (start is None) != (end is None):
            raise ValueError(f"source_spans[{index}] character bounds must be provided together")
        if start is not None and (not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start):
            raise ValueError(f"source_spans[{index}] character bounds are invalid")
        cell_range = span.get("cell_range")
        if cell_range is not None and (
            not isinstance(cell_range, str) or not re.fullmatch(r"[A-Z]+[1-9]\d*:[A-Z]+[1-9]\d*", cell_range)
        ):
            raise ValueError(f"source_spans[{index}].cell_range is invalid")
        if start is not None and cell_range is not None:
            raise ValueError(f"source_spans[{index}] cannot mix character and cell ranges")


def validate_locator(schema: str, locator: dict[str, Any]) -> None:
    if schema == "document/v1":
        pages = locator.get("pages")
        if not isinstance(pages, list) or not pages:
            raise ValueError("document/v1 locator requires pages")
        for index, page in enumerate(pages):
            bbox = page.get("bbox") if isinstance(page, dict) else None
            if not isinstance(page.get("page_no"), int) or page["page_no"] < 1:
                raise ValueError(f"locator.pages[{index}].page_no is invalid")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"locator.pages[{index}].bbox is invalid")
            x0, y0, x1, y1 = bbox
            if not all(isinstance(value, (int, float)) for value in bbox) or not (
                0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1
            ):
                raise ValueError(f"locator.pages[{index}].bbox is not normalized")
            if page.get("coordinate_space") != "page_normalized_top_left":
                raise ValueError(f"locator.pages[{index}].coordinate_space is invalid")
        return
    if schema == "document-logical/v1":
        refs = locator.get("source_refs")
        if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) and ref for ref in refs):
            raise ValueError("document-logical/v1 locator requires source_refs")
        if locator.get("coordinate_space") != "logical_document":
            raise ValueError("document-logical/v1 coordinate_space is invalid")
        return
    if schema == "spreadsheet/v1":
        for field in ("sheet_name", "sheet_ref"):
            if not isinstance(locator.get(field), str) or not locator[field]:
                raise ValueError(f"spreadsheet/v1 locator requires {field}")
        cell_range = locator.get("cell_range")
        if cell_range is not None and not re.fullmatch(r"[A-Z]+[1-9]\d*:[A-Z]+[1-9]\d*", cell_range):
            raise ValueError("spreadsheet/v1 cell_range is invalid")
        return
    raise ValueError(f"unsupported locator_schema_version: {schema}")


def validate_artifact_manifest(manifest: dict[str, Any]) -> None:
    missing = REQUIRED_ARTIFACTS.difference(manifest)
    if missing:
        raise ValueError(f"artifact_manifest is missing: {', '.join(sorted(missing))}")
    extra = set(manifest).difference(REQUIRED_ARTIFACTS | OPTIONAL_ARTIFACTS)
    if extra:
        raise ValueError(f"artifact_manifest has unsupported entries: {', '.join(sorted(extra))}")
    for name in REQUIRED_ARTIFACTS:
        descriptor = manifest[name]
        if not isinstance(descriptor, dict):
            raise ValueError(f"artifact_manifest.{name} must be an object")
        for field in ("uri", "sha256", "schema", "generator"):
            if not isinstance(descriptor.get(field), str) or not descriptor[field].strip():
                raise ValueError(f"artifact_manifest.{name}.{field} is required")
        digest = descriptor["sha256"].lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"artifact_manifest.{name}.sha256 must be lowercase SHA-256")


def validate_quality_report(report: dict[str, Any]) -> None:
    if report.get("passed") is not True:
        raise ValueError("quality_report.passed must be true before activation")
    hard_failures = report.get("hard_failures")
    if not isinstance(hard_failures, list) or hard_failures:
        raise ValueError("quality_report.hard_failures must be an empty list")
    if not isinstance(report.get("metrics"), dict):
        raise ValueError("quality_report.metrics is required")


def evidence_fingerprint(
    *, content_text: str, source_spans: list[dict[str, Any]], locator: dict[str, Any]
) -> str:
    return canonical_json_hash({
        "content_text": content_text,
        "source_spans": source_spans,
        "locator": locator,
    })


def build_evidence_key(
    *, parse_view_id: UUID, source_spans: list[dict[str, Any]],
    fragment_index: int, evidence_type: str,
) -> str:
    validate_source_spans(source_spans)
    if fragment_index < 0 or evidence_type not in EVIDENCE_TYPES:
        raise ValueError("invalid Evidence key components")
    return f"ev1:{parse_view_id}:{canonical_json_hash(source_spans)}:{fragment_index}:{evidence_type}"


def build_output_fingerprint(
    *, artifact_hashes: dict[str, str], evidence_keys: list[str]
) -> str:
    if not REQUIRED_ARTIFACTS.issubset(artifact_hashes) or set(artifact_hashes).difference(REQUIRED_ARTIFACTS | OPTIONAL_ARTIFACTS):
        raise ValueError("output fingerprint contains an unsupported Parser artifact set")
    return canonical_json_hash({"artifacts": artifact_hashes, "evidence_keys": evidence_keys})
