"""Lossless adapter from DoclingDocument to the KC Atom IR."""

from uuid import UUID
import hashlib
from typing import Any

from docling_core.types.doc import DoclingDocument

from .ir import Atom, AtomIr, AtomLocator, PageGeometry


_LABEL_TO_ATOM_TYPE = {
    "title": "TITLE_CANDIDATE",
    "section_header": "TITLE_CANDIDATE",
    "text": "TEXT",
    "paragraph": "TEXT",
    "handwritten_text": "TEXT",
    "document_index": "LIST_ITEM",
    "reference": "TEXT",
    "checkbox_selected": "TEXT",
    "checkbox_unselected": "TEXT",
    "form": "TEXT",
    "key_value_region": "TEXT",
    "grading_scale": "TEXT",
    "empty_value": "TEXT",
    "field_region": "TEXT",
    "field_heading": "TITLE_CANDIDATE",
    "field_item": "TEXT",
    "field_key": "TEXT",
    "field_value": "TEXT",
    "field_hint": "TEXT",
    "marker": "TEXT",
    "list_item": "LIST_ITEM",
    "table": "TABLE",
    "picture": "PICTURE",
    "chart": "PICTURE",
    "caption": "CAPTION",
    "formula": "FORMULA",
    "code": "CODE",
    "footnote": "FOOTNOTE",
    "page_header": "HEADER",
    "page_footer": "FOOTER",
}


class DoclingAdapterError(ValueError):
    """Docling output cannot be represented without losing provenance."""


class DoclingAtomNormalizer:
    def __init__(self, *, generator_version: str):
        if not generator_version.strip():
            raise ValueError("generator_version is required")
        self._generator_version = generator_version

    def normalize(self, *, document_version_id: UUID, document: DoclingDocument) -> AtomIr:
        pages, page_items = self._pages(document)
        atoms: list[Atom] = []
        seen_refs: set[str] = set()

        for reading_order, (item, _) in enumerate(document.iterate_items()):
            label = self._label(item)
            atom_type = _LABEL_TO_ATOM_TYPE.get(label)
            if atom_type is None:
                continue
            source_ref = str(getattr(item, "self_ref", "") or "").strip()
            if not source_ref:
                raise DoclingAdapterError(f"Docling {label} item has no self_ref")
            if source_ref in seen_refs:
                raise DoclingAdapterError(f"duplicate Docling self_ref: {source_ref}")
            locators = self._locators(item, page_items)
            content_text = self._content(item, atom_type, document)
            atom_id = self._atom_id(document_version_id, source_ref, atom_type)
            atoms.append(Atom(
                atom_id=atom_id,
                source_ref=source_ref,
                atom_type=atom_type,
                content_text=content_text,
                locators=locators,
                reading_order_hint=reading_order,
                original_label=label,
                style=self._style(item, document),
                confidence=self._confidence(item),
                provenance=(self._provenance(item, document),),
                repeated_region_key=self._repeated_region_key(atom_type, content_text),
            ))
            seen_refs.add(source_ref)
            for annotation_index, annotation in enumerate(getattr(item, "annotations", ()) or ()):
                annotation_text = str(getattr(annotation, "text", "") or "").strip()
                annotation_source = str(getattr(annotation, "provenance", "") or "").lower()
                if not annotation_text or not any(
                    marker in annotation_source for marker in ("ocr", "vlm", "visual")
                ):
                    continue
                derived_ref = f"{source_ref}::annotation:{annotation_index}"
                derived_id = self._atom_id(document_version_id, derived_ref, "VISUAL_DESCRIPTION")
                extractor = "OCR" if "ocr" in annotation_source else "VLM"
                atoms.append(Atom(
                    atom_id=derived_id, source_ref=derived_ref,
                    atom_type="VISUAL_DESCRIPTION", content_text=annotation_text,
                    locators=locators, reading_order_hint=reading_order,
                    original_label="visual_description", confidence=0.75,
                    provenance=({
                        "extractor": extractor, "derived": True,
                        "source_ref": source_ref, "annotation_provenance": annotation_source,
                        "adapter_version": self._generator_version,
                    },),
                ))
                seen_refs.add(derived_ref)

        result = AtomIr(
            document_version_id=document_version_id,
            pages=pages,
            atoms=tuple(atoms),
            generator={"name": "docling-atom-normalizer", "version": self._generator_version},
        )
        result.validate()
        return result

    @staticmethod
    def _pages(document: DoclingDocument) -> tuple[tuple[PageGeometry, ...], dict[int, Any]]:
        page_items: dict[int, Any] = {}
        pages: list[PageGeometry] = []
        for page in document.pages.values():
            page_no = int(page.page_no)
            page_items[page_no] = page
            pages.append(PageGeometry(page_no=page_no, width=float(page.size.width), height=float(page.size.height)))
        return tuple(sorted(pages, key=lambda value: value.page_no)), page_items

    @staticmethod
    def _label(item: Any) -> str:
        label = getattr(item, "label", "")
        return str(getattr(label, "value", label) or "").lower()

    @staticmethod
    def _content(item: Any, atom_type: str, document: DoclingDocument) -> str:
        if atom_type == "TABLE":
            try:
                return str(item.export_to_markdown(document) or "").strip()
            except Exception as exc:
                raise DoclingAdapterError(f"cannot serialize table {item.self_ref}: {exc}") from exc
        return str(getattr(item, "text", "") or "").strip()

    @staticmethod
    def _locators(item: Any, page_items: dict[int, Any]) -> tuple[AtomLocator, ...]:
        locators: list[AtomLocator] = []
        for provenance in getattr(item, "prov", ()) or ():
            page_no = int(provenance.page_no)
            page = page_items.get(page_no)
            if page is None or getattr(provenance, "bbox", None) is None:
                raise DoclingAdapterError(f"item {item.self_ref} has incomplete page provenance")
            original = provenance.bbox
            top_left = original.to_top_left_origin(float(page.size.height))
            normalized = top_left.normalized(page.size)
            x0, x1 = sorted((float(normalized.l), float(normalized.r)))
            y0, y1 = sorted((float(normalized.t), float(normalized.b)))
            locators.append(AtomLocator(
                page_no=page_no,
                bbox=(x0, y0, x1, y1),
                original_bbox={
                    "l": float(original.l), "t": float(original.t),
                    "r": float(original.r), "b": float(original.b),
                    "coord_origin": str(getattr(original.coord_origin, "value", original.coord_origin)),
                },
            ))
        if not locators:
            if page_items:
                raise DoclingAdapterError(f"item {item.self_ref} has no source locator")
            locators.append(AtomLocator(logical_ref=str(item.self_ref)))
        return tuple(locators)

    @staticmethod
    def _style(item: Any, document: DoclingDocument) -> dict[str, Any]:
        style: dict[str, Any] = {}
        formatting = getattr(item, "formatting", None)
        if formatting is not None:
            style["formatting"] = formatting.model_dump(mode="json")
        level = getattr(item, "level", None)
        if level is not None:
            style["declared_level"] = int(level)
        data = getattr(item, "data", None)
        if data is not None and hasattr(data, "table_cells"):
            parent_ref = getattr(getattr(item, "parent", None), "cref", None)
            sheet = next((
                group for group in getattr(document, "groups", ())
                if getattr(group, "self_ref", None) == parent_ref
                and str(getattr(getattr(group, "label", None), "value", getattr(group, "label", ""))) == "sheet"
            ), None)
            if sheet is not None:
                rows, columns = int(data.num_rows), int(data.num_cols)
                style["spreadsheet"] = {
                    "sheet_ref": sheet.self_ref,
                    "sheet_name": sheet.name,
                    "table_ref": item.self_ref,
                    "cell_range": DoclingAtomNormalizer._cell_range(rows, columns),
                    "row_start": 1, "row_end": rows,
                    "column_start": 1, "column_end": columns,
                    "cells": [{
                        "row_start": int(cell.start_row_offset_idx) + 1,
                        "row_end": int(cell.end_row_offset_idx),
                        "column_start": int(cell.start_col_offset_idx) + 1,
                        "column_end": int(cell.end_col_offset_idx),
                        "text": cell.text,
                        "column_header": bool(cell.column_header),
                        "row_header": bool(cell.row_header),
                    } for cell in data.table_cells],
                }
        return style

    @staticmethod
    def _cell_range(rows: int, columns: int) -> str:
        if rows < 1 or columns < 1:
            raise DoclingAdapterError("spreadsheet table has invalid dimensions")
        value, letters = columns, ""
        while value:
            value, remainder = divmod(value - 1, 26)
            letters = chr(65 + remainder) + letters
        return f"A1:{letters}{rows}"

    @staticmethod
    def _confidence(item: Any) -> float:
        values = [
            float(value)
            for provenance in (getattr(item, "prov", ()) or ())
            if (value := getattr(provenance, "confidence", None)) is not None
        ]
        return min(values) if values else 1.0

    def _provenance(self, item: Any, document: DoclingDocument) -> dict[str, Any]:
        annotations = []
        for annotation in getattr(item, "annotations", ()) or ():
            annotations.append({
                "kind": annotation.__class__.__name__,
                "provenance": getattr(annotation, "provenance", None),
            })
        return {
            "extractor": "DOCLING",
            "adapter_version": self._generator_version,
            "document_schema": getattr(document, "schema_name", None),
            "document_schema_version": getattr(document, "version", None),
            "source_ref": item.self_ref,
            "annotations": annotations,
        }

    @staticmethod
    def _atom_id(document_version_id: UUID, source_ref: str, atom_type: str) -> str:
        digest = hashlib.sha256(f"{document_version_id}|{source_ref}|{atom_type}".encode("utf-8")).hexdigest()
        return f"atom:{digest[:32]}"

    @staticmethod
    def _repeated_region_key(atom_type: str, content_text: str) -> str | None:
        if atom_type not in {"HEADER", "FOOTER"}:
            return None
        normalized = " ".join(content_text.lower().split())
        return f"repeated:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:24]}"
