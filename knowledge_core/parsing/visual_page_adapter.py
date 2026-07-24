"""将整页 VLM 结果融合进 KC Atom IR。"""

from dataclasses import replace
import hashlib
import re
from typing import TYPE_CHECKING
from uuid import UUID

from .ir import Atom, AtomIr, AtomLocator

if TYPE_CHECKING:
    from knowledge_core.workers.parser.visual_enricher import (
        VisualEnrichmentResult,
        VisualPageResult,
    )


_HEADING = re.compile(r"^(#{1,9})\s+(.+?)\s*$")
_LIST = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)、]\s*)(.+)$")
_IMAGE = re.compile(r"\[IMAGE:(.+?)]", re.IGNORECASE)


class VisualPageAtomAdapter:
    """按页面质量选择替换或补充 Docling Atom。"""

    def apply(
        self,
        atom_ir: AtomIr,
        enrichment: "VisualEnrichmentResult | None",
    ) -> AtomIr:
        if enrichment is None or not enrichment.page_results:
            return atom_ir

        atoms = list(atom_ir.atoms)
        for page_result in enrichment.page_results:
            page_atoms = self._page_atoms(atoms, page_result.page_no)
            if page_result.replace_docling:
                replaceable_page_atoms = [
                    atom
                    for atom in page_atoms
                    if self._belongs_only_to_page(
                        atom,
                        page_result.page_no,
                    )
                ]
                atoms = [
                    atom for atom in atoms
                    if not self._belongs_only_to_page(atom, page_result.page_no)
                ]
                atoms.extend(
                    self._fuse_replacement(
                        atom_ir.document_version_id,
                        replaceable_page_atoms,
                        page_result,
                    )
                )
                continue
            atoms = self._apply_heading_hints(atoms, page_atoms, page_result)
            atoms.extend(
                self._visual_semantic_atoms(
                    atom_ir.document_version_id,
                    page_atoms,
                    page_result,
                )
            )

        result = replace(
            atom_ir,
            atoms=tuple(atoms),
            generator={
                "name": "docling-visual-fusion",
                "version": enrichment.adapter_version,
            },
        )
        result.validate()
        return result

    def _fuse_replacement(
        self,
        document_version_id: UUID,
        docling_atoms: list[Atom],
        page: "VisualPageResult",
    ) -> list[Atom]:
        visual_atoms = self._parse_visual_markdown(
            document_version_id,
            page,
        )
        available = [
            atom
            for atom in docling_atoms
            if atom.atom_type not in {
                "HEADER",
                "FOOTER",
                "PICTURE",
                "VISUAL_DESCRIPTION",
            }
            and atom.content_text.strip()
        ]
        consumed: set[str] = set()
        fused: list[Atom] = []
        for visual_atom in visual_atoms:
            matched = self._matching_docling_atom(
                visual_atom,
                available,
                consumed,
            )
            if matched is None:
                fused.append(visual_atom)
                continue
            consumed.add(matched.atom_id)
            if visual_atom.atom_type == "TITLE_CANDIDATE":
                style = dict(matched.style)
                style["declared_level"] = visual_atom.style["declared_level"]
                fused.append(
                    replace(
                        matched,
                        atom_type="TITLE_CANDIDATE",
                        style=style,
                        provenance=(
                            *matched.provenance,
                            {
                                "extractor": "VLM",
                                "purpose": "PAGE_REPLACEMENT_HEADING",
                                "model": page.served_model_name,
                                "page_no": page.page_no,
                            },
                        ),
                    )
                )
            else:
                fused.append(matched)

        fused.extend(
            atom
            for atom in available
            if atom.atom_id not in consumed
            and self._requires_exact_preservation(atom)
        )
        return fused

    def _parse_visual_markdown(
        self,
        document_version_id: UUID,
        page: "VisualPageResult",
    ) -> list[Atom]:
        blocks = self._blocks(page.markdown)
        atoms: list[Atom] = []
        for index, (kind, text, level) in enumerate(blocks):
            atom_type = {
                "heading": "TITLE_CANDIDATE",
                "table": "TABLE",
                "list": "LIST_ITEM",
                "visual": "VISUAL_DESCRIPTION",
            }.get(kind, "TEXT")
            atoms.append(
                self._atom(
                    document_version_id=document_version_id,
                    page=page,
                    index=index,
                    atom_type=atom_type,
                    text=text,
                    declared_level=level,
                    purpose="PAGE_REPLACEMENT",
                )
            )
        return atoms

    def _apply_heading_hints(
        self,
        all_atoms: list[Atom],
        page_atoms: list[Atom],
        page: "VisualPageResult",
    ) -> list[Atom]:
        headings = [
            (self._normalize(text), level)
            for kind, text, level in self._blocks(page.markdown)
            if kind == "heading"
        ]
        if not headings:
            return all_atoms
        replacements: dict[str, Atom] = {}
        for atom in page_atoms:
            normalized = self._normalize(atom.content_text)
            match = next(
                (
                    level for heading, level in headings
                    if normalized == heading
                    or (
                        len(normalized) >= 6
                        and (normalized in heading or heading in normalized)
                    )
                ),
                None,
            )
            if match is None:
                continue
            style = dict(atom.style)
            style["declared_level"] = match
            replacements[atom.atom_id] = replace(
                atom,
                atom_type="TITLE_CANDIDATE",
                style=style,
                confidence=max(atom.confidence, 0.85),
                provenance=(
                    *atom.provenance,
                    {
                        "extractor": "VLM",
                        "purpose": "HEADING_CORRECTION",
                        "model": page.served_model_name,
                        "page_no": page.page_no,
                    },
                ),
            )
        return [replacements.get(atom.atom_id, atom) for atom in all_atoms]

    def _matching_docling_atom(
        self,
        visual_atom: Atom,
        candidates: list[Atom],
        consumed: set[str],
    ) -> Atom | None:
        visual_text = self._normalize(visual_atom.content_text)
        for candidate in candidates:
            if candidate.atom_id in consumed:
                continue
            candidate_text = self._normalize(candidate.content_text)
            if candidate_text == visual_text:
                return candidate
            if (
                visual_atom.atom_type == "TITLE_CANDIDATE"
                and min(len(candidate_text), len(visual_text)) >= 6
                and (
                    candidate_text in visual_text
                    or visual_text in candidate_text
                )
            ):
                return candidate
        return None

    @staticmethod
    def _requires_exact_preservation(atom: Atom) -> bool:
        if any(
            provenance.get("extractor") == "DEEPSEEK_OCR"
            for provenance in atom.provenance
        ):
            return True
        if atom.confidence < 0.75:
            return False
        if atom.atom_type in {"TABLE", "FORMULA", "CODE"}:
            return True
        return bool(re.search(r"\d", atom.content_text))

    def _visual_semantic_atoms(
        self,
        document_version_id: UUID,
        page_atoms: list[Atom],
        page: "VisualPageResult",
    ) -> list[Atom]:
        existing = {
            self._normalize(atom.content_text)
            for atom in page_atoms
            if atom.content_text.strip()
        }
        output: list[Atom] = []
        for index, (kind, text, _) in enumerate(self._blocks(page.markdown)):
            if kind != "visual" or self._normalize(text) in existing:
                continue
            output.append(
                self._atom(
                    document_version_id=document_version_id,
                    page=page,
                    index=index,
                    atom_type="VISUAL_DESCRIPTION",
                    text=text,
                    declared_level=None,
                    purpose="VISUAL_SEMANTIC_SUPPLEMENT",
                )
            )
        return output

    @staticmethod
    def _blocks(markdown: str) -> list[tuple[str, str, int | None]]:
        lines = markdown.replace("\r\n", "\n").split("\n")
        blocks: list[tuple[str, str, int | None]] = []
        paragraph: list[str] = []
        table: list[str] = []

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append(("text", " ".join(paragraph).strip(), None))
                paragraph.clear()

        def flush_table() -> None:
            if table:
                blocks.append(("table", "\n".join(table).strip(), None))
                table.clear()

        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("|") and line.endswith("|"):
                flush_paragraph()
                table.append(line)
                continue
            flush_table()
            if not line:
                flush_paragraph()
                continue
            heading = _HEADING.match(line)
            if heading:
                flush_paragraph()
                blocks.append(
                    ("heading", heading.group(2).strip(), len(heading.group(1)))
                )
                continue
            image = _IMAGE.fullmatch(line)
            if image:
                flush_paragraph()
                blocks.append(("visual", image.group(1).strip(), None))
                continue
            if line.startswith(">"):
                flush_paragraph()
                blocks.append(("visual", line.lstrip("> ").strip(), None))
                continue
            list_item = _LIST.match(line)
            if list_item:
                flush_paragraph()
                blocks.append(("list", line, None))
                continue
            paragraph.append(line)
        flush_table()
        flush_paragraph()
        return [block for block in blocks if block[1]]

    @staticmethod
    def _page_atoms(atoms: list[Atom], page_no: int) -> list[Atom]:
        return [
            atom for atom in atoms
            if any(locator.page_no == page_no for locator in atom.locators)
        ]

    @staticmethod
    def _belongs_only_to_page(atom: Atom, page_no: int) -> bool:
        page_numbers = {
            locator.page_no
            for locator in atom.locators
            if locator.page_no is not None
        }
        return page_numbers == {page_no}

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text).lower().strip()

    @staticmethod
    def _atom(
        *,
        document_version_id: UUID,
        page: "VisualPageResult",
        index: int,
        atom_type: str,
        text: str,
        declared_level: int | None,
        purpose: str,
    ) -> Atom:
        source_ref = f"#/visual-pages/{page.page_no}/blocks/{index}"
        digest = hashlib.sha256(
            f"{document_version_id}|{source_ref}|{atom_type}".encode("utf-8")
        ).hexdigest()
        style = (
            {"declared_level": declared_level}
            if declared_level is not None
            else {}
        )
        return Atom(
            atom_id=f"atom:{digest[:32]}",
            source_ref=source_ref,
            atom_type=atom_type,
            content_text=text,
            locators=(
                AtomLocator(
                    page_no=page.page_no,
                    bbox=(0.0, 0.0, 1.0, 1.0),
                ),
            ),
            reading_order_hint=index,
            original_label="visual_page_markdown",
            style=style,
            confidence=page.confidence,
            provenance=(
                {
                    "extractor": "VLM",
                    "purpose": purpose,
                    "model": page.served_model_name,
                    "page_no": page.page_no,
                    "selection_reasons": list(page.selection_reasons),
                    "locator_precision": "PAGE",
                },
            ),
        )
