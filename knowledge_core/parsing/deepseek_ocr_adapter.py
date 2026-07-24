"""将 DeepSeek OCR grounding 结果转换为可追溯 Atom。"""

from dataclasses import replace
import hashlib
from typing import TYPE_CHECKING
from uuid import UUID

from .ir import Atom, AtomIr, AtomLocator

if TYPE_CHECKING:
    from knowledge_core.workers.parser.deepseek_ocr import (
        DeepSeekOcrElement,
        DeepSeekOcrEnrichmentResult,
    )


_ELEMENT_TO_ATOM = {
    "text": "TEXT",
    "title": "TITLE_CANDIDATE",
    "table_caption": "CAPTION",
    "table": "TABLE",
    "image": "VISUAL_DESCRIPTION",
    "code": "CODE",
    "formula": "FORMULA",
}


class DeepSeekOcrAtomAdapter:
    """成功页替换 Docling 低质量页面，失败页保持原结果。"""

    def apply(
        self,
        atom_ir: AtomIr,
        enrichment: "DeepSeekOcrEnrichmentResult | None",
    ) -> AtomIr:
        if enrichment is None or not enrichment.page_results:
            return atom_ir
        atoms = list(atom_ir.atoms)
        for page_result in enrichment.page_results:
            atoms = [
                atom
                for atom in atoms
                if not self._belongs_only_to_page(
                    atom,
                    page_result.page_no,
                )
            ]
            atoms.extend(
                self._atoms(
                    atom_ir.document_version_id,
                    enrichment.served_model_name,
                    page_result.page_no,
                    page_result.elements,
                )
            )
        result = replace(
            atom_ir,
            atoms=tuple(atoms),
            generator={
                "name": "docling-deepseek-ocr",
                "version": enrichment.adapter_version,
            },
        )
        result.validate()
        return result

    @staticmethod
    def _atoms(
        document_version_id: UUID,
        served_model_name: str,
        page_no: int,
        elements: tuple["DeepSeekOcrElement", ...],
    ) -> list[Atom]:
        output: list[Atom] = []
        for index, element in enumerate(elements):
            atom_type = _ELEMENT_TO_ATOM[element.element_type]
            source_ref = f"#/deepseek-ocr/pages/{page_no}/elements/{index}"
            digest = hashlib.sha256(
                f"{document_version_id}|{source_ref}|{atom_type}".encode(
                    "utf-8"
                )
            ).hexdigest()
            output.append(
                Atom(
                    atom_id=f"atom:{digest[:32]}",
                    source_ref=source_ref,
                    atom_type=atom_type,
                    content_text=element.content_text,
                    locators=(
                        AtomLocator(
                            page_no=page_no,
                            bbox=element.bbox,
                        ),
                    ),
                    reading_order_hint=index,
                    original_label=element.element_type,
                    style=(
                        {"declared_level": 1}
                        if element.element_type == "title"
                        else {}
                    ),
                    confidence=0.9,
                    provenance=(
                        {
                            "extractor": "DEEPSEEK_OCR",
                            "model": served_model_name,
                            "page_no": page_no,
                            "coordinate_space": "grounding_0_999",
                        },
                    ),
                )
            )
        return output

    @staticmethod
    def _belongs_only_to_page(atom: Atom, page_no: int) -> bool:
        pages = {
            locator.page_no
            for locator in atom.locators
            if locator.page_no is not None
        }
        return pages == {page_no}
