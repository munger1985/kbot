"""Deterministic structure quality gate before Evidence publication."""

from dataclasses import asdict, dataclass
import re
from typing import Any

from .ir import AtomIr, IrValidationError, StructureIr
from .reading_order import ReadingOrderResult


@dataclass(frozen=True)
class StructureQualityReport:
    passed: bool
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, Any]
    evaluator_version: str = "kc-structure-quality/v1"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["hard_failures"] = list(self.hard_failures)
        value["warnings"] = list(self.warnings)
        return value


class StructureQualityEvaluator:
    def evaluate(
        self, atom_ir: AtomIr, reading_order: ReadingOrderResult, structure_ir: StructureIr
    ) -> StructureQualityReport:
        hard_failures: list[str] = []
        warnings: list[str] = []
        try:
            structure_ir.validate(atom_ir)
        except IrValidationError as exc:
            hard_failures.append(f"STRUCTURE_INVARIANT:{exc}")

        body_atoms = [
            atom for atom in atom_ir.atoms if atom.atom_type not in {"HEADER", "FOOTER"}
        ]
        text_atoms = [
            atom for atom in body_atoms
            if atom.atom_type not in {"PICTURE"}
        ]
        nonempty = [atom for atom in text_atoms if atom.content_text.strip()]
        if not body_atoms:
            hard_failures.append("EMPTY_DOCUMENT")
        if not nonempty:
            hard_failures.append("NO_EXTRACTED_TEXT")

        normalized = [self._normalize(atom.content_text) for atom in nonempty]
        duplicate_count = len(normalized) - len(set(normalized))
        duplicate_ratio = duplicate_count / max(1, len(normalized))
        gibberish_chars = sum(self._gibberish_count(atom.content_text) for atom in nonempty)
        total_chars = sum(len(atom.content_text) for atom in nonempty)
        gibberish_ratio = gibberish_chars / max(1, total_chars)
        low_order = [
            atom_id for atom_id, confidence in reading_order.confidence_by_atom.items()
            if confidence < 0.70
        ]
        headings = [node.heading for node in structure_ir.nodes if node.heading is not None]
        low_headings = [heading.atom_id for heading in headings if heading.confidence < 0.65]

        if duplicate_ratio > 0.25:
            warnings.append("HIGH_DUPLICATE_TEXT")
        if gibberish_ratio > 0.08:
            hard_failures.append("EXCESSIVE_GIBBERISH")
        elif gibberish_ratio > 0.02:
            warnings.append("POSSIBLE_GIBBERISH")
        if low_order:
            warnings.append("LOW_READING_ORDER_CONFIDENCE")
        if low_headings:
            warnings.append("LOW_HEADING_CONFIDENCE")

        owned_atom_ids = {
            atom_id for node in structure_ir.nodes for atom_id in node.atom_ids
        }
        expected_atom_ids = {atom.atom_id for atom in body_atoms}
        coverage = len(owned_atom_ids.intersection(expected_atom_ids)) / max(1, len(expected_atom_ids))
        if coverage < 1:
            hard_failures.append("ATOM_COVERAGE_INCOMPLETE")

        metrics = {
            "document": {
                "body_atom_count": len(body_atoms),
                "text_atom_count": len(text_atoms),
                "content_character_count": total_chars,
                "atom_coverage": coverage,
                "duplicate_text_ratio": duplicate_ratio,
                "gibberish_ratio": gibberish_ratio,
            },
            "reading_order": {
                "excluded_repeated_region_count": len(reading_order.excluded_atom_ids),
                "continuation_count": len(reading_order.continuation_of),
                "low_confidence_atom_ids": low_order,
            },
            "structure": {
                "node_count": len(structure_ir.nodes),
                "heading_count": len(headings),
                "low_confidence_heading_atom_ids": low_headings,
            },
            "pages": self._page_metrics(atom_ir, reading_order),
            "sections": [{
                "section_key": node.node_id,
                "heading_confidence": node.heading.confidence,
                "page_range": list(node.page_range) if node.page_range else None,
            } for node in structure_ir.nodes if node.heading is not None],
        }
        return StructureQualityReport(
            passed=not hard_failures,
            hard_failures=tuple(hard_failures), warnings=tuple(warnings), metrics=metrics,
        )

    def evaluate_evidence(
        self, base_report: StructureQualityReport, evidences: tuple[Any, ...], *, max_tokens: int,
    ) -> StructureQualityReport:
        hard_failures = list(base_report.hard_failures)
        warnings = list(base_report.warnings)
        if base_report.passed and not evidences:
            hard_failures.append("NO_RETRIEVABLE_EVIDENCE")
        keys = [evidence.evidence_key for evidence in evidences]
        if len(keys) != len(set(keys)):
            hard_failures.append("DUPLICATE_EVIDENCE_KEY")
        invalid_source = [evidence.evidence_key for evidence in evidences if not evidence.source_spans]
        invalid_locator = [evidence.evidence_key for evidence in evidences if not evidence.locator]
        if invalid_source:
            hard_failures.append("EVIDENCE_SOURCE_SPAN_MISSING")
        if invalid_locator:
            hard_failures.append("EVIDENCE_LOCATOR_MISSING")
        short = [evidence for evidence in evidences if evidence.token_count < 80]
        oversized = [
            evidence for evidence in evidences
            if evidence.token_count > max_tokens and evidence.evidence_type not in {"TABLE", "SHEET"}
        ]
        if oversized:
            hard_failures.append("EVIDENCE_TOKEN_LIMIT_EXCEEDED")
        metrics = dict(base_report.metrics)
        metrics["evidence"] = {
            "count": len(evidences),
            "short_count": len(short),
            "short_ratio": len(short) / max(1, len(evidences)),
            "oversized_count": len(oversized),
            "locator_complete_ratio": (len(evidences) - len(invalid_locator)) / max(1, len(evidences)),
            "source_span_complete_ratio": (len(evidences) - len(invalid_source)) / max(1, len(evidences)),
        }
        return StructureQualityReport(
            passed=not hard_failures, hard_failures=tuple(hard_failures),
            warnings=tuple(dict.fromkeys(warnings)), metrics=metrics,
        )

    @staticmethod
    def _page_metrics(atom_ir: AtomIr, reading_order: ReadingOrderResult) -> list[dict[str, Any]]:
        result = []
        for page in atom_ir.pages:
            atoms = [
                atom for atom in atom_ir.atoms
                if any(locator.page_no == page.page_no for locator in atom.locators)
            ]
            result.append({
                "page_no": page.page_no,
                "atom_count": len(atoms),
                "low_reading_order_count": sum(
                    1 for atom in atoms if reading_order.confidence_by_atom.get(atom.atom_id, 1) < 0.7
                ),
            })
        return result

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _gibberish_count(text: str) -> int:
        return sum(
            1 for character in text
            if character == "\ufffd" or (ord(character) < 32 and character not in "\n\r\t")
        )
