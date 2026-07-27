import unittest

from knowledge_core.parsing.evidence_planner import EvidencePlanner, EvidencePolicy
from knowledge_core.parsing.ir import Atom, AtomIr, AtomLocator, PageGeometry
from knowledge_core.parsing.quality import StructureQualityEvaluator
from knowledge_core.parsing.reading_order import ReadingOrderResolver
from knowledge_core.parsing.structure_builder import OutlineResolver


def atom(atom_id, atom_type, text, page, bbox, order, *, level=None):
    style = {"declared_level": level} if level else {}
    return Atom(
        atom_id=atom_id, source_ref=f"#/{atom_id}", atom_type=atom_type,
        content_text=text, locators=(AtomLocator(page_no=page, bbox=bbox),),
        reading_order_hint=order, original_label=atom_type.lower(), style=style,
        provenance=({"extractor": "DOCLING"},),
    )


def sample_ir():
    return AtomIr(
        document_version_id=10,
        pages=(
            PageGeometry(1, 100, 100),
            PageGeometry(2, 100, 100),
        ),
        atoms=(
            atom("title", "TITLE_CANDIDATE", "1 Overview", 1, (0.1, 0.02, 0.9, 0.08), 0, level=1),
            atom("left-1", "TEXT", "Left first.", 1, (0.05, 0.15, 0.42, 0.25), 1),
            atom("right-1", "TEXT", "Right first.", 1, (0.58, 0.12, 0.95, 0.22), 2),
            atom("left-2", "TEXT", "Left second", 1, (0.05, 0.30, 0.42, 0.40), 3),
            atom("right-2", "TEXT", "Right second", 1, (0.58, 0.28, 0.95, 0.38), 4),
            atom("continued", "TEXT", "continues here.", 2, (0.58, 0.05, 0.95, 0.15), 5),
            atom("footer", "FOOTER", "Confidential", 2, (0.1, 0.95, 0.9, 0.98), 6),
        ),
        generator={"name": "test", "version": "1"},
    )


class StructurePipelineTest(unittest.TestCase):
    def test_resolves_columns_continuation_and_outline(self):
        atoms = sample_ir()
        reading = ReadingOrderResolver().resolve(atoms)

        self.assertEqual(reading.ordered_atom_ids[:5], (
            "title", "left-1", "left-2", "right-1", "right-2",
        ))
        self.assertEqual(reading.continuation_of["continued"], "right-2")
        self.assertEqual(reading.excluded_atom_ids, ("footer",))

        structure = OutlineResolver().build(atoms, reading)
        section = next(node for node in structure.nodes if node.node_type == "SECTION")
        self.assertEqual(section.heading.level, 1)
        continued_paragraph = next(
            node for node in structure.nodes if "continued" in node.atom_ids
        )
        self.assertIn("right-2", continued_paragraph.atom_ids)

    def test_quality_and_evidence_are_traceable(self):
        atoms = sample_ir()
        reading = ReadingOrderResolver().resolve(atoms)
        structure = OutlineResolver().build(atoms, reading)
        quality = StructureQualityEvaluator().evaluate(atoms, reading, structure)

        self.assertTrue(quality.passed)
        self.assertEqual(quality.metrics["document"]["atom_coverage"], 1)

        evidences = EvidencePlanner(EvidencePolicy(min_tokens=1, target_tokens=20, max_tokens=50)).plan(
            parse_view_id=77, atom_ir=atoms, structure_ir=structure,
        )
        self.assertTrue(evidences)
        self.assertTrue(all(evidence.evidence_key.startswith("ev1:77:") for evidence in evidences))
        self.assertTrue(all(evidence.source_spans for evidence in evidences))
        self.assertTrue(all(evidence.locator["pages"] for evidence in evidences))


if __name__ == "__main__":
    unittest.main()
