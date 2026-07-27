import unittest

from knowledge_core.parsing import (
    Atom,
    AtomIr,
    AtomLocator,
    HeadingDecision,
    IrValidationError,
    PageGeometry,
    StructureIr,
    StructureNode,
)


def atom_ir() -> AtomIr:
    provenance = ({"extractor": "DOCLING", "version": "2.0"},)
    return AtomIr(
        document_version_id=301,
        pages=(PageGeometry(page_no=1, width=595, height=842),),
        atoms=(
            Atom(
                atom_id="atom:title", source_ref="#/texts/0", atom_type="TITLE_CANDIDATE",
                content_text="1 Deployment",
                locators=(AtomLocator(page_no=1, bbox=(0.1, 0.1, 0.8, 0.15)),),
                reading_order_hint=0, original_label="section_header", provenance=provenance,
            ),
            Atom(
                atom_id="atom:body", source_ref="#/texts/1", atom_type="TEXT",
                content_text="Install the service.",
                locators=(AtomLocator(page_no=1, bbox=(0.1, 0.2, 0.8, 0.3)),),
                reading_order_hint=1, original_label="text", provenance=provenance,
            ),
            Atom(
                atom_id="atom:footer", source_ref="#/texts/2", atom_type="FOOTER",
                content_text="Confidential",
                locators=(AtomLocator(page_no=1, bbox=(0.1, 0.95, 0.8, 0.98)),),
                reading_order_hint=2, original_label="footer", provenance=provenance,
                repeated_region_key="footer:confidential",
            ),
        ),
        generator={"name": "docling-adapter", "version": "1.0"},
    )


def structure_ir(atoms: AtomIr) -> StructureIr:
    return StructureIr(
        document_version_id=atoms.document_version_id,
        atom_ir_sha256=atoms.fingerprint(),
        nodes=(
            StructureNode(
                node_id="document:1", node_type="DOCUMENT", parent_node_id=None,
                ordinal=0, atom_ids=(), decision_provenance={"resolver": "outline/v1"},
            ),
            StructureNode(
                node_id="section:1", node_type="SECTION", parent_node_id="document:1",
                ordinal=1, atom_ids=("atom:title",),
                heading=HeadingDecision(
                    atom_id="atom:title", text="1 Deployment", level=1,
                    confidence=0.98, reasons=("numbering:1", "style:h1"),
                ),
                heading_path=("1 Deployment",), page_range=(1, 1),
                decision_provenance={"resolver": "outline/v1"},
            ),
            StructureNode(
                node_id="paragraph:1", node_type="PARAGRAPH", parent_node_id="section:1",
                ordinal=2, atom_ids=("atom:body",), heading_path=("1 Deployment",),
                page_range=(1, 1), decision_provenance={"builder": "semantic-block/v1"},
            ),
        ),
        generator={"name": "structure-pipeline", "version": "1.0"},
    )


class AtomIrTest(unittest.TestCase):
    def test_valid_atom_ir_has_stable_fingerprint(self):
        first = atom_ir()
        second = atom_ir()

        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_bbox_must_be_normalized_top_left(self):
        invalid = atom_ir()
        invalid_atom = Atom(**{
            **invalid.atoms[0].__dict__,
            "locators": (AtomLocator(page_no=1, bbox=(0.1, -0.1, 0.8, 0.15)),),
        })
        invalid = AtomIr(
            document_version_id=invalid.document_version_id,
            pages=invalid.pages,
            atoms=(invalid_atom, *invalid.atoms[1:]),
            generator=invalid.generator,
        )

        with self.assertRaisesRegex(IrValidationError, "normalized"):
            invalid.validate()


class StructureIrTest(unittest.TestCase):
    def test_valid_structure_owns_each_content_atom_once(self):
        atoms = atom_ir()
        structure = structure_ir(atoms)

        structure.validate(atoms)
        self.assertEqual(structure.fingerprint(atoms), structure.fingerprint(atoms))

    def test_structure_rejects_unowned_content(self):
        atoms = atom_ir()
        valid = structure_ir(atoms)
        without_paragraph = StructureIr(
            document_version_id=valid.document_version_id,
            atom_ir_sha256=valid.atom_ir_sha256,
            nodes=valid.nodes[:-1],
            generator=valid.generator,
        )

        with self.assertRaisesRegex(IrValidationError, "ownership mismatch"):
            without_paragraph.validate(atoms)

    def test_structure_rejects_duplicate_atom_ownership(self):
        atoms = atom_ir()
        valid = structure_ir(atoms)
        duplicate = StructureNode(
            node_id="paragraph:2", node_type="PARAGRAPH", parent_node_id="section:1",
            ordinal=3, atom_ids=("atom:body",),
            decision_provenance={"builder": "semantic-block/v1"},
        )
        invalid = StructureIr(
            document_version_id=valid.document_version_id,
            atom_ir_sha256=valid.atom_ir_sha256,
            nodes=(*valid.nodes, duplicate),
            generator=valid.generator,
        )

        with self.assertRaisesRegex(IrValidationError, "multiple primary owners"):
            invalid.validate(atoms)


if __name__ == "__main__":
    unittest.main()
