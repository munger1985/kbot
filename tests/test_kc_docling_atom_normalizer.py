import unittest
from types import SimpleNamespace

from docling_core.types.doc import BoundingBox, CoordOrigin, Size

from knowledge_core.parsing.docling_adapter import DoclingAdapterError, DoclingAtomNormalizer


class FakeDocument:
    schema_name = "DoclingDocument"
    version = "1.0"

    def __init__(self, items):
        self.pages = {
            1: SimpleNamespace(page_no=1, size=Size(width=100, height=100)),
            2: SimpleNamespace(page_no=2, size=Size(width=200, height=100)),
        }
        self._items = items

    def iterate_items(self):
        return iter((item, 0) for item in self._items)

    def export_to_dict(self, **kwargs):
        return {"schema_name": self.schema_name, "version": self.version, "item_count": len(self._items)}


def provenance(page_no=1, bbox=None):
    return SimpleNamespace(
        page_no=page_no,
        bbox=bbox or BoundingBox(
            l=10, t=20, r=50, b=40, coord_origin=CoordOrigin.TOPLEFT,
        ),
        confidence=0.9,
    )


def item(*, source_ref, label, text="", prov=None):
    return SimpleNamespace(
        self_ref=source_ref,
        label=label,
        text=text,
        prov=prov if prov is not None else [provenance()],
        formatting=None,
        annotations=[],
    )


class FakeTable(SimpleNamespace):
    def export_to_markdown(self, document):
        return "| Name | Value |\n| --- | --- |\n| A | 1 |"


class DoclingAtomNormalizerTest(unittest.TestCase):
    def test_normalizes_labels_coordinates_and_table_content(self):
        table = FakeTable(
            self_ref="#/tables/0", label="table", prov=[provenance(page_no=2)],
            formatting=None, annotations=[],
        )
        document = FakeDocument([
            item(source_ref="#/texts/0", label="section_header", text="1 Overview"),
            table,
            item(source_ref="#/texts/1", label="page_footer", text="Confidential"),
        ])

        result = DoclingAtomNormalizer(generator_version="1.0").normalize(
            document_version_id=301, document=document,
        )

        self.assertEqual([atom.atom_type for atom in result.atoms], [
            "TITLE_CANDIDATE", "TABLE", "FOOTER",
        ])
        self.assertEqual(result.atoms[0].locators[0].bbox, (0.1, 0.2, 0.5, 0.4))
        self.assertIn("| A | 1 |", result.atoms[1].content_text)
        self.assertTrue(result.atoms[2].repeated_region_key.startswith("repeated:"))

    def test_preserves_every_provenance_locator(self):
        text = item(
            source_ref="#/texts/0", label="text", text="Cross page",
            prov=[provenance(1), provenance(2)],
        )

        result = DoclingAtomNormalizer(generator_version="1.0").normalize(
            document_version_id=301, document=FakeDocument([text]),
        )

        self.assertEqual([locator.page_no for locator in result.atoms[0].locators], [1, 2])

    def test_rejects_paged_content_without_source_locator(self):
        document = FakeDocument([
            item(source_ref="#/texts/0", label="text", text="Lost", prov=[]),
        ])

        with self.assertRaisesRegex(DoclingAdapterError, "no source locator"):
            DoclingAtomNormalizer(generator_version="1.0").normalize(
                document_version_id=301, document=document,
            )

    def test_non_paged_document_uses_logical_source_ref(self):
        document = FakeDocument([
            item(source_ref="#/texts/0", label="text", text="Logical", prov=[]),
        ])
        document.pages = {}

        result = DoclingAtomNormalizer(generator_version="1.0").normalize(
            document_version_id=301, document=document,
        )

        self.assertEqual(result.atoms[0].locators[0].logical_ref, "#/texts/0")

    def test_visual_annotation_is_a_separate_derived_atom(self):
        picture = item(source_ref="#/pictures/0", label="picture", text="")
        picture.annotations = [SimpleNamespace(
            text="A deployment architecture diagram", provenance="vlm_inference",
        )]

        result = DoclingAtomNormalizer(generator_version="1.0").normalize(
            document_version_id=301, document=FakeDocument([picture]),
        )

        self.assertEqual([value.atom_type for value in result.atoms], [
            "PICTURE", "VISUAL_DESCRIPTION",
        ])
        self.assertEqual(result.atoms[1].provenance[0]["extractor"], "VLM")
        self.assertTrue(result.atoms[1].provenance[0]["derived"])


if __name__ == "__main__":
    unittest.main()
