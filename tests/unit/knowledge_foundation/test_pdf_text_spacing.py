"""PDF 原页字形回归：不需要 OCR、模型或数据库。"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "packages/knowledge_algorithms/src"), str(ROOT / "services/knowledge_core/src"),
               str(ROOT / "tests/unit/knowledge_algorithms")]

from knowledge_algorithm_fixtures import node, parse_profile, run_for
from knowledge_algorithms.contracts.content import ContentKind, ProvenanceKind
from knowledge_algorithms.contracts.identity import AppBackendBinding, Locator
from knowledge_algorithms.parsing.contracts import ExtractedRegion, SourceDescriptor
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_core.adapters.document_parsing import DoclingParsingAdapter
from knowledge_core.adapters.pdf_text_spacing import REVISION, restore_pdf_word_spacing


def draw_pdf(lines, *, word_gap=6):
    commands = ["BT /F1 14 Tf"]
    for row, words in enumerate(lines):
        x, y = 40, 350 - 30 * row
        for word in words.split():
            for char in word:
                commands.append(f"1 0 0 1 {x:.2f} {y} Tm ({char}) Tj")
                x += 8.4
            x += word_gap
    commands.append("ET")
    stream = "\n".join(commands).encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] /Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    )
    raw = b"%PDF-1.4\n"
    offsets = []
    for index, value in enumerate(objects, 1):
        offsets.append(len(raw))
        raw += f"{index} 0 obj\n".encode() + value + b"\nendobj\n"
    xref = len(raw)
    raw += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    raw += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets)
    return raw + f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def region(text, *, bounds=(.08, .08, .95, .22), kind=ContentKind.PARAGRAPH):
    original = replace(node(text=text, kind=kind), locators=(Locator(1, bounds),))
    return ExtractedRegion("fixture", (original,), 1.0)


def fragment(value):
    return " ".join("".join(value.split()))


def test_reconstructs_pdf_word_breaks_without_changing_node_or_location():
    lines = ("Test method for measuring", "engine speed accurately")
    expected = " ".join(lines)
    old = region(fragment(expected))
    repaired, = restore_pdf_word_spacing(draw_pdf(lines), [old])
    assert repaired == replace(old, nodes=(replace(old.nodes[0], text=expected),))


def test_preserves_paragraph_when_words_have_no_geometric_gap_or_content_does_not_match():
    original = region(fragment("Test method for measuring"))
    wrong = region(fragment("Test methods for measuring"))
    other_box = region(original.nodes[0].text, bounds=(.08, .60, .95, .90))
    assert restore_pdf_word_spacing(draw_pdf(("Test method for measuring",), word_gap=0), [original]) == [original]
    assert restore_pdf_word_spacing(draw_pdf(("Test method for measuring",)), [wrong, other_box]) == [wrong, other_box]


def test_does_not_rewrite_normal_text_code_or_ocr():
    exact = "Test method for measuring"
    normal = region(exact)
    code = region(fragment(exact), kind=ContentKind.CODE)
    base_ocr = region(fragment(exact))
    ocr_node = base_ocr.nodes[0]
    ocr = replace(base_ocr, nodes=(replace(ocr_node, provenance=replace(ocr_node.provenance, kind=ProvenanceKind.OCR)),))
    assert restore_pdf_word_spacing(draw_pdf((exact,)), [normal, code, ocr]) == [normal, code, ocr]


@pytest.mark.asyncio
async def test_docling_adapter_makes_recovered_text_canonical_before_parsing_completes():
    from docling_core.types.doc import BoundingBox, CoordOrigin, Size

    expected = "Test method for measuring"
    raw = draw_pdf((expected,))

    class Reader:
        descriptor = SourceDescriptor("app-a", "doc-v1", sha256(raw).hexdigest(),
                                      "application/pdf", len(raw), "source-1")

        async def chunks(self):
            yield raw

    bbox = BoundingBox(l=32, t=32, r=380, b=88, coord_origin=CoordOrigin.TOPLEFT)
    item = SimpleNamespace(self_ref="#/texts/0", label="text", text=fragment(expected),
                           prov=(SimpleNamespace(page_no=1, bbox=bbox),))
    document = SimpleNamespace(pages={1: SimpleNamespace(size=Size(width=400, height=400))},
                               iterate_items=lambda: iter(((item, 0),)))
    converter = SimpleNamespace(convert=lambda *_args, **_kwargs: SimpleNamespace(status="success", document=document))
    adapter = DoclingParsingAdapter(None, converter_factory=lambda: converter)
    reader, profile = Reader(), parse_profile(timeout_seconds=20)
    parsed = await ParseEngine(AppBackendBinding("app-a", "app-a"), (adapter,)).parse(
        reader, profile, run_for(reader, profile))
    paragraph, = parsed.nodes
    assert paragraph.text == expected
    position, = paragraph.locators
    assert position.page_no == 1
    assert position.bbox == pytest.approx((.08, .08, .95, .22))
    assert paragraph.provenance.revision.endswith(REVISION)
