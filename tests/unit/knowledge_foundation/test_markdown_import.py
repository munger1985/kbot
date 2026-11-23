"""Markdown 导入的通用排版与资源边界回归。"""

import asyncio
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "packages/knowledge_algorithms/src"), str(ROOT / "services/knowledge_core/src"),
               str(ROOT / "tests/unit/knowledge_algorithms")]

from knowledge_algorithm_fixtures import parse_profile, run_for, source
from knowledge_algorithms.contracts.content import ContentKind
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_core.adapters.document_parsing import DoclingParsingAdapter, validate_markdown


@pytest.fixture(scope="module")
def converter():
    return DoclingParsingAdapter(None).converter()


def convert(converter, markdown):
    from docling.datamodel.base_models import DocumentStream
    result = converter.convert(DocumentStream(name="source.md", stream=BytesIO(markdown.encode())))
    assert result.status.value == "success"
    return result.document


@pytest.mark.parametrize("tag", ["<br>", "<br/>", "<br />", "<BR  />"])
def test_body_breaks_do_not_hide_following_text_or_change_code(converter, tag):
    markdown = f"之前{tag}之后\n\n> {tag}\n> 下一段\n\n代码 `{tag}`\n\n"
    markdown += f"```html\n{tag}\n```\n"
    doc = convert(converter, markdown)
    assert [item.text for item in doc.texts[:3]] == ["之前", "之后", "下一段"]
    assert [code.text for code in doc.texts if code.label.value == "code"] == [tag, tag]


def test_attribute_free_tab_whitespace_and_escaped_tag_are_distinct(converter):
    markdown = "| A | B |\n| --- | --- |\n| 甲<br\t/>乙 | \\<br>、&lt;br&gt; |\n"
    table = convert(converter, markdown).tables[0]
    assert [cell.text for cell in table.data.table_cells[-2:]] == ["甲\n乙", "<br>、<br>"]


@pytest.mark.parametrize("markdown", [
    "| 名称 | 说明 |\r\n| :--- | ---: |\r\n| **甲**<br>乙<BR/>丙 | A \\| B、`<br>` |\r\n",
    "名称 | 说明\n--- | ---\n**甲**<br>乙<BR/>丙 | A \\| B、`<br>`\n",
])
def test_table_breaks_stay_in_the_same_cell(converter, markdown):
    doc = convert(converter, markdown)
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert (table.data.num_rows, table.data.num_cols) == (2, 2)
    assert [cell.text for cell in table.data.table_cells] == ["名称", "说明", "甲\n乙\n丙", "A | B、<br>"]

    reader = source(markdown)
    reader.descriptor = replace(reader.descriptor, mime_type="text/markdown")
    profile = parse_profile()
    adapter = DoclingParsingAdapter(None, converter_factory=lambda: converter)

    async def extract():
        return [batch async for batch in adapter.extract(reader, profile, run_for(reader, profile))]

    batches = asyncio.run(extract())
    assert len(batches[0].regions) == 1
    table_node = batches[0].regions[0].nodes[0]
    assert table_node.kind == ContentKind.TABLE
    assert "甲 乙 丙" in table_node.text
    assert table_node.locators[0].logical_ref == table.self_ref
    assert table_node.provenance.input_hash == reader.descriptor.sha256


@pytest.mark.parametrize("markdown", [
    "前<br class='extra'>后",
    "<br>\n<script>不允许</script>",
    "透明标签 <span>不允许</span>",
    "| A | B |\n| --- | --- |\n| 甲<br>乙 | ![图](https://example.invalid/x.png) |\n",
    "| A | B |\n| --- | --- |\n| 甲<br>乙 | ![图][img] |\n\n[img]: data:image/png;base64,AAAA",
    "| A | B |\n| --- | --- |\n| 甲<br>乙 | <img src='./x.png'> |\n",
])
def test_allowing_breaks_does_not_allow_other_html_or_unregistered_media(markdown):
    with pytest.raises(AlgorithmError) as error:
        validate_markdown(markdown.encode())
    assert error.value.code == ErrorCode.INVALID_CONTRACT
    assert error.value.stage == "convert"
