"""合成切片与 Markdown 安全回归，不使用数据库、真实模型或业务文件。"""

import asyncio
from base64 import b64encode
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [str(path) for path in ROOT.glob("services/*/src")]
sys.path.insert(0, str(ROOT / "tests/unit/knowledge_algorithms"))

from knowledge_algorithm_fixtures import AllowedReader, FakeModel, allowed, node, projection_profile, spec
from knowledge_algorithms.contracts.content import ContentKind
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.projections.builder import ProjectionBuilder, planned_unit_bytes
from knowledge_algorithms.projections.contracts import ProjectionBatch
from knowledge_algorithms.execution import Deadline
from knowledge_algorithms.projections.tables import markdown_table_spans
from knowledge_core.adapters.bounded_parse_executor import BoundedParseExecutor
from knowledge_core.adapters.document_parsing import validate_markdown


class ByteTokenizer:
    revision = "test-utf8"

    def count(self, text):
        return len(text.encode("utf-8"))


@pytest.mark.parametrize(("kind", "text"), [
    (ContentKind.PARAGRAPH, "内部控制专业知识" * 70 + "𠀀e\u0301"),
    (ContentKind.PARAGRAPH, "a" * 1461),
    (ContentKind.TABLE, "表格文本无可靠列" * 98),
    (ContentKind.TABLE, "| 列名与单位" + "甲" * 450 + " | B\\|1 | C |\n| --- | --- | --- |\n| "
        + "原始单元格" * 600 + " | 4 MPa | 说明 |\n| A | B | C |\n"),
], ids=["long-chinese", "long-english", "unknown-table-layout", "long-header-and-cell"])
def test_oversized_content_preserves_every_original_character(kind, text):
    async def verify():
        content = replace(allowed(node(kind=kind, text=text)), heading_path=("很长的章节名" * 500, "附录"))
        reader = AllowedReader((content,))
        model = FakeModel()
        builder = ProjectionBuilder(reader.binding, ByteTokenizer(), model)
        profile = projection_profile(max_tokens=1024, batch_size=64, representations=(spec(),))
        unconstrained = [batch async for batch in builder.build(reader, profile)]
        batch_bytes = max(planned_unit_bytes(item.unit, profile, 65_536) for b in unconstrained for item in b.units)
        model.calls.clear()
        count = await builder.plan_batch_count(reader, profile, max_batch_bytes=batch_bytes)
        batches = [batch async for batch in builder.build(reader, profile, max_batch_bytes=batch_bytes)]
        assert len(batches) == count >= 2
        assert all(len(call.text.encode("utf-8")) <= 1024 for call in model.calls)
        covered = set()
        units = [item.unit for batch in batches for item in batch.units]
        assert len({item.unit_key for item in units}) == len(units)
        for unit in units:
            assert unit.source_key == content.key
            assert unit.locators == content.locators
            for source in unit.sources:
                covered.update(range(source.source_start, source.source_end))
                assert unit.text[source.unit_start:source.unit_end] == text[source.source_start:source.source_end]
        assert covered == set(range(len(text)))
        if kind == ContentKind.PARAGRAPH:
            assert any(a.sources[-1].source_end > b.sources[0].source_start for a, b in zip(units, units[1:]))
    asyncio.run(verify())


def test_table_syntax_validates_delimiters_and_does_not_invent_columns():
    valid = "| A | B |\n| --- | --- |\n| x\\|y | z |\n"
    assert len(markdown_table_spans(valid)[1]) == 1
    assert markdown_table_spans(valid.replace("| z |", "| z | extra |")) == (None, ())
    native = "表 1 说明\n\n" + valid.replace("---", "-")
    header, rows = markdown_table_spans(native)
    assert native[slice(*header)].startswith("| A")
    assert native[slice(*rows[0])] == "| x\\|y | z |"


@pytest.mark.parametrize("text", [
    "范围 A < 5，B<3；普通标记 < 不是 HTML",
    "比较符 " + chr(96) + "<img src='http://invalid.local/x'>" + chr(96),
    "普通链接 <https://example.invalid/x>",
    "![内嵌](data:image/png;base64,AAAA)",
])
def test_markdown_preserves_safe_source(text):
    validate_markdown(text.encode())


@pytest.mark.parametrize("text", [
    "<img src='https://example.invalid/x'>", "行内 <span>HTML</span>",
    "![外部](https://example.invalid/x)", "![相对路径](./local.png)",
    "![引用][x]\n\n[x]: <data:image/png;base64,AAAA>",
])
def test_markdown_still_rejects_raw_html_and_unregistered_images(text):
    with pytest.raises(AlgorithmError) as error:
        validate_markdown(text.encode())
    assert error.value.stage == "convert"


def test_model_call_diagnostics_correlate_request_without_logging_image():
    from PIL import Image
    from loguru import logger

    raw = BytesIO()
    with Image.new("RGB", (963, 2)) as image:
        image.save(raw, "PNG")
    encoded = b64encode(raw.getvalue()).decode()
    call = AsyncMock(side_effect=RuntimeError("敏感的模型请求内容"))
    executor = BoundedParseExecutor(model_transcription=call)
    job = SimpleNamespace(run=SimpleNamespace(job_ref="job", parse_run_ref="run"), profile=object())
    context = SimpleNamespace(model_copy=lambda *, update: SimpleNamespace(**update))
    messages = []
    sink = logger.add(messages.append, level="INFO")
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(executor._transcribe(job, context, {"role": "vlm", "region": "figure", "image": encoded,
                "location": {"object": "a" * 64, "origin": "CONVERTER_FIGURE", "page": 2, "width": 963, "height": 2}}))
    finally:
        logger.remove(sink)
    request_id = call.await_args.args[1].request_id
    assert call.await_args.args[2] == {"role": "vlm", "region": "figure", "image": encoded}
    assert any(request_id in message and "(963, 2)" in message and "CONVERTER_FIGURE" in message for message in messages)
    assert all(encoded not in message and "敏感" not in message for message in messages)


def test_single_image_gap_preserves_source_completes_other_batches_and_keeps_old_checkpoint():
    visual = spec("MULTIVECTOR", "IMAGE", "IMAGE_TO_IMAGE")
    dense = spec()
    profile = projection_profile(representations=(dense, visual), batch_size=1)

    def figure(name):
        return replace(allowed(node(name, text="", kind=ContentKind.FIGURE)), content_handle=name)

    contents = (figure("known-image"), allowed(node("text", text="关键词段落")),
                figure("unsupported-image"), figure("later-image"))
    reader = AllowedReader(contents)

    class Model(FakeModel):
        async def represent(self, value, specification):
            if value.image_handle == "unsupported-image" and specification == visual:
                self.calls.append(value)
                raise AlgorithmError(ErrorCode.IMAGE_INPUT_UNSUPPORTED, "测试图片不支持", stage="visual_input")
            return await super().represent(value, specification)

    async def verify():
        old_model = FakeModel()
        old = ProjectionBuilder(reader.binding, ByteTokenizer(), old_model)
        first = []
        iterator = old.build(reader, profile, max_batch_bytes=65536)
        try:
            for _ in range(2):
                entry = await anext(iterator)
                wire = encode(entry)
                assert "skipped_image_spaces" not in wire["units"][0]
                first.append(decode(ProjectionBatch, wire))
        finally:
            await iterator.aclose()
        model = Model()
        builder = ProjectionBuilder(reader.binding, ByteTokenizer(), model)

        async def load_saved(sequence, deadline):
            return first[sequence-1] if sequence <= len(first) else None

        count = await builder.plan_batch_count(reader, profile, max_batch_bytes=65536)
        actual = [batch async for batch in builder.build(reader, profile, load_saved=load_saved, max_batch_bytes=65536)]
        assert count == len(actual) == 4 and actual[:2] == first
        assert [call.image_handle for call in model.calls] == ["unsupported-image", "later-image"]
        missing, = actual[2].units
        assert missing.unit.content_handle == "unsupported-image" and missing.unit.locators == contents[2].locators
        assert missing.skipped_image_spaces == {visual.space_id} and not missing.representations
        assert decode(ProjectionBatch, encode(actual[2])) == actual[2]
        assert actual[2].payload_hash == fingerprint(encode(actual[2].units))
        assert len(actual[3].units[0].representations) == 1

        # 同一空间的缺口可复用；换模型时必须用同一原图重新尝试。
        model.calls.clear()
        assert await builder.represent_unit(missing.unit, profile, Deadline(1), missing) == missing
        assert not model.calls
        changed = replace(visual, model_ref="new-image-model")
        rebuilt = await builder.represent_unit(missing.unit, replace(profile, representations=(changed,)),
                                               Deadline(1), missing)
        assert rebuilt.skipped_image_spaces is None and rebuilt.representations[0].space_id == changed.space_id
        assert [call.image_handle for call in model.calls] == ["unsupported-image"]
    asyncio.run(verify())


@pytest.mark.parametrize("role,status,code,expected", [
    ("IMAGE", 422, "IMAGE_INPUT_UNSUPPORTED", ErrorCode.IMAGE_INPUT_UNSUPPORTED),
    ("IMAGE", 422, "NATIVE_REPRESENTATION_UNAVAILABLE", None),
    ("IMAGE", 503, "IMAGE_INPUT_UNSUPPORTED", None),
    ("IMAGE", 409, "MODEL_REVISION_CHANGED", ErrorCode.MODEL_REVISION_CHANGED),
    ("TEXT", 422, "IMAGE_INPUT_UNSUPPORTED", None),
])
def test_only_confirmed_model_image_rejection_is_a_skippable_gateway_error(role, status, code, expected):
    from knowledge_algorithms.projections.contracts import ModelInput
    from knowledge_retrieval_app.adapters.knowledge_models import KnowledgeModelGateway
    from platform_clients.knowledge_computation import KnowledgeComputationClientError

    visual = replace(spec("MULTIVECTOR", "IMAGE", "IMAGE_TO_IMAGE"),
                     directions=frozenset({"IMAGE_TO_IMAGE", "TEXT_TO_IMAGE"}),
                     inference_ref="visual", binding_revision="a" * 64)
    failure = KnowledgeComputationClientError(status, code)
    client = SimpleNamespace(represent=AsyncMock(side_effect=failure))
    gateway = KnowledgeModelGateway(visual_client=client, dense_client=None, auth_context=None,
        image_reader=AsyncMock(return_value=(b"test-image", "image/png")), models={}, max_elements=8)
    value = ModelInput(role, text="查询") if role == "TEXT" else ModelInput(role, image_handle="original-image")

    with pytest.raises(AlgorithmError if expected else KnowledgeComputationClientError) as caught:
        asyncio.run(gateway.represent(value, visual))
    if expected:
        assert caught.value.code is expected
    else:
        assert caught.value is failure


def test_image_rejection_code_without_an_image_does_not_skip_text_embedding():
    async def verify():
        reader = AllowedReader((allowed(node("text", text="待生成文本向量")),))
        failure = AlgorithmError(ErrorCode.IMAGE_INPUT_UNSUPPORTED, "无效的文本拒绝", stage="projection_model")
        gateway = SimpleNamespace(represent=AsyncMock(side_effect=failure))
        builder = ProjectionBuilder(reader.binding, ByteTokenizer(), gateway)
        with pytest.raises(AlgorithmError) as caught:
            await anext(builder.build(reader, projection_profile(representations=(spec(),))))
        assert caught.value is failure
    asyncio.run(verify())


def test_image_exceeding_visual_transport_limit_is_not_sent():
    from knowledge_algorithms.projections.contracts import ModelInput
    from knowledge_retrieval_app.adapters.knowledge_models import KnowledgeModelGateway
    from platform_core.contracts.model import NATIVE_VISUAL_MAX_IMAGE_BYTES

    visual = replace(spec("MULTIVECTOR", "IMAGE", "IMAGE_TO_IMAGE"), inference_ref="visual",
                     binding_revision="a" * 64)
    client = SimpleNamespace(represent=AsyncMock())
    materialize = AsyncMock(return_value=(b"x" * (NATIVE_VISUAL_MAX_IMAGE_BYTES + 1), "image/png"))
    gateway = KnowledgeModelGateway(visual_client=client, dense_client=None, auth_context=None,
        image_reader=materialize, models={}, max_elements=8)
    with pytest.raises(AlgorithmError) as caught:
        asyncio.run(gateway.represent(ModelInput("IMAGE", image_handle="original-image", input_role="DOCUMENT"), visual))
    assert caught.value.code is ErrorCode.IMAGE_INPUT_UNSUPPORTED
    client.represent.assert_not_awaited()
