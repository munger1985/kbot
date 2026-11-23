"""验证装饰线、区域缺口和正常转写的边界。"""

import asyncio
import base64
from hashlib import sha256
from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
from zipfile import ZipFile

import pytest
from PIL import Image

from knowledge_algorithms.contracts.content import ContentKind, ProvenanceKind
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.contracts.identity import Locator
from knowledge_algorithms.parsing.contracts import ParseModelBinding, ParseStrategy, SourceDescriptor
from knowledge_algorithms.parsing.events import ParseRunIdentity
from knowledge_core.adapters.document_parsing import DoclingParsingAdapter, WorkerArtifactWriter
from knowledge_core.adapters.bounded_parse_executor import BoundedParseExecutor, OptionalTranscriptionBudget
from knowledge_core.adapters.model_transcription import (
    IMAGE_DIMENSIONS_ERROR,
    AuthorizedModelTranscription,
    ImageDimensionsUnsupported,
    TranscriptionBudgetExhausted,
    WorkerModelTranscriber,
)
from knowledge_retrieval_app.composition.knowledge_profiles import KnowledgeFoundationConfig
from platform_clients.knowledge_computation import KnowledgeComputationClientError

OFFICE_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def profile_for(strategy=ParseStrategy.AUTO):
    profile, _, _ = KnowledgeFoundationConfig().profiles()
    return profile.for_approval(strategy, ocr=ParseModelBinding(str(uuid4()), "a" * 64, "ocr"),
                               vlm=ParseModelBinding(str(uuid4()), "b" * 64, "vlm"))


def picture(name, *, size=(768, 2), varied=False, label="picture", semantic=False):
    image = Image.new("RGB", size, (50, 50, 50))
    image.putpixel((0, 0), (150, 150, 150))
    if varied:
        image.putpixel((size[0] // 2, size[1] // 2), (200, 200, 200))
    return SimpleNamespace(label=label, self_ref=name, meta={"description": "文字"} if semantic else None,
                           get_image=lambda doc: image)


def test_office_separator_keeps_artifact_and_other_figures_still_use_models(tmp_path):
    buffer = BytesIO()
    with ZipFile(buffer, "w"):
        pass
    data = buffer.getvalue()

    class Source:
        descriptor = SourceDescriptor("knowledge_retrieval", "doc-v1", sha256(data).hexdigest(),
                                      OFFICE_MIME, len(data), "source-1")

        async def chunks(self):
            yield data

    figures = [
        picture("separator"), picture("narrow_scan", varied=True),
        picture("chart", label="chart"), picture("captioned", semantic=True),
        picture("normal", size=(400, 200)),
    ]
    doc = SimpleNamespace(pages={}, iterate_items=lambda: ((figure, 0) for figure in figures))
    converter = SimpleNamespace(convert=lambda *args, **kwargs: SimpleNamespace(status="success", document=doc))
    writer = WorkerArtifactWriter(tmp_path, expires_at=2000000000, max_bytes=100000, max_total_bytes=100000)

    async def transcribe(image, *, role, **kwargs):
        if role == "ocr":
            return ""
        if min(image.size) <= 10:
            raise ImageDimensionsUnsupported()
        return "正常图片识别内容"

    model = AsyncMock(side_effect=transcribe)
    profile = profile_for()
    adapter = DoclingParsingAdapter(writer, converter_factory=lambda: converter,
                                    transcriber=SimpleNamespace(transcribe_region=model))

    async def extract():
        reader = Source()
        run = ParseRunIdentity("knowledge_retrieval", "job-1", "parse-1", 1, 1,
                               reader.descriptor.sha256, profile.profile_hash)
        return [batch async for batch in adapter.extract(reader, profile, run)]

    batch, = asyncio.run(extract())
    assert len(batch.assets) == len(figures) == len(writer.artifacts)
    assert [call.kwargs["role"] for call in model.await_args_list] == ["ocr", "vlm"] * 4
    coverage = {reg.region_key: reg.text_coverage for reg in batch.regions}
    assert coverage == {"separator": 1.0, "narrow_scan": 0.0, "chart": 0.0, "captioned": 0.0,
                        "normal": 1.0, "normal:model_transcription": 1.0}


def test_complex_region_retains_ocr_if_vlm_refuses_dimensions():
    from knowledge_core.adapters.document_parsing import bound_transcription_nodes

    reader = SimpleNamespace(descriptor=SimpleNamespace(
        owner_binding_ref="knowledge_retrieval", document_version_ref="doc-v1", sha256="d" * 64))
    profile = profile_for(ParseStrategy.VISUAL)
    model = AsyncMock(side_effect=["已有 OCR 文字", ImageDimensionsUnsupported()])
    fig = SimpleNamespace(kind=ContentKind.FIGURE, native_ref="complex", reading_order=0,
                          key=SimpleNamespace(object_key="figure-1"), locators=(Locator(logical_ref="complex"),))
    with Image.new("RGB", (500, 7)) as image:
        result = asyncio.run(bound_transcription_nodes(image, reader, SimpleNamespace(parse_run_ref="parse-1"),
            profile, figure=fig, transcriber=SimpleNamespace(transcribe_region=model), max_nodes=10, complex_layout=True))

    assert result.image_input_unsupported
    assert [(n.text, n.provenance.kind) for n in result.nodes] == [("已有 OCR 文字", ProvenanceKind.OCR)]


@pytest.mark.parametrize("exhaust_budget", [False, True])
def test_required_pdf_page_precedes_optional_figures_and_originals_survive(tmp_path, monkeypatch, exhaust_budget):
    from knowledge_core.adapters import document_parsing

    raw = b"%PDF-1.7 fixture"
    monkeypatch.setattr(document_parsing, "restore_pdf_word_spacing", lambda raw, regions: regions)

    class Source:
        descriptor = SourceDescriptor("knowledge_retrieval", "doc-v1", sha256(raw).hexdigest(),
                                      "application/pdf", len(raw), "source-1")

        async def chunks(self):
            yield raw

    bounds = SimpleNamespace(l=.1, t=.1, r=.3, b=.3)
    bbox = SimpleNamespace(to_top_left_origin=lambda _: SimpleNamespace(normalized=lambda _: bounds))
    on_page_one = [SimpleNamespace(page_no=1, bbox=bbox)]
    small = picture("small", size=(17, 17), varied=True)
    table_crop = picture("table_crop", size=(450, 89), semantic=True)
    later = picture("later", size=(90, 90))
    line = picture("line")
    for figure in (small, table_crop, later, line):
        figure.prov = on_page_one
    body = SimpleNamespace(label="text", self_ref="body", text="可靠原生正文", prov=on_page_one)
    scan = Image.new("RGB", (200, 300))
    size = SimpleNamespace(width=200, height=300)
    doc = SimpleNamespace(pages={1: SimpleNamespace(size=size),
                                2: SimpleNamespace(size=size, image=SimpleNamespace(pil_image=scan))},
                          iterate_items=lambda: ((n, 0) for n in (small, body, table_crop, later, line)))
    converter = SimpleNamespace(convert=lambda *args, **kwargs: SimpleNamespace(status="success", document=doc))
    profile = profile_for()
    writer = WorkerArtifactWriter(tmp_path, expires_at=2000000000, max_bytes=100000, max_total_bytes=100000)
    calls = []

    async def transcribe(image, *, role, region, **kwargs):
        calls.append((image.size, role, region))
        if region == "page":
            return "扫描页补回的文字"
        if role == "vlm":
            if exhaust_budget:
                raise TranscriptionBudgetExhausted()
            return "表格补回的文字"
        return ""

    async def extract():
        reader = Source()
        run = ParseRunIdentity("knowledge_retrieval", "job-1", "parse-1", 1, 1,
                               reader.descriptor.sha256, profile.profile_hash)
        adapter = DoclingParsingAdapter(writer, converter_factory=lambda: converter,
                                        transcriber=SimpleNamespace(transcribe_region=transcribe))
        return [batch async for batch in adapter.extract(reader, profile, run)]

    try:
        batch, = asyncio.run(extract())
    finally:
        scan.close()
    assert len(batch.assets) == len(writer.artifacts) == 5
    assert calls[:3] == [((200, 300), "ocr", "page"),
                         ((450, 89), "ocr", "figure"), ((450, 89), "vlm", "figure")]
    coverage = {reg.region_key: reg.text_coverage for reg in batch.regions}
    assert coverage["body"] == coverage["#/pages/2/image"] == coverage["line"] == 1
    if exhaust_budget:
        assert len(calls) == 3
        assert coverage["small"] == coverage["later"] == coverage["table_crop"] == 0
    else:
        assert calls[3:] == [((17, 17), "ocr", "figure"), ((90, 90), "ocr", "figure")]
        assert coverage["table_crop:model_transcription"] == 1


def test_optional_budget_cancels_only_its_window_and_keeps_real_failures():
    async def verify():
        loop = asyncio.get_running_loop()
        budget = OptionalTranscriptionBudget(timeout_seconds=0.05, deadline=loop.time() + 10)
        cancelled = asyncio.Event()

        async def blocked():
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        with pytest.raises(TranscriptionBudgetExhausted):
            await budget.run(blocked)
        assert cancelled.is_set()
        unused = AsyncMock()
        with pytest.raises(TranscriptionBudgetExhausted):
            await budget.run(unused)
        unused.assert_not_called()
        fresh = OptionalTranscriptionBudget(timeout_seconds=180, deadline=loop.time() + 180)
        with pytest.raises(TimeoutError):
            await fresh.run(AsyncMock(side_effect=TimeoutError()))
        late = OptionalTranscriptionBudget(timeout_seconds=180, deadline=loop.time() + 14)
        with pytest.raises(TranscriptionBudgetExhausted):
            await late.run(unused)

    asyncio.run(verify())
    profile = profile_for()
    assert BoundedParseExecutor._optional_request(profile, {
        "role": "ocr", "region": "figure", "image": "", "location": {
            "object": "", "page": 1, "origin": "CONVERTER_FIGURE", "width": 1, "height": 1, "optional": True},
    })
    with pytest.raises(AlgorithmError):
        BoundedParseExecutor._optional_request(profile, {
            "role": "ocr", "region": "page", "image": "", "location": {
                "object": "", "page": 1, "origin": "PAGE_IMAGE", "width": 1, "height": 1, "optional": True},
        })


def test_same_image_reuses_frozen_model_output_but_keeps_both_locations(monkeypatch):
    from io import StringIO
    from knowledge_core.adapters import model_transcription
    from knowledge_core.adapters.document_parsing import bound_transcription_nodes

    profile = profile_for()
    source = SimpleNamespace(descriptor=SimpleNamespace(mime_type="application/pdf",
        owner_binding_ref="knowledge_retrieval", document_version_ref="doc-v1", sha256="a" * 64))
    run = SimpleNamespace(parse_run_ref="parse-1")
    transcriber = WorkerModelTranscriber()
    replies = BytesIO(b'{"text":"same text"}\n')
    requests = StringIO()

    async def verify(image):
        nodes = []
        for page in (2, 4):
            fig = SimpleNamespace(kind=ContentKind.FIGURE, native_ref=f"#/pictures/{page}",
                key=SimpleNamespace(object_key=str(page)), reading_order=page,
                locators=(Locator(page, (.1, .1, .3, .3)),))
            parsed = await bound_transcription_nodes(image, source, run, profile, figure=fig,
                transcriber=transcriber, max_nodes=10, optional=True)
            nodes.extend(parsed.nodes)
        return nodes

    with monkeypatch.context() as patch:
        patch.setattr(model_transcription.sys, "stdin", SimpleNamespace(buffer=replies))
        patch.setattr(model_transcription.sys, "stdout", requests)
        with Image.new("RGB", (100, 100)) as image:
            nodes = asyncio.run(verify(image))
    request, = [json.loads(line)["model_request"] for line in requests.getvalue().splitlines()]
    assert request["location"]["page"] == 2
    assert [(node.text, node.locators[0].page_no, node.parent_key) for node in nodes] == [
        ("same text", 2, "2"), ("same text", 4, "4"),
    ]


def test_image_exceeding_transcription_transport_keeps_original_and_marks_gap(tmp_path, monkeypatch):
    from knowledge_core.adapters.document_parsing import NativeImageAdapter
    from knowledge_core.adapters import model_transcription

    buffer = BytesIO()
    with Image.new("RGB", (40, 40)) as image:
        image.save(buffer, "PNG")
    raw = buffer.getvalue()
    monkeypatch.setattr(model_transcription, "MAX_IMAGE_BYTES", len(raw) - 1)

    class Source:
        descriptor = SourceDescriptor("knowledge_retrieval", "doc-v1", sha256(raw).hexdigest(),
                                      "image/png", len(raw), "source-1")

        async def chunks(self):
            yield raw

    async def verify():
        reader, profile = Source(), profile_for()
        run = ParseRunIdentity("knowledge_retrieval", "job-1", "parse-1", 1, 1,
                               reader.descriptor.sha256, profile.profile_hash)
        writer = WorkerArtifactWriter(tmp_path, expires_at=2000000000, max_bytes=100000, max_total_bytes=100000)
        adapter = NativeImageAdapter(writer, transcriber=model_transcription.WorkerModelTranscriber())
        batch, = [value async for value in adapter.extract(reader, profile, run)]
        assert len(batch.regions) == len(batch.assets) == len(writer.artifacts) == 1
        assert batch.regions[0].text_coverage == 0.0
        assert batch.assets[0].sha256 == reader.descriptor.sha256

    asyncio.run(verify())


@pytest.mark.parametrize("role,status,code,expected", [
    ("vlm", 422, IMAGE_DIMENSIONS_ERROR, ImageDimensionsUnsupported),
    ("vlm", 500, IMAGE_DIMENSIONS_ERROR, ErrorCode.MODALITY_UNAVAILABLE),
    ("vlm", 422, "OTHER_BAD_REQUEST", ErrorCode.MODALITY_UNAVAILABLE),
    ("vlm", 409, "MODEL_REVISION_CHANGED", ErrorCode.MODEL_REVISION_CHANGED),
    ("ocr", 422, IMAGE_DIMENSIONS_ERROR, ErrorCode.MODALITY_UNAVAILABLE),
])
def test_only_confirmed_vlm_dimension_rejection_is_regional(role, status, code, expected):
    buffer = BytesIO()
    with Image.new("RGB", (400, 4)) as image:
        image.save(buffer, format="PNG")
    infer = AsyncMock(side_effect=KnowledgeComputationClientError(status, code))
    exception_type = expected if expected is ImageDimensionsUnsupported else AlgorithmError
    with pytest.raises(exception_type) as exc:
        asyncio.run(AuthorizedModelTranscription({role: SimpleNamespace(infer=infer)})(profile_for(), object(),
            {"role": role, "region": "figure", "image": base64.b64encode(buffer.getvalue()).decode("ascii")}))
    if exception_type is AlgorithmError:
        assert exc.value.code is expected
