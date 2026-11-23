from io import BytesIO
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageDraw

from knowledge_retrieval_app.adapters.knowledge_image_source import image_for_verification, verifies_image
from knowledge_retrieval_app.application.knowledge_retrieval import KnowledgeRetrievalService
from knowledge_retrieval_app.domain.knowledge_retrieval import RetrievalLimits


def _encoded(image, fmt="PNG"):
    data = BytesIO()
    image.save(data, format=fmt)
    return data.getvalue()


def _diagram(kind):
    diagram = Image.new("L", (180, 138), 255)
    d = ImageDraw.Draw(diagram)
    if kind == "schematic":
        d.rounded_rectangle((24, 21, 64, 56), radius=4, outline=0, width=3)
        d.line([(64, 42), (108, 42), (108, 108)], fill=0, width=3)
        d.ellipse((91, 89, 125, 123), outline=0, width=3)
        d.line([(24, 56), (71, 114), (23, 114)], fill=0, width=3)
        d.rectangle((109, 20, 153, 64), outline=0, width=3)
        d.line([(128, 27), (149, 56)], fill=0, width=2)
    else:
        d.ellipse((21, 16, 156, 123), outline=0, width=3)
        d.polygon([(90, 24), (148, 103), (32, 103)], outline=0, width=3)
        d.line([(61, 58), (116, 88)], fill=0, width=4)
        d.rectangle((72, 50, 105, 70), outline=0, width=3)
    return diagram


@pytest.mark.parametrize("kind", ["schematic", "chart"])
def test_source_verifies_screenshot_with_caption_scale_and_crop(kind):
    source = _diagram(kind)
    screenshot = Image.new("L", (300, 240), 255)
    screenshot.paste(source.resize((198, 152), Image.Resampling.BICUBIC), (40, 17))
    ImageDraw.Draw(screenshot).text((32, 198), "Figure 2: example caption and labels", fill=0)
    query = image_for_verification(_encoded(screenshot, "JPEG"))
    assert verifies_image(query, _encoded(source), similar=False)
    cropped = image_for_verification(_encoded(source.crop((8, 8, 155, 128))))
    assert verifies_image(cropped, _encoded(source), similar=False)
    negative = _diagram("chart" if kind == "schematic" else "schematic")
    assert not verifies_image(query, _encoded(negative), similar=False)
    assert not verifies_image(query, _encoded(negative), similar=True)


def test_unreadable_or_featureless_images_do_not_pass_visual_verification():
    blank = Image.new("L", (160, 100), 200)
    query = image_for_verification(_encoded(blank))
    assert not verifies_image(query, _encoded(blank), similar=False)
    assert not verifies_image(query, b"not an image", similar=True)


def test_similar_mode_can_keep_changed_figure_without_calling_it_the_source():
    source = _diagram("schematic")
    variant = source.copy()
    drawing = ImageDraw.Draw(variant)
    drawing.rectangle((0, 14, 68, 58), fill=255)
    drawing.ellipse((5, 18, 63, 54), outline=0, width=3)
    query = image_for_verification(_encoded(source))
    assert not verifies_image(query, _encoded(variant), similar=False)
    assert verifies_image(query, _encoded(variant), similar=True)


@pytest.mark.asyncio
async def test_verifier_filters_before_returning_hits_and_does_not_fill_result_count():
    source = _diagram("schematic")
    screenshot = Image.new("L", (280, 200), 255)
    screenshot.paste(source, (30, 11))
    ImageDraw.Draw(screenshot).text((40, 165), "Chart legend", fill=0)
    query = _encoded(screenshot)
    objects = {"wrong": _encoded(_diagram("chart")), "match": _encoded(source), "exact": query}
    calls = []

    async def read(handle, *, max_bytes):
        calls.append(handle)
        assert len(objects[handle]) <= max_bytes
        return objects[handle]

    service = SimpleNamespace(store=SimpleNamespace(read=read), limits=RetrievalLimits())
    items = {name: SimpleNamespace(mime_type="image/png", content_hash=sha256(raw).hexdigest(),
                byte_count=len(raw), content_handle=name) for name, raw in objects.items()}
    backend = SimpleNamespace(items=items, image_fingerprints={"query": (sha256(query).hexdigest(), None)})
    request = SimpleNamespace(query=SimpleNamespace(image_handles=("query",)), input_index=0,
                              channel=SimpleNamespace(limit=100))
    loader = AsyncMock(return_value=(query, "image/png"))
    verify = KnowledgeRetrievalService.image_verifier(service, backend, loader, "SOURCE")
    candidates = tuple(SimpleNamespace(key=name) for name in objects)

    hits, exhausted = await verify(request, candidates)
    assert exhausted and [hit.key for hit in hits] == ["match", "exact"]
    assert calls == ["wrong", "match"], "摘要一致的原图不需要再读像素"
    await verify(request, candidates)
    assert calls == ["wrong", "match"]
    loader.assert_awaited_once()
