"""跨库模型隔离、档位界线及发布表示空间回归。"""

import asyncio
import base64
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from PIL import Image

from knowledge_algorithms.contracts.content import ContentKind, ProvenanceKind
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.identity import Locator
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.contracts import ParseModelBinding, ParseStrategy
from knowledge_algorithms.projections.contracts import ModelInput
from knowledge_algorithms.projections.library_models import library_projection, published_profile
from knowledge_core.adapters.document_parsing import bound_transcription_nodes
from knowledge_core.adapters.model_transcription import AuthorizedModelTranscription, requested_binding
from knowledge_retrieval_app.adapters.knowledge_models import KnowledgeModelGateway
from knowledge_retrieval_app.composition.knowledge_profiles import KnowledgeFoundationConfig
from platform_clients.knowledge_computation import KnowledgeComputationClientError
from platform_core.contracts.knowledge_parse_approval import available_parse_bindings, bindings_for_strategy


def test_two_libraries_freeze_approved_ids_without_falling_back():
    parse, _, _ = KnowledgeFoundationConfig().profiles()
    first, second = uuid4(), uuid4()
    lookup = AsyncMock(side_effect=lambda role, identity: {
        "model_id": str(identity), "revision": "a" * 64, "inference_ref": str(identity),
    })

    async def workflow():
        current = await available_parse_bindings({"ocr": first}, lookup)
        profile = parse.for_approval(ParseStrategy.AUTO,
            ocr=ParseModelBinding(**bindings_for_strategy("AUTO", current)["ocr"]))
        changed = await available_parse_bindings({"ocr": second}, lookup)
        assert changed["ocr"]["model_id"] == str(second)
        assert requested_binding(profile, "ocr", "page").model_id == str(first)
        lookup.side_effect = TimeoutError()
        assert await available_parse_bindings({"ocr": first}, lookup) == {}
        with pytest.raises(ValueError, match="不可用"):
            bindings_for_strategy("AUTO", {})
    asyncio.run(workflow())


@pytest.mark.parametrize("strategy,region,called", [
    ("AUTO", "page", ["ocr"]), ("AUTO", "figure", ["ocr", "vlm"]), ("VISUAL", "page", ["ocr", "vlm"]),
])
def test_model_call_is_confined_to_approved_region(strategy, region, called):
    parse, _, _ = KnowledgeFoundationConfig().profiles()
    ocr = ParseModelBinding(str(uuid4()), "a" * 64, "ocr")
    vlm = ParseModelBinding(str(uuid4()), "b" * 64, "vlm")
    profile = parse.for_approval(ParseStrategy(strategy), ocr=ocr, vlm=vlm)
    model = AsyncMock(side_effect=["", "扫描表格"])
    kind = ContentKind.PAGE if region == "page" else ContentKind.FIGURE
    source = SimpleNamespace(descriptor=SimpleNamespace(
        owner_binding_ref="knowledge_retrieval", document_version_ref="1", sha256="d" * 64))
    figure = SimpleNamespace(kind=kind, locators=(Locator(page_no=1, bbox=(0., 0., 1., 1.))
        if region == "page" else Locator(page_no=1, bbox=(0., 0., .5, .5)),),
        native_ref="image", key=SimpleNamespace(object_key="source-image"), reading_order=0)
    with Image.new("RGB", (4, 4)) as image:
        result = asyncio.run(bound_transcription_nodes(image, source, SimpleNamespace(parse_run_ref="run"),
            profile, figure=figure, transcriber=SimpleNamespace(transcribe_region=model), max_nodes=10))
    assert not result.image_input_unsupported
    nodes = result.nodes
    assert [call.kwargs["role"] for call in model.await_args_list] == called
    if "vlm" in called:
        assert nodes[0].provenance.kind == ProvenanceKind.MODEL_TRANSCRIPTION
        assert nodes[0].provenance.generator == vlm.model_id
    else:
        assert nodes == ()


def test_model_revision_conflict_is_a_failed_task_not_an_empty_transcript():
    parse, _, _ = KnowledgeFoundationConfig().profiles()
    ocr = ParseModelBinding(str(uuid4()), "a" * 64, "ocr")
    profile = parse.for_approval(ParseStrategy.AUTO, ocr=ocr)
    client = SimpleNamespace(infer=AsyncMock(side_effect=KnowledgeComputationClientError(409, "MODEL_REVISION_CHANGED")))
    raw = BytesIO()
    with Image.new("RGB", (4, 4)) as image:
        image.save(raw, format="PNG")
    with pytest.raises(AlgorithmError) as error:
        asyncio.run(AuthorizedModelTranscription({"ocr": client})(profile, object(),
            {"role": "ocr", "region": "page", "image": base64.b64encode(raw.getvalue()).decode()}))
    assert error.value.code.value == "MODEL_REVISION_CHANGED"
    assert client.infer.await_args.kwargs["payload"]["expected_revision"] == ocr.revision


def test_search_uses_index_space_and_revision_not_a_new_library_binding():
    _, base, templates = KnowledgeFoundationConfig().profiles()
    left, right = str(uuid4()), str(uuid4())
    metadata = dict(kind="DENSE", input_kind="TEXT", dimension=2, model_revision="provider-weights",
        processor_revision="processor", input_template="template", normalization="L2", similarity="COSINE",
        directions=["TEXT_TO_TEXT"])

    async def resolve(role, model_id):
        return {"model_id": model_id, "revision": "a" * 64, "inference_ref": model_id, "representation": metadata}

    async def workflow():
        first = await library_projection(base, {"embedding": left}, resolve)
        second = await library_projection(base, {"embedding": right}, resolve)
        a, b = first.representations[0], second.representations[0]
        assert a.space_id != b.space_id
        assert replace(a, binding_revision="b" * 64).space_id != a.space_id
        only_first = published_profile(templates["default"], [{"generation_manifest": {"profile": encode(first)}}])
        assert [c.representation_spec for c in only_first.channels if c.representation_spec] == [a]
        assert not any(c.representation_spec for c in published_profile(templates["default"], []).channels)
        client = SimpleNamespace(represent=AsyncMock(return_value={**metadata, "vector": [1., 0.]}))
        gateway = KnowledgeModelGateway(visual_client=None, dense_client=client,
            auth_context=None, image_reader=None, models={}, max_elements=8)
        out = await gateway.represent(ModelInput("TEXT", text="query"), a)
        assert out.space_id == a.space_id
        wire = client.represent.await_args.args[0]
        assert wire["model_id"] == left
        assert wire["expected_revision"] == a.binding_revision
    asyncio.run(workflow())
