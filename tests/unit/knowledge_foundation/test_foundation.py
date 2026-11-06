"""新增基础能力的受控测试；没有真实文档、模型或开发数据库访问。"""

import asyncio
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [str(path) for path in ROOT.glob("services/*/src")]
sys.path[:0] = [str(ROOT / "tests/unit/knowledge_algorithms")]
from knowledge_algorithm_fixtures import (FakeBackend, CharacterTokenizer, allowed, node, parse_profile,
    provenance, retrieval_profile, run_for, spec, source)
from knowledge_algorithms.contracts.content import ContentKind
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.identity import AppBackendBinding, SourceSpan, fingerprint
from knowledge_algorithms.contracts.query import QueryInput, QueryMode
from knowledge_algorithms.contracts.retrieval import ChannelSpec, ChannelRequest
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.ports.text_scan import (AuthorizedTextSource, TextScanRequest, lexical_family_order)
from knowledge_algorithms.projections.contracts import SearchUnit, ProjectedUnit, Representation, UnitSource
from knowledge_algorithms.references.citations import CitationTarget, canonical_identifier, resolve_citations, citation_edges
from knowledge_algorithms.references.relations import RelationType, ResolvedRelation, direct_figure_manifest
from knowledge_algorithms.retrieval.engine import RetrievalEngine
from knowledge_algorithms.retrieval.exact import IndexedUnit, ExactIndex, lexical_tokens
from knowledge_algorithms.retrieval.maxsim import TokenMatrix
from knowledge_algorithms.retrieval.selection import PayloadCost
from agent_runtime.application.knowledge_delivery import KnowledgeEvidenceComposer, ModelEvidenceReply, EvidenceDeliveryError
from agent_runtime.adapters.knowledge_model import KnowledgeEvidenceModelGateway
from knowledge_core.adapters.document_parsing import WorkerArtifactWriter, NativeImageAdapter


def unit(name, text, representations=()):
    original = node(name, text)
    value = allowed(original)
    key = replace(original.key, projection_ref="projection-1")
    return IndexedUnit(ProjectedUnit(SearchUnit(key, original.key, key.object_key, original.kind,
        value.text, "", value.sources, value.locators, value.provenance, "view", "hash", None),
        representations), (("created_at", name),))


class ExactAndCitationTests(unittest.TestCase):
    def index(self, units):
        return ExactIndex(units, max_units=100, max_text_chars=10000, max_vector_elements=10000)

    def test_unicode_and_cjk_terms(self):
        self.assertIn("图像", lexical_tokens("ＡＩ 图像"))
        self.assertIn("ai", lexical_tokens("ＡＩ 图像"))

    def test_lexical_global_order_independent_of_input_order(self):
        units = [unit("a", "数据库连接"), unit("b", "图像检索 图像检索")]
        request = ChannelRequest(QueryInput(QueryMode.RELEVANCE, 2, ("TEXT",), text="图像检索"),
                                 retrieval_profile().channels[0], None, None)
        a, b = self.index(units), self.index(list(reversed(units)))
        self.assertEqual(a.statistics_ref, b.statistics_ref)
        self.assertEqual(a.search(request), b.search(request))
        self.assertEqual(a.search(request)[0].key, units[1].key)

    def test_explicit_budget_never_truncates_corpus(self):
        with self.assertRaises(AlgorithmError):
            ExactIndex([unit("a", "x"), unit("b", "y")], max_units=1,
                       max_text_chars=100, max_vector_elements=100)

    def test_native_multivector_mask_not_mean_pooled(self):
        specification = spec("MULTIVECTOR")
        a = unit("a", "a", (Representation(specification.space_id, matrix=TokenMatrix(specification.space_id,
            ((1., 0.), (-100., 0.)), (True, False))),))
        b = unit("b", "b", (Representation(specification.space_id, matrix=TokenMatrix(specification.space_id,
            ((0., 1.),), (True,))),))
        channel = ChannelSpec("multi", "multi", "MULTIVECTOR", "TEXT_TO_TEXT", 1., 10, 1., True, specification)
        representation = Representation(specification.space_id, matrix=TokenMatrix(specification.space_id,
            ((1., 0.),), (True,)))
        request = ChannelRequest(QueryInput(QueryMode.RELEVANCE, 2, ("TEXT",), text="x"), channel, representation, None)
        hits = self.index([a, b]).search(request)
        self.assertEqual(hits[0].key, a.key)
        self.assertEqual(hits[0].score, 1.)

    def test_cross_document_exact_doi_and_reverse(self):
        a = node("a").key
        b = replace(node("b").key, document_version_ref="other-version", parse_run_ref="other-parse")
        resolved = resolve_citations("参考 doi:10.1234/ABC。", a,
            (CitationTarget(b, ("https://doi.org/10.1234/abc",)),), policy_revision="v1")
        self.assertEqual(resolved[0].status, "RESOLVED")
        edge = citation_edges(resolved)[0]
        self.assertEqual(edge.relation_type, RelationType.CITES_DOCUMENT)
        self.assertIn(b, direct_figure_manifest(a, anchor_kind="TEXT", relations=(edge,),
                                               relation_pages_exhausted=True).linked_keys)
        self.assertIn(a, direct_figure_manifest(b, anchor_kind="TEXT", relations=(edge,),
                                               relation_pages_exhausted=True).linked_keys)

    def test_ambiguous_and_missing_citation_never_guessed(self):
        a = node("a").key
        b = replace(a, document_version_ref="b")
        c = replace(a, document_version_ref="c")
        targets = (CitationTarget(b, ("doi:10.1234/x",)), CitationTarget(c, ("doi:10.1234/x",)))
        values = resolve_citations("doi:10.1234/x doi:10.1234/y", a, targets, policy_revision="v1")
        self.assertEqual([value.status for value in values], ["AMBIGUOUS", "UNRESOLVED"])
        self.assertFalse(citation_edges(values))

    def test_cross_app_and_regular_cross_version_relation_rejected(self):
        a = node("a").key
        b = replace(a, owner_binding_ref="foreign")
        with self.assertRaises(AlgorithmError):
            resolve_citations("doi:10.1234/x", a, (CitationTarget(b, ("doi:10.1234/x",)),), policy_revision="v1")
        with self.assertRaises(AlgorithmError):
            ResolvedRelation("edge", RelationType.CAPTION_OF, a, replace(a, document_version_ref="other"), "bad")

    def test_urn_keeps_explicit_case_and_unknown_scheme_rejected(self):
        self.assertEqual(canonical_identifier("urn:example:Asset"), "urn:example:Asset")
        with self.assertRaises(ValueError):
            canonical_identifier("https://arbitrary.example/title")


class ImageParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_bytes_and_artifact_are_committed_to_exact_object(self):
        from PIL import Image
        data = BytesIO()
        Image.new("RGB", (3, 2), "red").save(data, "PNG")
        raw = data.getvalue()
        reader = source("x", owner="knowledge_retrieval")
        reader.descriptor = replace(reader.descriptor, mime_type="image/png", byte_count=len(raw),
                                    sha256=sha256(raw).hexdigest())
        async def chunks():
            yield raw
        reader.chunks = chunks
        profile = parse_profile()
        with tempfile.TemporaryDirectory() as temporary:
            writer = WorkerArtifactWriter(temporary, expires_at=2000000000, max_bytes=10000, max_total_bytes=20000)
            def artifacts(kind, payload):
                return tuple(writer.artifacts[item["local_key"]] for item in payload.get("objects", ())
                             if item.get("local_key") in writer.artifacts)
            engine = ParseEngine(AppBackendBinding("knowledge_retrieval", "knowledge_retrieval"),
                                  (NativeImageAdapter(writer),), artifact_provider=artifacts)
            events = [event async for event in engine.parse_events(reader, profile, run_for(reader, profile))]
            self.assertEqual(events[-1].event_type.value, "RUN_COMPLETED")
            artifact = next(artifact for event in events for artifact in event.artifact_handles)
            copied = (Path(temporary) / ("artifact-" + artifact.sha256)).read_bytes()
            self.assertEqual(sha256(copied).hexdigest(), artifact.sha256)
            self.assertEqual(len(copied), artifact.byte_count)

    async def test_pixel_budget_rejected_before_materializing_large_image(self):
        from PIL import Image
        data = BytesIO()
        Image.new("RGB", (4, 4)).save(data, "PNG")
        raw = data.getvalue()
        from knowledge_core.adapters.document_parsing import checked_image
        with self.assertRaises(AlgorithmError):
            checked_image(raw, max_pixels=8)


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def setup_evidence(self, *, image=False):
        backend = FakeBackend()
        self.raw = b"synthetic-image" if image else "合成证据".encode()
        self.other_raw = b"another source"
        p = backend.add("p", ContentKind.FIGURE if image else ContentKind.PARAGRAPH)
        q = backend.add("q")
        for key, raw in ((p, self.raw), (q, self.other_raw)):
            backend.items[key] = replace(backend.items[key], byte_count=len(raw),
                content_hash=sha256(raw).hexdigest(), estimated_cost=PayloadCost(1, int(key == p and image), len(raw)))
        backend.rows = {"lexical": ((p, 2), (q, 1))}
        self.evidence = await RetrievalEngine(backend, None).retrieve(
            QueryInput(QueryMode.RELEVANCE, 2, ("MIXED",), text="x"), backend.scope,
            retrieval_profile(payload_budget=PayloadCost(10000, 10, 10000)))
        self.p, self.q = p, q
        outer = self
        class Port:
            owner = backend.binding.binding_ref
            broken, revoked = set(), set()
            calls = 0
            async def content(inner, key, max_bytes):
                if key in inner.broken:
                    raise OSError("合成读取失败")
                yield outer.raw if key == p else outer.other_raw
            async def revalidate(inner, keys):
                inner.calls += 1
                return tuple(key for key in keys if key not in inner.revoked)
        class Gateway:
            calls = []
            async def generate(inner, **kwargs):
                inner.calls.append(kwargs)
                return ModelEvidenceReply("可核对来源 [D1]", ("D1",), 1000)
        class Recorder:
            events = []
            async def record(inner, run, turn, kind, value):
                inner.events.append((kind, value))
        self.port, self.gateway, self.recorder = Port(), Gateway(), Recorder()
        self.composer = KnowledgeEvidenceComposer(tokenizer=CharacterTokenizer(), image_cost=lambda *args: 8,
            model_gateway=self.gateway, artifact_recorder=self.recorder,
            max_payload_bytes=10000, max_materialized_items=20, max_cover_states=1000)

    async def compose(self, **kwargs):
        return await self.composer.compose(run_ref="run-1", turn_ref="turn-1", model_snapshot_ref="model/v1",
            evidence_set=self.evidence, content_port=self.port, question="问题", budget=PayloadCost(10000,10,10000),
            **kwargs)

    async def test_actual_text_bytes_sent_and_ordered_citations_audited(self):
        await self.setup_evidence()
        result = await self.compose()
        self.assertEqual(result.delivery.status, "SENT")
        self.assertEqual(self.gateway.calls[0]["evidence_parts"][0]["data"], self.raw)
        self.assertEqual([value.status for kind,value in self.recorder.events if kind == "MODEL_EVIDENCE_DELIVERY"],
                         ["PREPARED", "SENT"])
        self.assertEqual(result.answer.used_labels, ("D1",))

    async def test_image_only_payload_does_not_need_text_gate(self):
        await self.setup_evidence(image=True)
        result = await self.compose()
        self.assertEqual(result.delivery.status, "SENT")
        self.assertEqual(self.gateway.calls[0]["evidence_parts"][0]["kind"], "IMAGE")

    async def test_one_materialization_failure_preserves_other_sources(self):
        await self.setup_evidence()
        self.port.broken = {self.p}
        result = await self.compose()
        self.assertEqual(result.delivery.status, "SENT")
        self.assertEqual(self.gateway.calls[0]["evidence_parts"][0]["data"], self.other_raw)
        self.assertEqual(result.answer.coverage_status, "PARTIAL")

    async def test_malformed_model_receipt_is_unknown_not_prepared(self):
        await self.setup_evidence()
        async def malformed(**kwargs):
            return {}
        self.gateway.generate = malformed
        result = await self.compose()
        self.assertEqual(result.delivery.status, "UNKNOWN_DELIVERY")

    async def test_send_failure_not_zero_retrieval(self):
        await self.setup_evidence()
        async def failed(**kwargs):
            raise EvidenceDeliveryError("FAILED")
        self.gateway.generate = failed
        result = await self.compose()
        self.assertEqual(result.delivery.status, "FAILED")
        self.assertIsNone(result.answer)
        self.assertTrue(self.evidence.primary_hits)

    async def test_forged_citation_rejected_after_truthful_sent_record(self):
        await self.setup_evidence()
        async def forged(**kwargs):
            return ModelEvidenceReply("伪造 [D999]", ("D999",), 1000)
        self.gateway.generate = forged
        with self.assertRaises(AlgorithmError):
            await self.compose()
        self.assertEqual(self.recorder.events[-1][1].status, "SENT")

    async def test_search_only_does_not_materialize_or_call_model(self):
        await self.setup_evidence()
        result = await self.compose(search_only=True)
        self.assertEqual(result.delivery.status, "NOT_APPLICABLE")
        self.assertFalse(self.gateway.calls)

    async def test_second_revalidation_reselects_remaining_valid_group(self):
        await self.setup_evidence()
        async def revalidate(keys):
            self.port.calls += 1
            return tuple(key for key in keys if self.port.calls == 1 or key != self.p)
        self.port.revalidate = revalidate
        result = await self.compose()
        self.assertEqual(result.delivery.status, "SENT")
        self.assertEqual(self.gateway.calls[0]["evidence_parts"][0]["data"], self.other_raw)

    async def test_http_model_payload_has_inline_bytes_not_preview_url(self):
        import base64
        class Client:
            payload = None
            async def json(inner, *args, **kwargs):
                inner.payload, inner.path = kwargs["payload"], args[1]
                return {"choices":[{"message":{"content":"图片 [D1]"}}]}
        client = Client()
        gateway = KnowledgeEvidenceModelGateway(client=client, auth_context=None, model_snapshot_ref="m",
            model_payload={"served_model_name":"fixture", "max_tokens":77}, system_prompt="合成受控提示",
            supports_images=True, clock=lambda:1000)
        result = await gateway.generate(question="图里是什么", model_snapshot_ref="m",
            evidence_parts=({"label":"D1", "kind":"IMAGE", "data":b"image", "mime_type":"image/png"},))
        content = client.payload["messages"][1]["content"]
        self.assertEqual(content[-1]["image_url"]["url"], "data:image/png;base64," + base64.b64encode(b"image").decode())
        self.assertEqual(client.payload["max_tokens"],77)
        self.assertEqual(client.path,"/internal/v1/inference")
        from model_serving.vlm.schema import VLMRequest
        from uuid import uuid4
        payload = dict(client.payload)
        if not (ROOT/"database/oracle").is_dir():
            payload["model_id"] = uuid4()
            payload.pop("served_model_name")
        VLMRequest.model_validate(payload)
        self.assertEqual(result.used_labels,("D1",))


class ScannerTests(unittest.IsolatedAsyncioTestCase):
    async def scan(self, *, text="中文错误 ORA-123 和 --pre=x", patterns=("ORA-123",), candidates=(), mode="EXACT_RECALL",
                   max_bytes=10000, max_matches=100):
        if not shutil.which("rg"):
            self.skipTest("未安装独立可选 rg")
        from knowledge_text_scan.ripgrep import RipgrepTextScanner
        value = allowed(node("scan",text))
        key = replace(value.key, projection_ref="projection")
        source_value = AuthorizedTextSource(key,text,value.sources,sha256(text.encode()).hexdigest(),
                                           fingerprint(encode(value.sources)))
        backend = FakeBackend()
        class Provider:
            async def validate(inner,*args):
                return None
            async def sources(inner,*args):
                yield source_value
        with tempfile.TemporaryDirectory() as temporary:
            scanner = RipgrepTextScanner(binding=backend.binding, provider=Provider(),executable=shutil.which("rg"),
                                         temporary_root=temporary)
            request = TextScanRequest(mode, patterns,"ANY",candidates,100,max_bytes,max_matches,100000,2.)
            return await scanner.scan(request, backend.scope, backend.fixed_snapshot)

    async def test_unicode_byte_offsets_map_to_original_characters(self):
        result = await self.scan()
        self.assertTrue(result.exhaustive)
        match = result.matches[0]
        self.assertEqual(match.byte_start,len("中文错误 ".encode()))
        self.assertEqual(match.source_span.char_start,len("中文错误 "))

    async def test_pattern_option_injection_is_literal(self):
        result = await self.scan(patterns=("--pre=x",))
        self.assertEqual(len(result.matches),1)

    async def test_candidate_scan_does_not_widen_scope(self):
        result = await self.scan(mode="WITHIN_CANDIDATES")
        self.assertTrue(result.exhaustive)
        self.assertFalse(result.matches)

    async def test_budget_is_partial_not_complete_no_match(self):
        result = await self.scan(max_bytes=1)
        self.assertFalse(result.exhaustive)
        self.assertEqual(result.status,"PARTIAL")

    async def test_fulltext_and_scanner_get_one_lexical_rank(self):
        result = await self.scan(patterns=("ORA-123","--pre=x"))
        key = result.matches[0].key
        other = replace(key, object_key="another")
        ranked = lexical_family_order((other,key),result.matches,pattern_count=2,combination="ALL")
        self.assertEqual(ranked,(key,other))



class NativeModelAndOCRTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_visual_text_and_image_keep_token_mask(self):
        from model_serving.visual.model.colqwen2_local import ColQwen2Embedding, ColQwen2EmbeddingConfig
        class Tensor:
            def __init__(inner, value):
                inner.value = value
                def shape(item):
                    return (len(item),) + shape(item[0]) if isinstance(item,list) else ()
                inner.shape = shape(value)
                inner.ndim = len(inner.shape)
            def __getitem__(inner,key):
                return Tensor(inner.value[key])
            def to(inner,*args):
                return inner
            def detach(inner):
                return inner
            def cpu(inner):
                return inner
            def float(inner):
                return inner
            def tolist(inner):
                return inner.value
        processor = SimpleNamespace(process_queries=lambda queries:{"attention_mask":Tensor([[1,0]])})
        config = ColQwen2EmbeddingConfig(model_name="fixture",provider="fixture",dimension=2,
            model_revision="weights-v1",processor_revision="processor-v1",input_template="raw",
            retrieval_directions=("TEXT_TO_IMAGE",))
        model = ColQwen2Embedding(config)
        model._processor = processor
        class FakeModel:
            device = "cpu"
            def __call__(inner,**kwargs):
                return Tensor([[[2.,0.],[0.,0.]]])
        model._model = FakeModel()
        with patch.dict(sys.modules,{"torch":SimpleNamespace(inference_mode=lambda:nullcontext())}):
            result = await model.represent(input_kind="TEXT",text="图像问题",direction="TEXT_TO_IMAGE")
        self.assertEqual(result["vectors"],[[1.,0.],[0.,0.]])
        self.assertEqual(result["mask"],[True,False])
        self.assertEqual(result["model_revision"],"weights-v1")
        with self.assertRaises(ValueError):
            await model.embed("unused")
        with self.assertRaises(ValueError):
            await model.represent(input_kind="TEXT",text="x",direction="IMAGE_TO_IMAGE")

    async def test_local_ocr_preserves_word_coordinates_and_budget(self):
        from knowledge_core.adapters.local_ocr import parse_tsv
        header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        row = "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t92\t合成识别\n"
        values = parse_tsv(header+row,width=100,height=100,max_lines=10,max_text_bytes=100)
        self.assertEqual(values[0].bbox,(.1,.2,.4,.3))
        self.assertEqual(values[0].text,"合成识别")
        with self.assertRaises(AlgorithmError):
            parse_tsv(header+row,width=20,height=20,max_lines=10,max_text_bytes=100)

    async def test_image_ocr_is_transcription_with_explicit_parent_relation(self):
        from PIL import Image
        from knowledge_core.adapters.local_ocr import OCRLine
        data = BytesIO()
        Image.new("RGB",(10,10)).save(data,"PNG")
        raw = data.getvalue()
        reader = source("x",owner="knowledge_retrieval")
        reader.descriptor = replace(reader.descriptor,mime_type="image/png",byte_count=len(raw),sha256=sha256(raw).hexdigest())
        async def chunks():
            yield raw
        reader.chunks = chunks
        class OCR:
            revision = "synthetic-ocr/v1"
            async def transcribe(inner,image,profile):
                return (OCRLine("图内原生转录",(.1,.1,.5,.5)),)
        with tempfile.TemporaryDirectory() as temporary:
            writer = WorkerArtifactWriter(temporary,expires_at=2000000000,max_bytes=10000,max_total_bytes=20000)
            engine = ParseEngine(AppBackendBinding("knowledge_retrieval","knowledge_retrieval"),
                                  (NativeImageAdapter(writer,transcriber=OCR()),))
            profile = parse_profile()
            parsed = await engine.parse(reader,profile,run_for(reader,profile))
        text = next(value for value in parsed.nodes if value.text)
        figure = next(value for value in parsed.nodes if value.kind == ContentKind.FIGURE)
        self.assertEqual(text.provenance.kind.value,"OCR")
        self.assertEqual(text.parent_key,figure.key.object_key)
        manifest = direct_figure_manifest(text.key,anchor_kind="TEXT",relations=parsed.relations,relation_pages_exhausted=True)
        self.assertIn(figure.key,manifest.linked_keys)
        self.assertTrue(any(value.kind == RelationType.HAS_ASSET for value in manifest.relations
                            if hasattr(value,"kind")) or any(value.relation_type == RelationType.HAS_ASSET
                                                             for value in manifest.relations))

    async def test_docling_normalization_uses_exact_caption_parent_without_loading_models(self):
        from PIL import Image
        from knowledge_core.adapters.document_parsing import DoclingParsingAdapter
        reader = source("synthetic pdf",owner="knowledge_retrieval")
        reader.descriptor = replace(reader.descriptor,mime_type="application/pdf")
        items = {}
        class Ref:
            def __init__(inner,ref):
                inner.cref = ref
            def resolve(inner,document):
                return items[inner.cref]
        paragraph = SimpleNamespace(self_ref="#/texts/0",label="text",text="参见图 1-1 所示。",prov=[],parent=None)
        caption = SimpleNamespace(self_ref="#/texts/1",label="caption",text="图 1-1 结构示意",prov=[],parent=None)
        picture = SimpleNamespace(self_ref="#/pictures/0",label="picture",prov=[],parent=None,
            captions=[Ref("#/texts/1")],get_image=lambda document:Image.new("RGB",(3,2)))
        for item in (paragraph,caption,picture):
            items[item.self_ref] = item
        document = SimpleNamespace(pages={},iterate_items=lambda:iter([(paragraph,0),(caption,0),(picture,0)]))
        class Converter:
            def convert(inner,*args,**kwargs):
                return SimpleNamespace(status="success",document=document)
        with tempfile.TemporaryDirectory() as temporary:
            writer = WorkerArtifactWriter(temporary,expires_at=2000000000,max_bytes=10000,max_total_bytes=20000)
            adapter = DoclingParsingAdapter(writer,converter_factory=Converter)
            engine = ParseEngine(AppBackendBinding("knowledge_retrieval","knowledge_retrieval"),(adapter,))
            profile = parse_profile(timeout_seconds=15.)
            parsed = await engine.parse(reader,profile,run_for(reader,profile))
        self.assertTrue(any(relation.relation_type == RelationType.CAPTION_OF for relation in parsed.relations))
        self.assertTrue(any(relation.relation_type == RelationType.REFERS_TO for relation in parsed.relations))
        self.assertEqual(len(parsed.assets),1)


if __name__ == "__main__":
    unittest.main()
