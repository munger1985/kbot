"""基础能力补全的合成验收；不使用真实数据库、模型或业务文件。"""

import asyncio
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [str(path) for path in ROOT.glob("services/*/src")]
sys.path[:0] = [str(ROOT / "tests/unit/knowledge_algorithms"), str(Path(__file__).parent)]
import test_foundation as support
from knowledge_algorithm_fixtures import FakeBackend, CharacterTokenizer, allowed, node, retrieval_profile, spec
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.content import ContentKind
from knowledge_algorithms.contracts.query import QueryInput, QueryMode
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.projections.contracts import ModelInput, UnitSource
from knowledge_algorithms.references.citations import CitationTarget, document_citation_edges
from knowledge_algorithms.retrieval.engine import RetrievalEngine
from knowledge_algorithms.retrieval.fusion import ChannelShard
from knowledge_algorithms.retrieval.selection import CoverageTarget, CandidateTargetMatch, MatchStatus
from agent_runtime.adapters.app_knowledge import AppEvidenceContentPort
from agent_runtime.composition.knowledge import build_app_knowledge_registry, build_app_knowledge_plan


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_table_rows_preserve_headers_and_do_not_split_cells(self):
        from knowledge_algorithms.projections.tables import markdown_table_spans
        from knowledge_algorithms.projections.builder import ProjectionBuilder
        from knowledge_algorithm_fixtures import projection_profile
        text = "| 名称 | 单位 |\n| --- | --- |\n| 原子行一 | ms |\n| 原子行二 | MB |"
        header, rows = markdown_table_spans(text)
        content = replace(allowed(node("table", text)),kind=ContentKind.TABLE,table_header=header,table_rows=rows)
        builder = ProjectionBuilder(FakeBackend().binding,CharacterTokenizer(),None)
        spans = builder._chunks(content,replace(projection_profile(),max_tokens=45))
        self.assertGreater(len(spans),1)
        self.assertTrue(all(group[0]==header for group in spans))
        self.assertEqual(tuple(row for group in spans for row in group[1:]),rows)


    async def test_partial_empty_channel_is_not_exhaustive_no_match(self):
        backend = FakeBackend()
        async def search(request, scope, snapshot):
            return (ChannelShard("lexical", "lexical", "space", "native", "stats", snapshot, (),
                                  exhausted=False, status_reason="TEXT_SCAN_PARTIAL"),)
        backend.search_lexical = search
        result = await RetrievalEngine(backend, None).retrieve(
            QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="none"), backend.scope, retrieval_profile())
        self.assertEqual(result.status, "PARTIAL")
        self.assertNotEqual(result.reason, "NO_MATCH")

    async def test_revalidation_is_batched_and_identity_preserving(self):
        keys = tuple(node(str(index)).key for index in range(260))
        async def read(payload, **kwargs):
            self.assertLessEqual(len(payload["evidence_keys"]), 128)
            return {"items": [{"evidence_key": key, "status": "AVAILABLE"} for key in payload["evidence_keys"]]}
        client = SimpleNamespace(owner="app-a", read_manifest=AsyncMock(side_effect=read))
        evidence = SimpleNamespace(owner_binding_ref="app-a", evidence=tuple(SimpleNamespace(key=key) for key in keys))
        port = AppEvidenceContentPort(client=client, auth_context=None, retrieval_ref="receipt", evidence_set=evidence)
        self.assertEqual(await port.revalidate(keys), keys)
        self.assertEqual(client.read_manifest.await_count, 3)

    async def test_optional_disabled_does_not_resolve_paths_or_import_plugin(self):
        from knowledge_retrieval_app.composition.knowledge_text_scan import TextScanOptions
        with patch("pathlib.Path.resolve", side_effect=AssertionError("不应探测路径")):
            self.assertIsNone(TextScanOptions(enabled=False, executable="/nonexistent").factory(lambda query: ()))
        self.assertNotIn("knowledge_text_scan.ripgrep", sys.modules)

    async def test_dense_receipt_checks_revision_and_preserves_query_document_role(self):
        from knowledge_retrieval_app.adapters.knowledge_models import KnowledgeModelGateway
        specification = spec()
        receipt = {"vector": [1., 0.], "model_revision": specification.revision,
            "processor_revision": specification.processor_revision, "input_template": specification.input_template,
            "dimension": specification.dimension, "directions": list(specification.directions),
            "similarity": specification.similarity, "normalization": specification.normalization}
        client = SimpleNamespace(represent=AsyncMock(return_value=receipt))
        gateway = KnowledgeModelGateway(visual_client=None, dense_client=client, auth_context=None, image_reader=None,
            models={specification.space_id: {"model_ref": specification.model_ref,
                                             "wire_model_field": "served_model_name"}}, max_elements=100)
        result = await gateway.represent(ModelInput("TEXT", text="合成正文", input_role="DOCUMENT"), specification)
        self.assertEqual(result.vector, (1., 0.))
        self.assertEqual(client.represent.call_args.args[0]["input_role"], "DOCUMENT")
        client.represent.return_value = {**receipt, "model_revision": "different"}
        with self.assertRaises(AlgorithmError):
            await gateway.represent(ModelInput("TEXT", text="合成查询"), specification)

    async def test_dense_service_receipt_is_from_actual_instance_and_nonempty(self):
        from model_serving.embedding.representation import dense_representation
        config = SimpleNamespace(model_revision="r1", processor_revision="p1", input_template="t1",
                                 representation_dimension=2)
        model = SimpleNamespace(config=config, embed=AsyncMock(return_value=SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[3., 4.])])))
        result = await dense_representation(model, text="合成正文", input_role="DOCUMENT")
        self.assertEqual(result["vector"], [.6, .8])
        self.assertFalse(model.embed.call_args.kwargs["is_query"])
        config.model_revision = None
        with self.assertRaises(ValueError):
            await dense_representation(model, text="x", input_role="QUERY")

    async def test_numbered_document_citation_uses_explicit_bibliography_and_source_map(self):
        a, b = allowed(node("p", "参见文献 [1]。")), node("b").key
        b = replace(b, document_version_ref="another", parse_run_ref="another-run")
        bibliography = replace(allowed(node("refs", "[1] Author. doi:10.1234/example")),
                               heading_path=("References",))
        targets = (CitationTarget(b, ("10.1234/example",)),)
        edges = document_citation_edges((a, bibliography), targets, policy_revision="v2")
        paragraph_edges = [edge for edge in edges if edge.source_key == a.key]
        self.assertEqual(len(paragraph_edges), 1)
        self.assertEqual(paragraph_edges[0].source_span.char_start, a.text.index("[1]"))
        self.assertEqual(paragraph_edges[0].target_key, b)
        self.assertFalse(document_citation_edges((a,), targets, policy_revision="v2"))

    async def test_redacted_text_citation_maps_to_original_not_shifted_view(self):
        original = node("p", "秘密删除 参见 doi:10.1234/example")
        text = original.text[5:]
        content = replace(allowed(original), text=text,
            sources=(UnitSource(0, len(text), original.key, 5, len(original.text)),))
        target = replace(node("b").key, document_version_ref="another", parse_run_ref="another-run")
        edges = document_citation_edges((content,), (CitationTarget(target, ("10.1234/example",)),),
                                        policy_revision="v2")
        self.assertEqual(edges[0].source_span.char_start, original.text.index("doi:"))

    async def test_missing_materialized_coverage_candidate_does_not_empty_other_sources(self):
        await support.DeliveryTests.setup_evidence(self)
        target = CoverageTarget("topic", "SUBQUESTION", True, "fixture", "v1")
        self.evidence = replace(self.evidence, plan=replace(self.evidence.plan, targets=(target,)),
            matches=(CandidateTargetMatch("topic", self.p, MatchStatus.MATCHED, ("fixture",), "fixture"),))
        self.port.broken = {self.p}
        result = await support.DeliveryTests.compose(self)
        self.assertEqual(result.delivery.status, "SENT")
        self.assertEqual(tuple(value.evidence_key for value in result.delivery.prepared), (self.q,))

    async def test_cancelled_model_delivery_has_unknown_audit(self):
        await support.DeliveryTests.setup_evidence(self)
        entered = asyncio.Event()
        async def pending(**kwargs):
            entered.set()
            await asyncio.Event().wait()
        self.gateway.generate = pending
        task = asyncio.create_task(support.DeliveryTests.compose(self))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.recorder.events[-1][1].status, "UNKNOWN_DELIVERY")

    async def test_runtime_registry_plan_use_only_new_contracts(self):
        from datetime import datetime, timezone, timedelta
        from agent_runtime.domain.planning import PlanValidator, PlanLimits
        registry = build_app_knowledge_registry(clients={}, context_factory=None, query_factory=None,
                                                composer_factory=None, content_port_factory=None)
        plan = build_app_knowledge_plan(objective="合成检索", expires_at=datetime.now(timezone.utc)+timedelta(minutes=2))
        PlanValidator(skill_exists=registry.contains, capability_exists=lambda *args: False,
                      public_artifact_types={"KNOWLEDGE_COMPOSITION"}).validate(plan, PlanLimits())
        self.assertEqual(plan.tasks[0].expected_outputs, ("RETRIEVAL_EVIDENCE",))
        self.assertEqual(plan.tasks[1].depends_on, ("retrieve",))


class ParserResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_markup_external_resources_rejected_before_converter(self):
        from knowledge_core.adapters.document_parsing import DoclingParsingAdapter
        from knowledge_algorithm_fixtures import source, parse_profile, run_for
        cases = (("text/markdown", "![alt](https://example.invalid/x.png)"),
                 ("text/markdown", "![alt][x]\n[x]: https://example.invalid/x.png"),
                 ("text/markdown", "<img src='https://example.invalid/x.png'>"),
                 ("text/html", "<svg><image href='https://example.invalid/x'/></svg>"),
                 ("text/html", "<meta http-equiv='refresh' content='0;url=https://example.invalid'>"),
                 ("text/html", "<img src='https://example.invalid/x'>"))
        for mime, text in cases:
            with self.subTest(mime=mime, text=text):
                reader = source(text)
                reader.descriptor = replace(reader.descriptor, mime_type=mime)
                converter = unittest.mock.Mock(side_effect=AssertionError("外部资源不应进入转换器"))
                adapter = DoclingParsingAdapter(None, converter_factory=converter)
                with self.assertRaises((AlgorithmError, ValueError)):
                    _ = [value async for value in adapter.extract(reader, parse_profile(), run_for(reader, parse_profile()))]
                converter.assert_not_called()


    async def test_uncovered_physical_page_is_kept_as_page_image_not_silently_omitted(self):
        from PIL import Image
        from knowledge_core.adapters.document_parsing import DoclingParsingAdapter, WorkerArtifactWriter
        from knowledge_algorithm_fixtures import source, parse_profile, run_for
        reader = source("synthetic-page")
        reader.descriptor = replace(reader.descriptor,mime_type="application/pdf")
        image = Image.new("RGB",(8,10))
        page = SimpleNamespace(size=SimpleNamespace(width=8,height=10),image=SimpleNamespace(pil_image=image))
        document = SimpleNamespace(pages={1:page},iterate_items=lambda:iter(()))
        converter = SimpleNamespace(convert=lambda *args,**kwargs:SimpleNamespace(status="success",document=document))
        with tempfile.TemporaryDirectory() as temporary:
            writer = WorkerArtifactWriter(temporary,expires_at=2000000000,max_bytes=10000,max_total_bytes=10000)
            adapter = DoclingParsingAdapter(writer,converter_factory=lambda:converter)
            batches = [value async for value in adapter.extract(reader,parse_profile(),run_for(reader,parse_profile()))]
        self.assertEqual(batches[0].regions[0].nodes[0].kind,ContentKind.PAGE)
        self.assertEqual(batches[0].assets[0].locator.page_no,1)
        self.assertEqual(batches[0].regions[0].nodes[0].text,"")
        image.close()


class QualityAuditTests(unittest.TestCase):
    def runner(self):
        import importlib.util
        specification = importlib.util.spec_from_file_location("knowledge_quality_audit",
            ROOT/"tests/evaluation/knowledge_algorithms/audit_quality.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    def fixtures(self):
        manifest = {"schema_version":"knowledge.quality_manifest.v1","owner":"knowledge_retrieval",
            "dataset_revision":"synthetic/v1","configuration_hash":"frozen","split":"validation",
            "cases":[{"query_id":"q1","task":"TEXT_TO_TEXT","requested_count":1,
                      "relevant_keys":["a"],"forbidden_keys":["foreign"],"answer_required":True}],
            "thresholds":{"max_p95_ms":100,"by_task":{"TEXT_TO_TEXT":{
                "recall_at_k":1,"ndcg_at_k":1,"link_recall":1,"successful":1}}}}
        row = {"query_id":"q1","owner":"knowledge_retrieval","status":"COMPLETE","duration_ms":10,
               "primary_keys":["a"],"evidence_keys":["a"],"delivered_keys":["a"],"delivery_status":"SENT",
               "readable_within_budget":True}
        return manifest,row
    def test_all_queries_and_per_task_metrics_required(self):
        manifest,row = self.fixtures()
        report = self.runner().assess(manifest,[row])
        self.assertEqual(report["gate"],"PASS")
        self.assertFalse(report["production_quality_claim"])
        with self.assertRaises(ValueError):
            self.runner().assess(manifest,[])
    def test_failed_queries_and_silent_zero_are_never_dropped(self):
        manifest,row = self.fixtures()
        row.update(status="FAILED",primary_keys=[],delivered_keys=[],delivery_status="FAILED")
        report = self.runner().assess(manifest,[row])
        self.assertEqual(report["gate"],"FAIL")
        self.assertEqual(report["silent_zero_delivery"],1)
        self.assertEqual(report["by_task"]["TEXT_TO_TEXT"]["recall_at_k"],0)


class ScannerFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_exit_two_and_malformed_output_are_failure_not_complete(self):
        import shutil
        from knowledge_text_scan.ripgrep import RipgrepTextScanner
        from knowledge_algorithms.ports.text_scan import TextScanRequest
        from unittest.mock import Mock
        backend = FakeBackend()
        request = TextScanRequest("EXACT_RECALL",("literal",),"ANY",(),10,1000,10,10000,1.)
        with tempfile.TemporaryDirectory() as temporary:
            scanner = RipgrepTextScanner(binding=backend.binding,provider=None,executable=sys.executable,
                                         temporary_root=temporary)
            for code, lines in ((2,[b""]), (0,[b"not-json\n",b""])):
                process = SimpleNamespace(returncode=code,stdout=SimpleNamespace(readline=AsyncMock(side_effect=lines)),
                    stderr=SimpleNamespace(read=AsyncMock(return_value=b"")),wait=AsyncMock(return_value=code))
                with patch("asyncio.create_subprocess_exec",AsyncMock(return_value=process)):
                    with self.assertRaises(ValueError):
                        await scanner._run(Path(temporary)/"controlled",request,timeout=1.,output_limit=10000)

    async def test_cancel_terminates_child_and_reaps_process(self):
        from knowledge_text_scan.ripgrep import RipgrepTextScanner
        from knowledge_algorithms.ports.text_scan import TextScanRequest
        backend, entered = FakeBackend(), asyncio.Event()
        request = TextScanRequest("EXACT_RECALL",("literal",),"ANY",(),10,1000,10,10000,1.)
        async def pending():
            entered.set()
            await asyncio.Event().wait()
        process = SimpleNamespace(returncode=None,stdout=SimpleNamespace(readline=pending),
            stderr=SimpleNamespace(read=AsyncMock(return_value=b"")),wait=AsyncMock(return_value=-15))
        def terminate():
            process.returncode=-15
        process.terminate = unittest.mock.Mock(side_effect=terminate)
        with tempfile.TemporaryDirectory() as temporary:
            scanner = RipgrepTextScanner(binding=backend.binding,provider=None,executable=sys.executable,
                                         temporary_root=temporary)
            with patch("asyncio.create_subprocess_exec",AsyncMock(return_value=process)):
                task = asyncio.create_task(scanner._run(Path(temporary)/"controlled",request,timeout=1.,output_limit=10000))
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        process.terminate.assert_called_once()
        process.wait.assert_awaited()


if __name__ == "__main__":
    unittest.main()
