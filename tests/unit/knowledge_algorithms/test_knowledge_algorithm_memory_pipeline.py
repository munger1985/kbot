"""从解析内存结果到投影、召回的闭环；所有持久/模型端口均在内存。"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/knowledge_algorithms/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_algorithm_fixtures import (AllowedReader, CharacterTokenizer, FakeBackend, FakeModel, allowed,
    node, original, parse_profile, projection_profile, provenance, retrieval_profile, run_for, source, spec)
from knowledge_algorithms.contracts.content import ContentKind, VisualAsset
from knowledge_algorithms.contracts.identity import Locator, SourceSpan
from knowledge_algorithms.contracts.query import QueryInput, QueryMode
from knowledge_algorithms.contracts.retrieval import ChannelSpec, EvidenceItem
from knowledge_algorithms.parsing.contracts import ExtractedRegion, ExtractionBatch
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.parsing.plain_text import PlainTextAdapter
from knowledge_algorithms.projections.builder import ProjectionBuilder
from knowledge_algorithms.retrieval.engine import RetrievalEngine
from knowledge_algorithms.retrieval.selection import PayloadCost


class MemoryPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_parse_project_retrieve_same_exact_figures_both_directions(self):
        p = node("paragraph", "如图 1-1 所示。")
        f = node("figure", "", ContentKind.FIGURE, display_label="图 1-1")
        caption = node("caption", "结构示意", ContentKind.CAPTION, parent_key=f.key.object_key)
        asset = VisualAsset(original("asset"), f.key, "controlled-image", "a" * 64, "image/png", 128,
                            Locator(logical_ref="image"), provenance())
        class Adapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                async for _ in source.chunks():
                    pass
                yield ExtractionBatch((), (ExtractedRegion("all", (p, f, caption), 1.0),), (asset,))
        backend, model = FakeBackend(), FakeModel()
        reader, profile = source(), parse_profile()
        parsed = await ParseEngine(backend.binding, (Adapter(),)).parse(reader, profile, run_for(reader, profile))
        contents = [replace(allowed(n), content_handle="controlled-image") if n.kind == ContentKind.FIGURE
                    else allowed(n) for n in parsed.nodes]
        document = AllowedReader(contents)
        batches = [batch async for batch in ProjectionBuilder(backend.binding, CharacterTokenizer(), model).build(
            document, projection_profile(representations=(spec(), spec(input_kind="IMAGE", direction="TEXT_TO_IMAGE"))))]
        units = [item.unit for batch in batches for item in batch.units]
        by_source = {unit.source_key: unit for unit in units}
        def evidence(key, sources, kind, locators, spans=()):
            image = kind in {ContentKind.FIGURE, ContentKind.IMAGE}
            return EvidenceItem(key, sources, kind, locators, provenance(), "allowed-v1", "controlled-payload",
                sha256(key.stable_key.encode()).hexdigest(), "image/png" if image else "text/plain", 128,
                PayloadCost(1, int(image), 128), 2000000000, source_spans=spans)
        for n in parsed.nodes:
            backend.items[n.key] = evidence(n.key, (n.key,), n.kind, n.locators)
        backend.items[asset.key] = evidence(asset.key, (asset.key,), ContentKind.IMAGE, (asset.locator,))
        for unit in units:
            backend.items[unit.key] = evidence(unit.key, tuple(dict.fromkeys(s.source_key for s in unit.sources))
                or (unit.source_key,), unit.kind, unit.locators,
                tuple(SourceSpan(s.source_key.object_key, s.source_start, s.source_end) for s in unit.sources))
        backend.relations = parsed.relations
        primary_text, primary_figure = by_source[p.key].key, by_source[f.key].key
        backend.rows = {"lexical": ((primary_text, 1),)}
        engine = RetrievalEngine(backend, model)
        result = await engine.retrieve(QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="结构说明"),
            backend.scope, retrieval_profile())
        self.assertEqual(result.primary_hits[0].evidence_key, primary_text)
        self.assertTrue({f.key, caption.key, asset.key} <= set(result.groups[0].linked_keys))
        image_channel = ChannelSpec("image", "cross-modal", "DENSE", "TEXT_TO_IMAGE", 1, 10, 1, True,
                                   spec(input_kind="IMAGE", direction="TEXT_TO_IMAGE"))
        backend.rows = {"image": ((primary_figure, 1),)}
        reverse = await engine.retrieve(QueryInput(QueryMode.RELEVANCE, 1, ("FIGURE",), text="结构图"),
            backend.scope, retrieval_profile(channels=(image_channel,)))
        self.assertEqual(reverse.primary_hits[0].evidence_key, primary_figure)
        self.assertTrue({p.key, caption.key, asset.key} <= set(reverse.groups[0].linked_keys))
        self.assertTrue(reverse.plan.query_representation_hashes)
        self.assertTrue(all(i.payload_status == "DEFERRED" for i in reverse.evidence))
        self.assertTrue(reverse.context_selection.selected_evidence_keys)


if __name__ == "__main__":
    unittest.main()
